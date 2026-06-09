# Praetor Platform — Overview

This document set breaks down the **Local LLM Agent Platform** plan into individual phase files for tracking and execution. Below is the complete architecture summary from `plan.md`.

## Intent

A multi-agent platform where different agents collaborate on tasks, share memory, and call local models — triggered automatically from external systems (Vikunja, GitHub, cron, eventually chat/voice). No custom agent loops, no custom harness code, no custom dispatcher.

## Full Stack

| Layer | Tool | Stateful? | Where |
|-------|------|-----------|-------|
| Inference + queue | LiteLLM | No | k3s |
| Dispatch + triggers + tracking | Hatchet (Lite) | No (state in PG) | k3s |
| Agent harness + orchestration | PydanticAI + Pydantic Graph | No | inside Hatchet workers on k3s |
| Cross-session memory | Mem0 server | No (state in Qdrant + PG) | k3s |
| Vector store | Qdrant | **Yes** | Mac Mini — core stack |
| Relational state | PostgreSQL | **Yes** | Mac Mini — core stack (existing) |

**Stateful services live on Mac Mini. Everything else is a stateless k3s Deployment.**

Qdrant and Hatchet/Mem0 schemas both live in Mac Mini's existing PostgreSQL — separate databases, one server.

## Why These Choices

### Hatchet (not a custom dispatcher)

Hatchet is an open-source AI agent orchestration engine. It is exactly the dispatcher needed without writing one:

- **Triggers built-in:** cron schedules, webhooks (GitHub PR → worker), events API (any interface pushes an event, Hatchet dispatches the right worker), inter-service calls
- **Durable execution:** tasks survive worker crashes and restart from where they left off
- **Task tracking DB:** every run has status, logs, input, output, duration stored in PostgreSQL
- **Web UI:** full task history, retries, cancellation
- **Retry policies, timeouts, concurrency limits** — all config, no code
- **Hatchet Lite** = single Docker image + PostgreSQL only. No RabbitMQ, no Kafka.
- **Python SDK:** workers are decorated functions — minimal code

The "dispatcher" is now Hatchet. Writing one is off the table.

### PydanticAI (inside Hatchet workers)

- Native LiteLLM support
- 48% fewer tokens than CrewAI for equivalent tasks
- `Pydantic Graph` handles multi-agent state passing within a pipeline
- Stateless — fits naturally inside a Hatchet worker
- V1 stable API (Sep 2025), MIT licensed

### Mem0 + Qdrant

- Qdrant: stateful vector store, Docker on Mac Mini core stack
- Mem0: stateless server on k3s, backed by Qdrant + PostgreSQL
- All agents hit one Mem0 HTTP endpoint — concurrent writes handled by Mem0
- MCP server available: agents use `search_memory`/`add_memory` as native tools
- No Neo4j (Graphiti ruled out — requires Neo4j v5.26+)

## Trigger Architecture

```
External events
──────────────────────────────────────────────────
  Vikunja (webhook)     GitHub PR webhook   Any future interface
         │                    │                (chat, voice, API)
         │                    │                      │
         └────────────────────┴──────────────────────┘
                              │
                           Hatchet
                  ┌─── Cron scheduler ────────────────┐
                  │    Webhook receiver                │
                  │    Events API                      │
                  │    Task DB (PostgreSQL)             │
                  │    Web UI (history, logs, retries) │
                  └───────────────────────────────────┘
                              │
               ┌───────────────┼───────────────┐
               │               │               │
         Research           Coder          PR Reviewer
         Worker             Worker          Worker
               │               │               │
               └───────────────┴───────────────┘
                              │
                         PydanticAI
                    (harness, tool dispatch,
                     Pydantic Graph for multi-agent)
                              │
                     ┌─────────┴─────────┐
                     │                   │
                 LiteLLM              Mem0
                (inference)          (memory)
                     │                   │
            Model containers          Qdrant
```

### How triggers work in Hatchet

**Vikunja webhook** — Vikunja POSTs to `https://praetor.amer.dev/webhooks/vikunja` on `task.updated` and `task.created` events (Vikunja has no `task.label.added` event — label changes arrive as `task.updated`). A FastAPI adapter validates the `X-Vikunja-Signature` HMAC-SHA256, inspects the task's current label list, and pushes the appropriate Hatchet event. If both `ai-research` and `ai-go` labels are present, it pushes `pipeline:research_code` directly. Idempotency key `vikunja-{task_id}-{label_id}` prevents duplicate runs.

**GitHub PR webhook** — Hatchet receives the GitHub webhook directly, dispatches to PR Reviewer or Pipeline workers based on event type and labels. Same pattern as Vikunja but using GitHub's HMAC validation (`X-Hub-Signature-256`).

**Cron** — registered in code at worker startup. E.g., every 30s → push `research:poll` event with `task_id`, every 5min → `test:ping`. Cron expression as env var, configurable via ConfigMap.

## Where Things Live

### Mac Mini (stateful core stack)

- **PostgreSQL** — existing installation; hosts Qdrant + Hatchet/Mem0 schemas in separate databases
- **Qdrant** — vector store at `10.100.20.18:6333`, Docker via Komodo GitOps
- **Komodo** — manages stateful services (Qdrant, PostgreSQL) from `komodo-dean-gitops`

### k3s cluster (ArgoCD via k3s-dean-gitops)

New Deployments:
- **LiteLLM** — inference gateway
- **Hatchet Lite** — dispatch engine, UI, cron, webhooks (points at Mac Mini PG)
- **Mem0 server** — memory API (points at Mac Mini Qdrant + PG)
- **Agent workers** — one Deployment per agent type, pull tasks from Hatchet

## Phases Overview

| Phase | Name | Description | File |
|-------|------|-------------|------|
| 0 | Deployment Foundation | App-factory, infra-mcp, bws-mcp, GitHub apps, agent.md templates | [phase-0.md](./phase-0.md) |
| 1 | Inference | LiteLLM on k3s, model routing, runner config | [phase-1.md](./phase-1.md) |
| 2 | Storage | Qdrant on Mac Mini core stack via Komodo | [phase-2.md](./phase-2.md) |
| 3 | Dispatch | Hatchet Lite on k3s, stub worker, cron | [phase-3.md](./phase-3.md) |
| 4 | Memory | Mem0 server on k3s, memory tools for agents | [phase-4.md](./phase-4.md) |
| 5 | Research Agent | PydanticAI research agent via Vikunja webhooks | [phase-5.md](./phase-5.md) |
| 6 | Coder Agent | PydanticAI coder agent, sandbox, PR creation | [phase-6.md](./phase-6.md) |
| 7 | PR Reviewer + QA | GitHub events drive agents automatically | [phase-7.md](./phase-7.md) |
| 8 | Multi-Agent Pipeline | Pydantic Graph DAG combining research + code | [phase-8.md](./phase-8.md) |

## Non-Goals

This platform does **not** require:

- No custom dispatcher code
- No custom agent loop
- No custom task tracking DB schema
- No RabbitMQ / Kafka
- No Neo4j
- No n8n (Hatchet handles cron, webhooks, events natively)
- No Kubernetes-specific operator (Hatchet Lite is a single Docker image)
