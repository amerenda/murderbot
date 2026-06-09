# Phase 5 — Research Agent

**Goal:** Tag a Vikunja task `ai-research` → agent runs automatically → report in Mem0 → Vikunja task marked done.

## Pre-conditions

- Phases 1–4 complete and healthy
- SearXNG reachable from k3s at `https://searxng.amer.dev` (already deployed)
- `VIKUNJA_TOKEN` added to BWS manually (generate=false — it is a Vikunja API token, not a generated password). Add `vikunja-token` to BWS, then add `VIKUNJA_TOKEN` secret ref to `praetor.toml`

## What Gets Built — `praetor/agents/research/`

- `agent.py` — PydanticAI agent with tools:
  - `web_search(query)` → calls SearXNG REST API
  - `add_memory` / `search_memory` → Mem0 MCP tools
  - `update_vikunja_task(task_id, status, comment)` → calls Vikunja API
- `worker.py` — Hatchet worker handling `agent:research` events
- `Dockerfile`

## Trigger: Vikunja Webhook (Not Polling)

Vikunja does not have a `task.label.added` event. Label changes arrive as `task.updated`. The webhook adapter must detect label changes:

### 1. Webhook Registration Is IaC

Declared in `praetor.toml` as a `vikunja_webhooks` block, registered by `provision_app` via Tofu's `http` provider:

```toml
[[vikunja_webhooks]]
project_id = 21          # Mycroft project
target_url  = "https://praetor.amer.dev/webhooks/vikunja"
events      = ["task.updated", "task.created"]
secret_bws_key = "vikunja-webhook-secret"   # generate=true; Tofu creates and stores it
```

Tofu calls `PUT /api/v1/projects/{id}/webhooks` with the generated secret. Running `provision_app("praetor")` again is idempotent — it upserts the webhook registration.

### 2. Webhook Adapter

The adapter endpoint lives in `praetor/webhooks/vikunja.py` (a small FastAPI router mounted alongside the workers). It:
- Validates `X-Vikunja-Signature` against `VIKUNJA_WEBHOOK_SECRET` from env
- Inspects `data.task.labels` in the payload for known label IDs (14 = `ai-research`, 11 = `ai-go`, 13 = `ai-plan-only`)
- For each matching label, pushes the corresponding Hatchet event with `idempotency_key=f"vikunja-{task_id}-{label_id}"`
- If a task has BOTH labels 14 and 11 → pushes `pipeline:research_code` instead of individual events

### 3. Webhook Adapter Deployment

The webhook adapter is a component in `praetor.toml` — a lightweight FastAPI service exposed via Ingress at `https://praetor.amer.dev/webhooks/vikunja`. It does NOT run inside a Hatchet worker.

Webhook secret stored in BWS as `vikunja-webhook-secret` (generate=true in TOML). Tofu owns creation; `provision_app` handles registration.

### Deduplication

Events are pushed with `idempotency_key=f"vikunja-{task_id}-{label_id}"`. Duplicate `task.updated` deliveries for the same label do not create duplicate runs — Hatchet deduplicates on this key automatically.

**Label-to-event mapping:**
| Label ID | Label Name | Hatchet Event | Worker |
|----------|------------|---------------|--------|
| 14 | `ai-research` | `agent:research` | Research worker |
| 13 | `ai-plan-only` | `agent:plan` (future) | — |

**Ready conditions (all must pass):**
1. Create a Vikunja task with label `ai-research` → Hatchet UI shows `agent:research` run triggered within 5 seconds
2. Research agent completes and writes findings to Mem0 under `task-{id}` namespace
3. Vikunja task is marked done, comment contains the research summary or link to memory
