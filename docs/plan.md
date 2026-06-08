# Local LLM Agent Platform — Plan

## Current State

```
opencode → llama-proxy (:8089) → llama-server (:8088) → GPU (Qwen3.6-35B-A3B, 24GB VRAM)
```

Single machine, single model, single agent. LiteLLM install script exists but not deployed. Komodo GitOps directory at `~/komodo-dean-gitops/`.

---

## Target Architecture

```
                        ┌─────────────────────────────────────┐
                        │       LiteLLM Proxy (Inference)     │
                        │      Unified OpenAI-compatible API  │
                        └──┬──────────┬──────────┬───────────┘
                           │          │          │
                    ┌──────▼──┐  ┌────▼────┐  ┌──▼───┐
                    │ Runner A│  │ Runner B│  │Runner C│   ← any number of runners
                    │ (Docker)│  │(Docker) │  │(Docker)│
                    └─────────┘  └─────────┘  └───────┘
                           ▲          ▲           │
                        ┌────── Unified Queue / Scheduler (llm-manager) ──────┐
                        │  Priority queuing, load balancing, rate limiting    │
                        └─────────────────────────────────────────────────────┘

   ┌──────────┐     ┌──────────────┐     ┌────────────────┐
   │ Agent A  │────▶│ Unified Memory│◀────│ Skills Registry │
   │ (Docker) │     │   (MemoryOS) │     │   (JSON schema)  │
   └──────────┘     └──────────────┘     └────────────────┘
        │
   ┌──────────┐
   │ Agent B  │ ◀── agent-to-agent via shared memory
   │ (Docker) │
   └──────────┘
```

---

## Layer 1: Inference — LiteLLM Proxy

**Decision:** Deploy LiteLLM as the single entry point for all inference.

- Drop-in OpenAI-compatible API gateway, routes to any backend
- Handles auth, rate limiting, per-model routing
- Install script exists at `install-litellm.sh`
- One persistent container; config defines all models and backends
- All agents connect only to the LiteLLM endpoint — never directly to runners
- Adding a new runner = update config + restart LiteLLM

---

## Layer 2: Unified Queue / Scheduler — llm-manager

**Decision:** Start simple with LiteLLM's built-in routing/rate-limiting. If queue contention becomes real (multiple agents competing for the same GPU), layer in an external scheduler later.

Research findings on candidates:

| Option | Fit | Notes |
|--------|-----|-------|
| **llm-d** (Red Hat) | Strong | Predicted-latency scheduling, KV cache management, P/D disaggregation, autoscaling. Full stack but tied to Red Hat/K8s ecosystem. |
| **Kthena** (CNCF/Volcano) | Moderate | KV cache-aware, fairness scheduling, rate limiting. Newer/smaller community. Bundled into Volcano — adds complexity. |
| **Gateway API Inference Extension** | Emerging | CNCF standard but still maturing; not ready standalone. |

---

## Layer 3: Model Execution — Docker Containers

Each model runs as an independent always-on Docker container.

- One model per runner container. Switching models = stop old, start new, update LiteLLM config.
- Inference engine per model (llama.cpp for GGUF, vLLM/TGI depending on model size)
- `restart: unless-stopped` or managed by container orchestrator
- Container isolation — one crashing doesn't affect others

---

## Layer 4: Unified Memory — MemoryOS

**Decision:** Use MemoryOS as the shared memory layer. Fallback to Mem0 if stability issues arise.

MemoryOS provides a hierarchical model mapping well to multi-agent workflows:

| Level | Purpose | Agent use case |
|-------|---------|---------------|
| **Short-term** | Sliding window of recent conversation | Current task context — what the agent is working on right now |
| **Mid-term** | Transitional knowledge bridging sessions | Partial results from previous agents in a pipeline |
| **Long-term** | Persistent profiles + knowledge base | Shared facts, established patterns, reusable outputs across chains |

**Integration pattern:**
- MemoryOS runs as its own container (Python service)
- Exposes an API for agents to read/write memory
- Each agent has a namespace/identity so memories are attributable
- Output from Agent A → write to mid-term with tag `pipeline:task-X` → Agent B reads that tag

---

## Layer 5: Agent Model — Simple Task Execution

Agents follow a strict contract pattern:

```
INPUT:  { context, task, memory_refs }
OUTPUT: { result, memory_writes, next_agent_hint }
```

Each agent is a Docker container containing:
- Lightweight runtime (Python) that reads from stdin / API
- Calls the LLM via LiteLLM with the contract prompt
- Writes structured output to stdout / API
- Optionally writes to shared memory (MemoryOS)

**Communication:** Agent-to-agent via MemoryOS mid-term layer. No direct agent-to-agent calls — everything goes through shared memory. This gives decoupling, observability, and reusability.

### Orchestration: Centralized Orchestrator

A single process manages the pipeline. It reads a definition, launches agents in order, reads outputs from memory, and decides what happens next.

**Why centralized over agent-driven:**
- Lowest complexity — one orchestrator file, agents are dumb containers
- Highest debuggability — linear execution flow, easy to trace who did what when
- Clean failure handling — orchestrator catches crashes and decides retry/fallback
- Best fit for local model latency (1-5s/call) — no async event loop overhead

**Migration path:** Add agent-driven event routing later only if you encounter real needs that the central orchestrator can't handle.

---

## Layer 6: Skills — Unified Registry

Skills are declarative contracts (JSON schemas), not hardcoded functions. Both operational and LLM-native skills supported from day one.

### Skill Types

| Dimension | Operational Skills | LLM-Native Skills |
|-----------|-------------------|-------------------|
| What they do | Modify the environment (side effects) | Transform data within context (no side effects) |
| Examples | Run bash, read/write files, call APIs, query DBs | Summarize text, classify content, extract entities, translate |
| Determinism | Same input → same output | Same input → potentially different output |
| Speed | Fast: ms to seconds | Slow: 1-5s per LLM call on local model |
| Security risk | High — need sandboxing | Low — confined to context window |

### Composition Patterns (agents use both types together)

1. **Pipeline:** LLM-native transform → operational write
2. **Guard:** Operational read → LLM-native evaluate → operational action
3. **Loop:** Operational fetch → LLM-native process (repeat) → operational aggregate
4. **Router:** LLM-native classify → branch to different operational skills
5. **Validator:** Operational execute → LLM-native verify output quality

### Registry Architecture: Three Layers

**1. Global skills pool** — available to every agent (file ops, web search, summarization templates)
**2. Agent-specific skills** — per-agent customizations, can be promoted to global
**3. Skill composition layer** — chaining simple skills into complex workflows

### Implementation: Custom JSON Schema Registry → MCP Migration Path

Skills stored as JSON files in a shared volume (`/opt/skills/`). Agent runtime loads them at startup and exposes them via function calling / tool-use format. Global scope by default with sandboxing for operational skills.

Migration path to MCP (Model Context Protocol) when ready — it's becoming an open standard and both skill types map cleanly to MCP tools/resources/prompts.

---

## Data Flow

```
1. Orchestrator reads pipeline definition: [Agent A → Agent B → Agent C]

2. For each step:
   a. Load agent's context from MemoryOS (memory_refs)
   b. Attach available skills to the prompt
   c. Call LiteLLM with the task
   d. Parse structured output from the LLM
   e. Execute any tool calls (skills) — results written back into memory
   f. Write agent result to MemoryOS mid-term layer
   g. Orchestrator reads next_agent_hint, advances pipeline

3. All state is in MemoryOS — agents themselves are stateless containers
4. Any agent can be re-run independently by reading its input from memory
```

---

## Phased Implementation Plan

### Phase 1: Foundation (Inference Layer)
- Deploy LiteLLM proxy with current model + any new models
- Ensure all existing tooling routes through LiteLLM
- Containerize llama.cpp / other inference engines (one model per container)

### Phase 2: Memory Layer
- Deploy MemoryOS as a shared service (Python container)
- Define memory schema for agent outputs — what gets written, how it's tagged in mid-term layer
- Build the simple read/write client that agents will use
- Fallback plan: if MemoryOS has stability issues, swap to Mem0

### Phase 3: Skills Registry
- Create `/opt/skills/` shared volume with JSON schema skill definitions
- Implement initial operational skills (file operations, command execution with sandboxing)
- Implement initial LLM-native skills (summarization template, classification template)
- Build the skill execution engine: load schemas → expose to LLM via function calling → execute commands or prompt templates

### Phase 4: Agent Runtime + Orchestrator
- Define the agent container template: input → LiteLLM call → structured output → MemoryOS write
- Build centralized orchestrator (lightweight Python process): reads pipeline definition, manages agent lifecycle, handles failure/retry
- Connect orchestrator to MemoryOS for state management and skill registry

### Phase 5: Multi-Agent Pipelines + Always-On Infrastructure
- Define actual pipelines (what agents run, in what order)
- Containerize everything with proper Docker networking
- Set up always-on infrastructure with `restart: unless-stopped` compose files
- Add monitoring, logging, error recovery
- Later: add agent-driven event routing for exception paths if needed

---

## Decisions Made

1. **Scale:** One model per runner container. Adding runners = new containers + update LiteLLM config. No hardcoded limits.
2. **Agents:** Architecture first; specific agents defined later based on use cases.
3. **Memory:** MemoryOS as primary, Mem0 as fallback if stability issues arise.
4. **Orchestration:** Centralized orchestrator for main pipeline. Agent-driven event routing added later only if needed.
5. **Skills:** Both operational and LLM-native from day one. Custom JSON schema registry with MCP migration path. Global scope by default.
