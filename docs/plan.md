# Local LLM Agent Platform — Plan

## Intent

A multi-agent platform where different agents collaborate on tasks, share memory, and call local models — triggered automatically from external systems (Vikunja, GitHub, cron, eventually chat/voice). No custom agent loops, no custom harness code, no custom dispatcher.

---

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

---

## Why These Choices

### Hatchet (not a custom dispatcher)

Hatchet is an open-source AI agent orchestration engine. It is exactly the dispatcher the plan needed without writing one:

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

---

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

**GitHub PR webhook** — Hatchet receives the GitHub webhook directly, dispatches to PR Reviewer worker. Webhook secret validated via `X-Hub-Signature-256` HMAC before event is accepted.

**Future interfaces** — chat UI, voice, app: push a Hatchet event via the events API. One line of code on the interface side. Same worker handles it regardless of source.

### Label → agent mapping (Vikunja existing labels)

| Label | ID | Agent dispatched |
|-------|----|-----------------|
| `ai-research` | 14 | Research agent |
| `ai-go` | 11 | Coder agent |
| `ai-plan-only` | 13 | Research agent (report only, no code) |

---

## Agents

| Agent | Trigger | Tools | Output |
|-------|---------|-------|--------|
| **Research** | Vikunja `ai-research`, cron, event | web search, Mem0 read/write, Vikunja update | Report in Mem0, Vikunja task updated |
| **Coder** | Vikunja `ai-go`, event | git, file ops, shell, GitHub PR | Branch + PR opened, Vikunja updated |
| **PR Reviewer** | GitHub `pull_request.opened` webhook | GitHub API (read PR, post comments) | Review posted to PR |
| **QA** | Post-deploy event, cron | HTTP testing, Vikunja update | Pass/fail report, Vikunja task updated |

---

## Memory Design

One Mem0 instance, all agents use it. Scoping convention:

```python
# Agent's own persistent knowledge (facts, patterns learned)
mem0.add(content, agent_id="researcher")

# Shared task workspace — all agents on this task read/write here
mem0.add(content, agent_id=f"task-{task_id}")

# User-level facts that persist across all tasks
mem0.add(content, user_id="alex")
```

Within a single Pydantic Graph run, shared state passes through the typed `TaskState` dataclass — no Mem0 call needed for in-flight state. Mem0 is for persistence across restarts, cross-agent handoffs in separate workers, and long-term knowledge.

**Eviction policy:**
- `agent_id=f"task-{task_id}"` memories: deleted by the worker after the task completes (or fails permanently). A Hatchet cron runs weekly to purge orphaned task memories older than 7 days.
- `agent_id="researcher"` / `agent_id="coder"` etc.: no TTL — these are the agent's learned knowledge. Reviewed manually.
- `user_id="alex"`: no TTL — persistent personal knowledge graph.

---

## Infrastructure Layout

### Mac Mini — core stack (Komodo, Docker Compose)

New additions to the `core` stack:
- **Qdrant** — vector store for Mem0

Existing PostgreSQL gets new databases (provisioned by Tofu via `provision_app`, not manually):
- `hatchet` — Hatchet task state, execution history
- `mem0` — Mem0 relational state (memory metadata, history)

### k3s cluster (ArgoCD via k3s-dean-gitops)

New Deployments:
- **LiteLLM** — inference gateway
- **Hatchet Lite** — dispatch engine, UI, cron, webhooks (points at Mac Mini PG)
- **Mem0 server** — memory API (points at Mac Mini Qdrant + PG)
- **Agent workers** — one Deployment per agent type, pull tasks from Hatchet

---

## Phased Build

### Phase 0 — Deployment Foundation

This is a reference phase, not a build sequence. It defines the canonical patterns for deploying anything in this infrastructure. All subsequent phases follow these rules. An AI given any deployment task should apply these patterns without being told.

---

#### Secret Management — Hard Rules

1. **BWS is the single source of truth for all secrets.** No exceptions.
2. **Nothing secret goes in git.** No passwords, tokens, keys, or DSNs in any repo, ever.
3. **No hand-written `.env` files.** `.env` files are only valid if populated by BWS automation at runtime (e.g., Komodo's BWS sync, a CI step, or a BWS-aware startup script).
4. **Ansible takes exactly one secret as input:** `BWS_ACCESS_TOKEN` (read-only key). Everything else — postgres passwords, API keys, encryption keys — is read from BWS at runtime via that token.
5. **Generated secrets (passwords, keys) are created by OpenTofu** using `random_password` and written directly to BWS via the `bitwarden-secrets` provider. Humans never generate these.
6. **Manual secrets (third-party API keys, tokens)** are created by a human in the BWS UI. The TOML spec or Ansible playbook declares the BWS key name; it does not set the value.
7. **k3s secrets are delivered via ExternalSecrets operator.** ExternalSecrets reads from BWS (`ClusterSecretStore: bitwarden-secretstore`) and creates native k8s Secrets. Pods consume k8s Secrets — never env vars with inline values.

---

#### App Type Decision Tree

```
Need to deploy a service?
│
├─ Does it need local GPU / direct hardware access / persistent local storage?
│   └─ YES → Stateful app (Komodo on Mac Mini, Docker Compose)
│
└─ NO → Stateless app (k3s via app-factory)
        │
        ├─ Needs a PostgreSQL database? → Tofu provisions it (app-factory)
        ├─ Needs Docker volumes or host config on Mac Mini? → Ansible
        └─ Needs a Docker volume + PG on Mac Mini? → Ansible (stateful dependency)
```

---

#### Pattern A: Stateless App (k3s)

**Toolchain:** app-factory (Tofu + generate.py) → k3s-dean-gitops → ArgoCD

**Step 1 — Write the spec:**
Create `app-factory/apps/<name>.toml`. Declare:
- App name, domain, namespace
- Components (image, port, replicas, resources)
- Secrets (`bws_name`, `k8s_secret`, `k8s_key`, `generate: true/false`)
- Database (`type`, `name`, `host`, `extensions`, `password_secret`)
- UAT config (`enabled: true`, reduced replicas/resources)
- CI/CD (`repo`, `label`)

**Step 2 — Provision + generate (one command):**
```bash
BWS_ACCESS_TOKEN=<read-write token> make create-app APP=<name>
```
This runs:
1. **Tofu provision** — generates random passwords → writes to BWS; creates PostgreSQL role + database + extensions (prod and UAT)
2. **generate.py** — renders Jinja2 templates → writes k8s manifests to `k3s-dean-gitops/apps/<name>/`:
   - `deployment.yaml`, `service.yaml`, `externalsecret.yaml`, `ingress.yaml` (prod)
   - `deployment-uat.yaml`, UAT service, UAT ExternalSecret (if `uat.enabled`)
   - ARC runner values + ArgoCD app entries appended to `root-app.yaml`
   - UAT ApplicationSet entry

**Step 3 — Deploy:**
Commit and push the generated manifests in `k3s-dean-gitops`. ArgoCD auto-syncs. Done.

**Staging (UAT) environment:**
- Same namespace as prod, suffix `-uat` on resource names
- Separate PostgreSQL database (`<name>_uat`) and credentials
- Deployed automatically when a PR is open (via UAT ApplicationSet watching the `deploy:<name>` label on PRs)
- UAT DB can be seeded/reset: app repo CI runs a seed script on PR open or on a manual trigger

**Secret flow (stateless):**
```
Tofu (generate) → BWS (store)
                       ↓
            ExternalSecrets operator (k3s)
                       ↓
                  k8s Secret
                       ↓
                   Pod env var
```

---

#### Pattern B: Stateful App (Komodo on Mac Mini)

**Toolchain:** Tofu (databases + secrets) → komodo-dean-gitops → Komodo (deploy)

Ansible is **not** part of this flow. Ansible is run by a human when provisioning a new server (installing Docker, PostgreSQL, system packages) and for correcting configuration drift. It is never triggered by an AI agent and is never part of a deployment pipeline.

**Step 1 — Provision databases and secrets (Tofu):**
If the stateful service needs a PostgreSQL database or generated secrets, add a spec to `app-factory/apps/<name>.toml` and run `provision_app("<name>")`. Tofu creates the database/role in Mac Mini PostgreSQL and writes secrets to BWS, identical to the stateless path. If there is no database needed, skip this step.

Docker volumes do not need pre-creation. Named volumes declared in compose files are created automatically by Docker on first start and persist across redeploys. Do not use `external: true` unless the volume was created by a separate system. Do not use Ansible to pre-create volumes.

**Step 2 — Service definition (GitOps):**
Add the service to the appropriate Komodo stack in `komodo-dean-gitops/mac-mini-m4/<stack>/compose.yaml`.
- Secrets referenced as `${SECRET_NAME}` — values come from Komodo's BWS-synced env at deploy time
- No hardcoded values in compose files
- Volumes declared without `external: true` — Komodo/Docker creates them on first deploy
- New stacks: add `<stack>/compose.yaml` and register in Komodo

**Step 3 — Deploy:**
Commit and push to `komodo-dean-gitops`. Komodo detects the change and redeploys the stack.

**Secret flow (stateful):**
```
BWS (source of truth)
        ↓
Komodo BWS sync (at deploy time)
        ↓
Stack .env (runtime only, never committed)
        ↓
Docker Compose env vars
```

---

#### Toolchain Reference

| Tool | Repo | Role | When to use |
|------|------|------|-------------|
| OpenTofu | `app-factory/tofu/` | Secret generation + PostgreSQL provisioning | Any new app (stateless or stateful) that needs a DB or generated secrets |
| app-factory generate.py | `app-factory/generate/` | k8s manifest generation from TOML spec | Every stateless app |
| Ansible | `ansible-playbooks/` | **Server provisioning only** — installs packages, configures OS, provisions new nodes, corrects drift | Run by a human when a new server is added or configuration drifts. Never run by AI, never run in deploy pipelines |
| ArgoCD | k3s cluster | Syncs `k3s-dean-gitops` → k3s | Stateless app deployment |
| Komodo | Mac Mini | Deploys `komodo-dean-gitops` stacks | Stateful app deployment |
| ExternalSecrets | k3s cluster | BWS → k8s Secret sync | All k3s secret delivery |
| BWS | bitwarden.com | Single source of truth for all secrets | Everything |

---

#### What Gets Built in Phase 0

The goal is that an AI agent can be told "build a new stateless app called X" and know exactly what to do — not because it read docs, but because there is one tool to call and the tool enforces every rule.

---

##### 1. `infra-mcp` — MCP Server (new repo: `amerenda/infra-mcp`)

A Python `fastmcp` server exposing infrastructure operations as first-class MCP tools. Registered in `~/.claude.json` (stdio transport) so it is available in every agent session and to PydanticAI workers in later phases.

**Tools:**

| Tool | Inputs | What it does |
|------|--------|--------------|
| `scaffold_app` | `name`, `description`, `type: "stateless"\|"stateful"\|"auto"` | Infers type from description if `auto`. Stateless: generates `app-factory/apps/<name>.toml` from template. Stateful: generates compose service stub in `komodo-dean-gitops/<host>/<stack>/` AND a `app-factory/apps/<name>.toml` for DB/secret provisioning if needed. Returns paths created + what to fill in. |
| `provision_app` | `name` | Runs Tofu for the named app: generates secrets → writes to BWS; creates PostgreSQL role + database. Works for both stateless (also generates k8s manifests) and stateful (DB/secrets only, no manifests). Validates no inline secrets before running. |
| `open_deploy_pr` | `name`, `title` | Opens a PR on `k3s-dean-gitops` with the generated manifests. Returns PR URL. Never pushes directly to main. |
| `check_secret_hygiene` | `repo` (optional, defaults all gitops repos) | Runs `git grep` for hardcoded secret patterns. Returns violations or `"clean"`. |
| `get_app_status` | `name`, `type` | Stateless: queries ArgoCD API for sync + health status. Stateful: queries Komodo API for stack status. Returns human-readable + structured status. |
| `resolve_secret_name` | `name` | Looks up a BWS secret by human-readable name (e.g. `"qdrant-api-key"`) → returns the BWS secret ID. Never hardcode UUIDs in manifests or compose files; call this tool instead. |

**Type inference in `scaffold_app`:** The tool uses keyword matching to infer `stateless` vs `stateful`:
- Stateful signals: "GPU", "local storage", "filesystem", "persistent", "mac mini", "murderbot", "hardware"
- Everything else defaults to stateless (k3s)
- AI can always override with explicit `type` parameter

**Secret hygiene enforcement in `provision_app`:**
- Parses TOML before running; rejects if any `[[secrets]]` entry with `generate = false` has a non-empty `value` field
- Rejects if any component env var has a literal value that looks like a secret (entropy check)
- If violations found: prints exactly which field and what to do instead (point at BWS key name)

**app-factory TOML spec extensions needed (backlog for Phase 0 execution):**
- `configmap_mounts` — list of `{name, mount_path}` so `provision_app` bakes ConfigMap volume mounts into the generated Deployment rather than requiring a post-generation patch
- `command_args` — list of extra args appended to the container command
- `ephemeral_storage` — emptyDir size limit for pods that need scratch space (e.g., coder agent git clones)
- `security_context` — pod-level `readOnlyRootFilesystem`, `allowPrivilegeEscalation`, `runAsNonRoot`
- `secret_format` — optional `"base64_bytes:<n>"` generator type for secrets that must be base64-encoded random bytes (e.g., Hatchet encryption key) rather than alphanumeric passwords

---

##### 2. Agent Instruction Files (one per relevant repo)

Each repo gets two files:

- **`CLAUDE.md`** — Claude Code reads this automatically. Its only content is: `"Read agent.md for full guidelines."` This keeps Claude Code integrated while making the real instructions agent-agnostic.
- **`agent.md`** — The actual workflow doc. Any agent (Claude, opencode, future AI) reads this. An agent reading it should never need to ask "how do I deploy here?"

All future additions go in `agent.md`. `CLAUDE.md` is a one-line stub that never changes.

**`app-factory/agent.md`:**
- "All new apps start here. Use `scaffold_app` MCP tool or write a TOML spec manually."
- "The only command you run is `make create-app APP=<name>`. Never run `tofu apply` or `generate.py` directly."
- "No secret values in TOML. `generate = true` secrets are generated by Tofu. `generate = false` secrets must already exist in BWS."
- "Use `resolve_secret_name` MCP tool to look up a BWS secret ID from its name — never hardcode UUIDs."
- Link to `apps/template.toml.example`

**`k3s-dean-gitops/agent.md`:**
- "All manifests under `apps/<name>/generated/` are created by app-factory. Never hand-edit them — re-run `make create-app` instead."
- "Supplementary files (`configmap.yaml`, CRDs, additional Deployments) live in `apps/<name>/` alongside the generated subdirectory. These are hand-maintained and are not overwritten by app-factory."
- "No secrets. If you find yourself about to write a password or token in a YAML file, stop and use ExternalSecrets instead."
- "UAT manifests commit directly to main. Prod manifests require a human-approved PR."

**`komodo-dean-gitops/agent.md`:**
- "Stateful services only. If the service has no local state, it belongs in k3s-dean-gitops instead."
- "Secrets in compose files use `${SECRET_NAME}` syntax. Values come from Komodo's BWS sync at runtime."
- "To add a new service: run `provision_app` MCP tool (creates DB + secrets via Tofu), then add the compose service block here and push to main."
- "Use `resolve_secret_name` MCP tool to look up a secret by name — never hardcode BWS secret UUIDs in compose files or Komodo stack configs."
- "Named volumes declared in compose files are created by Docker on first start — do not pre-create them with Ansible or mark them `external: true`."
- "Never hardcode ports that conflict with `network_mode: host` services — check existing services first."

**`ansible-playbooks/agent.md`:**
- "Ansible is for server provisioning only: installing packages, configuring OS, adding users, provisioning new nodes, correcting configuration drift."
- "Ansible does NOT deploy services and is NOT called as part of any deployment pipeline. Deployment is Komodo (stateful) or ArgoCD (stateless)."
- "Docker volumes and PostgreSQL databases are NOT created by Ansible. Volumes are created by Docker on first compose start. Databases are created by Tofu via `provision_app`."
- "Every playbook reads exactly one secret from env: `BWS_ACCESS_TOKEN`. No other secrets as inputs."
- "All tasks must be idempotent."
- "Ansible is run by a human when adding a new server or correcting drift — never by an AI agent."
- Template for new playbook included inline.

---

##### 3. GitHub Apps (one-time bootstrap)

Two GitHub Apps give each actor a distinct identity in the GitHub UI — your commits, AI-generated commits, and AI reviews are all visually distinct without any convention enforcement.

> **Cannot be created with OpenTofu.** The `integrations/github` provider has no `github_app` resource. Apps are created once in the GitHub UI; credentials are then stored in BWS and referenced as `generate = false` secrets.

**`dean-coder`** — used by the coder agent to push commits and open PRs.
- Commits show as: _"authored by Alex, committed by dean-coder[bot]"_ (or fully bot-authored if agent generates the code)
- PRs opened by `dean-coder[bot]` are visually distinct from human-opened PRs
- Required permissions: `Contents: Read/Write`, `Pull requests: Read/Write`, `Metadata: Read`
- Install on: `amerenda` org

**`dean-reviewer`** — used by the PR reviewer agent to post review comments.
- Review submissions show as submitted by `dean-reviewer[bot]`
- Required permissions: `Pull requests: Read/Write`, `Contents: Read`, `Metadata: Read`
- Install on: `amerenda` org

**Creation steps (one-time, human):**
1. GitHub → Settings → Developer settings → GitHub Apps → New GitHub App
2. Create each app with the permissions above; disable webhooks (the app makes outbound calls only)
3. After creation: note the App ID; generate and download the private key PEM
4. Store in BWS:
   ```bash
   bws secret create dean-coder-app-id       "<APP_ID>"
   bws secret create dean-coder-private-key  "$(cat dean-coder.private-key.pem)"
   bws secret create dean-reviewer-app-id    "<APP_ID>"
   bws secret create dean-reviewer-private-key "$(cat dean-reviewer.private-key.pem)"
   ```
5. Install each app on the `amerenda` org: App page → Install App → `amerenda` → All repositories

**In `praetor.toml`** (both referenced as pre-existing secrets):
```toml
[[secrets]]
name = "CODER_APP_ID"
bws_key = "dean-coder-app-id"
generate = false

[[secrets]]
name = "CODER_APP_PRIVATE_KEY"
bws_key = "dean-coder-private-key"
generate = false

[[secrets]]
name = "REVIEWER_APP_ID"
bws_key = "dean-reviewer-app-id"
generate = false

[[secrets]]
name = "REVIEWER_APP_PRIVATE_KEY"
bws_key = "dean-reviewer-private-key"
generate = false
```

**Auth flow in agents** (same for both apps):
```python
import jwt, time, httpx

def get_installation_token(app_id: str, private_key_pem: str) -> str:
    now = int(time.time())
    payload = {"iss": app_id, "iat": now - 60, "exp": now + 540}
    jwt_token = jwt.encode(payload, private_key_pem, algorithm="RS256")
    # Get installation ID for the org
    inst = httpx.get(
        "https://api.github.com/app/installations",
        headers={"Authorization": f"Bearer {jwt_token}", "Accept": "application/vnd.github+json"}
    ).json()[0]["id"]
    # Exchange for short-lived installation token (valid 1 hour)
    resp = httpx.post(
        f"https://api.github.com/app/installations/{inst}/access_tokens",
        headers={"Authorization": f"Bearer {jwt_token}", "Accept": "application/vnd.github+json"}
    )
    return resp.json()["token"]
```
For git push: `https://x-access-token:{token}@github.com/amerenda/{repo}.git`

---

##### 4. `bws-mcp` — Secrets MCP Server (new repo: `amerenda/bws-mcp`)

A dedicated FastMCP server that wraps the BWS API with explicit, enforced permission tiers. Agents get exactly the access they need — no more.

**Three permission levels, enforced by which token the client presents:**

| Level | Can do | Cannot do |
|-------|--------|-----------|
| **Read** | Look up secrets by name or ID; list secret names | Create, update, or delete secrets |
| **Write** | Create a secret (upsert by name — create if missing, overwrite if exists) | Read secret values, list values, delete secrets |
| **None** | Nothing — tool calls return `403` | Everything |

> Write permission is intentionally write-only: an agent can store a value it generated without being able to exfiltrate existing secrets. This is the right scope for the coder agent (can write its own artifacts) and workers that need to provision credentials without reading them.

**Implementation:**
- Single FastMCP server with two token env vars: `BWS_READ_TOKEN` and `BWS_WRITE_TOKEN` (both are BWS access tokens with different project permissions, or the same token with the server enforcing the restriction in code)
- Client connects with either token — the server inspects which token was used and gates tool availability accordingly
- Alternative: two separate stdio server registrations pointing to the same binary but with different `--mode` flags (simpler and avoids token-sniffing in shared context)

**Tools:**

| Tool | Permission required | What it does |
|------|---------------------|--------------|
| `bws_get_secret` | Read | Returns the secret value for a given name or ID |
| `bws_list_secrets` | Read | Returns list of secret names (no values) in the project |
| `bws_upsert_secret` | Write | Creates secret if it doesn't exist; overwrites value if it does. Never returns the value. |

**Registration in `~/.claude.json`:**
```json
{
  "mcpServers": {
    "bws-read":  { "command": "bws-mcp", "args": ["--mode", "read"],  "env": { "BWS_ACCESS_TOKEN": "${BWS_READ_TOKEN}" } },
    "bws-write": { "command": "bws-mcp", "args": ["--mode", "write"], "env": { "BWS_ACCESS_TOKEN": "${BWS_WRITE_TOKEN}" } }
  }
}
```
Agents that should only write get `bws-write` in their MCP config. Infra operations (provision_app) use `bws-read`. The `bws-write` server never exposes `bws_get_secret`.

**Where each agent gets which access:**
- `infra-mcp` / `provision_app` — Read (needs to read secrets to validate, pass to Tofu)
- Coder agent — Write only (can store generated artifacts, cannot read existing secrets)
- Research agent — None (no BWS access needed)
- PR Reviewer agent — None
- Hatchet workers (general) — None by default; opt-in Write if they generate credentials

---

##### 5. `app-factory/apps/praetor.toml` (Phase 0 end-to-end test)

A real TOML spec for the `praetor` app (the platform being built in this plan). Running `make create-app APP=praetor` is the Phase 0 integration test — it proves the full toolchain works. The spec declares:
- PostgreSQL database (prod + UAT)
- A placeholder component (image TBD) — enough to generate valid manifests
- UAT enabled

This is also the first app that will be populated in Phases 1–8.

---

#### Phase 0 Ready Conditions

**MCP server:**
1. `infra-mcp` repo exists with `fastmcp` server; registered in `~/.claude.json` (and `~/.config/opencode/opencode.json`)
2. Agent can call `get_app_status` and get a response (proves MCP is live)
3. `check_secret_hygiene` runs against all gitops repos and returns clean

**End-to-end stateless path (via MCP):**
4. `scaffold_app("praetor", "stateless agent platform backend", "stateless")` creates a valid `app-factory/apps/praetor.toml`
5. `provision_app("praetor")` completes; verify explicitly:
   - `bws secret list | grep praetor` — generated secrets exist in BWS (DB password, app secret, etc.)
   - `psql -h 10.100.20.18 -U postgres -c "\l"` — shows `agent_platform` and `agent_platform_uat` databases
   - `k3s-dean-gitops/apps/praetor/generated/` directory exists with valid manifests
6. `open_deploy_pr("praetor", "phase-0: bootstrap praetor app")` opens a PR on k3s-dean-gitops
7. After PR merge: ArgoCD syncs the namespace and ExternalSecrets object; `kubectl get secret -n praetor` shows secrets populated from BWS; `get_app_status("praetor", "stateless")` returns healthy

**End-to-end stateful GitOps path (no Ansible):**
8. Add a compose service stub to `komodo-dean-gitops`, push to main → Komodo picks it up and deploys without any Ansible step (proves stateful deploy is pure GitOps)

**CLAUDE.md files:**
9. All four repos (`app-factory`, `k3s-dean-gitops`, `komodo-dean-gitops`, `ansible-playbooks`) have both `CLAUDE.md` (one-line stub: "Read agent.md") and `agent.md` (full guidelines) committed to main

---

### Phase 1 — Inference: LiteLLM on k3s

**Goal:** Single inference endpoint for all model calls. opencode and all future agents use one URL. Switching models is a ConfigMap change only.

**Current state:** `opencode → llama-proxy (:8089) → llama-server (:8088 on murderbot) → GPU`
**After:** `opencode / agents → LiteLLM (k3s, litellm.amer.dev) → llama-server (:8080 on murderbot)`

**Pre-conditions:**
- murderbot llama-server migrated to `:8080` (standard llama.cpp default — done as part of this phase; update start script and llama-proxy config)
- k3s cluster healthy, ArgoCD green
- `LITELLM_MASTER_KEY` secret created in Bitwarden

**What gets built:**

`LITELLM_MASTER_KEY` is a generated secret — Tofu creates it. Provision via MCP:

```
scaffold_app("litellm", "stateless OpenAI-compatible inference proxy", "stateless")
# Edit app-factory/apps/litellm.toml:
#   component: image ghcr.io/berriai/litellm:main-stable, port 4000
#   secret: bws_name "litellm-master-key", generate=true
#   no database
provision_app("litellm")   # secret → BWS, generates k3s-dean-gitops/apps/litellm/ skeleton
open_deploy_pr("litellm", "phase-1: deploy LiteLLM inference proxy")
```

Generated by app-factory: `namespace.yaml`, `deployment.yaml`, `service.yaml`, `externalsecret.yaml` (LITELLM_MASTER_KEY → `LITELLM_MASTER_KEY` env var), `ingressroute.yaml` (`litellm.amer.dev`), ArgoCD app entry.

Supplementary files (hand-maintained alongside generated manifests, not replacing them):
- `configmap.yaml` — LiteLLM model routing config (see multi-runner section below)
- Patch to generated `deployment.yaml`: mount the ConfigMap as `/app/config.yaml`, add `--config /app/config.yaml` to command args

**Multi-runner routing and load balancing:**

LiteLLM's router treats multiple backends for the same model name as a pool. Requests are distributed across all healthy backends automatically. Adding a new runner is a ConfigMap edit + ArgoCD sync — no code changes.

```yaml
model_list:
  # Each entry is one runner. Same model_name = same pool.
  - model_name: qwen3-35b
    litellm_params:
      model: openai/qwen3-35b
      api_base: http://10.100.20.19:8080/v1   # murderbot (RTX 4000)
      api_key: none
  - model_name: qwen3-35b
    litellm_params:
      model: openai/qwen3-35b
      api_base: http://10.100.20.25:8080/v1   # archlinux (RX 9070 XT) — add when ready
      api_key: none
  # Future runners: just add another entry here
  - model_name: ollama/*                        # passthrough all Ollama models
    litellm_params:
      model: ollama/*
      api_base: http://10.100.20.18:11434

router_settings:
  routing_strategy: least-busy   # routes to the backend with fewest active in-flight requests
  num_retries: 3
  retry_after: 5                 # seconds before retrying a failed backend
  cooldown_time: 60              # seconds a failed backend is excluded from rotation
  allowed_fails: 3               # failures before cooldown triggers
```

**Why `least-busy`:** LLM requests vary wildly in length. Round-robin sends every Nth request to each backend regardless of how long it'll take; `least-busy` tracks active request count per backend and always sends the next request to the one doing the least work. For 3+ heterogeneous GPU runners, this is almost always the right choice.

**Request queuing:** LiteLLM queues requests internally when all backends are saturated (configurable with `redis` for multi-replica LiteLLM deployments; single replica uses in-memory queue). Clients experience back-pressure as latency, not errors, until a backend becomes free.

**Adding a new runner:** Add a new `model_list` entry pointing to the new runner's endpoint. Update the ConfigMap, push to gitops. ArgoCD syncs and LiteLLM hot-reloads the config — no pod restart required.

**opencode config:** `OPENAI_BASE_URL` updated from `http://localhost:8080` → `https://litellm.amer.dev/v1`

**Retired:**
- `install-litellm.sh` — archived, superseded by gitops
- `llama-proxy.py` — stopped and removed from `start-opencode-stable.sh` (LiteLLM handles retries natively)

**Observability (wire up in this phase):**
LiteLLM exports Prometheus metrics at `/metrics` natively — token usage, latency, error rates per model. Add a `ServiceMonitor` or static scrape job to Prometheus so the metrics are available from day one. No custom code required.

**Ready conditions (all must pass):**
1. murderbot llama-server running on `:8080`; `curl -sf http://localhost:8080/health` returns OK
2. `curl -sf -H "Authorization: Bearer $LITELLM_MASTER_KEY" https://litellm.amer.dev/v1/models` returns JSON list including `qwen3-35b`
3. Test completion request returns a valid response end-to-end through LiteLLM
4. opencode starts a session and completes at least one tool call successfully
5. `pgrep -f llama-proxy` returns nothing — proxy retired
6. Multi-runner test: add archlinux endpoint to ConfigMap (or a duplicate murderbot entry), send 10 concurrent requests — LiteLLM distributes across backends; `GET /metrics` shows requests split across both deployments
7. Model swap test: add a model entry to ConfigMap → ArgoCD syncs → new model appears in `/v1/models` without touching any local scripts
8. ArgoCD app is synced and healthy

---

### Phase 2 — Storage: Qdrant on Mac Mini core stack

**Goal:** Qdrant running and reachable from k3s. `hatchet` and `mem0` databases are provisioned by Tofu in Phases 3 and 4 respectively — not here.

**Current state:** Mac Mini core stack has Technitium, PostgreSQL (pgvector/pg16, `network_mode: host`, port 5432, databases: `todo`, `agent_kb`), MongoDB. No Qdrant.

**Rule:** No manual steps. All service changes via GitOps (Komodo for Mac Mini, ArgoCD for k3s). Databases provisioned by Tofu. No Ansible in the deploy flow.

**Pre-conditions:**
- Mac Mini core stack healthy
- `QDRANT_API_KEY` secret provisioned in Bitwarden (via `provision_app("qdrant")` or manually with `bws secret create`)

**What gets built:**

In `komodo-dean-gitops/mac-mini-m4/core/compose.yaml` — add Qdrant service:
```yaml
qdrant:
  image: qdrant/qdrant:v1.13.6        # pin to current stable
  container_name: qdrant
  restart: unless-stopped
  network_mode: host                   # binds 0.0.0.0:6333 (REST) and 0.0.0.0:6334 (gRPC)
  environment:
    QDRANT__SERVICE__API_KEY: "${QDRANT_API_KEY}"
  volumes:
    - qdrant-data:/qdrant/storage
  healthcheck:
    test: ["CMD", "curl", "-sf", "http://localhost:6333/healthz"]
    interval: 30s
    timeout: 5s
    retries: 3

volumes:
  qdrant-data:    # Docker creates this named volume on first start — no pre-creation needed
```
Add `QDRANT_API_KEY` to the stack's Bitwarden-synced env.

In `komodo-dean-gitops/mac-mini-m4/postgres/init.sh` — add new databases for disaster-recovery fresh installs only (existing instance gets them via Tofu in Phases 3/4):
```bash
# Added for fresh-install bootstrap — existing instances: see Phase 3/4 provision_app
CREATE USER hatchet WITH PASSWORD '${HATCHET_POSTGRES_PASSWORD}';
CREATE DATABASE hatchet OWNER hatchet;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE USER mem0 WITH PASSWORD '${MEM0_POSTGRES_PASSWORD}';
CREATE DATABASE mem0 OWNER mem0;
CREATE EXTENSION IF NOT EXISTS vector;
```

**Execution order:** Commit the compose change to `komodo-dean-gitops` main → Komodo picks up the change and deploys Qdrant. Docker creates `qdrant-data` automatically on first container start.

**Ready conditions (all must pass):**
1. `curl -sf -H "api-key: $QDRANT_API_KEY" http://10.100.20.18:6333/healthz` returns `{"title":"qdrant - version x.x.x"}`
2. Qdrant reachable from inside a k3s pod: `kubectl run -it --rm --image=curlimages/curl test -- curl -sf -H "api-key: $KEY" http://10.100.20.18:6333/healthz`
3. Smoke test via Qdrant REST API: create a collection, insert one vector, delete it — no errors
4. Komodo shows core stack healthy after deploy

---

### Phase 3 — Dispatch: Hatchet on k3s

**Goal:** Events flow through Hatchet. A stub worker receives a trigger and logs it, visible in the Hatchet UI with full task history. No real agent yet.

**Pre-conditions:**
- Phase 2 complete (Qdrant running, Mac Mini PostgreSQL healthy)
- Hatchet encryption keysets generated and stored in BWS (see below — these are NOT simple passwords)

**Hatchet encryption keys — how to generate (one-time, before provisioning):**
Hatchet uses [Tink](https://github.com/google/tink) keysets for encryption and JWT signing. These must be generated with the `hatchet-admin` CLI, not `openssl`. Run this locally:
```bash
docker run --rm -v /tmp/hatchet-keys:/keys \
  ghcr.io/hatchet-dev/hatchet/hatchet-lite:latest \
  /hatchet-admin keyset create-local-keys --key-dir /keys
# Outputs: master.key, jwt-public.key, jwt-private.key
```
Store all three in BWS:
- `hatchet-master-keyset` ← contents of `master.key`
- `hatchet-jwt-public-keyset` ← contents of `jwt-public.key`
- `hatchet-jwt-private-keyset` ← contents of `jwt-private.key`
- `hatchet-cookie-secrets` ← `openssl rand -hex 16` (space-separated pair: run twice, join with space)

**What gets built:**

Provision via MCP:

```
scaffold_app("hatchet", "stateless k3s workflow engine with PostgreSQL backend", "stateless")
# Edit app-factory/apps/hatchet.toml:
#   component: image ghcr.io/hatchet-dev/hatchet/hatchet-lite:latest
#              ports: 8888 (HTTP/UI/REST), 7077 (gRPC — workers connect here)
#   secrets (all generate=false — created in BWS above):
#     SERVER_ENCRYPTION_MASTER_KEYSET       ← hatchet-master-keyset
#     SERVER_ENCRYPTION_JWT_PUBLIC_KEYSET   ← hatchet-jwt-public-keyset
#     SERVER_ENCRYPTION_JWT_PRIVATE_KEYSET  ← hatchet-jwt-private-keyset
#     SERVER_AUTH_COOKIE_SECRETS            ← hatchet-cookie-secrets
#   database: name "hatchet", host "10.100.20.18"
provision_app("hatchet")   # hatchet DB + role created in Mac Mini PG, manifests generated
open_deploy_pr("hatchet", "phase-3: deploy Hatchet workflow engine")
```

Supplementary files for the Hatchet deployment (alongside generated manifests):
- `configmap.yaml` — non-secret runtime config:
  ```yaml
  SERVER_URL: "https://hatchet.amer.dev"
  SERVER_GRPC_BROADCAST_ADDRESS: "hatchet.hatchet.svc.cluster.local:7077"
  SERVER_GRPC_BIND_ADDRESS: "0.0.0.0"
  SERVER_GRPC_PORT: "7077"
  SERVER_AUTH_COOKIE_DOMAIN: "hatchet.amer.dev"
  SERVER_AUTH_COOKIE_INSECURE: "f"
  SERVER_DEFAULT_ENGINE_VERSION: "V1"
  SERVER_MSGQUEUE_KIND: "postgres"
  DATABASE_URL: "postgresql://hatchet:<pw>@10.100.20.18:5432/hatchet"
  ```
  Note: `SERVER_GRPC_BROADCAST_ADDRESS` uses the in-cluster service name so workers in other namespaces resolve it without leaving the cluster.
- `service-grpc.yaml` — separate Service for gRPC port 7077 (ClusterIP, not exposed via Ingress)
- IngressRoute routes to port 8888 only (HTTP UI + REST API)

**Initial Hatchet setup (one-time, done in the web UI after first deploy):**
The Hatchet Lite image auto-runs migrations on startup — no separate migration job needed. After first deploy:
1. Navigate to `https://hatchet.amer.dev` → complete the first-run setup wizard (set admin email + password)
2. Go to Settings → API Tokens → create a token named `praetor-workers`
3. Store this token in BWS as `hatchet-client-token`
4. Add `HATCHET_CLIENT_TOKEN` secret (generate=false) to `praetor.toml` — workers read this env var to authenticate

**Worker gRPC networking:** Workers in the `praetor` namespace connect to `hatchet.hatchet.svc.cluster.local:7077`. This requires the `service-grpc.yaml` ClusterIP service above. Workers do NOT need to go through the Ingress — gRPC travels in-cluster.

**In `praetor` repo:**
- `workers/stub/main.py` — minimal Hatchet worker: initializes with `Hatchet()` (reads `HATCHET_CLIENT_TOKEN` from env), registers `test:ping` event handler, logs payload, returns `{"status": "ok"}`
- `workers/stub/Dockerfile`

Stub worker component added to `praetor.toml` → `provision_app("praetor")` → PR to k3s-dean-gitops.

Hatchet cron registered in code at worker startup:
- Every 5 minutes → push `test:ping` event with timestamp payload
- Cron expression as env var: `STUB_CRON_INTERVAL` (default `*/5 * * * *`) — configurable via ConfigMap

**Ready conditions (all must pass):**
1. `https://hatchet.amer.dev` loads the UI and login works with the admin credentials set in step 1 above
2. `hatchet` database exists in Mac Mini PostgreSQL with correct owner (created by `provision_app` Tofu)
3. Stub worker pod is running and shows as connected in Hatchet UI (Workers tab)
4. Manually push `test:ping` event via Hatchet REST API → run appears in UI with status `SUCCEEDED` and logged payload
5. Cron fires on schedule → run appears in UI automatically without manual trigger
6. Kill the stub worker pod mid-run → Hatchet retries and succeeds after pod restarts
7. `kubectl exec` into a test pod in `praetor` namespace: `nc -zv hatchet.hatchet.svc.cluster.local 7077` succeeds (proves gRPC reachable cross-namespace)
8. ArgoCD shows Hatchet app synced and healthy

---

### Phase 4 — Memory: Mem0 on k3s

**Goal:** Two Hatchet workers share a memory namespace via Mem0. One writes, the other reads.

**Pre-conditions:**
- Phase 2 complete (Qdrant running at `10.100.20.18:6333`, Mac Mini PostgreSQL healthy)
- Phase 3 complete (Hatchet healthy, `HATCHET_CLIENT_TOKEN` in BWS, stub worker connected)
- `MEM0_API_KEY` does not need manual creation — generated by Tofu (`generate=true`)

**What gets built:**

Mem0 needs access to Qdrant and PostgreSQL. The self-hosted Mem0 OSS server image is `ghcr.io/mem0ai/mem0:latest`. Provision via MCP:

```
scaffold_app("mem0", "stateless memory layer backed by Qdrant vector store and PostgreSQL", "stateless")
# Edit app-factory/apps/mem0.toml:
#   component: image ghcr.io/mem0ai/mem0:latest, port 8000
#   secrets: MEM0_API_KEY (generate=true)
#            QDRANT_API_KEY (generate=false — already in BWS from Phase 2)
#   database: name "mem0", host "10.100.20.18"  ← Tofu creates this DB
provision_app("mem0")   # MEM0_API_KEY → BWS, mem0 DB + role created in Mac Mini PG, manifests generated
open_deploy_pr("mem0", "phase-4: deploy Mem0 memory layer")
```

Supplementary ConfigMap alongside generated ExternalSecret:
```yaml
MEM0_VECTOR_STORE_PROVIDER: "qdrant"
MEM0_VECTOR_STORE_HOST: "10.100.20.18"
MEM0_VECTOR_STORE_PORT: "6333"
MEM0_LLM_PROVIDER: "litellm"
MEM0_LLM_MODEL: "qwen3.6"
MEM0_LLM_BASE_URL: "https://litellm.amer.dev/v1"
MEM0_EMBEDDER_PROVIDER: "ollama"
MEM0_EMBEDDER_MODEL: "nomic-embed-text"
MEM0_EMBEDDER_BASE_URL: "http://10.100.20.18:11434"
```

**Memory access in agents — use Python SDK, not MCP:**
The self-hosted Mem0 OSS server does not expose an MCP endpoint — that is OpenMemory (cloud product). Agents use the `mem0` Python SDK pointed at the self-hosted REST API:
```python
from mem0 import MemoryClient

# Configured via env vars: MEM0_BASE_URL, MEM0_API_KEY
memory = MemoryClient(host=os.environ["MEM0_BASE_URL"], api_key=os.environ["MEM0_API_KEY"])

# Wrapped as PydanticAI tools in common/memory_tools.py
def add_memory(content: str, agent_id: str) -> str:
    memory.add(content, agent_id=agent_id)
    return "stored"

def search_memory(query: str, agent_id: str) -> list[str]:
    results = memory.search(query, agent_id=agent_id)
    return [r["memory"] for r in results]
```
These two functions are registered as tools on every PydanticAI agent. `common/memory_tools.py` is the single shared location — no duplication per agent.

**Memory cleanup worker** — lightweight Hatchet cron worker added to `praetor.toml`:
```python
@hatchet.cron(os.environ.get("MEMORY_CLEANUP_CRON", "0 3 * * 0"))  # weekly Sun 3am
async def cleanup_task_memories(ctx):
    # Purge task-scoped memories older than 7 days
    all_memories = memory.get_all(agent_id_prefix="task-")
    cutoff = datetime.utcnow() - timedelta(days=7)
    for m in all_memories:
        if datetime.fromisoformat(m["created_at"]) < cutoff:
            memory.delete(m["id"])
```

Two test workers added to `praetor.toml` as components: `mem-write-worker` (`test:mem-write`), `mem-read-worker` (`test:mem-read`).
`provision_app("praetor")` → PR to k3s-dean-gitops.

**Ready conditions (all must pass):**
1. `curl -sf -H "Authorization: Bearer $MEM0_API_KEY" https://mem0.amer.dev/v1/memories/?agent_id=test` returns `[]` (empty, not an error)
2. `mem0` database exists in Mac Mini PostgreSQL with correct owner (created by `provision_app` Tofu)
3. Push `test:mem-write` event → `mem-write-worker` calls `memory.add(content, agent_id="test-shared")` → Hatchet run `SUCCEEDED`
4. Push `test:mem-read` event → `mem-read-worker` calls `memory.search("fact", agent_id="test-shared")` → returns the fact written in step 3 → run `SUCCEEDED`
5. Push 10 concurrent `test:mem-write` events → all 10 runs succeed, all 10 facts retrievable (concurrent write test)
6. Mem0 reads Qdrant: verify a collection named `mem0` exists in Qdrant at `10.100.20.18:6333`
7. Mem0 uses LiteLLM for memory extraction: check Mem0 pod logs show LiteLLM calls to `litellm.amer.dev` on memory adds
8. ArgoCD shows Mem0 app synced and healthy

---

### Phase 5 — Research Agent

**Goal:** Tag a Vikunja task `ai-research` → agent runs automatically → report in Mem0 → Vikunja task marked done.

**Pre-conditions:**
- Phases 1–4 complete and healthy
- SearXNG reachable from k3s at `https://searxng.amer.dev` (already deployed)
- `VIKUNJA_TOKEN` added to BWS manually (generate=false — it is a Vikunja API token, not a generated password). Add `vikunja-token` to BWS, then add `VIKUNJA_TOKEN` secret ref to `praetor.toml`

**What gets built — `praetor/agents/research/`:**
- `agent.py` — PydanticAI agent with tools:
  - `web_search(query)` → calls SearXNG REST API
  - `add_memory` / `search_memory` → Mem0 MCP tools
  - `update_vikunja_task(task_id, status, comment)` → calls Vikunja API
- `worker.py` — Hatchet worker handling `agent:research` events
- `Dockerfile`

**Trigger: Vikunja webhook (not polling)**
Vikunja does not have a `task.label.added` event. Label changes arrive as `task.updated`. The webhook adapter must detect label changes:

1. **Webhook registration is IaC** — declared in `praetor.toml` as a `vikunja_webhooks` block, registered by `provision_app` via Tofu's `http` provider:
   ```toml
   [[vikunja_webhooks]]
   project_id = 21          # Mycroft project
   target_url  = "https://praetor.amer.dev/webhooks/vikunja"
   events      = ["task.updated", "task.created"]
   secret_bws_key = "vikunja-webhook-secret"   # generate=true; Tofu creates and stores it
   ```
   Tofu calls `PUT /api/v1/projects/{id}/webhooks` with the generated secret. Running `provision_app("praetor")` again is idempotent — it upserts the webhook registration.

2. The adapter endpoint lives in `praetor/webhooks/vikunja.py` (a small FastAPI router mounted alongside the workers). It:
   - Validates `X-Vikunja-Signature` against `VIKUNJA_WEBHOOK_SECRET` from env
   - Inspects `data.task.labels` in the payload for known label IDs (14 = `ai-research`, 11 = `ai-go`, 13 = `ai-plan-only`)
   - For each matching label, pushes the corresponding Hatchet event with `idempotency_key=f"vikunja-{task_id}-{label_id}"`
   - If a task has BOTH labels 14 and 11 → pushes `pipeline:research_code` instead of individual events

3. The webhook adapter is a component in `praetor.toml` — a lightweight FastAPI service exposed via Ingress at `https://praetor.amer.dev/webhooks/vikunja`. It does NOT run inside a Hatchet worker.

Webhook secret stored in BWS as `vikunja-webhook-secret` (generate=true in TOML). Tofu owns creation; `provision_app` handles registration.

**Deduplication:** Events are pushed with `idempotency_key=f"vikunja-{task_id}-{label_id}"`. Duplicate `task.updated` deliveries for the same label do not create duplicate Hatchet runs.

Deploy: add `research-worker` component to `praetor.toml` → `provision_app("praetor")` → `open_deploy_pr("praetor", "phase-5: research worker")`

**Ready conditions (all must pass):**
1. `provision_app("praetor")` completes; Vikunja project settings show the webhook registered with `https://praetor.amer.dev/webhooks/vikunja` — no manual UI step
2. Create Vikunja task "Research: summarize recent k3s releases", label `ai-research` → Hatchet UI shows `agent:research` run within 30 seconds (not 15 minutes)
3. `GET https://mem0.amer.dev/v1/memories/?agent_id=task-<id>` returns the research report
4. Vikunja task is marked done with a comment containing the report summary
5. Add the same label twice → only one Hatchet run (idempotency key dedup)
6. Web search failure (bad query/network error) → agent retries, then fails gracefully with a Vikunja comment — no silent failure

---

### Phase 6 — Coder Agent

**Goal:** Tag a Vikunja task `ai-go` with a repo reference → draft PR opened on that repo.

**Pre-conditions:**
- Phases 1–4 complete
- `dean-coder` GitHub App created, installed on `amerenda` org, credentials in BWS (see Phase 0 bootstrap)

**What gets built — `praetor/agents/coder/`:**
- `agent.py` — PydanticAI agent with tools:
  - `git_clone(repo, branch)`, `git_commit(message)`, `git_push()`, `open_pr(title, body)`
  - `read_file(path)`, `write_file(path, content)`, `run_shell(cmd)` (scoped to `SCRATCH_DIR` emptyDir mount)
  - `search_memory(query, agent_id)` → calls `common/memory_tools.py`, reads research context from `task-{id}` namespace if it exists
- `worker.py` — Hatchet worker handling `agent:code` events
- `common/github_app.py` — shared helper: `get_installation_token(app_id, private_key)` → short-lived token used for git push and GitHub API calls
- Vikunja webhook adapter (Phase 5) already handles label 11 (`ai-go`) → no code change needed in adapter

Commits and PRs opened by this worker appear as `dean-coder[bot]` in the GitHub UI — distinct from human-authored commits at a glance.

**Coder worker TOML additions:**
```toml
[[components]]
name = "coder-worker"
...
ephemeral_storage = "4Gi"   # emptyDir scratch for git clones
[components.security_context]
run_as_non_root = true
allow_privilege_escalation = false
read_only_root_filesystem = true   # writes go to SCRATCH_DIR emptyDir only

[[components.env]]
name = "SCRATCH_DIR"
value = "/scratch"
[[components.env]]
name = "CODER_APP_ID"
secret_ref = { bws_key = "dean-coder-app-id" }
[[components.env]]
name = "CODER_APP_PRIVATE_KEY"
secret_ref = { bws_key = "dean-coder-private-key" }
```

Sandbox is enforced at the pod level — `run_shell` is scoped to `$SCRATCH_DIR` in code, and `readOnlyRootFilesystem` prevents writes anywhere else. The pod cannot access other k8s Secrets (RBAC limited to its own namespace).

Deploy: add `coder-worker` component to `praetor.toml` → `provision_app("praetor")` → `open_deploy_pr("praetor", "phase-6: coder worker")`

**Ready conditions (all must pass):**
1. Create Vikunja task "Add /healthz endpoint to ecdysis", label `ai-go`, repo `amerenda/ecdysis` in description
2. Within 30 seconds: Hatchet UI shows `agent:code` run triggered (webhook-based, not polling)
3. A draft PR exists on `amerenda/ecdysis` on a new branch, opened by `dean-coder[bot]`
4. PR description references the Vikunja task ID
5. Vikunja task is marked done with the PR link as a comment
6. Task with a broken/missing repo reference → agent fails gracefully with a Vikunja comment, no unhandled exception

---

### Phase 7 — PR Reviewer + QA

**Goal:** GitHub events drive agents automatically without Vikunja labels.

**Pre-conditions:**
- Phases 1–4 complete
- GitHub webhook can reach `praetor.amer.dev` (public ingress already set up)
- `dean-reviewer` GitHub App created, installed on `amerenda` org, credentials in BWS (see Phase 0 bootstrap)
- A PAT or GitHub App token with `admin:org_hook` scope available in BWS for Tofu to register the org webhook (can reuse `dean-reviewer` app if it has org hook permissions, or a separate deploy token)

**What gets built:**

**Webhook registration is IaC** — declared in `praetor.toml` as a `github_webhooks` block, registered by `provision_app` via Tofu's `github` provider:
```toml
[[github_webhooks]]
org            = "amerenda"
target_url     = "https://praetor.amer.dev/webhooks/github"
events         = ["pull_request"]
secret_bws_key = "github-webhook-secret"   # generate=true; Tofu creates and stores it
```
Tofu calls `POST /orgs/{org}/hooks` (GitHub Webhooks API). Running `provision_app` again is idempotent — it upserts the registration.

PR Reviewer — `praetor/agents/pr_reviewer/`:
- `agent.py` — PydanticAI agent: reads PR diff via GitHub API using `dean-reviewer` app token, posts structured review comment
- `worker.py` — Hatchet worker handling `github:pr_opened` events; reads `REVIEWER_APP_ID` + `REVIEWER_APP_PRIVATE_KEY` from env, calls `common/github_app.py` to get an installation token
- Webhook adapter at `praetor/webhooks/github.py` (same FastAPI app as Vikunja adapter):
  - Validates `X-Hub-Signature-256` HMAC-SHA256 against `GITHUB_WEBHOOK_SECRET` (stored in BWS)
  - On `pull_request.opened`: pushes `github:pr_opened` Hatchet event with `{repo, pr_number, pr_url, diff_url, author}`
  - Returns 401 on invalid signature, 200 on success — GitHub retries on non-2xx

Reviews submitted by this worker appear as `dean-reviewer[bot]` — distinct from human reviews and from `dean-coder[bot]` commits.

Add reviewer app credentials to TOML:
```toml
[[components.env]]
name = "REVIEWER_APP_ID"
secret_ref = { bws_key = "dean-reviewer-app-id" }
[[components.env]]
name = "REVIEWER_APP_PRIVATE_KEY"
secret_ref = { bws_key = "dean-reviewer-private-key" }
```

QA — `praetor/agents/qa/`:
- `agent.py` — PydanticAI agent: tests staging URL using `playwright` Python library (HTTP + UI flows, not just curl)
- `worker.py` — Hatchet worker handling `deploy:staging` events
- CI pipeline: push `deploy:staging` event to Hatchet after UAT deploy completes

**`deploy:staging` event payload schema** (defined here, used by all CI pipelines):
```json
{
  "app": "ecdysis",
  "staging_url": "https://ecdysis-uat.amer.dev",
  "pr_number": 42,
  "commit_sha": "abc1234",
  "vikunja_task_id": 99
}
```
CI pushes this event; QA agent reads `staging_url` from payload directly — no URL-derivation logic needed.

Deploy: add `pr-reviewer-worker` and `qa-worker` components to `praetor.toml` → `provision_app("praetor")` → `open_deploy_pr("praetor", "phase-7: pr-reviewer and qa workers")`

**Ready conditions (all must pass):**
1. `provision_app("praetor")` registers GitHub org webhook — `gh api /orgs/amerenda/hooks` shows it; no manual GitHub UI step
2. Send a spoofed webhook (wrong HMAC) to `praetor.amer.dev/webhooks/github` → returns 401, no Hatchet run created
3. Open a PR on any `amerenda` repo → Hatchet UI shows `github:pr_opened` run within 60 seconds
4. PR receives a review comment from `dean-reviewer[bot]` within 5 minutes of opening
5. Manually push `deploy:staging` event with valid payload → QA run appears and completes
6. QA result (pass/fail + details) is posted as a PR comment or Vikunja comment
7. PR Reviewer handles a deleted PR mid-run gracefully (no unhandled exception)
8. QA handles an unreachable staging URL gracefully (clear failure message, not a hung run)

---

### Phase 8 — Multi-Agent Pipeline (Pydantic Graph)

**Goal:** One Vikunja task labeled both `ai-research` and `ai-go` → research then coding in sequence, coder uses research output.

**Pre-conditions:**
- Phases 5 and 6 complete and stable

**What gets built — `praetor/pipelines/`:**
- `research_then_code.py` — Pydantic Graph with two typed nodes:
  - `ResearchNode`: runs research agent, writes output to `task-{id}` Mem0 namespace
  - `CoderNode`: reads `task-{id}` Mem0 namespace, runs coder agent with that context
- New Hatchet worker executing the pipeline as a DAG (Hatchet has native DAG support)
- The Phase 5 Vikunja webhook adapter already handles this: when a `task.updated` event arrives with BOTH label 14 and label 11 present, the adapter pushes `pipeline:research_code` instead of the individual `agent:research` and `agent:code` events. No new webhook code needed — update the adapter's label routing logic.

Deploy: add `pipeline-worker` component to `praetor.toml` → `provision_app("praetor")` → `open_deploy_pr("praetor", "phase-8: pipeline worker")`

**Ready conditions (all must pass):**
1. Create Vikunja task "Implement X feature", label both `ai-research` and `ai-go`
2. Hatchet UI shows pipeline run with two sequential steps: `research` then `code`, both `SUCCEEDED`
3. Opened PR description references findings from the research step
4. If research step fails, code step does not start → Vikunja task updated with failure reason
5. Pipeline re-run skips research if `task-{id}` memory already exists (idempotent research step)

---

## What This Does Not Require

- No custom dispatcher code
- No custom agent loop
- No custom task tracking DB schema
- No RabbitMQ / Kafka
- No Neo4j
- No n8n (Hatchet handles cron, webhooks, events natively)
- No Kubernetes-specific operator (Hatchet Lite is a single Docker image)
