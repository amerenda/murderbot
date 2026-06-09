# Phase 0 — Deployment Foundation

This is a reference phase, not a build sequence. It defines the canonical patterns for deploying anything in this infrastructure. All subsequent phases follow these rules. An AI given any deployment task should apply these patterns without being told.

---

## Secret Management — Hard Rules

1. **BWS is the single source of truth for all secrets.** No exceptions.
2. **Nothing secret goes in git.** No passwords, tokens, keys, or DSNs in any repo, ever.
3. **No hand-written `.env` files.** `.env` files are only valid if populated by BWS automation at runtime (e.g., Komodo's BWS sync, a CI step, or a BWS-aware startup script).
4. **Ansible takes exactly one secret as input:** `BWS_ACCESS_TOKEN` (read-only key). Everything else — postgres passwords, API keys, encryption keys — is read from BWS at runtime via that token.
5. **Generated secrets (passwords, keys) are created by OpenTofu** using `random_password` and written directly to BWS via the `bitwarden-secrets` provider. Humans never generate these.
6. **Manual secrets (third-party API keys, tokens)** are created by a human in the BWS UI. The TOML spec or Ansible playbook declares the BWS key name; it does not set the value.
7. **k3s secrets are delivered via ExternalSecrets operator.** ExternalSecrets reads from BWS (`ClusterSecretStore: bitwarden-secretstore`) and creates native k8s Secrets. Pods consume k8s Secrets — never env vars with inline values.

---

## App Type Decision Tree

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

## Pattern A: Stateless App (k3s)

**Toolchain:** app-factory (Tofu + generate.py) → k3s-dean-gitops → ArgoCD

### Step 1 — Write the spec

Create `app-factory/apps/<name>.toml`. Declare:
- App name, domain, namespace
- Components (image, port, replicas)
- Secrets (`generate=true` for random, `generate=false` + `bws_key` for existing)
- Database (Tofu creates DB + role in Mac Mini PostgreSQL)

### Step 2 — Provision + generate (one command)

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

### Step 3 — Deploy

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

## Pattern B: Stateful App (Komodo on Mac Mini)

**Toolchain:** Tofu (databases + secrets) → komodo-dean-gitops → Komodo (deploy)

Ansible is **not** part of this flow. Ansible is run by a human when provisioning a new server (installing Docker, PostgreSQL, system packages) and for correcting configuration drift. It is never triggered by an AI agent and is never part of a deployment pipeline.

### Step 1 — Provision databases and secrets (Tofu)

If the stateful service needs a PostgreSQL database or generated secrets, add a spec to `app-factory/apps/<name>.toml` and run `provision_app("<name>")`. Tofu creates the database/role in Mac Mini PostgreSQL and writes secrets to BWS, identical to the stateless path. If there is no database needed, skip this step.

Docker volumes do not need pre-creation. Named volumes declared in compose files are created automatically by Docker on first start and persist across redeploys. Do not use `external: true` unless the volume was created by a separate system. Do not use Ansible to pre-create volumes.

### Step 2 — Service definition (GitOps)

Add the service to the appropriate Komodo stack in `komodo-dean-gitops/mac-mini-m4/<stack>/compose.yaml`.
- Secrets referenced as `${SECRET_NAME}` — values come from Komodo's BWS-synced env at deploy time
- No hardcoded values in compose files
- Volumes declared without `external: true` — Komodo/Docker creates them on first deploy
- New stacks: add `<stack>/compose.yaml` and register in Komodo

### Step 3 — Deploy

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

## Toolchain Reference

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

## What Gets Built in Phase 0

The goal is that an AI agent can be told "build a new stateless app called X" and know exactly what to do — not because it read docs, but because there is one tool to call and the tool enforces every rule.

### 1. `infra-mcp` (new repo: `amerenda/infra-mcp`)

An MCP server that exposes two tools for managing this infrastructure:

**`provision_app(name)`** — Creates a new app spec in `app-factory/apps/<name>.toml`, runs Tofu provision, generates k8s manifests via generate.py. Returns the paths to all generated files. Handles both stateless (Pattern A) and stateful (Pattern B) apps based on an `app_type` field in the TOML spec.

**`resolve_secret_name(name)`** — Looks up a secret by name from BWS via the Bitwarden API using the service account token stored in the MCP server's env (`BWS_SERVICE_ACCOUNT_TOKEN`). Returns the value; used when compose files or config need to reference a specific secret without hardcoding UUIDs.

The MCP server runs as a stateless k3s Deployment, exposed at `https://infra-mcp.amer.dev`. It is the single entry point for infrastructure provisioning — no direct Tofu/generate.py calls from agents.

**In `praetor.toml`:**
```toml
[[secrets]]
name = "BWS_SERVICE_ACCOUNT_TOKEN"
bws_key = "infra-mcp-bws-token"
generate = false
```

### 2. `bws-mcp` — Secrets MCP Server (new repo: `amerenda/bws-mcp`)

A dedicated FastMCP server that wraps the BWS API with explicit, enforced permission tiers. Agents get exactly the access they need — no more.

**Three permission levels, enforced by which token the client presents:**

| Level | Can do | Cannot do |
|-------|--------|-----------|
| **Read** | Look up secrets by name or ID; list secret names | Create, update, or delete secrets |
| **Write** | Read + create/update/delete secrets (app-factory provisioner) | List all secrets; access other apps' secrets |
| **Admin** | Full BWS API access (only for infra-mcp internal use) | N/A |

Each app gets its own BWS service account with scoped permissions. The MCP server enforces these at the API layer — even if a token is leaked, it can only access what its permission tier allows.

**In `praetor.toml`:**
```toml
[[secrets]]
name = "BWS_SERVICE_ACCOUNT_TOKEN"
bws_key = "bws-mcp-admin-token"
generate = false
```

### 3. GitHub Apps (one-time bootstrap)

Two GitHub Apps give each actor a distinct identity in the GitHub UI — your commits, AI-generated commits, and AI reviews are all visually distinct without any convention enforcement.

> **Cannot be created with OpenTofu.** The `integrations/github` provider has no `github_app` resource. Apps are created once in the GitHub UI; credentials are then stored in BWS and referenced as `generate = false` secrets.

**`dean-coder`** — used by the coder agent to push commits and open PRs.
- Commits show as: _"authored by Alex, committed by dean-coder[bot]"_ (or future alias)
- Permissions: Contents (read/write), Metadata (read), Pull requests (read/write)

**`dean-reviewer`** — used by the PR reviewer agent to post review comments.
- Permissions: Pull requests (read/write), Metadata (read), Contents (read)
- No write access to code — can only comment and label, never push

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

### 4. Agent.md Templates (three repos)

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

### 5. Ingress + TLS (one-time bootstrap)

A single Traefik ingress gateway at `https://praetor.amer.dev` with wildcard TLS, plus DNS entries for each service subdomain.

**DNS records:**
| Subdomain | Points to | Purpose |
|-----------|-----------|---------|
| `hatchet.amer.dev` | Load balancer IP (Traefik) | Hatchet UI + API |
| `litellm.amer.dev` | Load balancer IP (Traefik) | LiteLLM proxy |
| `mem0.amer.dev` | Load balancer IP (Traefik) | Mem0 memory API |
| `praetor.amer.dev` | Load balancer IP (Traefik) | Vikunja webhook adapter + agent APIs |

**TLS:** Let's Encrypt wildcard cert for `*.amer.dev`. Managed by Traefik's ACME resolver. The same cert covers all subdomains.

**Ingress rules in `k3s-dean-gitops`:**
```yaml
# traefik ingress route (example)
apiVersion: traefik.io/v1alpha1
kind: IngressRoute
metadata:
  name: praetor-ingress
  namespace: praetor
spec:
  entryPoints: ["websecure"]
  tls:
    secretName: amer-dev-tls
  routes:
    - match: Host(`praetor.amer.dev`) && PathPrefix(`/webhooks/vikunja`)
      kind: Rule
      services:
        - name: vikunja-adapter
          port: 8000
    - match: Host(`hatchet.amer.dev`)
      kind: Rule
      services:
        - name: hatchet
          port: 8080
```

**Ready conditions (all must pass):**
1. `https://praetor.amer.dev` resolves and serves TLS
2. All four subdomains resolve to the load balancer IP
3. Traefik ACME cert is valid and auto-renews (check with `openssl s_client`)
4. Vikunja webhook URL (`https://praetor.amer.dev/webhooks/vikunja`) returns 200 from a test POST

---

### 6. Agent.md Templates — Phase 0 Summary

The agent.md files serve as the AI's onboarding documentation for each repo. They encode every rule and pattern so an AI can work autonomously without human hand-holding.

**Key rules across all three repos:**
- Secrets flow from BWS → ExternalSecrets/Komodo → pods/compose (never inline)
- UAT is auto-deployed; prod requires PR approval
- One tool to call (`provision_app`), one pattern per app type
- Idempotent operations everywhere

---

## Ready Conditions for Phase 0

1. `infra-mcp` deployed and responding at `https://infra-mcp.amer.dev`
2. `bws-mcp` deployed with three permission tiers enforced
3. Both GitHub Apps (`dean-coder`, `dean-reviewer`) created, installed on `amerenda` org, credentials in BWS
4. All three agent.md templates committed to their respective repos
5. Ingress + wildcard TLS working for all four subdomains
6. A test stateless app deployed end-to-end via `provision_app` → ArgoCD sync
7. A test stateful app deployed end-to-end via `provision_app` → Komodo deploy
