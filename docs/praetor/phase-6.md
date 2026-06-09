# Phase 6 — Coder Agent

**Goal:** Tag a Vikunja task `ai-go` with a repo reference → draft PR opened on that repo.

## Pre-conditions

- Phases 1–4 complete
- `dean-coder` GitHub App created, installed on `amerenda` org, credentials in BWS (see Phase 0 bootstrap)

## What Gets Built — `praetor/agents/coder/`

- `agent.py` — PydanticAI agent with tools:
  - `git_clone(repo, branch)`, `git_commit(message)`, `git_push()`, `open_pr(title, body)`
  - `read_file(path)`, `write_file(path, content)`, `run_shell(cmd)` (scoped to `SCRATCH_DIR` emptyDir mount)
  - `search_memory(query, agent_id)` → calls `common/memory_tools.py`, reads research context from `task-{id}` namespace if it exists
- `worker.py` — Hatchet worker handling `agent:code` events
- `common/github_app.py` — shared helper: `get_installation_token(app_id, private_key)` → short-lived token used for git push and GitHub API calls
- Vikunja webhook adapter (Phase 5) already handles label 11 (`ai-go`) → no code change needed in adapter

Commits and PRs opened by this worker appear as `dean-coder[bot]` in the GitHub UI — distinct from human-authored commits at a glance.

### Coder Worker TOML Additions

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

**Deploy:** add `coder-worker` component to `praetor.toml` → `provision_app("praetor")` → `open_deploy_pr("praetor", "phase-6: coder worker")`

## Ready Conditions for Phase 6

1. Create Vikunja task "Add /healthz endpoint to ecdysis", label `ai-go`, repo `amerenda/ecdysis` in description
2. Within 30 seconds: Hatchet UI shows `agent:code` run triggered (webhook-based, not polling)
3. A draft PR exists on `amerenda/ecdysis` on a new branch, opened by `dean-coder[bot]`
4. PR description references the Vikunja task ID
5. Vikunja task is marked done with the PR link as a comment
6. Task with a broken/missing repo reference → agent fails gracefully with a Vikunja comment, no unhandled exception
