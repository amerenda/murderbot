# Phase 7 — PR Reviewer + QA

**Goal:** GitHub events drive agents automatically without Vikunja labels.

## Pre-conditions

- Phases 1–4 complete
- GitHub webhook can reach `praetor.amer.dev` (public ingress already set up)
- `dean-reviewer` GitHub App created, installed on `amerenda` org, credentials in BWS (see Phase 0 bootstrap)
- A PAT or GitHub App token with `admin:org_hook` scope available in BWS for Tofu to register the org webhook (can reuse `dean-reviewer` app if it has org hook permissions, or a separate deploy token)

## What Gets Built

### Webhook Registration Is IaC

Declared in `praetor.toml` as a `github_webhooks` block, registered by `provision_app` via Tofu's `github` provider:

```toml
[[github_webhooks]]
org            = "amerenda"
target_url     = "https://praetor.amer.dev/webhooks/github"
events         = ["pull_request"]
secret_bws_key = "github-webhook-secret"   # generate=true; Tofu creates and stores it
```

Tofu calls `POST /orgs/{org}/hooks` (GitHub Webhooks API). Running `provision_app` again is idempotent — it upserts the registration.

### PR Reviewer — `praetor/agents/pr_reviewer/`

- `agent.py` — PydanticAI agent: reads PR diff via GitHub API using `dean-reviewer` app token, posts structured review comment
- `worker.py` — Hatchet worker handling `github:pr_opened` events; reads `REVIEWER_APP_ID` + `REVIEWER_APP_PRIVATE_KEY` from env, calls `common/github_app.py` to get an installation token
- Webhook adapter at `praetor/webhooks/github.py` (same FastAPI app as Vikunja adapter):
  - Validates `X-Hub-Signature-256` HMAC-SHA256 against `GITHUB_WEBHOOK_SECRET` (stored in BWS)
  - On `pull_request.opened`: pushes `github:pr_opened` Hatchet event with `{repo, pr_number, pr_url, diff_url, author}`
  - Returns 401 on invalid signature, 200 on success — GitHub retries on non-2xx

Reviews submitted by this worker appear as `dean-reviewer[bot]` — distinct from human reviews and from `dean-coder[bot]` commits.

### Reviewer App Credentials in TOML

```toml
[[components.env]]
name = "REVIEWER_APP_ID"
secret_ref = { bws_key = "dean-reviewer-app-id" }
[[components.env]]
name = "REVIEWER_APP_PRIVATE_KEY"
secret_ref = { bws_key = "dean-reviewer-private-key" }
```

### QA — `praetor/agents/qa/`

- `agent.py` — PydanticAI agent: tests staging URL using `playwright` Python library (HTTP + UI flows, not just curl)
- `worker.py` — Hatchet worker handling `deploy:staging` events
- CI pipeline: push `deploy:staging` event to Hatchet after UAT deploy completes

#### `deploy:staging` Event Payload Schema

Defined here, used by all CI pipelines that trigger QA. All fields are strings; optional ones may be omitted:

```json
{
  "repo": "amerenda/ecdysis",
  "pr_number": 42,
  "branch": "feature/my-feature",
  "deploy_url": "https://ecdysis-staging.amer.dev",
  "commit_sha": "abc123def",
  "author": "human-dev"
}
```

**Deploy:** add `reviewer-worker` and `qa-worker` components to `praetor.toml` → `provision_app("praetor")` → `open_deploy_pr("praetor", "phase-7: pr reviewer + qa worker")`

## Ready Conditions for Phase 7

1. Open a PR on `amerenda/ecdysis` → Hatchet UI shows `github:pr_opened` run within 5 seconds
2. Review comment appears on the PR from `dean-reviewer[bot]` with actionable feedback
3. Trigger `deploy:staging` event via CLI or CI pipeline → QA worker runs and posts results
4. Invalid GitHub webhook signature is rejected (401), valid one succeeds (200)
