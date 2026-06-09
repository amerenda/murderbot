# Phase 2 — Storage: Qdrant on Mac Mini Core Stack

**Goal:** Qdrant running and reachable from k3s. `hatchet` and `mem0` databases are provisioned by Tofu in Phases 3 and 4 respectively — not here.

**Current state:** Mac Mini core stack has Technitium, PostgreSQL (pgvector/pg16, `network_mode: host`, port 5432, databases: `todo`, `agent_kb`), MongoDB. No Qdrant.

**Rule:** No manual steps. All service changes via GitOps (Komodo for Mac Mini, ArgoCD for k3s). Databases provisioned by Tofu. No Ansible in the deploy flow.

## Pre-conditions

- Mac Mini core stack healthy
- `QDRANT_API_KEY` secret provisioned in Bitwarden (via `provision_app("qdrant")` or manually with `bws secret create`)

## What Gets Built

### Qdrant Service Definition

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

### PostgreSQL Init Script (Fresh Install Bootstrap Only)

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

## Execution Order

Commit the compose change to `komodo-dean-gitops` main → Komodo picks up the change and deploys Qdrant. Docker creates `qdrant-data` automatically on first container start.

## Ready Conditions for Phase 2

1. `curl -sf -H "api-key: $QDRANT_API_KEY" http://10.100.20.18:6333/healthz` returns `{"title":"qdrant - version x.x.x"}`
2. Qdrant reachable from inside a k3s pod: `kubectl run -it --rm --image=curlimages/curl test -- curl -sf -H "api-key: $KEY" http://10.100.20.18:6333/healthz`
3. Smoke test via Qdrant REST API: create a collection, insert one vector, delete it — no errors
4. Komodo shows core stack healthy after de
