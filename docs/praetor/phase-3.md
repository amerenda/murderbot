# Phase 3 — Dispatch: Hatchet on k3s

**Goal:** Events flow through Hatchet. A stub worker receives a trigger and logs it, visible in the Hatchet UI with full task history. No real agent yet.

## Pre-conditions

- Phase 2 complete (Qdrant running, Mac Mini PostgreSQL healthy)
- Hatchet encryption keysets generated and stored in BWS (see below — these are NOT simple passwords)

### Hatchet Encryption Keys — How to Generate (One-Time, Before Provisioning)

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

## What Gets Built

### Provisioning Hatchet Lite

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

### Supplementary Files for Hatchet Deployment

**`configmap.yaml`** — non-secret runtime config:
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

Note: `DATABASE_URL` is a runtime-generated value — Tofu creates the database and writes credentials to BWS; ExternalSecrets reads them into the pod as env vars, which are interpolated into DATABASE_URL at startup.

**gRPC ClusterIP service** — Hatchet's gRPC port (7077) is not exposed via Ingress. Workers in other namespaces need a ClusterIP service:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: hatchet-grpc
  namespace: hatchet
spec:
  selector:
    app: hatchet
  ports:
    - port: 7077
      targetPort: 7077
      protocol: TCP
```

**Worker gRPC networking:** Workers in the `praetor` namespace connect to `hatchet.hatchet.svc.cluster.local:7077`. This requires the `service-grpc.yaml` ClusterIP service above. Workers do NOT need to go through the Ingress — gRPC travels in-cluster.

### Stub Worker

**In `praetor` repo:**
- `workers/stub/main.py` — minimal Hatchet worker: initializes with `Hatchet()` (reads `HATCHET_CLIENT_TOKEN` from env), registers `test:ping` event handler, logs payload, returns `{"status": "ok"}`
- `workers/stub/Dockerfile`

Stub worker component added to `praetor.toml` → `provision_app("praetor")` → PR to k3s-dean-gitops.

Hatchet cron registered at worker startup:
- Every 5 minutes → push `test:ping` event with timestamp payload
- Cron expression as env var: `STUB_CRON_INTERVAL` (default `*/5 * * * *`) — configurable via ConfigMap

## Ready Conditions for Phase 3

1. `https://hatchet.amer.dev` loads the UI and login works with the admin credentials set in step 1 above
2. `hatchet` database exists in Mac Mini PostgreSQL with correct owner (created by `provision_app` Tofu)
3. Stub worker pod is running and shows as connected in Hatchet UI (Workers tab)
4. Manually push `test:ping` event via Hatchet REST API → run appears in UI with status `SUCCEEDED` and logged payload
5. Cron fires on schedule → run appears in UI automatically without manual trigger
6. Kill the stub worker pod mid-run → Hatchet retries and succeeds after pod restarts
7. `kubectl exec` into a test pod in `praetor` namespace: `nc -zv hatchet.hatchet.svc.cluster.local 7077` succeeds (proves gRPC reachable cross-namespace)
8. ArgoCD shows Hatchet app synced and healthy
