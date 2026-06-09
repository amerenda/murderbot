#!/usr/bin/env bash
# Install LiteLLM proxy on murderbot as a drop-in replacement for llm-agent.
#
# This script:
#   1. Stops and removes the existing llm-agent container
#   2. Creates /opt/litellm/ with config.yaml, .env, and compose files
#   3. Starts LiteLLM proxy on port 4000 (replaces agent's 8090)
#   4. Updates Traefik labels to route to the new port
#
# Usage: sudo bash install-liteollm.sh
set -euo pipefail

LITELLM_DIR="/opt/litellm"
COMPOSE_FILE="${LITELLM_DIR}/compose.yaml"
CONFIG_FILE="${LITELLM_DIR}/config.yaml"
ENV_FILE="${LITELLM_DIR}/.env"
GITOPS_ENV="/home/alex/komodo-dean-gitops/murderbot/llm/.env"

log() { echo "[liteollm] $*"; }

###############################################################################
# 1. Stop & remove existing llm-agent
###############################################################################
log "Stopping and removing existing llm-agent container..."
docker compose --profile nvidia-full -f /home/alex/komodo-dean-gitops/murderbot/llm/compose.yaml down 2>/dev/null || true
docker stop llm-agent 2>/dev/null || true
docker rm llm-agent 2>/dev/null || true

###############################################################################
# 2. Create LiteLLM directory and config files
###############################################################################
log "Creating ${LITELLM_DIR}..."
mkdir -p "$LITELLM_DIR"

# --- .env -------------------------------------------------------------------
cat > "$ENV_FILE" <<'ENVEOF'
# LiteLLM Proxy configuration — populated by GitOps / manual edit
# Required: at least one of these must be set.

# Primary API key for the proxy (used as Bearer token by clients)
LITELLM_MASTER_KEY=sk-litellm-master-key-change-me

# Optional: allow all models without per-model API keys
ALLOW_ALL_PROXY_ACCESS=true

# Ollama local endpoint
OLLAMA_BASE_URL=http://localhost:11434
ENVEOF

log "Created ${ENV_FILE}"

# --- config.yaml ------------------------------------------------------------
cat > "$CONFIG_FILE" <<'CFGEOF'
# LiteLLM Proxy configuration
#
# model routing rules — models not matched here fall through to Ollama.
models: []  # empty = no per-model keys required when ALLOW_ALL_PROXY_ACCESS=true

litellm_settings:
  # Pass-through settings
  set_api_base: true
  verbose: false

# Router rules: unmatched models → local Ollama
router:
  - model_names: ["*"]
    provider: ollama
    config:
      model: "*"
      api_base: ${OLLAMA_BASE_URL:-http://localhost:11434}

# Management API key (for /status, /key/generate endpoints)
general_settings:
  master_key: "${LITELLM_MASTER_KEY:-sk-litellm-master-key-change-me}"
  proxy_auth_protocol: "https"
  host: "0.0.0.0"
  port: 4000

# Model routing — explicit overrides for cloud providers go here
# Add entries like this when you have API keys:
#   - model_name: claude-sonnet-4-20250514
#     litellm_params:
#       model: anthropic/claude-sonnet-4-20250514
#       api_key: sk-ant-xxx

# Health check settings
health_check_settings:
  health_check_interval_seconds: 30
CFGEOF

log "Created ${CONFIG_FILE}"

# --- compose.yaml -----------------------------------------------------------
cat > "$COMPOSE_FILE" <<'CPEOF'
services:
  litellm-proxy:
    image: ghcr.io/berriai/litellm:main-stable
    container_name: litellm-proxy
    network_mode: host
    restart: unless-stopped
    env_file: .env
    volumes:
      - ./config.yaml:/app/config.yaml:ro
    command: >
      --config /app/config.yaml
      --port 4000
      --detailed_debug
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:4000/health/liveliness"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s

volumes:
  litellm-data:
CPEOF

log "Created ${COMPOSE_FILE}"

###############################################################################
# 3. Start LiteLLM proxy
###############################################################################
log "Starting LiteLLM proxy..."
cd "$LITELLM_DIR"
docker compose up -d

###############################################################################
# 4. Update Traefik routing (if applicable)
###############################################################################
log "Updating Traefik labels on llm-manager-backend to route via port 4000..."

# The existing agent was on port 8090; LiteLLM is on 4000.
# We update the backend's Traefik label so external traffic flows through LiteLLM.
# This assumes a Kubernetes service or docker container with Traefik labels.
BACKEND_POD=$(kubectl get pods -l app=llm-manager-backend -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)

if [[ -n "$BACKEND_POD" ]]; then
  kubectl annotate pod "$BACKEND_POD" \
    "traefik.http.routers.llm-agent.rule=Host(\`llm.amer.dev\`) && PathPrefix(\`/v1/chat/completions\`)" \
    --overwrite 2>/dev/null || true
  log "Updated Traefik routing for ${BACKEND_POD}"
else
  log "No llm-manager-backend pod found — skipping Traefik update."
  log "You may need to manually update the backend's route to point to port 4000."
fi

###############################################################################
# Done
###############################################################################
log ""
log "=== LiteLLM proxy is running ==="
log "Endpoint: http://10.100.20.19:4000/v1/chat/completions"
log "API key:  ${LITELLM_MASTER_KEY:-sk-litellm-master-key-change-me}"
log "Config:   ${CONFIG_FILE}"
log ""
log "Test with:"
log '  curl -H "Authorization: Bearer sk-litellm-master-key-change-me" \\\n' \
     '       -H "Content-Type: application/json" \\\n' \
     '       http://10.100.20.19:4000/v1/chat/completions \\\n' \
     '       -d '\''{"model":"qwen3:8b","messages":[{"role":"user","content":"hi"}]}'\'''
