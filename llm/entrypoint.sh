#!/bin/bash
# entrypoint.sh — llama-server container entry point
# Env vars match start-opencode-stable.sh conventions so switching is seamless.
set -euo pipefail

MODEL="${MODEL:-/mnt/models/llms/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
NGL="${NGL:-41}"
CTX="${CTX:-131072}"
PORT="${PORT:-8088}"
HOST="${HOST:-0.0.0.0}"
BATCH="${BATCH:-512}"
THREADS="${THREADS:-6}"
REPEAT_PENALTY="${REPEAT_PENALTY:-1.1}"
CACHE_REUSE="${CACHE_REUSE:-256}"
VERBOSE="${VERBOSE:-false}"
MTP_ENABLE="${MTP_ENABLE:-false}"
MTP_N_MAX="${MTP_N_MAX:-2}"

if [ ! -f "$MODEL" ]; then
  echo "ERROR: Model file not found: $MODEL" >&2
  echo "  Mount the model directory: -v /mnt/storage/models:/mnt/models:ro" >&2
  exit 1
fi

echo "Starting llama-server..."
echo "  Model:   $(basename "$MODEL")"
echo "  NGL:     $NGL  CTX: $CTX  Port: $PORT"
echo "  Template: /app/templates/froggeric-v20.jinja"
echo ""

SERVER_ARGS=(
    -m      "$MODEL"
    --device CUDA0
    -ngl    "$NGL"
    -fa     1
    -ctk    q4_0
    -ctv    q4_0
    -c      "$CTX"
    -b      "$BATCH"
    -t      "$THREADS"
    --host  "$HOST"
    --port  "$PORT"
    --jinja
    --chat-template-file /app/templates/froggeric-v20.jinja
    --chat-template-kwargs '{"auto_disable_thinking_with_tools": true, "max_tool_response_chars": 3000, "preserve_thinking": false}'
    --reasoning-budget 8192
    --cache-ram     0
    --cache-reuse   "$CACHE_REUSE"
    --metrics
    --timeout       0
    --parallel      1
    --repeat-penalty "$REPEAT_PENALTY"
    --temp   0.6
    --top-k  20
    --top-p  0.95
    --min-p  0.05
)

if [ "$VERBOSE" = "true" ]; then
    SERVER_ARGS+=(--verbose)
fi

if [ "$MTP_ENABLE" = "true" ]; then
    SERVER_ARGS+=(--spec-type draft-mtp --spec-draft-n-max "$MTP_N_MAX")
fi

exec llama-server "${SERVER_ARGS[@]}"
