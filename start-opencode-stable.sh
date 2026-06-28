#!/bin/bash
# start-opencode-stable.sh — Qwen3.6 + llama.cpp (native tool calling)
#
# Architecture:
#   opencode → LiteLLM (https://litellm.amer.dev/v1)  → llama-server (port 8088)
#   opencode → llama-server (port 8088, local fallback when LiteLLM key absent)
#
# Key flags:
#
#   1. --chat-template-file froggeric-chat-template.jinja      (CRITICAL)
#      Replaces the built-in GGUF template, which has multiple bugs in v20:
#        - Minja replace bug: silently drops entire user message payloads
#        - Empty-think poisoning: stacked <think></think> in history causes
#          model to abort tool calls ~80% of the time
#        - KV cache invalidation: spacing mismatch forces full re-prompt every turn
#        - Minja AST nesting: ~80% throughput drop on llama.cpp from deep loops
#      froggeric/Qwen-Fixed-Chat-Templates v20 (2026-06-05) fixes all of these.
#      Template: https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates
#
#   2. --chat-template-kwargs (replaces --reasoning off)
#      auto_disable_thinking_with_tools: true
#        — Disables <think> blocks specifically during tool calls. Thinking
#          is still available for non-tool responses (complex reasoning).
#          This replaces the old --reasoning off (which disabled ALL thinking).
#      max_tool_response_chars: 3000
#        — Truncates large tool responses at template render time. Replaces
#          the proxy's --max-tool-chars behavior without a separate process.
#          File reads that return 47KB+ are the main overflow offender.
#      preserve_thinking: false
#        — Strips past <think> blocks from conversation history. Prevents
#          the "empty-think poison" and saves tokens across multi-turn sessions.
#
#
# Models:
#   qwen36        (default) — Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf    (~21GB, fully on GPU)
#                             No MTP, NGL=40, CTX=131072
#   qwen36-mtp               — Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf (~22GB, fully on GPU)
#                             MTP enabled (~150 t/s), NGL=40, CTX=131072
#   qwen3-coder-next          — Qwen3-Coder-Next-UD-IQ3_XXS.gguf    (~26.5GB, partial CPU offload)
#                             NGL=40, CTX=32768 (VRAM-constrained)
#
# Usage:
#   ./start-opencode-stable.sh                              # default: Qwen3.6-35B (no MTP)
#   ./start-opencode-stable.sh --model qwen36-mtp           # MTP variant for speed
#   ./start-opencode-stable.sh --model qwen3-coder-next     # coding-focused model
#   ./start-opencode-stable.sh --restart                    # stop existing server first
#
# Env overrides:
#   CTX=196608 ./start-opencode-stable.sh                  # larger context (GPU has headroom up to ~229K)
#   CTX=65536 ./start-opencode-stable.sh                   # smaller context (saves VRAM)
#   NGL=38 ./start-opencode-stable.sh                      # fewer GPU layers
#   VERBOSE=true ./start-opencode-stable.sh                # verbose server logging
#   OPENCODE_RESERVED=80000 ./start-opencode-stable.sh     # less aggressive compaction
#
# VRAM context headroom (RTX 4000 Blackwell, 24 GB):
#   NOTE: observed actual free VRAM at CTX 196608 was only ~73 MB (table below is
#   approximate; model/llama.cpp overhead may vary). NVENC needs ~500 MB for
#   cuCtxCreate; Jellyfin GPU transcoding fails if <500 MB free.
#   CTX 131072 → ~1.1 GB free  ← current default (Jellyfin NVENC coexistence)
#   CTX 163840 → ~0.6 GB free  ← marginal for NVENC
#   CTX 196608 → ~0.1 GB free  ← NVENC will fail (cuCtxCreate OOM)
#   CTX 229376 → OOM risk
#   CTX 262144 → OOM risk
#   Override: CTX=196608 ./start-opencode-stable.sh  (when not running Jellyfin)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── DEFAULTS ────────────────────────────────────────────────────────────────
MODEL_VARIANT="qwen36"
RESTART=false

# ─── CLI ARG PARSING ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_VARIANT="$2"; shift 2 ;;
    --restart) RESTART=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--model qwen36|qwen36-mtp|qwen3-coder-next] [--restart]"
      echo ""
      echo "  qwen36           (default) Qwen3.6-35B-A3B no-MTP — 21GB, stable baseline"
      echo "  qwen36-mtp                 Qwen3.6-35B-A3B MTP    — 22GB, faster (~150 t/s)"
      echo "  qwen3-coder-next           Qwen3-Coder-Next IQ3   — 26.5GB, coding-focused"
      exit 0 ;;
    *) echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ─── MODEL RESOLUTION ────────────────────────────────────────────────────────
MODELS_DIR="/mnt/storage/models/llms"
MTP_ENABLE=false
MTP_N_MAX=2

case "$MODEL_VARIANT" in
  qwen36)
    MODEL="${MODELS_DIR}/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    MODEL_DISPLAY="Qwen3.6-35B (no-MTP, stable)"
    MTP_ENABLE=false
    NGL="${NGL:-41}"
    CTX="${CTX:-131072}"
    REPEAT_PENALTY=1.1
    ;;
  qwen36-mtp)
    MODEL="${MODELS_DIR}/qwen36/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf"
    MODEL_DISPLAY="Qwen3.6-35B MTP (fast)"
    MTP_ENABLE=true
    NGL="${NGL:-99}"
    CTX="${CTX:-131072}"
    REPEAT_PENALTY=1.1
    ;;
  qwen3-coder-next)
    MODEL="${MODELS_DIR}/qwen3-coder-next/Qwen3-Coder-Next-UD-IQ3_XXS.gguf"
    MODEL_DISPLAY="Qwen3-Coder-Next UD-IQ3_XXS"
    MTP_ENABLE=false
    NGL="${NGL_CODER:-${NGL:-40}}"
    CTX="${CTX:-32768}"
    REPEAT_PENALTY=1.0
    ;;
  *)
    echo "ERROR: Unknown --model '$MODEL_VARIANT'. Use: qwen36, qwen36-mtp, qwen3-coder-next" >&2
    exit 1 ;;
esac

# ─── OPENCODE TOKEN BUDGET ───────────────────────────────────────────────────
OPENCODE_CTX="${OPENCODE_CTX:-${CTX}}"
OPENCODE_OUTPUT="${OPENCODE_OUTPUT:-8192}"
OPENCODE_RESERVED="${OPENCODE_RESERVED:-85000}"

MODEL_NAME="$(basename "$MODEL")"

# ─── SERVER BINARY ───────────────────────────────────────────────────────────
SERVER="$HOME/claude/llama.cpp/build/bin/llama-server"
if [ ! -x "$SERVER" ]; then
  echo "ERROR: llama-server not found at $SERVER" >&2
  echo "  Rebuild: cd ~/claude/llama.cpp && cmake --build build -j\$(nproc)" >&2
  exit 1
fi

# ─── MODEL FILE CHECK ────────────────────────────────────────────────────────
if [ ! -f "$MODEL" ]; then
  echo "ERROR: Model file not found: $MODEL" >&2
  exit 1
fi

# ─── OPENCODE CHECK / INSTALL ────────────────────────────────────────────────
if ! command -v opencode &>/dev/null; then
  echo "OpenCode not found. Installing via npm to ~/.local..."
  npm install -g opencode-ai --prefix ~/.local
  export PATH="$HOME/.local/bin:$PATH"
  echo "OpenCode installed."
fi

# ─── SERVER CONFIG ───────────────────────────────────────────────────────────
FA=1
CTK="q4_0"
CTV="q4_0"
BATCH="${BATCH:-512}"
THREADS="${THREADS:-6}"
HOST="0.0.0.0"
PORT="${PORT:-8088}"
PARALLEL=1
VERBOSE="${VERBOSE:-false}"

# ─── HELPERS ─────────────────────────────────────────────────────────────────
RESTART_SENTINEL="/tmp/llama-server-restarting"

stop_server() {
  local PIDS
  PIDS=$(pgrep -f "llama-server" 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo "Stopping llama-server (PIDs: $PIDS)..."
    touch "$RESTART_SENTINEL"
    echo "$PIDS" | xargs kill
    sleep 4
    if pgrep -f "llama-server" > /dev/null 2>&1; then
      pgrep -f "llama-server" | xargs kill -9
      sleep 2
    fi
    echo "Stopped."
  else
    echo "No running llama-server — nothing to stop."
  fi
}


stop_comfyui() {
  local BATCH_PIDS
  BATCH_PIDS=$(pgrep -f "run-tuning-batch\.py|run-flux-batch\.py" 2>/dev/null || true)
  if [ -n "$BATCH_PIDS" ]; then
    echo "[OPENCODE] Stopping image generation batch..."
    echo "$BATCH_PIDS" | xargs kill 2>/dev/null || true
    sleep 1
  fi
  local PIDS
  PIDS=$(pgrep -f "main\.py --listen" 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo "[OPENCODE] Stopping ComfyUI/FLUX (freeing VRAM)..."
    echo "$PIDS" | xargs kill
    sleep 3
    if pgrep -f "main\.py --listen" > /dev/null 2>&1; then
      pgrep -f "main\.py --listen" | xargs kill -9
      sleep 1
    fi
    echo "[OPENCODE] ComfyUI stopped."
  fi
}

# ─── STOP IF RESTARTING ──────────────────────────────────────────────────────
if [ "$RESTART" = true ]; then
  stop_server
  echo "Waiting for VRAM to drain..."
  sleep 5
fi

# ─── LOG FILE ────────────────────────────────────────────────────────────────
BASE_LOG="/tmp/llama-server-stable-$(echo "$MODEL" | md5sum | cut -d' ' -f1).log"
LOG_MAX_MB=2048
LOG_KEEP=3

if [ -f "$BASE_LOG" ]; then
  LOG_SIZE_MB=$(( $(stat -c%s "$BASE_LOG") / 1024 / 1024 ))
  if [ "$LOG_SIZE_MB" -ge "$LOG_MAX_MB" ]; then
    for i in $(seq $((LOG_KEEP - 1)) -1 1); do
      [ -f "${BASE_LOG}.${i}" ] && mv "${BASE_LOG}.${i}" "${BASE_LOG}.$((i + 1))"
    done
    mv "$BASE_LOG" "${BASE_LOG}.1"
    echo "Rotated log (was ${LOG_SIZE_MB} MB) → ${BASE_LOG}.1"
  fi
fi

# ─── CHECK FOR EXISTING SERVER ───────────────────────────────────────────────
if pgrep -f "llama-server" > /dev/null 2>&1; then
  echo "A llama-server is already running on :$PORT"

  CURRENT_MODEL=$(curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || true)

  if [ "$(basename "${CURRENT_MODEL:-}")" = "$MODEL_NAME" ]; then
    echo "Already serving $MODEL_NAME — connecting directly."
  else
    echo "Server is serving a different model ($(basename "${CURRENT_MODEL:-unknown}"))."
    echo "Stopping it to load $MODEL_NAME..."
    stop_server
    echo "Waiting for VRAM to drain..."
    sleep 5
  fi
fi

# ─── FREE VRAM (stop ComfyUI if running) ─────────────────────────────────────
if pgrep -f "main\.py --listen|run-tuning-batch\.py|run-flux-batch\.py" > /dev/null 2>&1; then
  stop_comfyui
fi

# ─── START LLAMA-SERVER (if not already running) ─────────────────────────────
if ! pgrep -f "llama-server" > /dev/null 2>&1; then
  echo "Starting llama-server (stability mode)..."
  echo "  Model:    $MODEL_NAME"
  echo "  Display:  $MODEL_DISPLAY"
  echo "  Template: GGUF embedded (Unsloth-patched, no override)"
  echo "  Thinking: off during tool calls (auto_disable_thinking_with_tools)"
  if [ "$MTP_ENABLE" = true ]; then
    echo "  MTP:      enabled (--spec-type draft-mtp --spec-draft-n-max $MTP_N_MAX)"
  else
    echo "  MTP:      disabled"
  fi
  echo "  ngl=$NGL  fa=$FA  ctk=$CTK  ctv=$CTV"
  echo "  ctx=$CTX  repeat_penalty=$REPEAT_PENALTY"
  echo "  Listen:   http://$HOST:$PORT"
  echo "  Log:      $BASE_LOG"
  echo ""

  VERBOSE_FLAG=""
  [ "$VERBOSE" = "true" ] && VERBOSE_FLAG="--verbose"

  SERVER_ARGS=(
    -m      "$MODEL"
    --device CUDA0
    -ngl    "$NGL"
    -fa     "$FA"
    -ctk    "$CTK"
    -ctv    "$CTV"
    -c      "$CTX"
    -b      "$BATCH"
    -t      "$THREADS"
    --host  "$HOST"
    --port  "$PORT"
    --jinja
    --chat-template-file "$SCRIPT_DIR/templates/froggeric-v20.jinja"
    --chat-template-kwargs '{"auto_disable_thinking_with_tools": true, "max_tool_response_chars": 3000, "preserve_thinking": false}'
    # Note: auto_disable_thinking_with_tools only suppresses thinking when < 3 tool results
    # are present in the conversation (template v20 logic). Once 3+ tool results accumulate,
    # thinking re-enables automatically so the model can synthesize complex research output.
    --metrics
    --timeout       0
    --parallel      "$PARALLEL"
    --cache-reuse   256
    --repeat-penalty "$REPEAT_PENALTY"
    --temp   0.6
    --top-k  20
    --top-p  0.95
    --min-p  0.05
    $VERBOSE_FLAG
  )

  if [ "$MTP_ENABLE" = true ]; then
    SERVER_ARGS+=(--spec-type draft-mtp --spec-draft-n-max "$MTP_N_MAX")
  fi

  "$SERVER" "${SERVER_ARGS[@]}" >> "$BASE_LOG" 2>&1 &
  SERVER_PID=$!
  disown "$SERVER_PID"  # survive terminal SIGHUP

  # Crash watcher — suppressed if a restart sentinel is present (intentional kill)
  (
    while kill -0 "$SERVER_PID" 2>/dev/null; do sleep 5; done
    if [ -f "$RESTART_SENTINEL" ]; then
      rm -f "$RESTART_SENTINEL"
      exit 0
    fi
    echo "" >&2
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
    echo "ERROR: llama-server (PID $SERVER_PID) has died!" >&2
    echo "Last log entries:" >&2
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
    tail -40 "$BASE_LOG" >&2
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
  ) &
  WATCHER_PID=$!

  # Wait for server ready
  echo "Waiting for llama-server to load model..."
  ATTEMPTS=0
  until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
    if [ "$ATTEMPTS" -ge 120 ]; then
      echo "ERROR: Timeout waiting for server after 240s." >&2
      echo "Check $BASE_LOG for details." >&2
      exit 1
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "ERROR: Server exited unexpectedly. Check $BASE_LOG." >&2
      tail -30 "$BASE_LOG" >&2
      exit 1
    fi
    sleep 2
    ATTEMPTS=$((ATTEMPTS + 1))
  done
  echo "llama-server ready."
  echo ""
fi

# ─── LITELLM KEY ─────────────────────────────────────────────────────────────
# Auto-fetch from k8s secret if not already set in env
if [ -z "${LITELLM_MASTER_KEY:-}" ]; then
  LITELLM_MASTER_KEY=$(kubectl get secret -n litellm litellm-secrets \
    -o jsonpath='{.data.master-key}' 2>/dev/null | base64 -d 2>/dev/null || true)
fi

# ─── OPENCODE CONFIG ─────────────────────────────────────────────────────────
PERMS='"read":"allow","edit":"allow","glob":"allow","grep":"allow","list":"allow","bash":"allow","task":"allow","external_directory":"allow","todowrite":"allow","webfetch":"allow","websearch":"allow","repo_clone":"allow","repo_overview":"allow","lsp":"allow","doom_loop":"allow","skill":"allow","question":"deny"'
AGENT_PROMPT='Proceed with tasks autonomously without stopping mid-task to ask for confirmation or check-ins. Never ask '"'"'did you sync?'"'"', '"'"'is it pushed?'"'"', '"'"'should I continue?'"'"', or any similar check-in. If you said you will do something, do it immediately. Only stop if you are completely blocked and need information only the user can provide.'
COMPACTION='"auto":true,"prune":true,"reserved":'"${OPENCODE_RESERVED}"

if [ -n "${LITELLM_MASTER_KEY:-}" ]; then
  # ── LiteLLM path: opencode → https://litellm.amer.dev/v1 ──────────────────
  OPENCODE_CONFIG_JSON=$(cat << OPENCODE_JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "litellm/qwen3-35b",
  "permission": {${PERMS}},
  "agent": {"build": {"prompt": "${AGENT_PROMPT}"}},
  "provider": {
    "litellm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LiteLLM (litellm.amer.dev)",
      "options": {
        "baseURL": "https://litellm.amer.dev/v1",
        "apiKey": "${LITELLM_MASTER_KEY}"
      },
      "models": {
        "qwen3-35b": {
          "name": "Qwen3.6-35B via LiteLLM",
          "limit": {
            "context": ${OPENCODE_CTX},
            "input": ${OPENCODE_CTX},
            "output": ${OPENCODE_OUTPUT}
          }
        }
      }
    }
  },
  "compaction": {${COMPACTION}}
}
OPENCODE_JSON
)
  mkdir -p ~/.config/opencode
  echo "$OPENCODE_CONFIG_JSON" > ~/.config/opencode/opencode.json
  echo "OpenCode config ready (LiteLLM mode)"
  echo "  model:      qwen3-35b via https://litellm.amer.dev/v1"
  echo "  ctx:        $OPENCODE_CTX tokens  max-output: $OPENCODE_OUTPUT"
  echo "  compact:    auto, reserved=$OPENCODE_RESERVED"
  echo "  perms:      allow (all tool calls auto-approved)"
  echo ""

else
  # ── Local path: opencode → llama-proxy → llama-server ─────────────────────
  OPENCODE_CONFIG_JSON=$(cat << OPENCODE_JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "llamacpp/${MODEL_NAME}",
  "permission": {${PERMS}},
  "agent": {"build": {"prompt": "${AGENT_PROMPT}"}},
  "provider": {
    "llamacpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-server (stable)",
      "options": {
        "baseURL": "http://127.0.0.1:${PORT}/v1",
        "apiKey": "local"
      },
      "models": {
        "${MODEL_NAME}": {
          "name": "${MODEL_DISPLAY}",
          "limit": {
            "context": ${OPENCODE_CTX},
            "input": ${OPENCODE_CTX},
            "output": ${OPENCODE_OUTPUT}
          }
        }
      }
    }
  },
  "compaction": {${COMPACTION}}
}
OPENCODE_JSON
)
  mkdir -p ~/.config/opencode
  echo "$OPENCODE_CONFIG_JSON" > ~/.config/opencode/opencode.json
  echo "OpenCode config ready (local mode — LiteLLM not available)"
  echo "  model:      $MODEL_DISPLAY"
  echo "  server ctx: $CTX  opencode ctx: $OPENCODE_CTX  max-output: $OPENCODE_OUTPUT"
  echo "  compact:    auto, reserved=$OPENCODE_RESERVED"
  echo "  perms:      allow (all tool calls auto-approved)"
  if [ "$OPENCODE_PORT" = "$PROXY_PORT" ]; then
    echo "  server:     http://127.0.0.1:$PORT"
  echo ""
fi

# ─── LAUNCH OPENCODE ─────────────────────────────────────────────────────────
echo "Starting opencode..."
OPENCODE_CONFIG_CONTENT="$OPENCODE_CONFIG_JSON" opencode

# Cleanup
kill "${WATCHER_PID:-}" 2>/dev/null || true
