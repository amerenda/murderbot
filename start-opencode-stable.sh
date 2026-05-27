#!/bin/bash
# start-opencode-stable.sh — Qwen3.6 + llama.cpp + overflow-recovery proxy
#
# Architecture:
#   opencode → llama-proxy (port 8089) → llama-server (port 8088)
#
#   llama-proxy.py (this dir) intercepts /v1/chat/completions requests. If
#   llama-server returns 400 (prompt exceeds context window), the proxy strips
#   the oldest/largest messages and retries transparently. This lets opencode
#   use the full server context without crashing on overflows.
#
# Key flags:
#
#   1. --reasoning off                                         (CRITICAL)
#      Disables Qwen3 thinking mode. Without this, the model generates reasoning
#      text before the tool call and the parser fails on "text before <tool_call>".
#      (Replaced deprecated --chat-template-kwargs '{"enable_thinking": false}'
#      + --reasoning auto from earlier builds.)
#
#   2. llama-proxy.py on port 8089 (auto-started)
#      Catches 400 context-overflow errors. Strips largest tool messages first
#      (file reads are the biggest offenders), retries up to 10 times.
#      No opencode changes needed. See llama-proxy.py for full details.
#
#   3. OPENCODE_CTX = CTX (full server context, proxy is the safety net)
#      opencode's token estimator (GPT cl100k_base) underestimates Qwen3 usage,
#      BUT the proxy catches overflows before they crash. Compaction (reserved=40000)
#      fires early to keep context healthy; proxy is last resort.
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
#   ./start-opencode-stable.sh --restart                    # stop existing server + proxy first
#
# Env overrides:
#   CTX=196608 ./start-opencode-stable.sh                  # larger context (GPU has headroom up to ~229K)
#   CTX=65536 ./start-opencode-stable.sh                   # smaller context (saves VRAM)
#   NGL=38 ./start-opencode-stable.sh                      # fewer GPU layers
#   VERBOSE=true ./start-opencode-stable.sh                # verbose server logging
#   OPENCODE_RESERVED=60000 ./start-opencode-stable.sh     # earlier compaction trigger
#   NO_PROXY=true ./start-opencode-stable.sh               # skip proxy, point opencode directly at server
#
# VRAM context headroom (RTX 4000 Blackwell, 24 GB):
#   CTX 131072 → 2.4 GB free
#   CTX 163840 → 1.9 GB free
#   CTX 196608 → 1.4 GB free   ← current default (dedicated single-workload GPU)
#   CTX 229376 → 0.9 GB free   ← tight but usable
#   CTX 262144 → 0.4 GB free   ← risky, not recommended

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
    NGL="${NGL:-40}"
    CTX="${CTX:-196608}"
    REPEAT_PENALTY=1.1
    ;;
  qwen36-mtp)
    MODEL="${MODELS_DIR}/qwen36/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf"
    MODEL_DISPLAY="Qwen3.6-35B MTP (fast)"
    MTP_ENABLE=true
    NGL="${NGL:-40}"
    CTX="${CTX:-196608}"
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

# ─── PROXY + OPENCODE TOKEN BUDGET ──────────────────────────────────────────
# llama-proxy.py catches 400 context-overflow errors and retries with fewer
# messages. This means OPENCODE_CTX can equal the server CTX — no artificial
# reduction needed. Compaction (reserved) fires early to keep context healthy;
# the proxy is the emergency safety net for spikes (e.g. 47KB file reads).
#
# OPENCODE_CTX      = full server CTX (proxy handles overflow)
# OPENCODE_OUTPUT   = max tokens per response
# OPENCODE_RESERVED = compaction trigger: fires at CTX - reserved estimated tokens
#                     opencode's GPT tokenizer underestimates, so reserved=40000
#                     ensures compaction fires well before the real limit.
OPENCODE_CTX="${OPENCODE_CTX:-${CTX}}"
OPENCODE_OUTPUT="${OPENCODE_OUTPUT:-8192}"
OPENCODE_RESERVED="${OPENCODE_RESERVED:-40000}"

# ─── PROXY CONFIG ─────────────────────────────────────────────────────────────
PROXY_PORT=8089
PROXY_LOG="/tmp/llama-proxy.log"
NO_PROXY="${NO_PROXY:-false}"
PROXY_SCRIPT="$SCRIPT_DIR/llama-proxy.py"

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
HOST="127.0.0.1"
PORT=8088
PARALLEL=1
VERBOSE="${VERBOSE:-false}"

# ─── HELPERS ─────────────────────────────────────────────────────────────────
stop_server() {
  local PIDS
  PIDS=$(pgrep -f "llama-server" 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo "Stopping llama-server (PIDs: $PIDS)..."
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

stop_proxy() {
  local PIDS
  PIDS=$(pgrep -f "python3.*llama-proxy\.py" 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo "Stopping llama-proxy (PIDs: $PIDS)..."
    echo "$PIDS" | xargs kill 2>/dev/null || true
    sleep 1
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
  stop_proxy
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
  echo "  Thinking: disabled (--reasoning off)"
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
    --reasoning     off
    --metrics
    --timeout       0
    --parallel      "$PARALLEL"
    --repeat-penalty "$REPEAT_PENALTY"
    $VERBOSE_FLAG
  )

  if [ "$MTP_ENABLE" = true ]; then
    SERVER_ARGS+=(--spec-type draft-mtp --spec-draft-n-max "$MTP_N_MAX")
  fi

  "$SERVER" "${SERVER_ARGS[@]}" >> "$BASE_LOG" 2>&1 &
  SERVER_PID=$!

  # Crash watcher
  (
    while kill -0 "$SERVER_PID" 2>/dev/null; do sleep 5; done
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
  until curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; do
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

# ─── START PROXY ─────────────────────────────────────────────────────────────
# Start proxy before writing config so we know the actual port opencode will use.
PROXY_PID=""
OPENCODE_PORT="$PORT"  # default: direct to server

if [ "$NO_PROXY" != "true" ]; then
  if ! [ -f "$PROXY_SCRIPT" ]; then
    echo "WARNING: llama-proxy.py not found at $PROXY_SCRIPT — running without proxy" >&2
  else
    stop_proxy
    echo "Starting llama-proxy..."
    python3 "$PROXY_SCRIPT" \
      --upstream "http://127.0.0.1:${PORT}" \
      --port "$PROXY_PORT" \
      >> "$PROXY_LOG" 2>&1 &
    PROXY_PID=$!
    sleep 0.5
    if ! kill -0 "$PROXY_PID" 2>/dev/null; then
      echo "WARNING: llama-proxy failed to start — running without proxy" >&2
    else
      OPENCODE_PORT="$PROXY_PORT"
      echo "llama-proxy ready (PID $PROXY_PID, log: $PROXY_LOG)"
    fi
  fi
else
  echo "Proxy disabled (NO_PROXY=true) — opencode → llama-server directly"
fi
echo ""

# ─── OPENCODE CONFIG ─────────────────────────────────────────────────────────
OPENCODE_CONFIG_JSON=$(cat << OPENCODE_JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "llamacpp/${MODEL_NAME}",
  "permission": "allow",
  "provider": {
    "llamacpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-server (stable)",
      "options": {
        "baseURL": "http://127.0.0.1:${OPENCODE_PORT}/v1",
        "apiKey": "local"
      },
      "models": {
        "${MODEL_NAME}": {
          "name": "${MODEL_DISPLAY}",
          "limit": {
            "context": ${OPENCODE_CTX},
            "output": ${OPENCODE_OUTPUT}
          }
        }
      }
    }
  },
  "compaction": {
    "auto": true,
    "prune": true,
    "reserved": ${OPENCODE_RESERVED}
  }
}
OPENCODE_JSON
)

mkdir -p ~/.config/opencode
echo "$OPENCODE_CONFIG_JSON" > ~/.config/opencode/opencode.json

echo "OpenCode config ready"
echo "  model:        $MODEL_DISPLAY"
echo "  server ctx:   $CTX tokens"
echo "  opencode ctx: $OPENCODE_CTX tokens  (proxy safety net for overflow)"
echo "  max output:   $OPENCODE_OUTPUT tokens"
echo "  compact:      auto, reserved=$OPENCODE_RESERVED, prune=true  (fires at $(( OPENCODE_CTX - OPENCODE_RESERVED )) estimated tokens)"
echo "  perms:        allow (all tool calls auto-approved)"
if [ "$OPENCODE_PORT" = "$PROXY_PORT" ]; then
  echo "  proxy:        http://127.0.0.1:$PROXY_PORT → http://127.0.0.1:$PORT"
else
  echo "  proxy:        DISABLED (direct to server)"
fi
echo ""

# ─── LAUNCH OPENCODE ─────────────────────────────────────────────────────────
echo "Starting opencode..."
OPENCODE_CONFIG_CONTENT="$OPENCODE_CONFIG_JSON" opencode

# Cleanup
[ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null || true
stop_proxy
kill "${WATCHER_PID:-}" 2>/dev/null || true
