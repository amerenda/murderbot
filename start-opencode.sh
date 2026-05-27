#!/bin/bash
# Start llama-server and launch OpenCode — coding agent with reliable context management
#
# OpenCode (opencode.ai, 120k+ stars): built-in auto-compaction with configurable
# reserved buffer — prevents the compact-request-overflow circuit-breaker problem
# that plagued openclaude with local models.
#
# Models:
#   qwen36 (default) — Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf (~22GB, fully on GPU)
#                      NGL=40, CTX=131072 (default), max confirmed-safe=196608
#                      MTP enabled (~150 t/s). q4_0/q4_0 KV cache fits at 196k.
#   qwen3-coder-next — Qwen3-Coder-Next-UD-IQ3_XXS.gguf (~26.5GB, partial CPU offload)
#                      Specifically tuned for coding agents and tool use.
#                      Needs ~22GB VRAM + ~5GB RAM. Uses CTX=32768 to save VRAM.
#                      NGL default=40; confirmed: 22.8GB used / 1.1GB free on 24GB VRAM.
#
# Download qwen3-coder-next if not yet present:
#   mkdir -p /mnt/storage/models/llms/qwen3-coder-next
#   HF_HOME=/mnt/storage/.hf-cache wget -c \
#     -O /mnt/storage/models/llms/qwen3-coder-next/Qwen3-Coder-Next-UD-IQ3_XXS.gguf \
#     "https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-UD-IQ3_XXS.gguf"
#
# Usage:
#   ./start-opencode.sh                              # default: Qwen3.6-35B MTP
#   ./start-opencode.sh --model qwen3-coder-next    # switch to Qwen3-Coder-Next
#   ./start-opencode.sh --restart                    # stop existing llama-server first
#   ./start-opencode.sh --restart --model qwen3-coder-next
#
# Env overrides (apply to any model):
#   CTX=49152 ./start-opencode.sh                   # override context window
#   NGL=38 ./start-opencode.sh                      # override GPU layers (qwen36)
#   NGL_CODER=36 ./start-opencode.sh --model qwen3-coder-next  # if NGL=40 OOMs after llama.cpp update

set -euo pipefail

# ─── DEFAULTS ────────────────────────────────────────────────────────────────
MODEL_VARIANT="qwen36"
RESTART=false

# ─── CLI ARG PARSING ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_VARIANT="$2"; shift 2 ;;
    --restart) RESTART=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--model qwen36|qwen3-coder-next] [--restart]"
      echo ""
      echo "  qwen36           (default) Qwen3.6-35B-A3B-MTP  — 22GB, fully on GPU"
      echo "  qwen3-coder-next           Qwen3-Coder-Next IQ3  — 26.5GB, coding-focused"
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
    MODEL="${MODELS_DIR}/qwen36/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf"
    MODEL_DISPLAY="Qwen3.6-35B MTP (local)"
    MTP_ENABLE=true
    NGL="${NGL:-40}"
    # 131072 (128k) default — confirmed safe with q4_0/q4_0 KV cache on 24GB VRAM.
    # Max confirmed: CTX=196608 ./start-opencode.sh  (full 192k window, 2GB headroom)
    # Note: 65536 was only needed to work around openclaude's broken autocompact math.
    # OpenCode knows the exact context from config — no 128k fallback assumption.
    CTX="${CTX:-131072}"
    REPEAT_PENALTY=1.1
    ;;
  qwen3-coder-next)
    MODEL="${MODELS_DIR}/qwen3-coder-next/Qwen3-Coder-Next-UD-IQ3_XXS.gguf"
    MODEL_DISPLAY="Qwen3-Coder-Next UD-IQ3_XXS (local)"
    MTP_ENABLE=false
    # NGL=40: confirmed working on 24GB VRAM. VRAM at load: 22836 used / 1151 free MiB.
    # KV cache is pre-allocated at slot init, so 1151 MiB is true headroom after all allocs.
    # OOM history: NGL=54 (26913 MiB), NGL=44 (24134 MiB) — both over 23754 MiB limit.
    # If you OOM (e.g. after llama.cpp update): NGL_CODER=36 ./start-opencode.sh --model qwen3-coder-next
    NGL="${NGL_CODER:-${NGL:-40}}"
    # 32k context keeps KV cache small (~2GB) so the partial CPU offload is manageable.
    CTX="${CTX:-32768}"
    REPEAT_PENALTY=1.0  # Qwen recommends no repeat penalty for Coder-Next
    ;;
  *)
    echo "ERROR: Unknown --model '$MODEL_VARIANT'. Use: qwen36, qwen3-coder-next" >&2
    exit 1 ;;
esac

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
  if [ "$MODEL_VARIANT" = "qwen3-coder-next" ]; then
    echo ""
    echo "Download it:"
    echo "  mkdir -p $(dirname "$MODEL")"
    echo "  HF_HOME=/mnt/storage/.hf-cache wget -c \\"
    echo "    -O $MODEL \\"
    echo "    \"https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-UD-IQ3_XXS.gguf\""
    echo ""
    echo "Download in progress? Check: ls -lh $(dirname "$MODEL")/"
  fi
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
REASONING="${REASONING:-auto}"

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
BASE_LOG="/tmp/llama-server-opencode-$(echo "$MODEL" | md5sum | cut -d' ' -f1).log"
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
  echo "Starting llama-server..."
  echo "  Model:   $MODEL_NAME"
  echo "  Display: $MODEL_DISPLAY"
  if [ "$MTP_ENABLE" = true ]; then
    echo "  MTP:     enabled (--spec-type draft-mtp --spec-draft-n-max $MTP_N_MAX)"
  else
    echo "  MTP:     disabled"
  fi
  echo "  ngl=$NGL  fa=$FA  ctk=$CTK  ctv=$CTV"
  echo "  ctx=$CTX  repeat_penalty=$REPEAT_PENALTY"
  echo "  Listen:  http://$HOST:$PORT"
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
    --metrics
    --timeout       0
    --reasoning         "$REASONING"
    --reasoning-format  deepseek
    --parallel          "$PARALLEL"
    --repeat-penalty    "$REPEAT_PENALTY"
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

# ─── OPENCODE CONFIG ─────────────────────────────────────────────────────────
# Write global opencode config pointing to the running llama-server.
#
# Key design choices vs openclaude:
#   - limit.context is set EXPLICITLY so opencode knows the real window (not 128k guess)
#   - compaction.reserved=20000 guarantees compact request never overflows context:
#       compact fires at CTX - 20000 (reserved) tokens, compact request fits in CTX
#   - compaction.prune=true removes old tool outputs first (cheap tokens before LLM compact)
#   - compaction.auto=true keeps running indefinitely — no circuit breaker that disables it
#   - permission=allow is the documented global-allow shorthand (opencode.ai/docs/permissions):
#       equivalent to { "*": "allow" } — overrides all defaults (last-match-wins rule order)
#       auto-approves every tool call in the TUI: read, edit, bash, glob, grep, doom_loop, etc.
#       (--dangerously-skip-permissions is the equivalent flag on `opencode run` only)
#   - OPENCODE_CONFIG_CONTENT env var is used to pass config inline at launch time.
#       This is precedence level 6 (highest user-controllable), overriding any project-level
#       .opencode/opencode.json files that may exist in the working directory tree.

# Build config JSON into a variable (used both for the file and OPENCODE_CONFIG_CONTENT)
OPENCODE_CONFIG_JSON=$(cat << OPENCODE_JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "llamacpp/${MODEL_NAME}",
  "permission": "allow",
  "provider": {
    "llamacpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-server (local)",
      "options": {
        "baseURL": "http://127.0.0.1:${PORT}/v1",
        "apiKey": "local"
      },
      "models": {
        "${MODEL_NAME}": {
          "name": "${MODEL_DISPLAY}",
          "limit": {
            "context": ${CTX},
            "output": 8192
          }
        }
      }
    }
  },
  "compaction": {
    "auto": true,
    "prune": true,
    "reserved": 20000
  }
}
OPENCODE_JSON
)

# Write to global config (reference copy)
mkdir -p ~/.config/opencode
echo "$OPENCODE_CONFIG_JSON" > ~/.config/opencode/opencode.json

echo "OpenCode config ready"
echo "  model:    $MODEL_DISPLAY"
echo "  context:  $CTX tokens"
echo "  compact:  auto, reserved=20000, prune=true"
echo "  perms:    allow (all tool calls auto-approved)"
echo ""

# ─── LAUNCH OPENCODE ─────────────────────────────────────────────────────────
echo "Starting opencode..."
# OPENCODE_CONFIG_CONTENT overrides any project-level .opencode/opencode.json in the working dir tree
OPENCODE_CONFIG_CONTENT="$OPENCODE_CONFIG_JSON" opencode

# Cleanup
kill "${WATCHER_PID:-}" 2>/dev/null || true
