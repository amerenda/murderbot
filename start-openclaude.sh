#!/bin/bash
# Start llama-server for OpenClaude — Qwen3.6-35B-A3B (MoE)
#
# Benchmark results (RTX PRO 4000 Blackwell, 24 GB VRAM):
#   tg (generation):  91.5 t/s  — 4x faster than 27B dense (only 3B active params/token)
#   pp (prefill):   2714 t/s  at short ctx
#   max safe ctx:   262k tokens (full window) with q4_0/q4_0
#
# Chosen config: q4_0/q4_0 — near-best speed, full 262k context headroom
#
# Usage:
#   ./start-openclaude.sh                          # default model (Qwen3.6-35B-A3B)
#   ./start-openclaude.sh --model /path/to/model.gguf
#   ./start-openclaude.sh --name CustomName        # set server name separately
#   ./start-openclaude.sh --restart              # restart llama-server, then start openclaude
#   ./start-openclaude.sh --uncensored            # use uncensored system prompt
#   ./start-openclaude.sh --prompt-file /path/to/prompt.txt  # custom system prompt file

set -euo pipefail

RESTART=false         # set by --restart; stop existing server before starting
PROMPT_TEXT=""        # inline system prompt (--system-prompt <text>)
PROMPT_FILE=""        # path to system prompt file (--prompt-file <path> or --uncensored)
DEFAULT_PROMPTS_DIR="$HOME/claude"  # where safe/unsafe prompts live


# ─── CLI ARG PARSING ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_PATH="$2"; shift 2 ;;
    --name)  MODEL_NAME_CLI="$2";   shift 2 ;;
    --restart) RESTART=true; shift ;;
    --system-prompt) PROMPT_TEXT="$2"; shift 2 ;;
    --uncensored) PROMPT_FILE="${DEFAULT_PROMPTS_DIR}/openclaude-unsafe-prompt.txt"; shift ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--model /path/to/model.gguf] [--name ServerName] [--restart] [--uncensored] [--system-prompt <text>] [--prompt-file <path>]"
      exit 0 ;;
    *)       echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ─── STOP LLAMA-SERVER ────────────────────────────────────────────────────────
stop_server() {
  local PIDS
  PIDS=$(pgrep -f "llama-server" 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo "Stopping llama-server (PIDs: $PIDS)..."
    echo "$PIDS" | xargs kill
    sleep 3
    if pgrep -f "llama-server" > /dev/null 2>&1; then
      echo "$(pgrep -f 'llama-server')" | xargs kill -9
      sleep 1
    fi
    echo "Stopped."
  else
    echo "No running llama-server found — nothing to stop."
  fi
}

# ─── STOP COMFYUI + IMAGE GENERATION (free VRAM for LLM) ────────────────────
stop_comfyui() {
  # Kill tuning/batch scripts first so they don't requeue jobs
  local BATCH_PIDS
  BATCH_PIDS=$(pgrep -f "run-tuning-batch\.py|run-flux-batch\.py" 2>/dev/null || true)
  if [ -n "$BATCH_PIDS" ]; then
    echo "[OPENCLAUDE] Stopping image generation batch (PIDs: $BATCH_PIDS)..."
    echo "$BATCH_PIDS" | xargs kill 2>/dev/null || true
    sleep 1
  fi

  local PIDS
  PIDS=$(pgrep -f "main\.py --listen" 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo "[OPENCLAUDE] Stopping ComfyUI / FLUX (free VRAM for Qwen3.6 ~24GB)..."
    echo "$PIDS" | xargs kill
    sleep 3
    if pgrep -f "main\.py --listen" > /dev/null 2>&1; then
      pgrep -f "main\.py --listen" | xargs kill -9
      sleep 1
    fi
    echo "[OPENCLAUDE] ComfyUI stopped."
  else
    echo "[OPENCLAUDE] No running ComfyUI found — nothing to stop."
  fi
}

# ─── STOP EXISTING SERVER IF RESTARTING ───────────────────────────────────────
if [ "$RESTART" = true ]; then
  stop_server
fi

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
DEFAULT_MODEL="/mnt/storage/models/llama.cpp/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
MODEL="${MODEL_PATH:-$DEFAULT_MODEL}"               # CLI --model or env, else default
SERVER="$HOME/claude/llama.cpp/build/bin/llama-server"

# Derive model name from filename if not explicitly set via --name
if [ -n "${MODEL_NAME_CLI:-}" ]; then
    MODEL_NAME="$MODEL_NAME_CLI"                     # user-served name
elif [ -z "${MODEL_NAME:-}" ]; then
    MODEL_NAME="$(basename "$MODEL")"                # auto from path
fi

# ─── MMProJ (multimodal projector) — auto-detect for Gemma4 ──────────────────
MM_PROJ_PATH="${MMPROJ:-}"                          # override via env MMPROJ, or detect
if [ -z "$MM_PROJ_PATH" ] && [[ "$(basename "$MODEL")" == "Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf" ]]; then
    MM_PROJ_PATH="/mnt/storage/models/text/gemma-balanced/mmproj-Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-f16.gguf"
fi

# ─── MODEL-SPECIFIC PARAMS ──────────────────────────────────────────────────
NGL="${NGL:-40}"            # GPU layers (env override: NGL=38 ./start-openclaude.sh)
FA=1                        # Flash attention — required for this model size
CTK="q4_0"                  # KV cache key quant   — best speed + max context
CTV="q4_0"                  # KV cache value quant — best speed + max context
BATCH="${BATCH:-512}"       # Prompt batch size (env override: BATCH=2048)
THREADS="${THREADS:-6}"     # CPU threads
HOST="0.0.0.0"
PORT=8088                   # OpenClaude expects http://127.0.0.1:8088/v1
PARALLEL=1                  # 1 slot — openclaude is the only client; avoids KV cache exhaustion
REPEAT_PENALTY=1.1          # mild repeat penalty

# Context window — auto-tune based on model
if [[ "$(basename "$MODEL")" == *"Gemma4"* ]]; then
    CTX="${CTX:-106608}"      # Gemma4: smaller context; override via env CTX to change (e.g. CTX=32768)
elif [[ "$(basename "$MODEL")" == "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" ]]; then
    CTX="${CTX:-196608}"    # Qwen3.6: 192k — benchmark confirmed safe; full 262k also works
else
    CTX="${CTX:-32768}"     # Generic fallback
fi

# Logging — per-model log file so you can keep both running for comparison
BASE_LOG="/tmp/llama-server-$(echo "$MODEL" | md5sum | cut -d' ' -f1).log"
LOG_MAX_MB=2048             # rotate when log exceeds this size
LOG_KEEP=3                  # number of old logs to keep

VERBOSE="${VERBOSE:-false}"    # set VERBOSE=true to enable llama-server verbose logging
REASONING="${REASONING:-auto}" # auto/on/off — think blocks routed to reasoning_content (not fed back to model)
EFFORT="${EFFORT:-high}"       # low/medium/high/max — reasoning depth per turn
CONTINUE="${CONTINUE:-false}"  # set CONTINUE=true to resume last session


# ─── LOG ROTATION ────────────────────────────────────────────────────────────
if [ -f "$BASE_LOG" ]; then
  LOG_SIZE_MB=$(( $(stat -c%s "$BASE_LOG") / 1024 / 1024 ))
  if [ "$LOG_SIZE_MB" -ge "$LOG_MAX_MB" ]; then
    for i in $(seq $((LOG_KEEP - 1)) -1 1); do
      [ -f "${BASE_LOG}.${i}" ] && mv "${BASE_LOG}.${i}" "${BASE_LOG}.$((i + 1))"
    done
    mv "$BASE_LOG" "${BASE_LOG}.1"
    echo "Rotated log (was ${LOG_SIZE_MB}MB) → ${BASE_LOG}.1"
  fi
fi

# ─── STOP EXISTING INSTANCE (different model) ────────────────────────────────
if pgrep -f "llama-server" > /dev/null 2>&1; then
  echo "A llama-server is already running on :$PORT"

  # Extract the actual model ID (full path) from the API response.
  CURRENT_MODEL_PATH=$(curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || true)

  if [ "$CURRENT_MODEL_PATH" = "$MODEL" ]; then
    # Same model — just connect
    echo "Running server is already serving $(basename "$MODEL") — connecting."
    export OPENAI_API_KEY="local"
    export OPENAI_BASE_URL="http://$HOST:$PORT/v1"
    exec openclaude \
      --provider openai \
      --model    "$MODEL_NAME" \
      --dangerously-skip-permissions
  fi

  # Different model — stop and restart
  echo "Running server is serving a different model ($(basename "$CURRENT_MODEL_PATH"))."
  echo "Stopping it to start $(basename "$MODEL")..."
  stop_server
fi

# ─── STOP COMFYUI (if running) ──────────────────────────────────────────────
# Kill ComfyUI/FLUX (and any batch scripts) to free VRAM for Qwen3.6 (~24GB needed)
if pgrep -f "main\.py --listen|run-tuning-batch\.py|run-flux-batch\.py" > /dev/null 2>&1; then
  stop_comfyui
fi

# ─── START LLAMA-SERVER ──────────────────────────────────────────────────────
echo "Starting llama-server..."
echo "  Model:   $(basename "$MODEL")"
[ -n "$MM_PROJ_PATH" ] && echo "  MMProj:  $(basename "$MM_PROJ_PATH")"
echo "  ngl=$NGL  fa=$FA  ctk=$CTK  ctv=$CTV"
echo "  Context: $CTX tokens"
echo "  Listen:  http://$HOST:$PORT"
echo ""

VERBOSE_FLAG=""
[ "$VERBOSE" = "true" ] && VERBOSE_FLAG="--verbose"

# Build server command — mmproj is optional, only passed if set
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
  --metrics
  --reasoning         "$REASONING"
  --reasoning-format  deepseek
  --parallel          "$PARALLEL"
  --repeat-penalty    "$REPEAT_PENALTY"
  $VERBOSE_FLAG
)

# Add mmproj if available (multimodal projector)
[ -n "$MM_PROJ_PATH" ] && SERVER_ARGS+=(--mmproj "$MM_PROJ_PATH")

"${SERVER}" "${SERVER_ARGS[@]}" >> "$BASE_LOG" 2>&1 &
SERVER_PID=$!

# ─── WAIT FOR SERVER READY ─────────────────────────────────────────────
echo "Waiting for llama-server to load model..."
ATTEMPTS=0
until curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; do
  if [ "$ATTEMPTS" -ge 60 ]; then
    echo "ERROR: Timeout waiting for server after 120s." >&2
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

# ─── START OPENCLAUDE ────────────────────────────────────────────────
export OPENAI_API_KEY="local"
export OPENAI_BASE_URL="http://$HOST:$PORT/v1"
export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=75  # compact at 75% of context window
CONTINUE_FLAG=""
[ "$CONTINUE" = "true" ] && CONTINUE_FLAG="--continue"

# ─── SYSTEM PROMPT (inline or from file, mutually exclusive) ──────────
SYSTEM_PROMPT_FLAGS=()
if [ -n "$PROMPT_TEXT" ] && [ -n "$PROMPT_FILE" ]; then
  echo "ERROR: Cannot use both --system-prompt and --prompt-file/--uncensored at the same time." >&2
  exit 1
fi
if [ -n "$PROMPT_TEXT" ]; then
  SYSTEM_PROMPT_FLAGS=(--system-prompt "$PROMPT_TEXT")
elif [ -n "$PROMPT_FILE" ]; then
  if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: Prompt file not found: $PROMPT_FILE" >&2
    exit 1
  fi
  SYSTEM_PROMPT_FLAGS=(--system-prompt-file "$PROMPT_FILE")
fi

# Watch for server crashes and surface the log immediately
(
  while kill -0 "$SERVER_PID" 2>/dev/null; do sleep 5; done
  echo "" >&2
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
  echo "ERROR: llama-server (PID $SERVER_PID) has died!" >&2
  echo "Last log entries from $BASE_LOG:" >&2
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
  tail -40 "$BASE_LOG" >&2
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
) &
WATCHER_PID=$!

openclaude \
  --provider openai \
  --model    "$MODEL_NAME" \
  --effort   "$EFFORT" \
  $CONTINUE_FLAG \
  "${SYSTEM_PROMPT_FLAGS[@]}" \
  --dangerously-skip-permissions

kill "$WATCHER_PID" 2>/dev/null || true

