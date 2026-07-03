#!/bin/bash
# Start llama-server for OpenClaude — Qwen3.6-35B-A3B (MoE)
#
# llama.cpp built from source at ~/claude/llama.cpp (CUDA 12.8, sm_120a Blackwell)
# Models on /mnt/storage/models/llms/qwen36/:
#   mtp  →  Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf   [default] ~21.8 GB  — MTP heads baked in
#   base →  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf        ~21.3 GB  — standard base
#
# MTP (Multi-Token Prediction):
#   --spec-type draft-mtp --spec-draft-n-max 2 applied automatically when MODEL_VARIANT=mtp
#   Expected speedup: ~1.4–2.2× generation vs base, zero accuracy loss
#
# Benchmark results (RTX PRO 4000 Blackwell, 24 GB VRAM):
#   tg (generation):  91.5 t/s base / 150+ t/s MTP expected
#   pp (prefill):   2714 t/s at short ctx
#   max safe ctx:   196k tokens with q4_0/q4_0 (default ctx reduced to 64k for tool-call reliability)
#
# Usage:
#   ./start-openclaude.sh                          # default (MTP model)
#   ./start-openclaude.sh --model base             # use base model (no MTP)
#   ./start-openclaude.sh --model mtp              # use MTP model (explicit)
#   ./start-openclaude.sh --model /path/to/model.gguf  # arbitrary path (no MTP)
#   ./start-openclaude.sh --name CustomName        # set server name separately
#   ./start-openclaude.sh --restart                # stop existing server, start fresh
#   ./start-openclaude.sh --uncensored             # use uncensored system prompt
#   ./start-openclaude.sh --prompt-file /path/to/prompt.txt
#   ./start-openclaude.sh --system-prompt "..."    # inline system prompt

set -euo pipefail

RESTART=false
PROMPT_TEXT=""
PROMPT_FILE=""
MODEL_VARIANT="mtp"            # mtp | base | <full path>
MODEL_NAME_CLI=""
DEFAULT_PROMPTS_DIR="$HOME/claude"

# ─── CLI ARG PARSING ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_VARIANT="$2"; shift 2 ;;
    --name)
      MODEL_NAME_CLI="$2"; shift 2 ;;
    --restart)
      RESTART=true; shift ;;
    --system-prompt)
      PROMPT_TEXT="$2"; shift 2 ;;
    --uncensored)
      PROMPT_FILE="${DEFAULT_PROMPTS_DIR}/openclaude-unsafe-prompt.txt"; shift ;;
    --prompt-file)
      PROMPT_FILE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--model mtp|base|/path/to/model.gguf] [--name Name] [--restart]"
      echo "          [--uncensored] [--system-prompt <text>] [--prompt-file <path>]"
      exit 0 ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ─── RESOLVE MODEL PATH ──────────────────────────────────────────────────────
MODELS_DIR="/mnt/storage/models/llms/qwen36"
MTP_ENABLE=false

case "$MODEL_VARIANT" in
  mtp)
    MODEL="${MODELS_DIR}/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf"
    MTP_ENABLE=true ;;
  base)
    MODEL="${MODELS_DIR}/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    MTP_ENABLE=false ;;
  /*)
    # Absolute path — caller decides; MTP disabled
    MODEL="$MODEL_VARIANT"
    MTP_ENABLE=false ;;
  *)
    echo "ERROR: Unknown --model value '$MODEL_VARIANT'. Use: mtp, base, or an absolute path." >&2
    exit 1 ;;
esac

if [ ! -f "$MODEL" ]; then
  echo "ERROR: Model file not found: $MODEL" >&2
  if [ "$MODEL_VARIANT" = "mtp" ]; then
    echo ""
    echo "MTP model not yet downloaded. Options:"
    echo "  1) Wait for download: hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF \\"
    echo "       Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --local-dir /mnt/storage/models/llms/qwen36-mtp-stage"
    echo "     then: mv /mnt/storage/models/llms/qwen36-mtp-stage/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \\"
    echo "            $MODEL"
    echo "  2) Use base model now: $0 --model base"
  fi
  exit 1
fi

# ─── SERVER BINARY ───────────────────────────────────────────────────────────
# Built from source: ~/claude/llama.cpp  (CUDA 12.8, sm_120a Blackwell, MTP support)
SERVER="$HOME/claude/llama.cpp/build/bin/llama-server"
if [ ! -x "$SERVER" ]; then
  echo "ERROR: llama-server not found at $SERVER" >&2
  echo "  Rebuild: cd ~/claude/llama.cpp && cmake --build build -j\$(nproc)" >&2
  exit 1
fi

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
    echo "[OPENCLAUDE] Stopping image generation batch (PIDs: $BATCH_PIDS)..."
    echo "$BATCH_PIDS" | xargs kill 2>/dev/null || true
    sleep 1
  fi
  local PIDS
  PIDS=$(pgrep -f "main\.py --listen" 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo "[OPENCLAUDE] Stopping ComfyUI/FLUX (freeing VRAM)..."
    echo "$PIDS" | xargs kill
    sleep 3
    if pgrep -f "main\.py --listen" > /dev/null 2>&1; then
      pgrep -f "main\.py --listen" | xargs kill -9
      sleep 1
    fi
    echo "[OPENCLAUDE] ComfyUI stopped."
  else
    echo "[OPENCLAUDE] No ComfyUI running — skipping."
  fi
}

# ─── STOP IF RESTARTING ──────────────────────────────────────────────────────
if [ "$RESTART" = true ]; then
  stop_server
fi

# ─── MODEL NAME (for openclaude --model flag) ─────────────────────────────────
if [ -n "$MODEL_NAME_CLI" ]; then
  MODEL_NAME="$MODEL_NAME_CLI"
else
  MODEL_NAME="$(basename "$MODEL")"
fi

# ─── SERVER CONFIGURATION ────────────────────────────────────────────────────
NGL="${NGL:-40}"            # GPU layers — 40 covers all layers of Qwen3.6-35B-A3B MoE
FA=1                        # Flash attention — required at this model size
CTK="q4_0"                  # KV cache key quant   — best speed + max context
CTV="q4_0"                  # KV cache value quant
CTX="${CTX:-65536}"          # 64k context — keeps tool calling reliable; override via CTX=131072 if needed
BATCH="${BATCH:-512}"
THREADS="${THREADS:-6}"
HOST="0.0.0.0"
PORT=8088                   # OpenClaude expects http://127.0.0.1:8088/v1
PARALLEL=1                  # single slot — only one client; prevents KV cache exhaustion
REPEAT_PENALTY=1.1

VERBOSE="${VERBOSE:-false}"
REASONING="${REASONING:-auto}"  # auto/on/off
EFFORT="${EFFORT:-high}"        # low/medium/high/max
CONTINUE="${CONTINUE:-false}"

# MTP speculative decoding settings
MTP_N_MAX="${MTP_N_MAX:-2}"   # draft tokens per step; 2 is sweet spot for 35B-A3B

# Logging — per-model log file
BASE_LOG="/tmp/llama-server-$(echo "$MODEL" | md5sum | cut -d' ' -f1).log"
LOG_MAX_MB=2048
LOG_KEEP=3

# ─── LOG ROTATION ────────────────────────────────────────────────────────────
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

  CURRENT_MODEL_PATH=$(curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || true)

  # Compare by basename — llama-server may return full path or basename depending on how it was started
  if [ "$(basename "${CURRENT_MODEL_PATH:-}")" = "$(basename "$MODEL")" ]; then
    echo "Already serving $(basename "$MODEL") — connecting."
    export OPENAI_API_KEY="local"
    export OPENAI_BASE_URL="http://$HOST:$PORT/v1"
    exec openclaude \
      --provider openai \
      --model    "$MODEL_NAME" \
      --dangerously-skip-permissions
  fi

  echo "Running server is serving a different model ($(basename "${CURRENT_MODEL_PATH:-unknown}"))."
  echo "Stopping it to start $(basename "$MODEL")..."
  stop_server
  # Wait for VRAM to drain before allocating for the new model
  echo "Waiting for VRAM to drain..."
  sleep 5
fi

# ─── FREE VRAM (stop ComfyUI if running) ─────────────────────────────────────
if pgrep -f "main\.py --listen|run-tuning-batch\.py|run-flux-batch\.py" > /dev/null 2>&1; then
  stop_comfyui
fi

# ─── PRINT STARTUP INFO ──────────────────────────────────────────────────────
echo "Starting llama-server..."
echo "  Binary:  $SERVER"
echo "  Model:   $(basename "$MODEL")"
if [ "$MTP_ENABLE" = true ]; then
  echo "  MTP:     enabled (--spec-type draft-mtp --spec-draft-n-max $MTP_N_MAX)"
else
  echo "  MTP:     disabled"
fi
echo "  ngl=$NGL  fa=$FA  ctk=$CTK  ctv=$CTV"
echo "  Context: $CTX tokens"
echo "  Jinja:   enabled"
echo "  Listen:  http://$HOST:$PORT"
echo ""

# ─── BUILD SERVER ARGS ───────────────────────────────────────────────────────
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
  --reasoning         "$REASONING"
  --reasoning-format  deepseek
  --parallel          "$PARALLEL"
  --repeat-penalty    "$REPEAT_PENALTY"
  --no-context-shift
  --timeout           0
  $VERBOSE_FLAG
)

# Append MTP speculative decoding args
if [ "$MTP_ENABLE" = true ]; then
  SERVER_ARGS+=(
    --spec-type        draft-mtp
    --spec-draft-n-max "$MTP_N_MAX"
  )
fi

# ─── LAUNCH LLAMA-SERVER ────────────────────────────────────────────────────
"${SERVER}" "${SERVER_ARGS[@]}" >> "$BASE_LOG" 2>&1 &
SERVER_PID=$!

# ─── WAIT FOR SERVER READY ──────────────────────────────────────────────────
echo "Waiting for llama-server to load model (MTP models take a few extra seconds)..."
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

# ─── SYSTEM PROMPT ───────────────────────────────────────────────────────────
SYSTEM_PROMPT_FLAGS=()
if [ -n "$PROMPT_TEXT" ] && [ -n "$PROMPT_FILE" ]; then
  echo "ERROR: Cannot use both --system-prompt and --prompt-file/--uncensored." >&2
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

# ─── CRASH WATCHER ───────────────────────────────────────────────────────────
(
  while kill -0 "$SERVER_PID" 2>/dev/null; do sleep 5; done
  echo "" >&2
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
  echo "ERROR: llama-server (PID $SERVER_PID) has died!" >&2
  echo "Last log from $BASE_LOG:" >&2
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
  tail -40 "$BASE_LOG" >&2
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
) &
WATCHER_PID=$!

# ─── LAUNCH OPENCLAUDE ───────────────────────────────────────────────────────
export OPENAI_API_KEY="local"
export OPENAI_BASE_URL="http://$HOST:$PORT/v1"
# --- Autocompact tuning ---
#
# Problem: OpenClaude defaults to 128k context for unknown local models. With 128k assumed,
# the autocompact threshold is ~54k tokens. At 54k client tokens the compact request
# is ~57k tokens and barely fits in 65536 → compact fails → circuit breaker trips
# after 3 failures → autocompact disabled → session grows to 66-71k → overflow.
#
# Fix 1: Register the actual context window via CLAUDE_CODE_OPENAI_CONTEXT_WINDOWS.
# This env var takes a JSON map; prefix matching applies, so "Qwen3.6" matches both
# MTP and base model filenames. With CTX=65536:
#   effectiveContext = 65536 - 20000 (reserved for summary) = 45536
#   autocompactThreshold = 45536 - 13000 (buffer for compact overhead) = 32536
# Compact fires at 32536 tokens. The compact request is ~32536 + overhead ≈ 33k,
# leaving 32k headroom in the 65536 window. Compact succeeds reliably.
#
# Fix 2: Do NOT set CLAUDE_AUTOCOMPACT_PCT_OVERRIDE below 72.
# PCT=50 would compute floor(45536 * 0.5) = 22768, which is BELOW the base system
# prompt overhead (~26k tokens). Autocompact would fire on turn 1 and thrash.
# The natural cap (effectiveContext - 13000 = 32536) is the correct threshold.
# Setting PCT > 71.4% hits the cap anyway, so just leave PCT_OVERRIDE unset.
export CLAUDE_CODE_OPENAI_CONTEXT_WINDOWS="{\"Qwen3.6\": $CTX}"
export CLAUDE_CODE_OPENAI_FALLBACK_CONTEXT_WINDOW=$CTX  # belt-and-suspenders

CONTINUE_FLAG=""
[ "$CONTINUE" = "true" ] && CONTINUE_FLAG="--continue"

openclaude \
  --provider openai \
  --model    "$MODEL_NAME" \
  --effort   "$EFFORT" \
  $CONTINUE_FLAG \
  "${SYSTEM_PROMPT_FLAGS[@]}" \
  --dangerously-skip-permissions

kill "$WATCHER_PID" 2>/dev/null || true
