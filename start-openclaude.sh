#!/bin/bash
# Start OpenClaude — sglang Docker stack (LiteLLM bypassed; direct connection to sglang)
#
# Architecture: openclaude → sglang (port 30000) → GPU
# Compose dir:  /home/alex/claude/projects/sglang-stack/
#
# Profiles / model names:
#   qwen27b  →  QuantTrio/Qwen3.6-27B-AWQ     [default]  ~29k token context
#     mem-fraction-static=0.93, max-running-requests=1
#     model: 18.22GB VRAM, KV: ~1.76GB (28,961 slots), Mamba: 1.58GB, free: ~1GB
#   qwen35b  →  QuantTrio/Qwen3.6-35B-A3B-AWQ [broken: sglang v0.5.12 AWQ MoE bug]
#
# Usage:
#   ./start-openclaude.sh                          # default (35B MoE)
#   ./start-openclaude.sh --model 27b              # 27B dense
#   ./start-openclaude.sh --model 35b              # 35B MoE (explicit)
#   ./start-openclaude.sh --restart                # tear down stack, start fresh
#   ./start-openclaude.sh --uncensored             # use uncensored system prompt
#   ./start-openclaude.sh --prompt-file /path/to/prompt.txt
#   ./start-openclaude.sh --system-prompt "..."    # inline system prompt

set -euo pipefail

COMPOSE_DIR="/home/alex/claude/projects/sglang-stack"
DEFAULT_PROMPTS_DIR="$HOME/claude"

RESTART=false
PROFILE=""        # qwen27b | qwen35b — set by --model; defaults to qwen35b below
PROMPT_TEXT=""
PROMPT_FILE=""

# ─── CLI ARG PARSING ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      case "$2" in
        27b) PROFILE="qwen27b" ;;
        35b) PROFILE="qwen35b" ;;
        *)   echo "Unknown model '$2'. Use 27b or 35b."; exit 1 ;;
      esac
      shift 2 ;;
    --restart)    RESTART=true; shift ;;
    --system-prompt) PROMPT_TEXT="$2"; shift 2 ;;
    --uncensored) PROMPT_FILE="${DEFAULT_PROMPTS_DIR}/openclaude-unsafe-prompt.txt"; shift ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--model 27b|35b] [--restart] [--uncensored] [--system-prompt <text>] [--prompt-file <path>]"
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Defaults
PROFILE="${PROFILE:-qwen27b}"
EFFORT="${EFFORT:-high}"
CONTINUE="${CONTINUE:-false}"
DEBUG="${DEBUG:-false}"

# Derive LiteLLM model alias and expected sglang model-path from profile
if [ "$PROFILE" = "qwen27b" ]; then
  MODEL_NAME="QuantTrio/Qwen3.6-27B-AWQ"
  SGLANG_MODEL_ID="QuantTrio/Qwen3.6-27B-AWQ"
else
  MODEL_NAME="QuantTrio/Qwen3.6-35B-A3B-AWQ"
  SGLANG_MODEL_ID="QuantTrio/Qwen3.6-35B-A3B-AWQ"
fi

# ─── HELPERS ────────────────────────────────────────────────────────────────

stop_stack() {
  echo "Stopping sglang stack..."
  docker compose -f "$COMPOSE_DIR/docker-compose.yml" down
  echo "Stack stopped."
}

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
    echo "[OPENCLAUDE] Stopping ComfyUI / FLUX (free VRAM for Qwen3.6 ~24 GB)..."
    echo "$PIDS" | xargs kill
    sleep 3
    if pgrep -f "main\.py --listen" > /dev/null 2>&1; then
      pgrep -f "main\.py --listen" | xargs kill -9
      sleep 1
    fi
    echo "[OPENCLAUDE] ComfyUI stopped."
  else
    echo "[OPENCLAUDE] No running ComfyUI found — skipping."
  fi
}

# ─── RESTART IF REQUESTED ───────────────────────────────────────────────────
if [ "$RESTART" = true ]; then
  stop_stack
fi

# ─── CHECK IF SGLANG IS ALREADY RUNNING ─────────────────────────────────────
NEED_START=true
if curl -sf "http://localhost:30000/v1/models" > /dev/null 2>&1; then
  # sglang is up — check which model it's serving
  CURRENT_MODEL=$(curl -sf "http://localhost:30000/v1/models" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || true)

  if [ "$CURRENT_MODEL" = "$SGLANG_MODEL_ID" ]; then
    echo "sglang already serving $MODEL_NAME — connecting openclaude."
    NEED_START=false
  else
    echo "sglang is serving a different model ($(basename "$CURRENT_MODEL"))."
    echo "Switching to $MODEL_NAME..."
    stop_stack
  fi
fi

# ─── START STACK (if needed) ─────────────────────────────────────────────────
if [ "$NEED_START" = true ]; then
  # Free VRAM: stop ComfyUI if running
  if pgrep -f "main\.py --listen|run-tuning-batch\.py|run-flux-batch\.py" > /dev/null 2>&1; then
    stop_comfyui
  fi

  echo "Starting sglang stack..."
  echo "  Profile: $PROFILE"
  echo "  Model:   $MODEL_NAME  ($SGLANG_MODEL_ID)"
  echo "  sglang:  http://localhost:30000"
  echo "  proxy:   http://localhost:4000"
  echo ""
  docker compose -f "$COMPOSE_DIR/docker-compose.yml" --profile "$PROFILE" up -d

  # ─── WAIT FOR SGLANG ───────────────────────────────────────────────────
  # Uses /v1/models (not /health — /health triggers inference and times out during load)
  echo "Waiting for sglang to load model (may take several minutes on first run)..."
  ATTEMPTS=0
  until curl -sf "http://localhost:30000/v1/models" > /dev/null 2>&1; do
    if [ "$ATTEMPTS" -ge 300 ]; then
      echo "ERROR: Timeout waiting for sglang after 600s." >&2
      echo "Check logs: docker compose -f $COMPOSE_DIR/docker-compose.yml logs" >&2
      exit 1
    fi
    sleep 2
    ATTEMPTS=$((ATTEMPTS + 1))
  done
  echo "sglang ready."

  # LiteLLM proxy skipped — connecting directly to sglang on port 30000
  echo ""
fi

# ─── SYSTEM PROMPT ──────────────────────────────────────────────────────────
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

# ─── DEBUG FLAGS ─────────────────────────────────────────────────────────────
DEBUG_FLAGS=()
if [ "$DEBUG" = "true" ]; then
  mkdir -p /tmp/openclaude
  DEBUG_FLAGS=(--debug --debug-file /tmp/openclaude/debug.log)
fi

# ─── LAUNCH OPENCLAUDE ───────────────────────────────────────────────────────
export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="http://localhost:30000/v1"  # direct to sglang, bypass LiteLLM
export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=55   # compact at 55% (~108k tokens)
# OpenClaude auto-sets NODE_OPTIONS with 8GB heap — override before it runs
export NODE_OPTIONS="--max-old-space-size=16384"

CONTINUE_FLAG=""
[ "$CONTINUE" = "true" ] && CONTINUE_FLAG="--continue"

openclaude \
  --provider openai \
  --model    "$MODEL_NAME" \
  --effort   "$EFFORT" \
  $CONTINUE_FLAG \
  "${SYSTEM_PROMPT_FLAGS[@]}" \
  ${DEBUG_FLAGS[@]+"${DEBUG_FLAGS[@]}"} \
  --dangerously-skip-permissions
