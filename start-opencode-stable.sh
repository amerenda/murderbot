#!/bin/bash
# start-opencode-stable.sh — Qwen3.6 + llama.cpp (native tool calling)
#
# Architecture:
#   opencode → llama-proxy (port 8089) → llama-server (port 8088)
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
#   3. llama-proxy.py — ENABLED by default (NO_PROXY=false)
#      Intercepts HTTP 400 context overflows, strips+retries transparently.
#      Also pre-truncates tool messages before they reach the server (backup
#      to max_tool_response_chars, which truncates at Jinja render time).
#      Exposes Prometheus metrics at :8089/metrics.
#      Disable with: NO_PROXY=true ./start-opencode-stable.sh
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
#   OPENCODE_RESERVED=80000 ./start-opencode-stable.sh     # less aggressive compaction
#   NO_PROXY=true ./start-opencode-stable.sh               # skip proxy, point opencode directly at server
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
    CTX="${CTX:-32768}"
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

# ─── PROXY + OPENCODE TOKEN BUDGET ──────────────────────────────────────────
# llama-proxy.py catches 400 context-overflow errors and retries with fewer
# messages. This means OPENCODE_CTX can equal the server CTX — no artificial
# reduction needed. Compaction (reserved) fires early to keep context healthy;
# the proxy is the emergency safety net for spikes (e.g. 47KB file reads).
#
# OPENCODE_CTX      = full server CTX (proxy handles overflow)
# OPENCODE_OUTPUT   = max tokens per response
# OPENCODE_RESERVED = compaction trigger: fires at CTX - reserved actual tokens
#
# WHY reserved=100000 (threshold = 31072 tokens):
#   The proxy strips messages transparently before requests reach the server.
#   The server therefore reports back a "small" prompt_tokens (e.g. 40-70K
#   after stripping). OpenCode uses this server-reported count to decide
#   when to compact. With reserved=70000, threshold=61072 — close enough to
#   the stripped context size that compaction never fires (server reports 50K,
#   threshold is 61K, so opencode thinks context is fine).
#
#   With reserved=100000, threshold=31072. Any request that needed stripping
#   reports 40K+ actual tokens from the server → exceeds threshold → compaction
#   fires and summarises the session. After compaction, context drops to ~10K
#   and the cycle resets cleanly. Without stripping (fresh session) the server
#   reports <30K, no compaction.
#
#   NOTE: llama-server reports actual Qwen3 token counts (not GPT estimates),
#   so the threshold comparison here is in real tokens, not tiktoken estimates.
OPENCODE_CTX="${OPENCODE_CTX:-${CTX}}"
OPENCODE_OUTPUT="${OPENCODE_OUTPUT:-8192}"
OPENCODE_RESERVED="${OPENCODE_RESERVED:-85000}"

# ─── PROXY CONFIG ─────────────────────────────────────────────────────────────
PROXY_PORT=8089
PROXY_LOG="/tmp/llama-proxy.log"
NO_PROXY="${NO_PROXY:-false}"
PROXY_SCRIPT="$SCRIPT_DIR/proxy/llama-proxy.py"

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
PORT="${PORT:-9110}"
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
    --metrics
    --timeout       0
    --parallel      "$PARALLEL"
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
    MAX_TOOL_CHARS="${MAX_TOOL_CHARS:-2000}"
    python3 "$PROXY_SCRIPT" \
      --upstream "http://127.0.0.1:${PORT}" \
      --port "$PROXY_PORT" \
      --max-tool-chars "$MAX_TOOL_CHARS" \
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
  "permission": {
    "read": "allow",
    "edit": "allow",
    "glob": "allow",
    "grep": "allow",
    "list": "allow",
    "bash": "allow",
    "task": "allow",
    "external_directory": "allow",
    "todowrite": "allow",
    "webfetch": "allow",
    "websearch": "allow",
    "repo_clone": "allow",
    "repo_overview": "allow",
    "lsp": "allow",
    "doom_loop": "allow",
    "skill": "allow",
    "question": "deny"
  },
  "agent": {
    "build": {
      "prompt": "Proceed with tasks autonomously without stopping mid-task to ask for confirmation or check-ins. Never ask 'did you sync?', 'is it pushed?', 'should I continue?', or any similar check-in. If you said you will do something, do it immediately. Only stop if you are completely blocked and need information only the user can provide."
    }
  },
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
