#!/bin/bash
# Start ComfyUI with FLUX.1-dev GGUF for image generation on murderbot (RTX 4000 Blackwell)
# Kills llama-server (:8088), installs custom nodes if missing, starts ComfyUI (:8188)
#
# Brain: Mac Mini M4 (10.100.20.18) running Ollama (llama3.1:8b) for prompt expansion
# Artist: Murderbot (10.100.20.19) running ComfyUI with FLUX GGUF on GPU
#
# Usage:
#   ./start-flux.sh                          # start ComfyUI with FLUX
#   ./start-flux.sh --restart                # restart with new params or models loaded
#   STEPS=30 FLUX_GUIDANCE_STRENGTH=4.0 ./start-flux.sh  # tune quality via env vars

set -euo pipefail

RESTART=false
if [[ "${1:-}" == "--restart" ]]; then
  RESTART=true
fi

# ─── ENV VARS (quality tuning) ──────────────────────────────────────────────
FLUX_GUIDANCE_STRENGTH="${FLUX_GUIDANCE_STRENGTH:-3.5}"   # FluxGuidance strength (default ~3.5)
STEPS="${STEPS:-25}"                                      # Sampling steps (sweet spot 20-30)
RESOLUTION="${RESOLUTION:-1024}"                          # Image size: RESOLUTIONxRESOLUTION
SEED="${SEED:---1}"                                       # -1 = random; set to fixed number for reproducibility

# Model paths
FLUX_DIR="/mnt/storage/models/image/flux1-dev-gguf"
FLUX_MODEL="${FLUX_DIR}/flux1-dev-Q4_K.gguf"
T5_MODEL="${FLUX_DIR}/t5xxl-Q4_K.gguf"
CLIP_MODEL="${FLUX_DIR}/clip_l-Q8_0.gguf"

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
COMFYUI_DIR="$HOME/claude/comfyui"
SERVER="$HOME/claude/llama.cpp/build/bin/llama-server"
OLLAMA_HOST="10.100.20.18"    # Mac Mini M4 Ollama brain
OLLAMA_PORT=11434
COMFYUI_PORT=8188

# ─── KILL LLAMA-SERVER (free VRAM for FLUX) ─────────────────────────────────
stop_llama_server() {
  local PIDS
  PIDS=$(pgrep -f "llama-server" 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo "[FLUX] Stopping llama-server (VRAM hog, free ~24GB for FLUX)..."
    echo "$PIDS" | xargs kill
    sleep 3
    if pgrep -f "llama-server" > /dev/null 2>&1; then
      echo "$(pgrep -f 'llama-server')" | xargs kill -9
      sleep 1
    fi
    echo "[FLUX] llama-server stopped."
  else
    echo "[FLUX] No running llama-server found — nothing to stop."
  fi
}

# ─── KILL COMFYUI IF RUNNING (for --restart) ────────────────────────────────
stop_comfyui() {
  local PIDS
  # Match the actual process: "python main.py --listen" started from comfyui dir
  PIDS=$(pgrep -f "main\.py --listen" 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo "[FLUX] Stopping ComfyUI..."
    echo "$PIDS" | xargs kill
    sleep 3
    if pgrep -f "main\.py --listen" > /dev/null 2>&1; then
      echo "$(pgrep -f 'main\.py --listen')" | xargs kill -9
      sleep 1
    fi
    echo "[FLUX] ComfyUI stopped."
  else
    echo "[FLUX] No running ComfyUI found — nothing to stop."
  fi
}

# ─── PRE-START CHECKS ──────────────────────────────────────────────────────
# Check FLUX model file exists
if [ ! -f "$FLUX_MODEL" ]; then
  echo "ERROR: FLUX GGUF model not found at $FLUX_MODEL" >&2
  echo "Run download-flux.sh first:" >&2
  echo "  bash ~/claude/download-flux.sh" >&2
  exit 1
fi

# Stop existing server — on restart kill both ComfyUI and llama-server
# (both compete for the same GPU VRAM; only one can run at a time)
if [ "$RESTART" = true ]; then
  stop_llama_server   # ensure llama-server is stopped too
  stop_comfyui
else
  stop_llama_server
fi

# ─── INSTALL CUSTOM NODES (first-run check) ────────────────────────────────
CUSTOM_NODES_DIR="$COMFYUI_DIR/custom_nodes"

install_if_missing() {
  local REPO_URL="$1"
  local BRANCH="${2:-main}"
  local DIR_NAME

  DIR_NAME=$(basename "$REPO_URL" .git)
  local TARGET="$CUSTOM_NODES_DIR/$DIR_NAME"

  if [ -d "$TARGET" ]; then
    echo "[FLUX] $DIR_NAME already installed — updating..."
    cd "$TARGET" && git pull origin "$BRANCH" > /dev/null 2>&1 || true
    # Install/update pip deps if requirements.txt exists
    if [ -f "requirements.txt" ]; then
      source "$COMFYUI_DIR/venv/bin/activate" 2>/dev/null || true
      pip install -r requirements.txt > /dev/null 2>&1 || true
      deactivate 2>/dev/null || true
    fi
  else
    echo "[FLUX] Cloning $DIR_NAME custom nodes..."
    git clone "git@github.com:${REPO_URL}.git" "$TARGET" --branch "$BRANCH" --depth 1 > /dev/null 2>&1
    # Install deps if requirements.txt exists
    if [ -f "$TARGET/requirements.txt" ]; then
      cd "$TARGET" && source "$COMFYUI_DIR/venv/bin/activate" 2>/dev/null || true
      pip install -r requirements.txt > /dev/null 2>&1 || true
      deactivate 2>/dev/null || true
    fi
  fi
}

# ComfyUI-GGUF (city96) — load GGUF diffusion models in ComfyUI
install_if_missing "city96/ComfyUI-GGUF" main
# comfyui-ollama (stavsap) — remote Ollama nodes for prompt expansion via Mac Mini brain
install_if_missing "stavsap/comfyui-ollama" v2

echo "[FLUX] Custom nodes ready."

# ─── VERIFY OLLAMA BRAIN REACHABLE ──────────────────────────────────────────
if ! curl -sf "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; then
  echo "WARNING: Ollama brain (${OLLAMA_HOST}:${OLLAMA_PORT}) not reachable." >&2
  echo "ComfyUI will start but prompt expansion via OllamaGenerateV2 won't work." >&2
else
  MODEL_COUNT=$(curl -sf "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/tags" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('models',[])))" 2>/dev/null || echo "?")
  echo "[FLUX] Ollama brain reachable — $MODEL_COUNT model(s) loaded."
fi

# ─── START COMFYUI ──────────────────────────────────────────────────────────
echo ""
echo "=== Starting ComfyUI with FLUX.1-dev GGUF ==="
echo "  FluxModel:   $(basename "$FLUX_MODEL") ($(du -h "$FLUX_MODEL" | cut -f1))"
[ -f "$T5_MODEL" ] && echo "  T5 Encoder:    $(basename "$T5_MODEL")"
[ -f "$CLIP_MODEL" ] && echo "  CLIP-L:        $(basename "$CLIP_MODEL")"
echo ""
echo "  Quality params:"
echo "    FluxGuidance: ${FLUX_GUIDANCE_STRENGTH}"
echo "    Steps:         $STEPS"
echo "    Resolution:    ${RESOLUTION}x${RESOLUTION}"
[ "$SEED" != "---1" ] && echo "    Seed:           $SEED (fixed)" || echo "    Seed:           random"
echo ""
echo "  Brain:     Ollama at http://${OLLAMA_HOST}:${OLLAMA_PORT}"
echo "  ComfyUI:   http://0.0.0.0:${COMFYUI_PORT}"
echo ""

cd "$COMFYUI_DIR"
source "$COMFYUI_DIR/venv/bin/activate"

python main.py \
  --listen 0.0.0.0 \
  --port "$COMFYUI_PORT" \
  --disable-auto-launch \
  2>&1 &
COMFYUI_PID=$!

echo "ComfyUI starting (PID: $COMFYUI_PID)..."

# ─── WAIT FOR READY ────────────────────────────────────────────────────────
ATTEMPTS=0
MAX_ATTEMPTS=90   # ~3 min for model load
until curl -sf "http://localhost:${COMFYUI_PORT}/system_stats" > /dev/null 2>&1; do
  if [ "$ATTEMPTS" -ge "$MAX_ATTEMPTS" ]; then
    echo "ERROR: Timeout waiting for ComfyUI after ~3 min." >&2
    exit 1
  fi
  if ! kill -0 "$COMFYUI_PID" 2>/dev/null; then
    echo "ERROR: ComfyUI exited unexpectedly." >&2
    exit 1
  fi
  sleep 2
  ATTEMPTS=$((ATTEMPTS + 1))
done

echo ""
echo "[FLUX] ComfyUI is ready!"
echo "  Web UI:   http://localhost:${COMFYUI_PORT}"
echo "  Remote:   https://comfy.amer.dev"
echo ""

# ─── WATCHER (log on crash) ──────────────────────────────────────────────
(
  while kill -0 "$COMFYUI_PID" 2>/dev/null; do sleep 5; done
  echo "" >&2
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
  echo "ComfyUI (PID $COMFYUI_PID) has died!" >&2
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
) &
WATCHER_PID=$!

# Wait for ComfyUI to be manually stopped
wait "$COMFYUI_PID" 2>/dev/null || true
kill "$WATCHER_PID" 2>/dev/null || true
