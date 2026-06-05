#!/bin/bash
# Verify vLLM works on this machine before committing to the 18GB NVFP4 model.
# Uses Qwen3-4B (BF16, ~8GB) — tests flashinfer + basic serving, fast to load.
#
# Run this first. Once it serves a response, Ctrl+C and run start-opencode-vllm.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_MODEL_ID="Qwen/Qwen3-4B"
TEST_MODEL_PATH="/mnt/storage/models/hf/Qwen3-4B"
PORT="${VLLM_PORT:-8181}"
VENV_DIR="$SCRIPT_DIR/.venv-vllm"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/vllm-test-$(date +%Y%m%d-%H%M%S).log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "==> Log file: ${LOG_FILE}"
echo "==> Started at: $(date)"

# ─── PRE-FLIGHT: GPU STATE ───────────────────────────────────────────────────
echo ""
echo "==> GPU state (pre-launch):"
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,temperature.gpu \
    --format=csv,noheader,nounits | awk -F',' '{
    printf "  GPU: %s | VRAM: %s MiB total | %s MiB free | %s MiB used | %s°C\n", $1,$2,$3,$4,$5
}'

# ─── STOP llama.cpp ──────────────────────────────────────────────────────────
echo "==> Stopping llama.cpp..."
pids=$(pgrep -f "llama-server" 2>/dev/null || true)
if [ -n "$pids" ]; then
    echo "$pids" | xargs kill 2>/dev/null || true
    sleep 2
    pgrep -f "llama-server" 2>/dev/null | xargs kill -9 2>/dev/null || true
    echo "  ✓ llama-server stopped"
else
    echo "  ✓ No llama-server running"
fi
pids=$(pgrep -f "llama-proxy\.py" 2>/dev/null || true)
[ -n "$pids" ] && echo "$pids" | xargs kill 2>/dev/null || true
sleep 1

# ─── DOWNLOAD TEST MODEL ─────────────────────────────────────────────────────
if [ ! -f "$TEST_MODEL_PATH/config.json" ]; then
    echo "==> Downloading ${TEST_MODEL_ID} (~8GB)..."
    mkdir -p "$TEST_MODEL_PATH"
    hf download "${TEST_MODEL_ID}" --local-dir "$TEST_MODEL_PATH" 2>&1 | tail -5
else
    echo "✓ Test model: ${TEST_MODEL_PATH}"
fi

# ─── vLLM SETUP ──────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating venv at ${VENV_DIR}..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "")
if [ -z "$VLLM_VER" ] || [ "${VLLM_FORCE_UPDATE:-false}" = "true" ]; then
    echo "==> Installing vLLM nightly (Blackwell sm_120 / flashinfer support)..."
    pip install --quiet --upgrade pip
    uv pip install -U vllm \
        --torch-backend=auto \
        --extra-index-url https://wheels.vllm.ai/nightly 2>&1 | tail -10
    VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)")
fi
echo "✓ vLLM: $VLLM_VER"

# ─── CUDA TOOLKIT SHIM (see start-opencode-vllm.sh for explanation) ──────────
CU13="${VENV_DIR}/lib/python3.13/site-packages/nvidia/cu13"
if [ -d "$CU13" ]; then
    ln -sfn "$CU13/lib" "$CU13/lib64" 2>/dev/null || true
    ln -sf "libcudart.so.13" "$CU13/lib/libcudart.so" 2>/dev/null || true
    mkdir -p "$CU13/lib/stubs"
    ln -sf "/usr/lib/x86_64-linux-gnu/libcuda.so.1" "$CU13/lib/stubs/libcuda.so" 2>/dev/null || true
fi
export CUDA_HOME="${CU13}"
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

# ─── SMOKE TEST ──────────────────────────────────────────────────────────────
echo ""
echo "┌─────────────────────────────────────────────────────────┐"
echo "│  vLLM TEST — Qwen3-4B (BF16)                           │"
echo "│  Verifies: install, flashinfer, basic server            │"
echo "│  Port ${PORT} — Ctrl+C once it says 'Application startup'  │"
echo "└─────────────────────────────────────────────────────────┘"
echo ""

vllm serve "${TEST_MODEL_PATH}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --dtype bfloat16 \
    --attention-backend flashinfer \
    --gpu-memory-utilization 0.70 \
    --max-model-len 16384 \
    --max-num-seqs 4 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --served-model-name "qwen3-4b-test" \
    2>&1
