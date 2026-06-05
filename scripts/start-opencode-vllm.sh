#!/bin/bash
# Start vLLM server for Qwen3.6-35B-A3B-NVFP4 on RTX PRO 4000 Blackwell (24GB)
#
# Fixed from original:
#   - Stops llama.cpp before starting (prevents GPU VRAM contention)
#   - Uses vllm nightly (required for Blackwell sm_120 NVFP4 + flashinfer support)
#   - max-model-len 32768 (was 192000 → OOM: KV cache for 192K doesn't fit in ~3GB headroom)
#   - vllm serve (was python3 -m vllm.entrypoints.api.server — deprecated)
#   - --attention-backend flashinfer (Blackwell FlashAttention via FlashInfer)
#   - --moe-backend marlin (NVFP4 MoE kernel — required for modelopt quant)
#   - --async-scheduling (reduced scheduling overhead)
#
# MTP speculative decoding: set VLLM_MTP=1 to enable (experimental, requires nightly)
#
# Memory notes:
#   NVFP4 weights: ~18 GB. At 0.85 util = 20.8 GB budget → ~2.8 GB for KV cache.
#   If you hit OOM on startup, lower VLLM_CTX (e.g. 16384) or VLLM_GPU_UTIL (e.g. 0.80).
#   There is no smaller weight format for this model on HF — NVFP4 is already 4-bit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="Qwen3.6-35B-A3B-NVFP4"
LOCAL_MODEL_PATH="/mnt/storage/models/hf/${MODEL_NAME}"
PORT="${VLLM_PORT:-8181}"
VENV_DIR="$SCRIPT_DIR/.venv-vllm"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/vllm-$(date +%Y%m%d-%H%M%S).log"

mkdir -p "$LOG_DIR"

# Tee all output to a timestamped log file so crashes can be diagnosed
exec > >(tee -a "$LOG_FILE") 2>&1

echo "==> Log file: ${LOG_FILE}"
echo "==> Started at: $(date)"

# ─── CLEANUP TRAP ────────────────────────────────────────────────────────────
_cleanup() {
    echo ""
    echo "==> Shutting down ($(date))..."
    if [ -n "${DMON_PID:-}" ]; then
        kill "$DMON_PID" 2>/dev/null || true
    fi
}
trap _cleanup EXIT INT TERM

# ─── STOP llama.cpp (free VRAM before vLLM loads ~18GB of weights) ───────────
echo "==> Stopping llama.cpp..."
pids=$(pgrep -f "llama-server" 2>/dev/null || true)
if [ -n "$pids" ]; then
    echo "  Stopping llama-server (PIDs: $pids)..."
    echo "$pids" | xargs kill 2>/dev/null || true
    sleep 2
    pgrep -f "llama-server" 2>/dev/null | xargs kill -9 2>/dev/null || true
    echo "  ✓ llama-server stopped"
else
    echo "  ✓ No llama-server running"
fi
pids=$(pgrep -f "llama-proxy\.py" 2>/dev/null || true)
if [ -n "$pids" ]; then
    echo "  Stopping llama-proxy (PIDs: $pids)..."
    echo "$pids" | xargs kill 2>/dev/null || true
    echo "  ✓ llama-proxy stopped"
fi
sleep 1

# ─── PRE-FLIGHT: GPU MEMORY CHECK ────────────────────────────────────────────
echo ""
echo "==> GPU state (pre-launch):"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu \
    --format=csv,noheader,nounits | awk -F',' '{
    printf "  GPU:  %s\n", $1
    printf "  Driver: %s\n", $2
    printf "  VRAM: %s MiB total | %s MiB free | %s MiB used\n", $3, $4, $5
    printf "  Temp: %s°C | Util: %s%%\n", $6, $7
}'

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
NEED_MIB=19000  # 18GB weights + 1GB headroom for loading
if [ "$FREE_MIB" -lt "$NEED_MIB" ]; then
    echo ""
    echo "ERROR: Only ${FREE_MIB} MiB VRAM free — need at least ${NEED_MIB} MiB." >&2
    echo "  Kill any GPU processes and try again." >&2
    nvidia-smi | tail -20
    exit 1
fi
echo "  ✓ ${FREE_MIB} MiB free — sufficient for model load"

# ─── BACKGROUND GPU MONITOR ──────────────────────────────────────────────────
# Logs VRAM and GPU util every 5s so we can see the ramp-up and any spike before a crash
echo ""
echo "==> Starting background GPU monitor (logged to ${LOG_FILE})..."
(
    while true; do
        nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null \
            | awk -F',' '{printf "[GPU] %s | used=%s MiB free=%s MiB util=%s%% temp=%s°C\n", $1,$2,$3,$4,$5}'
        sleep 5
    done
) &
DMON_PID=$!
echo "  Monitor PID: ${DMON_PID}"

# ─── MODEL CHECK ─────────────────────────────────────────────────────────────
if [ ! -f "$LOCAL_MODEL_PATH/config.json" ]; then
    echo "ERROR: Model not found at ${LOCAL_MODEL_PATH}" >&2
    echo "  Run: hf download nvidia/${MODEL_NAME} --local-dir ${LOCAL_MODEL_PATH}" >&2
    exit 1
fi
MODEL_SIZE=$(du -sh "$LOCAL_MODEL_PATH" 2>/dev/null | cut -f1)
echo "✓ Model: ${LOCAL_MODEL_PATH} (${MODEL_SIZE})"

# ─── vLLM SETUP ──────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating venv at ${VENV_DIR}..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "")
if [ -z "$VLLM_VER" ] || [ "${VLLM_FORCE_UPDATE:-false}" = "true" ]; then
    echo "==> Installing vLLM nightly (Blackwell sm_120 / NVFP4 / flashinfer)..."
    pip install --quiet --upgrade pip
    uv pip install -U vllm \
        --torch-backend=auto \
        --extra-index-url https://wheels.vllm.ai/nightly 2>&1 | tail -10
    VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)")
fi
echo "✓ vLLM: $VLLM_VER"
echo "✓ Python: $(python3 --version)"
echo "✓ CUDA: $(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo 'unknown')"
echo "✓ PyTorch: $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo 'unknown')"

# ─── CUDA TOOLKIT SHIM ───────────────────────────────────────────────────────
# The pip-installed nvidia/cu13 package has lib/ not lib64/, and libcudart.so.13
# not libcudart.so. FlashInfer JIT expects {CUDA_HOME}/lib64/ and standard .so
# names. We wire them up inside the venv so this is recreated on every venv build.
# NOTE: proper fix is to add cuda-toolkit-13-x to Ansible (driver supports 13.2),
# which makes all of this unnecessary.
CU13="${VENV_DIR}/lib/python3.13/site-packages/nvidia/cu13"
if [ -d "$CU13" ]; then
    ln -sfn "$CU13/lib" "$CU13/lib64" 2>/dev/null || true
    ln -sf "libcudart.so.13" "$CU13/lib/libcudart.so" 2>/dev/null || true
    mkdir -p "$CU13/lib/stubs"
    ln -sf "/usr/lib/x86_64-linux-gnu/libcuda.so.1" "$CU13/lib/stubs/libcuda.so" 2>/dev/null || true
    echo "✓ CUDA_HOME shim: ${CU13}"
fi
export CUDA_HOME="${CU13}"
# Disable CCCL strict version check: cu13 headers are 13.0, nvcc binary is 13.2
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

# vLLM verbose logging to help diagnose startup crashes
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# ─── LAUNCH ──────────────────────────────────────────────────────────────────
# gpu-memory-utilization 0.85 (was 0.90): gives ~20.8GB budget for 18GB weights +
# KV cache. More conservative = less chance of driver OOM panic on Blackwell.
# If you see "No available memory for the cache blocks", lower VLLM_CTX or
# reduce VLLM_GPU_UTIL further.
MAX_LEN="${VLLM_CTX:-16384}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"

SPEC_ARGS=()
if [ "${VLLM_MTP:-0}" = "1" ]; then
    SPEC_ARGS=(--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}')
    echo "  MTP: enabled (num_speculative_tokens=3)"
fi

echo ""
echo "┌─────────────────────────────────────────────────────────┐"
echo "│  vLLM · Qwen3.6-35B-A3B-NVFP4                         │"
echo "│  RTX PRO 4000 Blackwell · 24 GB VRAM · Port ${PORT}      │"
printf "│  max-model-len: %-6s · gpu-util: %-4s · KV: fp8      │\n" "${MAX_LEN}" "${GPU_UTIL}"
echo "└─────────────────────────────────────────────────────────┘"
echo ""
echo "==> Launching vLLM at $(date)..."

vllm serve "${LOCAL_MODEL_PATH}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --dtype auto \
    --quantization modelopt \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_LEN}" \
    --max-num-seqs 4 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --async-scheduling \
    --enable-prefix-caching \
    --served-model-name "qwen3.6-35b-a3b-nvfp4" \
    "${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"}" \
    2>&1
