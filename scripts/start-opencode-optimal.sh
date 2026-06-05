#!/bin/bash
# start-opencode-optimal.sh
# Qwen3.6-35B-A3B-NVFP4 on RTX PRO 4000 Blackwell SM120a (24 GB GDDR7)
#
# Usage:
#   bash start-opencode-optimal.sh                        # default mode
#   bash start-opencode-optimal.sh --min-memory           # guaranteed-start mode
#   bash start-opencode-optimal.sh --min-memory --debug   # verbose crash-diagnosis mode
#
# Env overrides: VLLM_PORT  VLLM_CTX  VLLM_GPU_UTIL  VLLM_FORCE_UPDATE=true
#
# ──── RESEARCH NOTES (2026-06-05) ─────────────────────────────────────────────
#
# ── CONFIRMED VRAM NUMBERS (from crash logs 2026-06-05) ──────────────────────
#   GPU total:        24467 MiB (nvidia-smi)
#   CUDA context:      ~485 MiB overhead, so vLLM sees 23982 MiB
#   Weights load peak: 22430 MiB (during Marlin repack)
#   Weights settled:   21326 MiB (temp buffer freed post-repack)
#   Weight fraction:   21326 / 23982 = 88.9% of usable VRAM
#   KV budget at 0.96: 23023 - 21326 = 1697 MiB
#   KV budget at 0.99: 23742 - 21326 = 2416 MiB
#
# WHY MARLIN IS EXPECTED (not a bug):
#   SM120 lacks tcgen05 — no native W4A4 FP4 tensor cores. NVFP4 MoE always uses
#   Marlin W4A16 on SM120. True for all vLLM versions and all NVFP4 engines.
#   The native CUTLASS NVFP4 MoE path fails on SM120 (TMA WS grouped GEMM init,
#   GH vllm-project/vllm#31085, confirmed Apr 2026).
#
# WHY WE OOM WITHOUT --enforce-eager:
#   Marlin repacks NVFP4 weights: 18 GiB on disk → peaks at ~22430 MiB in VRAM.
#   torch.compile adds ~1350 MiB permanently → 23790 MiB → 197 MiB left.
#   KV profiling OOM-kills EngineCore with no log entry (kernel OOM kill).
#   Confirmed in both crash runs (vllm-20260605-093245.log, vllm-20260605-094311.log).
#
# THE FIX — --enforce-eager:
#   Skips torch.compile. Weights settle at ~21326 MiB.
#   Cost: 40–70% throughput reduction vs compiled. Necessary on single 24 GB GPU.
#
# WHY KV BLOCKS ARE SO LARGE (164 MiB each):
#   vLLM sets attention block size = 2096 tokens to match Mamba page size (hybrid model).
#   fp8 KV block = 2096 × 8 KV-heads × 128 head-dim × 40 layers × 2 (K+V) × 1 byte
#               = 171,786,240 bytes ≈ 164 MiB per block.
#   This means VERY few blocks fit: at KV budget 1697 MiB → only ~10 blocks.
#   10 blocks = 10 × 2096 = 20,960 tokens KV capacity total.
#   Use fp8 KV (not auto/bf16) — bf16 blocks are 328 MiB each, 2× worse.
#
# WHY --num-gpu-blocks-override IS REQUIRED (profiling hang fix):
#   KV profiling runs a forward pass with max_num_batched_tokens to measure activation
#   memory. After KV allocation (23900 MiB), only 87 MiB free. Activation forward pass
#   needs more → CUDA OOM → EngineCore hangs silently. APIServer loops "Waiting for
#   EngineCore to start..." forever. Fix: bypass profiling by specifying block count.
#   (Confirmed from vllm-optimal-20260605-113216.log)
#
# WHY GPU UTIL IS HIGH:
#   Weights = 88.9% of VRAM. KV budget = util×23982 - 21326. At util=0.85: KV = -820.
#   Must use util ≥ 0.93 to have any KV budget. Use 0.99 for maximum KV space.
#   Min-memory = small context + single seq, NOT low GPU util.
#
# WHY --cpu-offload-gb CAN HELP (untested with Marlin NVFP4):
#   Offloads weight tensors to CPU RAM. For MoE where only 3/35B params (~9%) are
#   active per token, most expert weights sit idle — ideal for CPU offload.
#   Each GiB offloaded frees 1 GiB of VRAM → ~6 more KV blocks.
#   VLLM_CPU_OFFLOAD_GB env var controls this (default 0 = disabled).
#   Try VLLM_CPU_OFFLOAD_GB=3 to enable 16K ctx mode (needs 32 blocks = 5242 MiB,
#   only viable if 3 GiB freed). NOTE: may not work with Marlin-repacked NVFP4
#   weights — test and validate. Latency hit: PCIe transfer per active layer.
#
# WHY NOT AWQ OR GPTQ:
#   All W4A16 formats on SM120 use Marlin → same VRAM footprint. NVFP4 Marlin is
#   ~17% faster than AWQ Marlin (lower bandwidth pressure). Already downloaded.
#
# WHY NIGHTLY WHEELS:
#   Stable pip wheels lack SM120 arch flags (GH #35432). Nightly includes SM120.
#
# GPU CLOCK WARNING (headless SM120):
#   RTX PRO 4000 does NOT auto-boost clocks on headless servers. Run once as root:
#     sudo nvidia-smi -pm 1 && sudo nvidia-smi -i 0 --lock-gpu-clocks=2000,2000
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="Qwen3.6-35B-A3B-NVFP4"
LOCAL_MODEL_PATH="/mnt/storage/models/hf/${MODEL_NAME}"
PORT="${VLLM_PORT:-8181}"
VENV_DIR="$SCRIPT_DIR/.venv-vllm"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/vllm-optimal-$(date +%Y%m%d-%H%M%S).log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

# ─── FLAG PARSING ─────────────────────────────────────────────────────────────
MIN_MEMORY=false
DEBUG=false
for arg in "$@"; do
    case "$arg" in
        --min-memory) MIN_MEMORY=true ;;
        --debug)      DEBUG=true ;;
    esac
done

# ─── TIMESTAMP HELPER ─────────────────────────────────────────────────────────
# ts [label] message — always printed; prefixes HH:MM:SS.mmm
ts() {
    local label="${1:-INFO}"
    shift
    printf '[%s] [%s] %s\n' "$(date +%H:%M:%S.%3N)" "$label" "$*"
}

ts INFO "Log file: ${LOG_FILE}"
ts INFO "Started at: $(date)"
ts INFO "Args: $*"

# ─── DEBUG SETUP ──────────────────────────────────────────────────────────────
if [ "$DEBUG" = "true" ]; then
    ts DEBUG "DEBUG mode enabled"
    ts DEBUG "  VLLM_LOGGING_LEVEL=DEBUG"
    ts DEBUG "  CUDA_LAUNCH_BLOCKING=1  (synchronous CUDA ops — exact OOM stack, slower load)"
    ts DEBUG "  GPU monitor interval: 2s"
    export VLLM_LOGGING_LEVEL=DEBUG
    export CUDA_LAUNCH_BLOCKING=1
    MONITOR_INTERVAL=2
else
    export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
    MONITOR_INTERVAL=5
fi

# ─── MODE SETUP ───────────────────────────────────────────────────────────────
# CPU offload: each GiB freed from GPU weights = ~6 more fp8 KV blocks.
# Requires VLLM_CPU_OFFLOAD_GB > 0. Untested with Marlin NVFP4 — validate first.
CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-0}"

if [ "$MIN_MEMORY" = "true" ]; then
    ts INFO "Mode: --min-memory  (ctx=4096, seqs=1, fp8 KV, util=0.96, 3 blocks — startup test)"
    MAX_LEN="${VLLM_CTX:-4096}"
    GPU_UTIL="${VLLM_GPU_UTIL:-0.96}"
    MAX_SEQS=1
    MAX_BATCHED_TOKENS=2048
    KV_DTYPE="fp8"
    # 3 blocks × 2096 tokens/block = 6288 tokens KV capacity (covers ctx=4096 + margin).
    # --num-gpu-blocks-override bypasses the KV profiling forward pass entirely.
    # Without it, profiling hangs: runs 2048-token activations with only 87 MiB free.
    EXTRA_ARGS=(--num-gpu-blocks-override 3)
else
    # Default mode: ctx=8192, 2 seqs, fp8 KV, 10 blocks.
    # Why 8192 not 16384: at util=0.99, KV budget = 2416 MiB.
    #   fp8 block = 164 MiB. 10 blocks = 1640 MiB ✓ (fits).
    #   For 16384 ctx with 4 seqs: needs 32 blocks = 5242 MiB → DOES NOT FIT without
    #   cpu_offload_gb. Set VLLM_CPU_OFFLOAD_GB=3 to unlock 16K mode (frees ~18 blocks).
    ts INFO "Mode: default  (ctx=8192, seqs=2, fp8 KV, util=0.99, 10 blocks)"
    MAX_LEN="${VLLM_CTX:-8192}"
    GPU_UTIL="${VLLM_GPU_UTIL:-0.99}"
    MAX_SEQS="${VLLM_MAX_SEQS:-2}"
    MAX_BATCHED_TOKENS=4096
    KV_DTYPE="fp8"
    EXTRA_ARGS=(
        --num-gpu-blocks-override 10
        --enable-chunked-prefill
        --enable-prefix-caching
    )
fi

if [ "${CPU_OFFLOAD_GB}" != "0" ]; then
    ts INFO "CPU offload: ${CPU_OFFLOAD_GB} GiB (VLLM_CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB})"
    ts INFO "  Note: untested with Marlin NVFP4 — watch for startup errors"
fi

# ─── CLEANUP TRAP ─────────────────────────────────────────────────────────────
_cleanup() {
    ts INFO "Shutting down ($(date))..."
    [ -n "${DMON_PID:-}" ] && kill "$DMON_PID" 2>/dev/null || true
}
trap _cleanup EXIT INT TERM

# ─── STOP llama.cpp ───────────────────────────────────────────────────────────
ts INFO "=== STEP: stop llama.cpp ==="
pids=$(pgrep -f "llama-server" 2>/dev/null || true)
if [ -n "$pids" ]; then
    ts INFO "Stopping llama-server PIDs: $pids"
    echo "$pids" | xargs kill 2>/dev/null || true
    sleep 2
    pgrep -f "llama-server" 2>/dev/null | xargs kill -9 2>/dev/null || true
    ts INFO "llama-server stopped"
else
    ts INFO "No llama-server running"
fi
pids=$(pgrep -f "llama-proxy\.py" 2>/dev/null || true)
if [ -n "$pids" ]; then
    echo "$pids" | xargs kill 2>/dev/null || true
    ts INFO "llama-proxy stopped"
fi
sleep 1

# ─── PRE-FLIGHT: GPU STATE ────────────────────────────────────────────────────
ts INFO "=== STEP: GPU pre-flight check ==="
nvidia-smi \
    --query-gpu=name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu,clocks.gr \
    --format=csv,noheader,nounits | awk -F',' -v ts="$(date +%H:%M:%S.%3N)" '{
    printf "[%s] [GPU]  GPU:    %s\n",  ts, $1
    printf "[%s] [GPU]  Driver: %s\n",  ts, $2
    printf "[%s] [GPU]  VRAM:   %s MiB total | %s MiB free | %s MiB used\n", ts, $3, $4, $5
    printf "[%s] [GPU]  Temp:   %s°C | Util: %s%%\n", ts, $6, $7
    printf "[%s] [GPU]  Clock:  %s MHz\n", ts, $8
}'

if [ "$DEBUG" = "true" ]; then
    ts DEBUG "=== STEP: debug GPU process dump ==="
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader 2>/dev/null | while IFS= read -r line; do
        ts DEBUG "  GPU process: $line"
    done
    ts DEBUG "System RAM: $(free -h | awk '/^Mem/{print $2 " total, " $3 " used, " $7 " available"}')"
fi

CLOCK_MHZ=$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits | tr -d ' ')
if [ "${CLOCK_MHZ:-0}" -lt 1000 ] 2>/dev/null; then
    ts INFO "WARNING: GPU clock is only ${CLOCK_MHZ} MHz — throughput will be ~6x lower!"
    ts INFO "  Fix (run as root, once per boot):"
    ts INFO "    sudo nvidia-smi -pm 1 && sudo nvidia-smi -i 0 --lock-gpu-clocks=2000,2000"
fi

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
NEED_MIB=19000
if [ "${FREE_MIB:-0}" -lt "$NEED_MIB" ]; then
    ts INFO "ERROR: Only ${FREE_MIB} MiB VRAM free — need at least ${NEED_MIB} MiB."
    nvidia-smi | tail -20
    exit 1
fi
BUDGET_MIB=$(awk "BEGIN{printf \"%d\", ${GPU_UTIL} * 24467}")
KV_HEADROOM=$(awk "BEGIN{printf \"%d\", ${GPU_UTIL} * 24467 - 22440}")
ts INFO "${FREE_MIB} MiB free — util=${GPU_UTIL} → budget=${BUDGET_MIB} MiB → KV headroom=~${KV_HEADROOM} MiB"

# ─── BACKGROUND GPU MONITOR ───────────────────────────────────────────────────
ts INFO "=== STEP: start GPU monitor (${MONITOR_INTERVAL}s interval) ==="
(
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null \
            | awk -F',' '{printf "[GPU] %s | used=%s MiB free=%s MiB util=%s%% temp=%s°C\n",$1,$2,$3,$4,$5}'
        sleep "${MONITOR_INTERVAL}"
    done
) &
DMON_PID=$!
ts INFO "GPU monitor PID: ${DMON_PID}"

# ─── MODEL CHECK ──────────────────────────────────────────────────────────────
ts INFO "=== STEP: model check ==="
if [ ! -f "$LOCAL_MODEL_PATH/config.json" ]; then
    ts INFO "ERROR: Model not found at ${LOCAL_MODEL_PATH}"
    ts INFO "  Download: hf download nvidia/${MODEL_NAME} --local-dir ${LOCAL_MODEL_PATH}"
    exit 1
fi
MODEL_SIZE=$(du -sh "$LOCAL_MODEL_PATH" 2>/dev/null | cut -f1)
ts INFO "Model: ${LOCAL_MODEL_PATH} (${MODEL_SIZE})"

# ─── vLLM SETUP ───────────────────────────────────────────────────────────────
ts INFO "=== STEP: venv setup ==="
if [ ! -d "$VENV_DIR" ]; then
    ts INFO "Creating venv at ${VENV_DIR}..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
ts INFO "Activated venv: ${VENV_DIR}"

VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "")
if [ -z "$VLLM_VER" ] || [ "${VLLM_FORCE_UPDATE:-false}" = "true" ]; then
    ts INFO "Installing vLLM nightly (stable pip lacks SM120 arch flags — GH #35432)..."
    pip install --quiet --upgrade pip
    uv pip install -U vllm \
        --torch-backend=auto \
        --extra-index-url https://wheels.vllm.ai/nightly 2>&1 | tail -10
    VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)")
fi
ts INFO "vLLM:    ${VLLM_VER}"
ts INFO "Python:  $(python3 --version 2>&1)"
ts INFO "CUDA:    $(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo 'unknown')"
ts INFO "PyTorch: $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo 'unknown')"

# ─── CUDA TOOLKIT SHIM ────────────────────────────────────────────────────────
ts INFO "=== STEP: CUDA toolkit shim ==="
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
CU13="${VENV_DIR}/lib/python${PY_VER}/site-packages/nvidia/cu13"
if [ -d "$CU13" ]; then
    ln -sfn "$CU13/lib" "$CU13/lib64" 2>/dev/null || true
    ln -sf "libcudart.so.13" "$CU13/lib/libcudart.so" 2>/dev/null || true
    mkdir -p "$CU13/lib/stubs"
    ln -sf "/usr/lib/x86_64-linux-gnu/libcuda.so.1" "$CU13/lib/stubs/libcuda.so" 2>/dev/null || true
    ts INFO "CUDA_HOME shim applied: ${CU13}"
else
    ts INFO "CUDA_HOME shim skipped (${CU13} not found)"
fi
export CUDA_HOME="${CU13}"
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export FLASHINFER_DISABLE_VERSION_CHECK=1
# Seen in NVIDIA's own forum posts for Qwen3.6-35B-A3B on SM-class GPUs.
# Enables atomic-add path in Marlin kernels; no-op if not needed, harmless.
export VLLM_MARLIN_USE_ATOMIC_ADD=1

# ─── LAUNCH ───────────────────────────────────────────────────────────────────
ts INFO "=== STEP: build vllm serve args ==="
SERVE_ARGS=(
    "${LOCAL_MODEL_PATH}"
    --port "${PORT}"
    --tensor-parallel-size 1
    --trust-remote-code
    --dtype auto
    --quantization modelopt
    --kv-cache-dtype "${KV_DTYPE}"
    --attention-backend flashinfer
    --gpu-memory-utilization "${GPU_UTIL}"
    --max-model-len "${MAX_LEN}"
    --max-num-seqs "${MAX_SEQS}"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
    --async-scheduling
    --enforce-eager
    --served-model-name "qwen3.6-35b-a3b-nvfp4"
)
if [ "${CPU_OFFLOAD_GB}" != "0" ]; then
    SERVE_ARGS+=(--cpu-offload-gb "${CPU_OFFLOAD_GB}")
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    SERVE_ARGS+=("${EXTRA_ARGS[@]}")
fi

ts INFO "=== STEP: launch vllm serve ==="
ts INFO "Full command:"
ts INFO "  vllm serve ${SERVE_ARGS[*]}"

echo ""
KV_BUDGET=$(awk "BEGIN{printf \"%d\", ${GPU_UTIL} * 23982 - 21326}")
if [ "$MIN_MEMORY" = "true" ]; then
    NUM_BLOCKS=3
else
    NUM_BLOCKS=10
fi

echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  vLLM · Qwen3.6-35B-A3B-NVFP4  (Marlin W4A16 on SM120a)   │"
printf "│  RTX PRO 4000 · 24 GB · Port %-4s                          │\n" "${PORT}"
printf "│  ctx: %-6s  util: %-4s  seqs: %-2s  KV: fp8  eager: on    │\n" \
    "${MAX_LEN}" "${GPU_UTIL}" "${MAX_SEQS}"
printf "│  blocks: %-3s  KV budget: ~%-5s MiB  cpu_offload: %-3s GiB  │\n" \
    "${NUM_BLOCKS}" "${KV_BUDGET}" "${CPU_OFFLOAD_GB}"
printf "│  debug: %-5s                                               │\n" "${DEBUG}"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""
echo "  Expected VRAM sequence (with --enforce-eager):"
echo "    Weights load:   ~21332 MiB"
echo "    Marlin repack:  ~22430 MiB (peak), settles at ~21326 MiB"
echo "    KV blocks:      ${NUM_BLOCKS} × 164 MiB = $(( NUM_BLOCKS * 164 )) MiB (fp8, 2096 tok/block)"
echo "    Total settled:  ~$(( 21326 + NUM_BLOCKS * 164 )) MiB (budget: ~$(awk "BEGIN{printf \"%d\", ${GPU_UTIL} * 23982}") MiB at util=${GPU_UTIL})"
echo "    Profiling:      SKIPPED (--num-gpu-blocks-override avoids activation OOM hang)"
echo "    Success marker: 'Application startup complete'"
echo ""
if [ "${CPU_OFFLOAD_GB}" != "0" ]; then
echo "  CPU offload active: ${CPU_OFFLOAD_GB} GiB of expert weights → CPU RAM"
echo "    Latency cost: PCIe transfer per active MoE layer (acceptable for coding use)"
fi
echo ""
ts INFO "Launching vllm serve..."

vllm serve "${SERVE_ARGS[@]}" 2>&1
