# ─── Build stage ─────────────────────────────────────────────────────────────
# Compiles llama-server for SM_120a-real (NVIDIA RTX PRO 4000, Blackwell arch).
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake ninja-build \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

ARG LLAMA_CPP_REF=b9292
WORKDIR /build
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} https://github.com/ggml-org/llama.cpp .

# The devel image has CUDA stubs but no real driver. Add the stubs dir to ldconfig
# so the linker can resolve libcuda.so.1 transitively (libggml-cuda.so → libcuda.so.1).
# The real driver is injected at runtime by the NVIDIA container runtime.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf && \
    ldconfig

RUN cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="120a-real" \
        -DLLAMA_CURL=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja && \
    cmake --build build -j$(nproc) --target llama-server

# ─── Runtime stage ───────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/build/bin/llama-server /usr/local/bin/llama-server

WORKDIR /app
COPY templates/froggeric-v20.jinja templates/froggeric-v20.jinja
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8088

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=300s \
    CMD curl -sf http://localhost:8088/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
