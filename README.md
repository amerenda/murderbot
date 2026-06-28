# murderbot

Scripts, templates, and Docker image source for running local LLMs on murderbot (NVIDIA RTX 4000 Blackwell, 24 GB VRAM).

## Stack

```
opencode → LiteLLM ($LITELLM_BASE_URL /v1) → llama-server (:8088, Docker/Komodo) → GPU
```

- **Inference**: [llama.cpp](https://github.com/ggerganov/llama.cpp) compiled into `amerenda/murderbot-llm` Docker image (`llm/Dockerfile`, CUDA 12.8, sm_120a)
- **Deployment**: Komodo manages the container via `komodo-dean-gitops/murderbot/llm/compose.yaml`
- **Model**: Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf — MoE, ~21 GB on GPU
- **Template**: `llm/templates/froggeric-v20.jinja` — baked into the image at `/app/templates/`
- **Agent**: [opencode](https://opencode.ai) — coding agent harness

## Production deployment

The llama-server container is managed by **Komodo** — not the shell scripts in this repo.

- Compose spec: [`komodo-dean-gitops/murderbot/llm/compose.yaml`](https://github.com/amerenda/komodo-dean-gitops)
- Image: `amerenda/murderbot-llm:latest` (built from `llm/Dockerfile` in this repo)
- To update server flags or the Jinja template: edit `llm/entrypoint.sh` or `llm/templates/froggeric-v20.jinja`, build and push a new image, then restart the Komodo stack

## Local dev / opencode

`start-opencode-stable.sh` is for local development — it runs llama-server directly (not via Docker) and wires up opencode:

```bash
./start-opencode-stable.sh                   # default: Qwen3.6-35B
./start-opencode-stable.sh --restart         # kill existing server first, then start
./start-opencode-stable.sh --model qwen36-mtp   # MTP variant (~150 t/s, needs MTP GGUF)
CTX=229376 ./start-opencode-stable.sh        # larger context (tight but usable, ~0.9 GB free)
LITELLM_BASE_URL=https://my-litellm.example.com/v1 \  # custom LiteLLM endpoint
  ./start-opencode-stable.sh
```

The script:
1. Starts `llama-server` on `:8088` (waits until ready, only when no LiteLLM key is set)
2. Writes `~/.config/opencode/opencode.json` pointing at LiteLLM (or direct if key absent)
3. Launches `opencode`

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_BASE_URL` | `https://litellm.amer.dev/v1` | LiteLLM proxy endpoint (OpenAI-compatible). The LiteLLM key is auto-fetched from the k8s secret if `$LITELLM_MASTER_KEY` is unset. |
| `LITELLM_MASTER_KEY` | *(auto-fetched)* | Master API key for the LiteLLM proxy. Set manually to skip the kubectl lookup. |
| `CTX` | `131072` (varies by model) | Server context length in tokens. OpenCode uses CTX − 32768 as its own limit. |
| `OPENCODE_OUTPUT` | `16384` | Maximum output tokens for a single generation. |
| `OPENCODE_RESERVED` | `8192` | Safety margin subtracted from context so the model never hits the wall mid-generation. |

## Migration: removed llama-proxy.py (overflow proxy)

**What changed:** The local `proxy/llama-proxy.py` process (port 8089) that handled context overflow by splitting long responses has been retired. Context overflow now returns an error to the caller instead of being silently truncated or split.

**Why it was removed:** LiteLLM at `https://litellm.amer.dev/v1` provides a more reliable, centrally-managed inference proxy with built-in retry logic, metrics, and multi-backend routing. Running a local overflow-splitting proxy added complexity for marginal benefit — the template-level `max_tool_response_chars: 3000` in `--chat-template-kwargs` already truncates oversized tool responses at render time.

**What this means for you:**
- **If you were relying on the proxy for long file reads or large tool outputs**: those will now be truncated at 3000 chars by the template (same behavior as before, but without the split-and-retry loop). If you need larger responses, increase `OPENCODE_OUTPUT` and consider using a local LiteLLM instance that can route to a backend with more context.
- **If you were using the proxy for context overflow recovery**: the new path is opencode → LiteLLM → llama-server. LiteLLM handles retries automatically; if the server OOMs or drops a connection, LiteLLM retries up to its configured `num_retries`. No manual intervention needed.
- **To set up your own LiteLLM proxy**: deploy [LiteLLM](https://litellm.vercel.app/) (Docker image `ghcr.io/berriai/litellm:main-stable`) and point `$LITELLM_BASE_URL` at it. The proxy needs a master key passed via the `LITELLM_MASTER_KEY` env var or k8s secret.

## Models

| Variant | File | VRAM | Notes |
|---------|------|------|-------|
| `qwen36` (default) | `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | ~21.3 GB | Stable baseline |
| `qwen36-mtp` | `Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf` | ~21.8 GB | MTP speculative decoding ~150 t/s |
| `qwen3-coder-next` | `Qwen3-Coder-Next-UD-IQ3_XXS.gguf` | ~26.5 GB | Partial CPU offload, CTX=32768 |

Models live at `/mnt/storage/models/llms/`.

## VRAM headroom (CTX defaults to 196608)

| CTX | KV cache | Free VRAM |
|-----|----------|-----------|
| 131072 | ~1.0 GB | ~2.4 GB |
| 196608 | ~1.6 GB | **~1.4 GB ← default** |
| 229376 | ~2.1 GB | ~0.9 GB (tight) |
| 262144 | ~2.6 GB | ~0.4 GB (risky) |

KV cache uses `q4_0` quantization (`-ctk q4_0 -ctv q4_0`).

## Key flags (llm/entrypoint.sh)

- `--jinja --chat-template-file /app/templates/froggeric-v20.jinja` — uses the froggeric-v20 template baked into the image instead of the GGUF-embedded one.
- `--chat-template-kwargs '{"auto_disable_thinking_with_tools": true, ...}'` — suppresses `<think>` blocks during initial tool-selection rounds; re-enables thinking once ≥3 tool results have accumulated so the model can synthesize complex output.
- Flash attention (`-fa 1`) — enabled for performance.
- `-ctk q4_0 -ctv q4_0` — KV cache quantization to reduce VRAM usage.

## Architecture diagram (post-migration)

```
opencode ──→ LiteLLM ($LITELLM_BASE_URL /v1) ──→ llama-server (:8088, Docker/Komodo) ──→ GPU
  │              │
  │          retries on failure
  │          Prometheus metrics at /metrics
```

When no `$LITELLM_MASTER_KEY` is available, the script falls back to running `llama-server` locally:

```
opencode ──→ llama-server (:8088, local) ──→ GPU
```

This fallback path does **not** include any overflow proxy — large responses are truncated at the template level.
