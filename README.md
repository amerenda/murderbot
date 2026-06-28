# murderbot

Scripts, templates, and Docker image source for running local LLMs on murderbot (NVIDIA RTX 4000 Blackwell, 24 GB VRAM).

## Stack

```
opencode → LiteLLM (https://litellm.amer.dev/v1) → llama-server (:8088, Docker/Komodo) → GPU
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
```

The script:
1. Starts `llama-server` on `:8088` (waits until ready)
2. Writes `~/.config/opencode/opencode.json` pointing at LiteLLM (or direct if key absent)
3. Launches `opencode`

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

## Other scripts

| Script | Purpose |
|--------|---------|
| `start-flux.sh` | Start ComfyUI / FLUX image generation stack |
| `run-passenger.sh` | Run the Passenger image prompt expansion service |
| `passenger.py` | Passenger service — Ollama-backed prompt expansion |
| `run-*.py` | Image generation batch runners (SDXL, lora sweeps, etc.) |
| `test-prompt.py` | Test prompt strategies against the local LLM |

## Files

| File | Notes |
|------|-------|
| `llm/Dockerfile` | Builds `amerenda/murderbot-llm` image — compiled llama-server for SM_120a |
| `llm/entrypoint.sh` | Container entry point — server flags, model path, template kwargs |
| `llm/templates/froggeric-v20.jinja` | **Active template** — baked into the image at build time |
| `start-opencode-stable.sh` | Local dev launcher — runs llama-server directly (not Docker) |
| `templates/froggeric-v20.jinja` | Mirror of `llm/templates/` — keep in sync; used by local dev script |
| `CLAUDE.md` | opencode system instructions — file reading limits, tool discipline |
