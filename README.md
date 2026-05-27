# murderbot

Scripts and tooling for running local LLMs on murderbot (NVIDIA RTX 4000 Blackwell, 24 GB VRAM).

## Stack

```
opencode → llama-proxy (:8089) → llama-server (:8088) → GPU
```

- **Inference**: [llama.cpp](https://github.com/ggerganov/llama.cpp) built from source (`~/claude/llama.cpp`, CUDA 12.8, sm_120a)
- **Model**: Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf — MoE, ~21 GB on GPU
- **Agent**: [opencode](https://opencode.ai) — coding agent harness
- **Proxy**: `llama-proxy.py` — context overflow recovery (see below)

## Quick start

```bash
./start-opencode-stable.sh                   # default: Qwen3.6-35B, 196K context
./start-opencode-stable.sh --restart         # kill existing server first, then start
./start-opencode-stable.sh --model qwen36-mtp   # MTP variant (~150 t/s, needs MTP GGUF)
CTX=229376 ./start-opencode-stable.sh        # larger context (tight but usable, ~0.9 GB free)
NO_PROXY=true ./start-opencode-stable.sh     # skip proxy, point opencode directly at server
```

The script:
1. Starts `llama-server` on `:8088` (waits until ready)
2. Starts `llama-proxy.py` on `:8089`
3. Writes `~/.config/opencode/opencode.json` pointing at the proxy
4. Launches `opencode`
5. Kills proxy on exit

## llama-proxy.py — overflow recovery

### The problem

opencode uses the GPT `cl100k_base` tiktoken tokenizer for all OpenAI-compatible providers. Qwen3's XML tool-call format tokenizes 2–4× heavier than GPT estimates. On top of that, `--jinja` injects the full tools array into the system message at render time — those tokens are invisible to opencode's counter. Result: opencode sends requests that exceed llama-server's context window → immediate HTTP 400, session dead.

### The solution

The proxy intercepts every `/v1/chat/completions` request. On a 400 response it:

1. Strips the 2 largest `tool` role messages from conversation history (file reads are the worst offenders) plus the `assistant` message that called the tool
2. Retries the request
3. Repeats up to 10 times, falling back to removing oldest messages if no tool messages remain
4. Always protects: the system prompt (`messages[0]`) and the last 4 messages (current turn)

The proxy is transparent on the happy path — it streams bytes through with no overhead.

### Observed behavior

- Caught 3 overflows in a 15-minute session with no crashes
- Each overflow stripped 45–54 KB of old tool output (file reads)
- Without the proxy these would have been hard 400s killing the session
- Logs to `/tmp/llama-proxy.log`

```
[proxy] 400 overflow → stripped 2 msgs (53,951 bytes, retry 1/10, 32 msgs remain)
```

### Manual usage

```bash
python3 llama-proxy.py                                      # default: :8089 → :8088
python3 llama-proxy.py --upstream http://127.0.0.1:8088 --port 8089
python3 llama-proxy.py --quiet                              # suppress log output
```

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

## Key flags

- `--reasoning off` — disables Qwen3 thinking mode. Without this the model generates reasoning text before `<tool_call>` and the parser fails.
- `--jinja` — uses the GGUF-embedded Unsloth-patched Jinja template (includes `enable_thinking`, `preserve_thinking`, `developer` role fixes).
- Flash attention (`-fa 1`) — enabled for performance.

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
| `start-opencode-stable.sh` | **Main launcher** — use this |
| `start-opencode.sh` | Original launcher — do not modify |
| `llama-proxy.py` | Overflow recovery proxy |
| `qwen36-stable.jinja` | Reference copy of GGUF-embedded Jinja template (not used at runtime) |
| `CLAUDE.md` | opencode system instructions — file reading limits, tool discipline |
