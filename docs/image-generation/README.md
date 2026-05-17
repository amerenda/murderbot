# Image Generation on Murderbot

Murderbot (RTX PRO 4000 Blackwell, 24GB VRAM) runs [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for image generation using FLUX.1-dev and Illustrious XL. The Mac Mini M4 (10.100.20.18) runs Ollama for prompt expansion.

## Quick Start

```bash
# Start ComfyUI
cd ~/claude/projects/murderbot
./start-flux.sh

# Open in browser
# Local:  http://localhost:8188
# Remote: https://comfy.amer.dev
```

## Pipelines

### FLUX.1-dev — Photorealistic

Best for photorealistic images, portraits, landscapes, and product shots.

| Workflow | Description |
|---|---|
| `flux-ollama-prompt-expand` | General photorealistic generation with Ollama prompt expansion |
| `flux-ultrarealistic-lora` | Same as above + UltraRealistic LoRA for maximum detail |
| `fluxed-up-nsfw-ollama` | Fluxed Up NSFW fine-tune (fp8 safetensors) |

**Models loaded:**
- UNET: `flux1-dev-Q8_0.gguf` (12GB, near-lossless quality)
- T5 encoder: `t5xxl_fp16.safetensors` (9.4GB, full precision)
- CLIP-L: `clip_l.safetensors` (235MB)
- VAE: `ae.safetensors`
- **Total VRAM: ~22GB**

**Tuned settings (from 110-image benchmark):**
- Sampler: `euler`, Scheduler: `beta`
- CFG: `1.0` (FLUX ignores CFG — leave at 1.0)
- Guidance: `3.5` general, `2.5` portraits, `7.0` vivid/high-contrast, `5.0` complex scenes
- Steps: `20` general, `35–40` with UltraRealistic LoRA

### Illustrious XL — Anime / Artistic

Best for anime, illustration, and artistic styles.

| Workflow | Description |
|---|---|
| `illustrious-ollama-prompt-expand` | Anime/artistic generation with Ollama prompt expansion |

**Models loaded:**
- Checkpoint: `waiIllustriousSDXL_v170.safetensors`
- **Total VRAM: ~8GB** (leaves room to run alongside small models)

**Settings:**
- Sampler: `euler_ancestral`, Scheduler: `karras`
- CFG: `7.0`
- Steps: `25`
- Negative prompt: `worst quality, low quality, bad quality, lowres, blurry, ugly, deformed, bad anatomy, watermark`

## Post-Processing Workflows

These workflows take a generated image as input and enhance it. Run them after generating with one of the pipelines above.

### 4x Upscale — `flux-upscale-4x`

Upscales any image 4× using NMKD-Siax (detail-preserving ESRGAN model). Turns 1024×1024 output into a sharp 4096×4096 image.

1. Load `flux-upscale-4x` workflow in ComfyUI
2. Drag your generated image onto the **LoadImage** node
3. Queue — output saved as `upscaled_4x_*.png`

**Model:** `4x_NMKD-Siax_200k.pth` — no VRAM required beyond base ComfyUI overhead.

### Face Detailer — `flux-face-detail`

Detects faces in an image and re-generates each at high resolution using FLUX, then composites back. Fixes blurry or degraded faces in portraits.

1. Load `flux-face-detail` workflow in ComfyUI
2. Drag your generated image onto the **LoadImage** node
3. Queue — output saved as `face_detailed_*.png`

**First run:** Impact Pack will auto-download `face_yolov8m.pt` (face detector) and `sam_vit_b_01ec64.pth` (SAM segmentation model). This takes ~1–2 min.

**Settings inside FaceDetailer:**
- `denoise`: 0.45 — controls how much the face is regenerated (lower = preserve more of original)
- `steps`: 20
- Face crop size: 512px

## LoRAs

LoRAs are stored in `/mnt/storage/models/image/loras/flux/`.

| File | Size | Use |
|---|---|---|
| `UltraRealPhoto.safetensors` | 2GB | Maximum photorealism, better anatomy and skin. Use at strength 0.85. |
| `Flux_Portrait_Realism.safetensors` | 292MB | Portrait-focused realism enhancement. |

The `flux-ultrarealistic-lora` workflow uses `UltraRealPhoto.safetensors` at strength 0.85 with tuned settings (`dpmpp_2m`, 35 steps, guidance 2.5).

To add a LoRA to any workflow, insert a **LoraLoader** node between the UNET loader and KSampler.

## ControlNet

ControlNet lets you guide image generation with a reference image — preserving structure (edges, depth, pose) while changing style or content.

**Models** are in `/mnt/storage/models/image/controlnet/`. Download with:

```bash
~/claude/manual_runs/download-flux-controlnet.sh
```

| File | Type | Size | Use |
|---|---|---|---|
| `flux-canny-controlnet-v3.safetensors` | Canny | ~1.5GB | Edge/structure guidance from a reference image |
| `flux-depth-controlnet-v3.safetensors` | Depth | ~1.5GB | Depth map guidance for 3D structure |

**VRAM note:** ControlNet adds ~1.5GB. To stay within 24GB, the `flux-controlnet-canny.json` workflow uses `flux1-dev-Q4_K.gguf` (6.5GB) instead of Q8_0 (12GB):

```
Q4_K (6.5GB) + T5 fp16 (9.4GB) + CLIP (0.2GB) + ControlNet (1.5GB) ≈ 18GB  ✓
```

**Workflow:** `flux-controlnet-canny.json`

1. Load the workflow in ComfyUI
2. Drag a reference image onto the **LoadImage** node
3. `CannyEdgePreprocessor` (from comfyui_controlnet_aux) extracts edges
4. `ControlNetApplyAdvanced` applies them to the conditioning at `strength=0.8`
5. Edit the prompt in CLIPTextEncodeFlux nodes
6. Queue — the output will follow the reference image's structure

**Tuning ControlNet strength:**
- `0.3–0.5` — loose guidance, more creative freedom
- `0.6–0.8` — balanced structure + style
- `0.9–1.0` — strict structure, less variation

**Custom nodes required:** `Fannovel16/comfyui_controlnet_aux` (auto-installed by `start-flux.sh`)

## Comparing Results with Passenger

[Passenger](../../passenger.html) is a tournament-mode comparison tool. It shows two images side-by-side and you pick the winner. The loser is eliminated. One image per subject survives.

```bash
# Compare the default tuning batch output
./run-passenger.sh

# Compare a specific output directory
./run-passenger.sh ~/claude/comfyui/output/quality-comparison
./run-passenger.sh ~/claude/comfyui/output/tuning2
```

Open `http://localhost:8189` in a browser. Keyboard shortcuts: `1`/`←` left wins, `2`/`→` right wins, `Space` undo, `s` skip.

### manifest.json format

Every batch script must write a `manifest.json` to its output directory. Passenger won't start without it.

```json
{
  "runs": [
    {
      "subject": "woman-portrait",
      "preset":  "ultrareal",
      "image_file": "/abs/path/to/qc__woman-portrait__ultrareal_00001_.png",
      "steps": 35,
      "guidance": 2.5,
      "sampler": "dpmpp_2m"
    }
  ],
  "generated_at": "2026-05-17T09:00:00"
}
```

**Required fields per run:** `subject` (tournament category), `preset` (label shown in UI), `image_file` (absolute or relative path — basename is derived from it). Optional: `steps`, `guidance`, `sampler` (shown on hover).

### Image + sidecar naming convention

Batch scripts must follow this naming convention so Passenger can find the sidecar metadata:

```
{prefix}__{subject}__{preset}_00001_.png    ← image (ComfyUI adds counter)
{prefix}__{subject}__{preset}_00001_.json   ← sidecar with steps/guidance/sampler
```

The sidecar filename must match the image filename with `.json` replacing `.png`. Passenger derives the sidecar path from `image_file` automatically.

| Script | Prefix | Example filename |
|---|---|---|
| `run-tuning-batch.py` | none | `asian-woman__A__20steps__3.5guid__euler__beta__seed1002_00001_.png` |
| `run-quality-comparison.py` | `qc__` | `qc__woman-portrait__ultrareal_00001_.png` |
| `run-sdxl-batch.py` | `sdxl__` | `sdxl__portrait__preset-A_00001_.png` |

## Batch Generation

Batch runners generate multiple images for systematic comparison. All batch scripts support `OLLAMA_MODEL=<model>` to switch the prompt expansion model:

```bash
# Use llama3.3:70b for better prompt expansion (if available on Ollama)
OLLAMA_MODEL=llama3.3:70b python3 run-quality-comparison.py
```

### Quality Comparison

5 subjects × 3 variants (base, UltraRealistic LoRA, Portrait Realism LoRA) = 15 images.

```bash
python3 run-quality-comparison.py
python3 run-quality-comparison.py --output-dir /path/to/out
```

Output: `~/claude/comfyui/output/quality-comparison/`

### LoRA Strength Sweep

5 subjects × 4 strengths (0.0, 0.35, 0.6, 0.85) = 20 images. Tests a single LoRA at different strengths with strength 0.0 as the no-LoRA baseline.

```bash
python3 run-lora-strength-sweep.py
python3 run-lora-strength-sweep.py --lora UltraRealPhoto.safetensors
python3 run-lora-strength-sweep.py --output-dir /path/to/out
```

Output: `~/claude/comfyui/output/lora-strength-sweep/`

### Aspect Ratio Test

5 subjects × 2 resolutions (1024×1024 square, 832×1216 portrait) = 10 images. Tests whether portrait subjects look better at native 2:3 aspect.

```bash
python3 run-aspect-ratio-test.py
python3 run-aspect-ratio-test.py --output-dir /path/to/out
```

Output: `~/claude/comfyui/output/aspect-ratio-test/`

### FLUX Tuning Batch

Runs 10 subjects × 11 presets (110 images) to compare parameters.

```bash
cd ~/claude/projects/murderbot

# Default (flux1-dev-Q8_0.gguf, t5xxl_fp16)
python3 run-tuning-batch.py

# Custom model
python3 run-tuning-batch.py --model flux1-dev-Q4_K.gguf

# With LoRA
python3 run-tuning-batch.py --lora UltraRealPhoto.safetensors --lora-strength 0.85

# Portrait aspect ratio
python3 run-tuning-batch.py --width 832 --height 1216 --output-dir output/portrait-batch
```

Output: `~/claude/comfyui/output/tuning2/`
Gallery: open `index.html` in that directory.

### SDXL/FLUX Comparison Batch

Compares presets for Illustrious XL or Fluxed Up NSFW.

```bash
# Illustrious XL
python3 run-sdxl-batch.py --pipeline sdxl

# Fluxed Up NSFW (FLUX safetensors pipeline)
python3 run-sdxl-batch.py --pipeline flux
```

Output: `~/claude/comfyui/output/sdxl-batch/`

## Tuning Reference

From 110-image benchmark (2026-05-17), votes across 10 subjects:

| Subject type | Guidance | Notes |
|---|---|---|
| Portraits / people | 2.5–3.5 | Lower = more natural skin/faces |
| Vivid, high-contrast | 7.0 | Birds, food, product shots |
| Complex indoor scenes | 5.0 + 30 steps | Architecture, interiors |
| Landscapes | 3.5 | |

Scheduler: always `beta`. Sampler: `euler` (9/10 subjects). Steps: `20` for speed, `35–40` with UltraRealistic LoRA.

## Models on Disk

```
/mnt/storage/models/image/
├── flux1-dev-gguf/
│   ├── flux1-dev-Q8_0.gguf      # 12GB  — primary FLUX model
│   ├── flux1-dev-Q4_K.gguf      # 6.5GB — fallback if VRAM tight
│   ├── t5xxl_fp16.safetensors   # 9.4GB — T5 text encoder (full precision)
│   ├── t5xxl_fp8_e4m3fn.safetensors  # 4.6GB — T5 fp8 (fallback)
│   ├── clip_l.safetensors       # 235MB
│   └── clip_l-Q8_0.gguf        # 125MB (alternative CLIP)
├── flux-nsfw/
│   └── fluxedUpFluxNSFW_90FP8.safetensors  # 11GB — Fluxed Up NSFW fine-tune
├── illustrious/
│   └── waiIllustriousSDXL_v170.safetensors
├── loras/
│   ├── flux/
│   │   ├── UltraRealPhoto.safetensors       # 2GB  — UltraRealistic LoRA V2
│   │   └── Flux_Portrait_Realism.safetensors  # 292MB
│   └── illustrious/
│       └── (illustrious loras)
└── upscale/
    └── 4x_NMKD-Siax_200k.pth   # 64MB — 4x ESRGAN upscaler
```

## Tournament Results

Winners picked via Passenger (tournament-mode elimination). Each category compares all variants head-to-head until one remains.

### Quality Comparison Round 1 — 2026-05-17 (llama3.1:8b prompts)

5 subjects × 3 variants (base, UltraRealistic LoRA @ 0.85, Portrait Realism LoRA @ 0.85)

| Subject | Winner | Notes |
|---|---|---|
| woman-portrait | base | LoRAs over-processed; base more natural |
| man-portrait | base | |
| food | base | |
| night-street | portrait-realism | LoRA added mood/atmosphere |
| interior | portrait-realism | LoRA helped scene depth |

**Takeaway:** Base FLUX with tuned settings wins for isolated subjects (portraits, food). LoRA helps complex multi-element scenes.

### Quality Comparison Round 2 — 2026-05-17 (mistral-nemo:12b prompts)

Same 5 subjects × 3 variants. Switched prompt model to `mistral-nemo:12b` for better expansion quality.

| Subject | Winner | Notes |
|---|---|---|
| woman-portrait | portrait-realism | Better expanded prompts surfaced LoRA benefit |
| man-portrait | ultrareal | |
| food | ultrareal | |
| night-street | ultrareal | |
| interior | base | LoRA post-processing hurts natural interior light |

**Takeaway:** With better prompts, UltraReal LoRA is competitive (wins 3/5 vs 0/5 in Round 1). Interior scenes still favor base. The prompt model matters — `mistral-nemo:12b` unlocked quality that `llama3.1:8b` couldn't produce.

### Prompt Model — Mac Mini M4

Ollama runs on Mac Mini M4 (16GB unified memory):

| Model | VRAM | Status |
|---|---|---|
| `llama3.1:8b` | ~4.7GB | Default baseline |
| `mistral-nemo:12b` | ~7.2GB | **Current default** — better creative prose, ~1.2GB headroom |
| `llama3.3:70b` | ~40GB | Does not fit |

## Architecture

```
Mac Mini M4 (10.100.20.18)          Murderbot (10.100.20.19)
┌──────────────────────┐            ┌────────────────────────────┐
│ Ollama               │            │ ComfyUI :8188              │
│ mistral-nemo:12b     │◄───────────│ OllamaGenerateV2           │
│ Prompt expansion     │            │                            │
└──────────────────────┘            │ FLUX.1-dev Q8_0 (12GB)    │
                                    │ T5 fp16 (9.4GB)            │
                                    │ CLIP-L (235MB)             │
                                    │ ~22GB VRAM / 24GB total    │
                                    └────────────────────────────┘
```

The Mac Mini handles all language work (prompt expansion via Ollama). Murderbot handles all image generation on GPU.
