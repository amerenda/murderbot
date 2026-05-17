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

Batch runners generate multiple images for systematic comparison.

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

## Architecture

```
Mac Mini M4 (10.100.20.18)          Murderbot (10.100.20.19)
┌──────────────────────┐            ┌────────────────────────────┐
│ Ollama               │            │ ComfyUI :8188              │
│ llama3.1:8b          │◄───────────│ OllamaGenerateV2           │
│ Prompt expansion     │            │                            │
└──────────────────────┘            │ FLUX.1-dev Q8_0 (12GB)    │
                                    │ T5 fp16 (9.4GB)            │
                                    │ CLIP-L (235MB)             │
                                    │ ~22GB VRAM / 24GB total    │
                                    └────────────────────────────┘
```

The Mac Mini handles all language work (prompt expansion via Ollama). Murderbot handles all image generation on GPU.
