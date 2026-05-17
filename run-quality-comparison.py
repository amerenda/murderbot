#!/usr/bin/env python3
"""
Quality comparison batch: same subjects × same seeds × 3 variants.

Compares base FLUX against UltraRealistic LoRA and Portrait Realism LoRA
on 5 hand-picked subjects. Generates a 3-column HTML gallery.

Usage:
  python3 run-quality-comparison.py
  python3 run-quality-comparison.py --output-dir /path/to/out
"""

import argparse
import json
import shutil
import time
import requests
from datetime import datetime
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────

COMFYUI        = "http://localhost:8188"
OLLAMA_HOST    = "http://10.100.20.18:11434"
OLLAMA_MODEL   = "llama3.1:8b"
OUTPUT_DIR     = Path("/home/alex/claude/comfyui/output/quality-comparison")
COMFYUI_OUTPUT = Path("/home/alex/claude/comfyui/output")

FLUX_MODEL = "flux1-dev-Q8_0.gguf"

OLLAMA_SYSTEM = (
    "You are a prompt engineer for photorealistic AI image generation. "
    "Take the user's simple description and expand it into a rich, detailed visual prompt. "
    "Include: specific lighting setup (golden hour, studio strobes, Rembrandt lighting, soft window light), "
    "camera and lens metadata (e.g. 'shot on Sony A7R V, 85mm f/1.4, shallow depth of field'), "
    "atmosphere, surface textures, colors, and composition. "
    "Add post-processing style where natural (e.g. Lightroom color grade, subtle film grain, RAW finish). "
    "Reply ONLY with the enhanced prompt — no explanations, no preamble, no quotes."
)

# ─── SUBJECTS (5) ────────────────────────────────────────────────────────────
# Hand-picked for maximum LoRA visibility: portraits (skin/face), textures, mood lighting

SUBJECTS = [
    {"id": "woman-portrait",  "seed": 1002, "prompt": "close-up portrait of a young woman, soft window light"},
    {"id": "man-portrait",    "seed": 1003, "prompt": "portrait of a middle-aged man in a coffee shop"},
    {"id": "food",            "seed": 1006, "prompt": "bowl of tonkotsu ramen with chashu pork and soft boiled egg"},
    {"id": "night-street",    "seed": 1009, "prompt": "rainy street in Tokyo at night with neon reflections"},
    {"id": "interior",        "seed": 1007, "prompt": "cozy living room with fireplace and bookshelves, evening"},
]

# ─── VARIANTS (3) ────────────────────────────────────────────────────────────
# Each variant uses same seed per subject for a direct apples-to-apples comparison.

VARIANTS = [
    {
        "id":       "base",
        "note":     "Base FLUX Q8_0",
        "steps":    20,
        "guidance": 3.5,
        "sampler":  "euler",
        "scheduler":"beta",
        "lora":     None,
        "lora_strength": 0.0,
    },
    {
        "id":       "ultrareal",
        "note":     "UltraRealistic LoRA v2",
        "steps":    35,
        "guidance": 2.5,
        "sampler":  "dpmpp_2m",
        "scheduler":"beta",
        "lora":     "UltraRealPhoto.safetensors",
        "lora_strength": 0.85,
    },
    {
        "id":       "portrait-realism",
        "note":     "Portrait Realism LoRA",
        "steps":    35,
        "guidance": 2.5,
        "sampler":  "dpmpp_2m",
        "scheduler":"beta",
        "lora":     "Flux_Portrait_Realism.safetensors",
        "lora_strength": 0.85,
    },
]

TOTAL = len(SUBJECTS) * len(VARIANTS)

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def expand_prompt(base_prompt):
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json={
        "model": OLLAMA_MODEL,
        "prompt": base_prompt,
        "system": OLLAMA_SYSTEM,
        "stream": False,
    }, timeout=60)
    resp.raise_for_status()
    return resp.json()["response"].strip()

def build_comfy_prompt(base_prompt, expanded_prompt, variant, seed, save_prefix):
    guidance  = variant["guidance"]
    clip_src  = ["4", 0]
    model_src = ["7", 0]

    nodes = {
        "4":  {"class_type": "DualCLIPLoaderGGUF",  "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp16.safetensors", "type": "flux"}},
        "7":  {"class_type": "UnetLoaderGGUF",       "inputs": {"unet_name": FLUX_MODEL}},
        "8":  {"class_type": "EmptyLatentImage",     "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "10": {"class_type": "VAELoader",            "inputs": {"vae_name": "ae.safetensors"}},
        "11": {"class_type": "VAEDecode",            "inputs": {"samples": ["9", 0], "vae": ["10", 0]}},
        "12": {"class_type": "SaveImage",            "inputs": {"images": ["11", 0], "filename_prefix": save_prefix}},
    }

    if variant["lora"]:
        nodes["13"] = {"class_type": "LoraLoader", "inputs": {
            "model":          ["7", 0],
            "clip":           ["4", 0],
            "lora_name":      variant["lora"],
            "strength_model": variant["lora_strength"],
            "strength_clip":  variant["lora_strength"],
        }}
        model_src = ["13", 0]
        clip_src  = ["13", 1]

    nodes["5"] = {"class_type": "CLIPTextEncodeFlux", "inputs": {"clip": clip_src, "clip_l": base_prompt,  "t5xxl": expanded_prompt, "guidance": guidance}}
    nodes["6"] = {"class_type": "CLIPTextEncodeFlux", "inputs": {"clip": clip_src, "clip_l": "",           "t5xxl": "",              "guidance": guidance}}
    nodes["9"] = {"class_type": "KSampler",           "inputs": {
        "model": model_src, "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["8", 0],
        "seed": seed, "steps": variant["steps"], "cfg": 1.0,
        "sampler_name": variant["sampler"], "scheduler": variant["scheduler"], "denoise": 1.0,
    }}
    return nodes

def queue_prompt(prompt):
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": prompt}, timeout=30)
    r.raise_for_status()
    return r.json()["prompt_id"]

def wait_for_completion(prompt_id, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{COMFYUI}/history/{prompt_id}", timeout=10)
        if r.status_code == 200:
            hist = r.json()
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("completed"):
                    return entry
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI error: {status.get('messages', [])}")
        time.sleep(4)
    raise TimeoutError(f"Timed out after {timeout}s waiting for {prompt_id}")

def get_output_image_path(history_entry):
    for node_outputs in history_entry.get("outputs", {}).values():
        if "images" in node_outputs:
            img = node_outputs["images"][0]
            sub = img.get("subfolder", "")
            return COMFYUI_OUTPUT / sub / img["filename"] if sub else COMFYUI_OUTPUT / img["filename"]
    raise RuntimeError("No image output found in history")

# ─── HTML GALLERY ─────────────────────────────────────────────────────────────

def generate_gallery(results):
    variant_headers = "".join(
        f'<th>'
        f'<div class="vl">{v["id"]}</div>'
        f'<div class="vd">{v["steps"]}st · {v["guidance"]}g · {v["sampler"]}<br>'
        f'{"LoRA: " + v["lora"].replace(".safetensors","") + " @ " + str(v["lora_strength"]) if v["lora"] else "no LoRA"}'
        f'</div></th>'
        for v in VARIANTS
    )

    rows = ""
    for subject in SUBJECTS:
        sid   = subject["id"]
        cells = ""
        for variant in VARIANTS:
            key = f"{sid}__{variant['id']}"
            r   = results.get(key)
            if r and r.get("image_file") and Path(r["image_file"]).exists():
                img_rel  = Path(r["image_file"]).name
                gen_time = f"{r['generation_time_seconds']:.0f}s" if r.get("generation_time_seconds") else "?"
                exp      = r.get("expanded_prompt", "")
                exp_tip  = (exp[:300] + "…") if len(exp) > 300 else exp
                cells += (
                    f'<td><div class="cell">'
                    f'<a href="{img_rel}" target="_blank">'
                    f'<img src="{img_rel}" loading="lazy" title="{exp_tip}"></a>'
                    f'<div class="cm">{gen_time}</div>'
                    f'</div></td>'
                )
            else:
                err = r.get("error", "not run") if r else "not run"
                cells += f'<td><div class="cell miss">{err[:60]}</div></td>'

        rows += (
            f'<tr><th class="sl">{sid}<br>'
            f'<small>{subject["prompt"]}</small></th>'
            f'{cells}</tr>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Quality Comparison — {datetime.now().strftime('%Y-%m-%d')}</title>
<style>
  body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:16px}}
  h1{{font-size:1.1rem;margin-bottom:2px}}
  p.meta{{color:#777;font-size:0.75rem;margin-bottom:12px}}
  table{{border-collapse:collapse;width:100%}}
  th,td{{border:1px solid #2a2a2a;padding:4px;vertical-align:top}}
  th.sl{{text-align:left;font-size:0.7rem;width:130px;background:#1a1a1a;white-space:normal;line-height:1.4}}
  th.sl small{{color:#888;font-weight:normal;display:block;margin-top:3px}}
  th{{background:#1a1a1a;font-size:0.7rem;text-align:center}}
  .vl{{font-size:0.95rem;font-weight:bold;margin-bottom:2px}}
  .vd{{font-size:0.6rem;color:#999;line-height:1.4}}
  .cell img{{width:100%;display:block;border-radius:2px}}
  .cell img:hover{{opacity:0.8;cursor:pointer}}
  .cm{{font-size:0.6rem;color:#777;text-align:center;padding-top:2px}}
  .miss{{background:#1e1010;color:#664;text-align:center;padding:40px 4px;font-size:0.65rem}}
</style>
</head>
<body>
<h1>Quality Comparison — Base vs LoRAs</h1>
<p class="meta">
  {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
  {len([r for r in results.values() if r.get('image_file')])} / {TOTAL} images &nbsp;·&nbsp;
  {len(SUBJECTS)} subjects · {len(VARIANTS)} variants · 1024×1024 · fixed seed per subject
</p>
<table>
  <thead><tr><th>Subject</th>{variant_headers}</tr></thead>
  <tbody>{rows}</tbody>
</table>
<p style="margin-top:12px;font-size:0.65rem;color:#444">
  Hover image for expanded prompt · Click to open full size · .json sidecar per image
</p>
</body>
</html>"""

# ─── MAIN ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Quality comparison: base FLUX vs LoRA variants")
    p.add_argument("--output-dir", default=None, help="Override output directory")
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    done    = 0

    print(f"Quality Comparison — {TOTAL} images ({len(SUBJECTS)} subjects × {len(VARIANTS)} variants)")
    print(f"Output: {outdir}\n")

    # Expand prompts once per subject (same expanded prompt used across all variants)
    print("Expanding prompts via Ollama...")
    expanded = {}
    for subject in SUBJECTS:
        try:
            expanded[subject["id"]] = expand_prompt(subject["prompt"])
            print(f"  ✓ {subject['id']}")
        except Exception as e:
            expanded[subject["id"]] = subject["prompt"]
            print(f"  ✗ {subject['id']}: Ollama failed ({e}), using base prompt")
    print()

    for subject in SUBJECTS:
        sid          = subject["id"]
        seed         = subject["seed"]
        base_prompt  = subject["prompt"]
        exp_prompt   = expanded[sid]

        for variant in VARIANTS:
            vid = variant["id"]
            key = f"{sid}__{vid}"
            done += 1
            prefix = f"qc__{sid}__{vid}"

            print(f"[{done}/{TOTAL}] {sid} / {vid}  (seed {seed})")

            try:
                t0    = time.time()
                nodes = build_comfy_prompt(base_prompt, exp_prompt, variant, seed, prefix)
                pid   = queue_prompt(nodes)
                entry = wait_for_completion(pid)
                gen   = time.time() - t0

                src = get_output_image_path(entry)
                dst = outdir / src.name
                shutil.copy2(src, dst)

                results[key] = {
                    "subject":                  sid,
                    "preset":                   vid,   # passenger reads this field
                    "variant":                  vid,
                    "seed":                     seed,
                    "expanded_prompt":          exp_prompt,
                    "generation_time_seconds":  gen,
                    "image_file":               str(dst),
                    "steps":                    variant["steps"],
                    "guidance":                 variant["guidance"],
                    "sampler":                  variant["sampler"],
                    "scheduler":                variant["scheduler"],
                    "lora":                     variant.get("lora"),
                    "lora_strength":            variant.get("lora_strength"),
                }

                sidecar = dst.with_suffix(".json")
                sidecar.write_text(json.dumps(results[key], indent=2))
                print(f"  → {src.name}  ({gen:.0f}s)")

            except Exception as e:
                results[key] = {"subject": sid, "variant": vid, "error": str(e)}
                print(f"  ✗ ERROR: {e}")

    # Write manifest.json (passenger-compatible format)
    ok_runs = [r for r in results.values() if r.get("image_file")]
    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "runs":         ok_runs,
        "generated_at": datetime.now().isoformat(),
    }, indent=2))

    # Write gallery
    gallery_html = generate_gallery(results)
    gallery_path = outdir / "index.html"
    gallery_path.write_text(gallery_html)

    ok  = len(ok_runs)
    err = len([r for r in results.values() if r.get("error")])
    print(f"\n{'─'*50}")
    print(f"Done: {ok} images  |  {err} errors")
    print(f"Gallery:  {gallery_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Compare:  ./run-passenger.sh {outdir}")


if __name__ == "__main__":
    main()
