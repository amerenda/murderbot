#!/usr/bin/env python3
"""
LoRA strength sweep: same subjects × same seeds × 4 LoRA strengths.

Tests how Portrait Realism LoRA strength affects output — 0.0 is the baseline
(LoRA loaded but disabled), 0.35/0.6/0.85 dial up the effect.
All variants use dpmpp_2m/beta/35 steps/guidance 2.5 to isolate strength only.

Usage:
  python3 run-lora-strength-sweep.py
  python3 run-lora-strength-sweep.py --lora UltraRealPhoto.safetensors
  python3 run-lora-strength-sweep.py --output-dir /path/to/out
  OLLAMA_MODEL=llama3.3:70b python3 run-lora-strength-sweep.py
"""

import argparse
import json
import os
import shutil
import time
import requests
from datetime import datetime
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────

COMFYUI        = "http://localhost:8188"
OLLAMA_HOST    = "http://10.100.20.18:11434"
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "mistral-nemo:12b")
OUTPUT_DIR     = Path("/home/alex/claude/comfyui/output/lora-strength-sweep")
COMFYUI_OUTPUT = Path("/home/alex/claude/comfyui/output")

FLUX_MODEL    = "flux1-dev-Q8_0.gguf"
DEFAULT_LORA  = "Flux_Portrait_Realism.safetensors"

_PROSE = (
    "Write as a single flowing paragraph of 80–130 words. "
    "No bullet points, no headers, no markdown, no quotes. "
    "Do not start with 'Capture', 'Create', or 'Imagine'. Start directly with the subject."
)

_SYSTEMS = {
    "portrait": (
        "You are a portrait photographer writing a generation prompt. "
        "Expand into a detailed prompt: subject appearance (expression, skin tone, hair texture); "
        "lighting (key type/angle, fill ratio, rim light, catchlights); "
        "camera body and prime lens (Sony A7R V 85mm f/1.4 or Canon R5 135mm f/2); "
        "aperture, shutter speed, ISO; bokeh, skin pore detail; "
        "post-processing (Lightroom, subtle retouch, color grade). " + _PROSE
    ),
    "food": (
        "You are a food photographer writing a generation prompt. "
        "Expand into a detailed prompt: plating, garnishes, surface material (slate, marble, wood); "
        "lighting (window, overhead hero, backlit), specular highlights, steam or condensation; "
        "lens (Canon 100mm f/2.8 macro or 85mm), aperture, depth of field; color temperature; "
        "post-processing (warm Lightroom grade, clarity, vignette). " + _PROSE
    ),
    "street": (
        "You are a street photographer writing a generation prompt. "
        "Expand into a detailed prompt: time/weather, urban details (signage, puddles, crowds); "
        "light sources and color temperatures; camera/lens (Leica Q2, Fujifilm X-T5); "
        "exposure (ISO grain vs frozen motion); color grade direction. " + _PROSE
    ),
    "interior": (
        "You are an interior photographer writing a generation prompt. "
        "Expand into a detailed prompt: room features, materials, textures (wood, fabric, stone); "
        "light sources and color temperature balance; camera/lens (Sony 16-35mm f/2.8); "
        "styling details (art, plants, candles); post-processing (Lightroom, shadow lift). " + _PROSE
    ),
    "generic": (
        "You are a photographer writing a generation prompt for FLUX.1 image generation. "
        "Expand into a rich photorealistic prompt: lighting setup (type, angle, modifiers); "
        "camera and lens with exposure settings; surface textures; atmosphere; post-processing. " + _PROSE
    ),
}

_ROUTING = [
    (["portrait", "woman", "man", "person", "face"], "portrait"),
    (["food", "ramen", "dish", "meal", "plate", "burger", "sushi", "coffee"], "food"),
    (["street", "tokyo", "city", "neon", "alley", "sidewalk"], "street"),
    (["interior", "room", "living", "bedroom", "fireplace", "office"], "interior"),
]

def _get_system(prompt):
    lower = prompt.lower()
    for keywords, name in _ROUTING:
        if any(k in lower for k in keywords):
            return _SYSTEMS[name]
    return _SYSTEMS["generic"]

# ─── SUBJECTS ────────────────────────────────────────────────────────────────

SUBJECTS = [
    {"id": "woman-portrait",  "seed": 1002, "prompt": "close-up portrait of a young woman, soft window light"},
    {"id": "man-portrait",    "seed": 1003, "prompt": "portrait of a middle-aged man in a coffee shop"},
    {"id": "food",            "seed": 1006, "prompt": "bowl of tonkotsu ramen with chashu pork and soft boiled egg"},
    {"id": "night-street",    "seed": 1009, "prompt": "rainy street in Tokyo at night with neon reflections"},
    {"id": "interior",        "seed": 1007, "prompt": "cozy living room with fireplace and bookshelves, evening"},
]

# ─── STRENGTH VARIANTS ───────────────────────────────────────────────────────
# All use the same sampler/scheduler/steps/guidance to isolate LoRA strength only.
# strength=0.0 is the no-LoRA baseline at identical settings.

STRENGTHS = [0.0, 0.35, 0.6, 0.85]
STEPS     = 35
GUIDANCE  = 2.5
SAMPLER   = "dpmpp_2m"
SCHEDULER = "beta"

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def expand_prompt(base_prompt):
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json={
        "model": OLLAMA_MODEL,
        "prompt": base_prompt,
        "system": _get_system(base_prompt),
        "stream": False,
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()

def build_comfy_prompt(base_prompt, expanded_prompt, seed, save_prefix, lora, strength):
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

    if strength > 0.0:
        nodes["13"] = {"class_type": "LoraLoader", "inputs": {
            "model":          ["7", 0],
            "clip":           ["4", 0],
            "lora_name":      lora,
            "strength_model": strength,
            "strength_clip":  strength,
        }}
        model_src = ["13", 0]
        clip_src  = ["13", 1]

    nodes["5"] = {"class_type": "CLIPTextEncodeFlux", "inputs": {"clip": clip_src, "clip_l": base_prompt,  "t5xxl": expanded_prompt, "guidance": GUIDANCE}}
    nodes["6"] = {"class_type": "CLIPTextEncodeFlux", "inputs": {"clip": clip_src, "clip_l": "",           "t5xxl": "",              "guidance": GUIDANCE}}
    nodes["9"] = {"class_type": "KSampler",           "inputs": {
        "model": model_src, "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["8", 0],
        "seed": seed, "steps": STEPS, "cfg": 1.0,
        "sampler_name": SAMPLER, "scheduler": SCHEDULER, "denoise": 1.0,
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
    raise TimeoutError(f"Timed out after {timeout}s")

def get_output_image_path(history_entry):
    for node_outputs in history_entry.get("outputs", {}).values():
        if "images" in node_outputs:
            img = node_outputs["images"][0]
            sub = img.get("subfolder", "")
            return COMFYUI_OUTPUT / sub / img["filename"] if sub else COMFYUI_OUTPUT / img["filename"]
    raise RuntimeError("No image output found")

# ─── HTML GALLERY ─────────────────────────────────────────────────────────────

def generate_gallery(results, lora):
    variant_headers = "".join(
        f'<th><div class="vl">strength {s}</div>'
        f'<div class="vd">{"no LoRA" if s == 0.0 else f"LoRA @ {s}"}<br>'
        f'{STEPS}st · {GUIDANCE}g · {SAMPLER}</div></th>'
        for s in STRENGTHS
    )

    rows = ""
    for subject in SUBJECTS:
        sid   = subject["id"]
        cells = ""
        for s in STRENGTHS:
            key = f"{sid}__s{s}"
            r   = results.get(key)
            if r and r.get("image_file") and Path(r["image_file"]).exists():
                img_rel  = Path(r["image_file"]).name
                gen_time = f"{r['generation_time_seconds']:.0f}s" if r.get("generation_time_seconds") else "?"
                cells += (
                    f'<td><div class="cell">'
                    f'<a href="{img_rel}" target="_blank">'
                    f'<img src="{img_rel}" loading="lazy"></a>'
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
<title>LoRA Strength Sweep — {datetime.now().strftime('%Y-%m-%d')}</title>
<style>
  body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:16px}}
  h1{{font-size:1.1rem;margin-bottom:2px}}
  p.meta{{color:#777;font-size:0.75rem;margin-bottom:12px}}
  table{{border-collapse:collapse;width:100%}}
  th,td{{border:1px solid #2a2a2a;padding:4px;vertical-align:top}}
  th.sl{{text-align:left;font-size:0.7rem;width:130px;background:#1a1a1a;line-height:1.4}}
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
<h1>LoRA Strength Sweep — {Path(lora).stem}</h1>
<p class="meta">
  {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
  {len([r for r in results.values() if r.get('image_file')])} / {len(SUBJECTS) * len(STRENGTHS)} images &nbsp;·&nbsp;
  {len(SUBJECTS)} subjects · {len(STRENGTHS)} strengths · 1024×1024 · {STEPS}st {SAMPLER}/{SCHEDULER} g{GUIDANCE}
</p>
<table>
  <thead><tr><th>Subject</th>{variant_headers}</tr></thead>
  <tbody>{rows}</tbody>
</table>
</body>
</html>"""

# ─── MAIN ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LoRA strength sweep")
    p.add_argument("--lora",       default=DEFAULT_LORA, help=f"LoRA filename (default: {DEFAULT_LORA})")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    lora   = args.lora
    outdir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    total = len(SUBJECTS) * len(STRENGTHS)
    done  = 0
    results = {}

    print(f"LoRA Strength Sweep — {total} images ({len(SUBJECTS)} subjects × {len(STRENGTHS)} strengths)")
    print(f"LoRA:   {lora}")
    print(f"Model:  {OLLAMA_MODEL}")
    print(f"Output: {outdir}\n")

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
        sid         = subject["id"]
        seed        = subject["seed"]
        base_prompt = subject["prompt"]
        exp_prompt  = expanded[sid]

        for strength in STRENGTHS:
            key    = f"{sid}__s{strength}"
            done  += 1
            label  = f"s{strength}"
            prefix = f"lss__{sid}__{label}"

            print(f"[{done}/{total}] {sid} / strength {strength}  (seed {seed})")

            try:
                t0    = time.time()
                nodes = build_comfy_prompt(base_prompt, exp_prompt, seed, prefix, lora, strength)
                pid   = queue_prompt(nodes)
                entry = wait_for_completion(pid)
                gen   = time.time() - t0

                src = get_output_image_path(entry)
                dst = outdir / src.name
                shutil.copy2(src, dst)

                results[key] = {
                    "subject":                 sid,
                    "preset":                  label,
                    "lora":                    lora,
                    "lora_strength":           strength,
                    "seed":                    seed,
                    "steps":                   STEPS,
                    "guidance":                GUIDANCE,
                    "sampler":                 SAMPLER,
                    "scheduler":               SCHEDULER,
                    "expanded_prompt":         exp_prompt,
                    "generation_time_seconds": gen,
                    "image_file":              str(dst),
                }

                (dst.with_suffix(".json")).write_text(json.dumps(results[key], indent=2))
                print(f"  → {src.name}  ({gen:.0f}s)")

            except Exception as e:
                results[key] = {"subject": sid, "preset": label, "error": str(e)}
                print(f"  ✗ ERROR: {e}")

    ok_runs = [r for r in results.values() if r.get("image_file")]
    (outdir / "manifest.json").write_text(json.dumps({
        "runs": ok_runs,
        "generated_at": datetime.now().isoformat(),
    }, indent=2))

    (outdir / "index.html").write_text(generate_gallery(results, lora))

    ok  = len(ok_runs)
    err = len(results) - ok
    print(f"\n{'─'*50}")
    print(f"Done: {ok} images  |  {err} errors")
    print(f"Compare: ./run-passenger.sh {outdir}")


if __name__ == "__main__":
    main()
