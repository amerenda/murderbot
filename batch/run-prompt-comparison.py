#!/usr/bin/env python3
"""
Prompt strategy comparison: generic expansion vs subject-type-routed expansion.

Same subjects × same seeds × 2 prompt strategies = 10 images.
Both use identical FLUX settings (euler/beta/20 steps/guidance 3.5) so the only
variable is the prompt given to FLUX. Lets you see directly whether better
prompt expansion produces better images.

Usage:
  python3 run-prompt-comparison.py
  python3 run-prompt-comparison.py --output-dir /path/to/out
  OLLAMA_MODEL=mistral-nemo:12b python3 run-prompt-comparison.py
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
OUTPUT_DIR     = Path("/home/alex/claude/comfyui/output/prompt-comparison")
COMFYUI_OUTPUT = Path("/home/alex/claude/comfyui/output")
FLUX_MODEL     = "flux1-dev-Q8_0.gguf"

STEPS    = 20
GUIDANCE = 3.5
SAMPLER  = "euler"
SCHEDULER = "beta"

# ─── SUBJECTS ────────────────────────────────────────────────────────────────

SUBJECTS = [
    {"id": "woman-portrait",  "seed": 1002, "prompt": "close-up portrait of a young woman, soft window light"},
    {"id": "man-portrait",    "seed": 1003, "prompt": "portrait of a middle-aged man in a coffee shop"},
    {"id": "food",            "seed": 1006, "prompt": "bowl of tonkotsu ramen with chashu pork and soft boiled egg"},
    {"id": "night-street",    "seed": 1009, "prompt": "rainy street in Tokyo at night with neon reflections"},
    {"id": "interior",        "seed": 1007, "prompt": "cozy living room with fireplace and bookshelves, evening"},
]

# ─── PROMPT STRATEGIES ───────────────────────────────────────────────────────

_PROSE = (
    "Write as a single flowing paragraph of 80–130 words. "
    "No bullet points, no headers, no markdown, no quotes. "
    "Do not start with 'Capture', 'Create', or 'Imagine'. "
    "Start directly with the subject."
)

SYSTEM_GENERIC = (
    "You are a prompt engineer for photorealistic AI image generation. "
    "Expand the description into a rich visual prompt including: lighting setup, "
    "camera and lens (e.g. Sony A7R V, 85mm f/1.4), atmosphere, surface textures, "
    "and post-processing style (Lightroom, film grain, RAW). " + _PROSE
)

SYSTEM_PORTRAIT = (
    "You are a portrait photographer writing a generation prompt. "
    "Expand into a detailed prompt covering: subject appearance (expression, skin tone, hair texture); "
    "lighting (key type/angle, fill ratio, rim light, catchlights in eyes); "
    "camera body and prime lens (Sony A7R V 85mm f/1.4, Canon R5 135mm f/2, or similar); "
    "aperture, shutter speed, ISO; shallow bokeh, skin pore detail; "
    "post-processing (Lightroom, subtle retouch, color grade). " + _PROSE
)

SYSTEM_FOOD = (
    "You are a food photographer writing a generation prompt. "
    "Expand into a detailed prompt covering: plating and garnishes; surface material (slate, marble, wood); "
    "lighting direction and quality, specular highlights, steam or condensation; "
    "lens (Canon 100mm f/2.8 macro or 85mm), aperture, depth of field; color temperature; "
    "post-processing (warm Lightroom grade, clarity, vignette). " + _PROSE
)

SYSTEM_STREET = (
    "You are a street photographer writing a generation prompt. "
    "Expand into a detailed prompt covering: time of day and weather; urban details (signage, puddles, crowds); "
    "practical light sources (neon, streetlamps, headlights) and their color temperatures; "
    "camera and lens (Leica Q2 28mm, Fujifilm X-T5 35mm f/2, or similar); "
    "exposure (high ISO grain, motion blur vs frozen), color grade direction. " + _PROSE
)

SYSTEM_INTERIOR = (
    "You are an interior photographer writing a generation prompt. "
    "Expand into a detailed prompt covering: room features, furniture, surface materials (wood, fabric, stone); "
    "light sources and color temperature balance (window direction, practical lamps); "
    "camera and lens (Sony 16-35mm f/2.8 or 24mm tilt-shift); "
    "styling details (art, plants, candles); post-processing (Lightroom, shadow lift, clean highlights). " + _PROSE
)

ROUTING = [
    (["portrait", "woman", "man", "person", "face"], SYSTEM_PORTRAIT),
    (["food", "ramen", "dish", "meal", "plate", "burger", "sushi"], SYSTEM_FOOD),
    (["street", "tokyo", "city", "neon", "alley", "sidewalk"], SYSTEM_STREET),
    (["interior", "room", "living", "bedroom", "fireplace", "office"], SYSTEM_INTERIOR),
]

def get_routed_system(prompt):
    lower = prompt.lower()
    for keywords, system in ROUTING:
        if any(k in lower for k in keywords):
            return system
    return SYSTEM_GENERIC

STRATEGIES = [
    {"id": "generic", "label": "Generic expansion",       "get_system": lambda p: SYSTEM_GENERIC},
    {"id": "routed",  "label": "Subject-type routed",     "get_system": get_routed_system},
]

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def expand_prompt(base_prompt, system):
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json={
        "model": OLLAMA_MODEL,
        "prompt": base_prompt,
        "system": system,
        "stream": False,
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()

def build_comfy_prompt(base_prompt, expanded_prompt, seed, save_prefix):
    return {
        "4":  {"class_type": "DualCLIPLoaderGGUF",  "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp16.safetensors", "type": "flux"}},
        "5":  {"class_type": "CLIPTextEncodeFlux",   "inputs": {"clip": ["4", 0], "clip_l": base_prompt,    "t5xxl": expanded_prompt, "guidance": GUIDANCE}},
        "6":  {"class_type": "CLIPTextEncodeFlux",   "inputs": {"clip": ["4", 0], "clip_l": "",              "t5xxl": "",              "guidance": GUIDANCE}},
        "7":  {"class_type": "UnetLoaderGGUF",       "inputs": {"unet_name": FLUX_MODEL}},
        "8":  {"class_type": "EmptyLatentImage",     "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "9":  {"class_type": "KSampler",             "inputs": {
            "model": ["7", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["8", 0],
            "seed": seed, "steps": STEPS, "cfg": 1.0,
            "sampler_name": SAMPLER, "scheduler": SCHEDULER, "denoise": 1.0,
        }},
        "10": {"class_type": "VAELoader",            "inputs": {"vae_name": "ae.safetensors"}},
        "11": {"class_type": "VAEDecode",            "inputs": {"samples": ["9", 0], "vae": ["10", 0]}},
        "12": {"class_type": "SaveImage",            "inputs": {"images": ["11", 0], "filename_prefix": save_prefix}},
    }

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

def generate_gallery(results):
    strat_headers = "".join(
        f'<th><div class="vl">{s["id"]}</div>'
        f'<div class="vd">{s["label"]}<br>{STEPS}st · {GUIDANCE}g · {SAMPLER}</div></th>'
        for s in STRATEGIES
    )

    rows = ""
    for subject in SUBJECTS:
        sid   = subject["id"]
        cells = ""
        for strat in STRATEGIES:
            key = f"{sid}__{strat['id']}"
            r   = results.get(key)
            if r and r.get("image_file") and Path(r["image_file"]).exists():
                img_rel  = Path(r["image_file"]).name
                gen_time = f"{r['generation_time_seconds']:.0f}s" if r.get("generation_time_seconds") else "?"
                exp      = r.get("expanded_prompt", "")
                exp_tip  = (exp[:300] + "…") if len(exp) > 300 else exp
                exp_tip  = exp_tip.replace('"', "&quot;")
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

    total = len(SUBJECTS) * len(STRATEGIES)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Prompt Strategy Comparison — {datetime.now().strftime('%Y-%m-%d')}</title>
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
<h1>Prompt Strategy Comparison</h1>
<p class="meta">
  {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
  {len([r for r in results.values() if r.get('image_file')])} / {total} images &nbsp;·&nbsp;
  model: {OLLAMA_MODEL} &nbsp;·&nbsp; hover image for expanded prompt
</p>
<table>
  <thead><tr><th>Subject</th>{strat_headers}</tr></thead>
  <tbody>{rows}</tbody>
</table>
</body>
</html>"""

# ─── MAIN ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Compare prompt expansion strategies")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    total = len(SUBJECTS) * len(STRATEGIES)
    done  = 0
    results = {}

    print(f"Prompt Strategy Comparison — {total} images ({len(SUBJECTS)} subjects × {len(STRATEGIES)} strategies)")
    print(f"Model:  {OLLAMA_MODEL}")
    print(f"Output: {outdir}\n")

    print("Expanding prompts via Ollama...")
    expanded = {}
    for subject in SUBJECTS:
        sid = subject["id"]
        expanded[sid] = {}
        for strat in STRATEGIES:
            system = strat["get_system"](subject["prompt"])
            try:
                expanded[sid][strat["id"]] = expand_prompt(subject["prompt"], system)
                print(f"  ✓ {sid} / {strat['id']}")
            except Exception as e:
                expanded[sid][strat["id"]] = subject["prompt"]
                print(f"  ✗ {sid} / {strat['id']}: Ollama failed ({e}), using base prompt")
    print()

    for subject in SUBJECTS:
        sid         = subject["id"]
        seed        = subject["seed"]
        base_prompt = subject["prompt"]

        for strat in STRATEGIES:
            key    = f"{sid}__{strat['id']}"
            done  += 1
            prefix = f"pc__{sid}__{strat['id']}"
            exp_prompt = expanded[sid][strat["id"]]

            print(f"[{done}/{total}] {sid} / {strat['id']}  (seed {seed})")

            try:
                t0    = time.time()
                nodes = build_comfy_prompt(base_prompt, exp_prompt, seed, prefix)
                pid   = queue_prompt(nodes)
                entry = wait_for_completion(pid)
                gen   = time.time() - t0

                src = get_output_image_path(entry)
                dst = outdir / src.name
                shutil.copy2(src, dst)

                results[key] = {
                    "subject":                 sid,
                    "preset":                  strat["id"],
                    "strategy":                strat["label"],
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
                print(f"    prompt: {exp_prompt[:120]}…")

            except Exception as e:
                results[key] = {"subject": sid, "preset": strat["id"], "error": str(e)}
                print(f"  ✗ ERROR: {e}")

    ok_runs = [r for r in results.values() if r.get("image_file")]
    (outdir / "manifest.json").write_text(json.dumps({
        "runs": ok_runs,
        "generated_at": datetime.now().isoformat(),
    }, indent=2))

    (outdir / "index.html").write_text(generate_gallery(results))

    ok  = len(ok_runs)
    err = len(results) - ok
    print(f"\n{'─'*50}")
    print(f"Done: {ok} images  |  {err} errors")
    print(f"Compare: ./run-passenger.sh {outdir}")


if __name__ == "__main__":
    main()
