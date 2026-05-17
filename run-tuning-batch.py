#!/usr/bin/env python3
"""
FLUX quality tuning batch runner.
Generates images across subjects × presets with full metadata sidecars
and an HTML gallery for review.

Usage:
  python3 run-tuning-batch.py
  python3 run-tuning-batch.py --model juggernaut-flux-Q4_K.gguf
  python3 run-tuning-batch.py --lora flux-realism.safetensors --lora-strength 0.85
  python3 run-tuning-batch.py --width 832 --height 1216    # portrait aspect ratio
  python3 run-tuning-batch.py --model mymodel.gguf --lora mylora.safetensors --output-dir /path/to/out
"""

import argparse
import json
import os
import shutil
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────

COMFYUI        = "http://localhost:8188"
OLLAMA_HOST    = "http://10.100.20.18:11434"
OLLAMA_MODEL   = "llama3.1:8b"
OUTPUT_DIR     = Path("/home/alex/claude/comfyui/output/tuning2")
COMFYUI_OUTPUT = Path("/home/alex/claude/comfyui/output")

DEFAULT_FLUX_MODEL = "flux1-dev-Q4_K.gguf"

OLLAMA_SYSTEM = (
    "You are a prompt engineer for photorealistic AI image generation. "
    "Take the user's simple description and expand it into a rich, detailed visual prompt. "
    "Include: specific lighting setup (golden hour, studio strobes, Rembrandt lighting, soft window light), "
    "camera and lens metadata (e.g. 'shot on Sony A7R V, 85mm f/1.4, shallow depth of field'), "
    "atmosphere, surface textures, colors, and composition. "
    "Add post-processing style where natural (e.g. Lightroom color grade, subtle film grain, RAW finish). "
    "Reply ONLY with the enhanced prompt — no explanations, no preamble, no quotes."
)

# ─── SUBJECTS (10) ───────────────────────────────────────────────────────────

SUBJECTS = [
    {"id": "asian-woman",  "seed": 1002, "prompt": "portrait of a young Asian woman in a park"},
    {"id": "bird",         "seed": 1008, "prompt": "red-tailed hawk in mid-flight against a blue sky"},
    {"id": "cat",          "seed": 1001, "prompt": "a tabby cat sitting on a sunny windowsill"},
    {"id": "cityscape",    "seed": 1005, "prompt": "city skyline at dusk seen from across the water"},
    {"id": "dog",          "seed": 1010, "prompt": "a golden retriever running through autumn leaves"},
    {"id": "food",         "seed": 1006, "prompt": "bowl of tonkotsu ramen with chashu pork and soft boiled egg"},
    {"id": "interior",     "seed": 1007, "prompt": "cozy living room with fireplace and bookshelves"},
    {"id": "landscape",    "seed": 1004, "prompt": "mountain valley at golden hour with a river"},
    {"id": "night-street", "seed": 1009, "prompt": "rainy street in Tokyo at night with neon reflections"},
    {"id": "white-man",    "seed": 1003, "prompt": "portrait of a middle-aged white man in a coffee shop"},
]

# ─── PRESETS (11) → 110 images ───────────────────────────────────────────────

PRESETS = [
    {"id": "A",  "steps": 20, "guidance": 3.5, "sampler": "euler",            "scheduler": "beta",   "note": "baseline"},
    {"id": "B",  "steps": 25, "guidance": 3.5, "sampler": "euler",            "scheduler": "beta",   "note": "+5 steps"},
    {"id": "C",  "steps": 30, "guidance": 3.5, "sampler": "euler",            "scheduler": "beta",   "note": "+10 steps"},
    {"id": "D",  "steps": 35, "guidance": 3.5, "sampler": "euler",            "scheduler": "beta",   "note": "+15 steps"},
    {"id": "E",  "steps": 20, "guidance": 2.5, "sampler": "euler",            "scheduler": "beta",   "note": "low guidance"},
    {"id": "F",  "steps": 20, "guidance": 5.0, "sampler": "euler",            "scheduler": "beta",   "note": "mid guidance"},
    {"id": "G",  "steps": 20, "guidance": 7.0, "sampler": "euler",            "scheduler": "beta",   "note": "high guidance"},
    {"id": "H",  "steps": 30, "guidance": 5.0, "sampler": "euler",            "scheduler": "beta",   "note": "steps+guidance"},
    {"id": "I",  "steps": 25, "guidance": 3.5, "sampler": "euler",            "scheduler": "simple", "note": "simple scheduler"},
    {"id": "J",  "steps": 25, "guidance": 3.5, "sampler": "dpmpp_2m",         "scheduler": "beta",   "note": "dpmpp_2m"},
    {"id": "K",  "steps": 25, "guidance": 3.5, "sampler": "euler_ancestral",  "scheduler": "beta",   "note": "ancestral"},
]

TOTAL = len(SUBJECTS) * len(PRESETS)

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def make_filename(subject_id, preset, seed, model, lora):
    model_tag = Path(model).stem
    if model_tag == Path(DEFAULT_FLUX_MODEL).stem:
        model_tag = ""
    lora_tag = f"__{Path(lora).stem}" if lora else ""
    model_prefix = f"__{model_tag}" if model_tag else ""
    return (
        f"{subject_id}"
        f"__{preset['id']}"
        f"__{preset['steps']}steps"
        f"__{preset['guidance']}guid"
        f"__{preset['sampler']}"
        f"__{preset['scheduler']}"
        f"__seed{seed}"
        f"{model_prefix}"
        f"{lora_tag}"
    )

def expand_prompt(base_prompt):
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json={
        "model": OLLAMA_MODEL,
        "prompt": base_prompt,
        "system": OLLAMA_SYSTEM,
        "stream": False,
    }, timeout=60)
    resp.raise_for_status()
    return resp.json()["response"].strip()

def build_comfy_prompt(base_prompt, expanded_prompt, preset, seed, save_prefix,
                       model=DEFAULT_FLUX_MODEL, lora=None, lora_strength=0.8,
                       width=1024, height=1024):
    guidance = preset["guidance"]
    # clip and model source nodes — remapped if LoRA is inserted
    clip_src = ["4", 0]
    model_src = ["7", 0]

    nodes = {
        "4":  {"class_type": "DualCLIPLoaderGGUF",  "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux"}},
        "7":  {"class_type": "UnetLoaderGGUF",       "inputs": {"unet_name": model}},
        "8":  {"class_type": "EmptyLatentImage",     "inputs": {"width": width, "height": height, "batch_size": 1}},
        "10": {"class_type": "VAELoader",            "inputs": {"vae_name": "ae.safetensors"}},
        "11": {"class_type": "VAEDecode",            "inputs": {"samples": ["9", 0], "vae": ["10", 0]}},
        "12": {"class_type": "SaveImage",            "inputs": {"images": ["11", 0], "filename_prefix": save_prefix}},
    }

    if lora:
        nodes["13"] = {"class_type": "LoraLoader", "inputs": {
            "model": ["7", 0], "clip": ["4", 0],
            "lora_name": lora,
            "strength_model": lora_strength,
            "strength_clip": lora_strength,
        }}
        model_src = ["13", 0]
        clip_src  = ["13", 1]

    nodes["5"] = {"class_type": "CLIPTextEncodeFlux", "inputs": {"clip": clip_src, "clip_l": base_prompt,  "t5xxl": expanded_prompt, "guidance": guidance}}
    nodes["6"] = {"class_type": "CLIPTextEncodeFlux", "inputs": {"clip": clip_src, "clip_l": "",            "t5xxl": "",              "guidance": guidance}}
    nodes["9"] = {"class_type": "KSampler",           "inputs": {
        "model": model_src, "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["8", 0],
        "seed": seed, "steps": preset["steps"], "cfg": 1.0,
        "sampler_name": preset["sampler"], "scheduler": preset["scheduler"], "denoise": 1.0,
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
    outputs = history_entry.get("outputs", {})
    for node_outputs in outputs.values():
        if "images" in node_outputs:
            img = node_outputs["images"][0]
            subfolder = img.get("subfolder", "")
            filename = img["filename"]
            if subfolder:
                return COMFYUI_OUTPUT / subfolder / filename
            return COMFYUI_OUTPUT / filename
    raise RuntimeError("No image output found in history")

# ─── HTML GALLERY ─────────────────────────────────────────────────────────────

def generate_gallery(results):
    preset_ids  = [p["id"] for p in PRESETS]
    subject_ids = [s["id"] for s in SUBJECTS]

    preset_headers = "".join(
        f'<th><div class="pl">{p["id"]}</div>'
        f'<div class="pd">{p["steps"]}st · {p["guidance"]}g<br>'
        f'{p["sampler"]}<br>{p["scheduler"]}<br>'
        f'<em>{p["note"]}</em></div></th>'
        for p in PRESETS
    )

    rows = ""
    for sid in subject_ids:
        subject = next(s for s in SUBJECTS if s["id"] == sid)
        cells = ""
        for pid in preset_ids:
            key = f"{sid}__{pid}"
            r = results.get(key)
            if r and r.get("image_file") and Path(r["image_file"]).exists():
                img_rel = Path(r["image_file"]).name
                exp = r.get("expanded_prompt", "")
                exp_short = (exp[:300] + "…") if len(exp) > 300 else exp
                gen_time = f"{r['generation_time_seconds']:.0f}s" if r.get("generation_time_seconds") else "?"
                cells += (
                    f'<td><div class="cell">'
                    f'<a href="{img_rel}" target="_blank">'
                    f'<img src="{img_rel}" loading="lazy" title="{exp_short}"></a>'
                    f'<div class="cm">{gen_time}</div>'
                    f'</div></td>'
                )
            else:
                err = r.get("error", "not run") if r else "not run"
                cells += f'<td><div class="cell miss">{err[:40]}</div></td>'
        rows += (
            f'<tr><th class="sl">{sid}<br>'
            f'<small>{subject["prompt"][:45]}…</small></th>'
            f'{cells}</tr>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FLUX Tuning — {datetime.now().strftime('%Y-%m-%d')}</title>
<style>
  body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:16px}}
  h1{{font-size:1.1rem;margin-bottom:2px}}
  p.meta{{color:#777;font-size:0.75rem;margin-bottom:12px}}
  table{{border-collapse:collapse;width:100%}}
  th,td{{border:1px solid #2a2a2a;padding:3px;vertical-align:top}}
  th.sl{{text-align:left;font-size:0.7rem;width:110px;background:#1a1a1a;white-space:nowrap}}
  th.sl small{{color:#777;font-weight:normal}}
  th{{background:#1a1a1a;font-size:0.7rem;text-align:center}}
  .pl{{font-size:1rem;font-weight:bold}}
  .pd{{font-size:0.6rem;color:#999;line-height:1.3;margin-top:2px}}
  .cell img{{width:100%;display:block;border-radius:2px}}
  .cell img:hover{{opacity:0.8;cursor:pointer}}
  .cm{{font-size:0.6rem;color:#777;text-align:center;padding-top:2px}}
  .miss{{background:#1e1010;color:#664;text-align:center;padding:20px 4px;font-size:0.65rem;min-height:60px}}
</style>
</head>
<body>
<h1>FLUX Tuning Batch</h1>
<p class="meta">
  {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
  {len([r for r in results.values() if r.get('image_file')])} images &nbsp;·&nbsp;
  {len(SUBJECTS)} subjects × {len(PRESETS)} presets &nbsp;·&nbsp;
  1024×1024 &nbsp;·&nbsp; fixed seed per subject
</p>
<table>
  <thead><tr><th>Subject / Prompt</th>{preset_headers}</tr></thead>
  <tbody>{rows}</tbody>
</table>
<p style="margin-top:12px;font-size:0.65rem;color:#444">
  Hover image for expanded prompt · Click to open full size · .json sidecar per image has full metadata
</p>
</body>
</html>"""

# ─── MAIN ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FLUX quality tuning batch runner")
    p.add_argument("--model",         default=DEFAULT_FLUX_MODEL,
                   help=f"FLUX GGUF model filename in ComfyUI model paths (default: {DEFAULT_FLUX_MODEL})")
    p.add_argument("--lora",          default=None,
                   help="LoRA filename to apply (safetensors, in ComfyUI loras/ dir)")
    p.add_argument("--lora-strength", type=float, default=0.8,
                   help="LoRA strength applied to model and CLIP (default: 0.8)")
    p.add_argument("--width",         type=int, default=1024, help="Image width (default: 1024)")
    p.add_argument("--height",        type=int, default=1024, help="Image height (default: 1024)")
    p.add_argument("--output-dir",    default=None,
                   help="Override output directory (default: tuning2/)")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    results    = {}
    job_num    = 0
    start_time = time.time()

    model_label = args.model
    lora_label  = f" + LoRA {args.lora} @ {args.lora_strength}" if args.lora else ""
    res_label   = f"{args.width}×{args.height}"

    print(f"FLUX Tuning Batch — {TOTAL} images ({len(SUBJECTS)} subjects × {len(PRESETS)} presets)")
    print(f"Model:   {model_label}{lora_label}")
    print(f"Output:  {out_dir}  |  Resolution: {res_label}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 65)

    for subject in SUBJECTS:
        sid         = subject["id"]
        seed        = subject["seed"]
        base_prompt = subject["prompt"]

        print(f"\n[{sid}] Expanding prompt via Ollama...")
        try:
            expanded = expand_prompt(base_prompt)
            print(f"  → {expanded[:120]}...")
        except Exception as e:
            print(f"  ! Ollama failed ({e}) — using base prompt")
            expanded = base_prompt

        for preset in PRESETS:
            job_num += 1
            pid     = preset["id"]
            fname   = make_filename(sid, preset, seed, args.model, args.lora)
            key     = f"{sid}__{pid}"

            # ETA
            elapsed  = time.time() - start_time
            avg_secs = elapsed / (job_num - 1) if job_num > 1 else 40
            remaining = avg_secs * (TOTAL - job_num + 1)
            eta      = datetime.now() + timedelta(seconds=remaining)

            print(
                f"\n[{job_num:3d}/{TOTAL}] {sid} {pid}  "
                f"{preset['steps']}steps {preset['guidance']}g "
                f"{preset['sampler']}/{preset['scheduler']}  "
                f"ETA {eta.strftime('%H:%M')}"
            )

            subfolder = out_dir.name
            save_prefix = f"{subfolder}/{fname}"
            comfy_prompt = build_comfy_prompt(
                base_prompt, expanded, preset, seed, save_prefix,
                model=args.model, lora=args.lora,
                lora_strength=args.lora_strength,
                width=args.width, height=args.height,
            )

            try:
                t0        = time.time()
                prompt_id = queue_prompt(comfy_prompt)
                print(f"  queued {prompt_id[:8]}...", end=" ", flush=True)

                history  = wait_for_completion(prompt_id)
                gen_time = time.time() - t0
                print(f"done in {gen_time:.0f}s")

                # Locate saved file and strip ComfyUI's _00001_ counter suffix
                src = get_output_image_path(history)
                dst = out_dir / f"{fname}.png"
                shutil.move(str(src), str(dst))

                meta = {
                    "subject":                  sid,
                    "preset":                   pid,
                    "preset_note":              preset["note"],
                    "steps":                    preset["steps"],
                    "guidance":                 preset["guidance"],
                    "sampler":                  preset["sampler"],
                    "scheduler":                preset["scheduler"],
                    "seed":                     seed,
                    "model":                    args.model,
                    "lora":                     args.lora,
                    "lora_strength":            args.lora_strength if args.lora else None,
                    "width":                    args.width,
                    "height":                   args.height,
                    "base_prompt":              base_prompt,
                    "expanded_prompt":          expanded,
                    "generation_time_seconds":  round(gen_time, 1),
                    "prompt_id":                prompt_id,
                    "image_file":               str(dst),
                    "generated_at":             datetime.now().isoformat(),
                }
                (out_dir / f"{fname}.json").write_text(json.dumps(meta, indent=2))
                results[key] = meta

            except Exception as e:
                print(f"  FAILED: {e}")
                results[key] = {"subject": sid, "preset": pid, "error": str(e)}

    # ── Gallery + manifest ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Generating gallery...")
    (out_dir / "index.html").write_text(generate_gallery(results))
    (out_dir / "manifest.json").write_text(
        json.dumps({"runs": list(results.values()), "generated_at": datetime.now().isoformat()}, indent=2)
    )

    total_time = time.time() - start_time
    succeeded  = sum(1 for r in results.values() if "image_file" in r)
    print(f"Done! {succeeded}/{TOTAL} images in {total_time/60:.1f} min")
    print(f"Gallery: {out_dir}/index.html")

if __name__ == "__main__":
    main()
