#!/usr/bin/env python3
"""
SDXL/Illustrious and FLUX-safetensors batch runner.
Generates images across subjects × presets with full metadata sidecars
and an HTML gallery for review.

Pipeline auto-selected by --pipeline flag:
  sdxl  — CheckpointLoaderSimple → CLIPTextEncode → KSampler  (Illustrious, Pony, etc.)
  flux  — UNETLoader + DualCLIPLoaderGGUF → CLIPTextEncodeFlux → KSampler  (Fluxed Up, etc.)

Usage:
  python3 run-sdxl-batch.py
  python3 run-sdxl-batch.py --model novaAnimeXL_ilV190.safetensors
  python3 run-sdxl-batch.py --model unholyDesireMixSinister_v80.safetensors
  python3 run-sdxl-batch.py --model fluxedUpFluxNSFW_90FP8.safetensors --pipeline flux
  python3 run-sdxl-batch.py --lora mylora.safetensors --lora-strength 0.85
  python3 run-sdxl-batch.py --width 832 --height 1216    # portrait
  python3 run-sdxl-batch.py --output-dir /path/to/out
"""

import argparse
import json
import shutil
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────

COMFYUI        = "http://localhost:8188"
OLLAMA_HOST    = "http://10.100.20.18:11434"
OLLAMA_MODEL   = "llama3.1:8b"
OUTPUT_DIR     = Path("/home/alex/claude/comfyui/output/sdxl-batch")
COMFYUI_OUTPUT = Path("/home/alex/claude/comfyui/output")

DEFAULT_SDXL_MODEL = "waiIllustriousSDXL_v170.safetensors"
DEFAULT_FLUX_MODEL = "fluxedUpFluxNSFW_90FP8.safetensors"

DEFAULT_NEGATIVE = (
    "worst quality, low quality, bad quality, lowres, blurry, jpeg artifacts, "
    "ugly, deformed, bad anatomy, watermark, signature, extra limbs"
)

OLLAMA_SYSTEM_SDXL = (
    "You are a prompt engineer for anime and artistic AI image generation using Illustrious XL. "
    "Take the user's simple description and expand it into a rich, detailed prompt. "
    "Include: character appearance, art style, lighting, composition, atmosphere, and visual mood. "
    "Add quality booster tags at the end: masterpiece, best quality, very aesthetic, absurdres. "
    "Reply ONLY with the enhanced prompt — no explanations, no preamble, no quotes."
)

OLLAMA_SYSTEM_FLUX = (
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

# ─── PRESETS ─────────────────────────────────────────────────────────────────

PRESETS_SDXL = [
    {"id": "A", "steps": 20, "cfg": 7.0, "sampler": "euler_ancestral", "scheduler": "karras",  "note": "baseline"},
    {"id": "B", "steps": 25, "cfg": 7.0, "sampler": "euler_ancestral", "scheduler": "karras",  "note": "+5 steps"},
    {"id": "C", "steps": 30, "cfg": 7.0, "sampler": "euler_ancestral", "scheduler": "karras",  "note": "+10 steps"},
    {"id": "D", "steps": 20, "cfg": 5.0, "sampler": "euler_ancestral", "scheduler": "karras",  "note": "low cfg"},
    {"id": "E", "steps": 20, "cfg": 9.0, "sampler": "euler_ancestral", "scheduler": "karras",  "note": "high cfg"},
    {"id": "F", "steps": 25, "cfg": 7.0, "sampler": "dpmpp_2m",        "scheduler": "karras",  "note": "dpm++2m"},
    {"id": "G", "steps": 25, "cfg": 7.0, "sampler": "dpmpp_sde",       "scheduler": "karras",  "note": "dpm++sde"},
    {"id": "H", "steps": 30, "cfg": 7.0, "sampler": "dpmpp_2m",        "scheduler": "karras",  "note": "dpm++2m 30st"},
]

PRESETS_FLUX = [
    {"id": "A", "steps": 20, "cfg": 1.0, "guidance": 3.5, "sampler": "euler",   "scheduler": "beta",   "note": "baseline"},
    {"id": "B", "steps": 25, "cfg": 1.0, "guidance": 3.5, "sampler": "euler",   "scheduler": "beta",   "note": "+5 steps"},
    {"id": "C", "steps": 30, "cfg": 1.0, "guidance": 3.5, "sampler": "euler",   "scheduler": "beta",   "note": "+10 steps"},
    {"id": "D", "steps": 20, "cfg": 1.0, "guidance": 5.0, "sampler": "euler",   "scheduler": "beta",   "note": "mid guidance"},
    {"id": "E", "steps": 20, "cfg": 1.0, "guidance": 7.0, "sampler": "euler",   "scheduler": "beta",   "note": "high guidance"},
]

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def make_filename(subject_id, preset, seed, model, lora):
    model_stem = Path(model).stem
    lora_tag   = f"__{Path(lora).stem}" if lora else ""
    cfg_val    = preset.get("cfg", 1.0)
    guidance   = preset.get("guidance", "")
    guid_tag   = f"__{guidance}guid" if guidance else f"__{cfg_val}cfg"
    return (
        f"{subject_id}"
        f"__{preset['id']}"
        f"__{preset['steps']}steps"
        f"{guid_tag}"
        f"__{preset['sampler']}"
        f"__{preset['scheduler']}"
        f"__seed{seed}"
        f"__{model_stem}"
        f"{lora_tag}"
    )

def expand_prompt(base_prompt, system_prompt):
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json={
        "model": OLLAMA_MODEL,
        "prompt": base_prompt,
        "system": system_prompt,
        "stream": False,
    }, timeout=60)
    resp.raise_for_status()
    return resp.json()["response"].strip()

def build_comfy_prompt_sdxl(positive, negative, preset, seed, save_prefix,
                             model=DEFAULT_SDXL_MODEL, lora=None, lora_strength=0.8,
                             width=1024, height=1024):
    clip_src  = ["4", 1]
    model_src = ["4", 0]
    vae_src   = ["4", 2]

    nodes = {
        "4":  {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}},
        "7":  {"class_type": "EmptyLatentImage",        "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8":  {"class_type": "KSampler",               "inputs": {
                   "model": model_src, "positive": ["5", 0], "negative": ["6", 0],
                   "latent_image": ["7", 0], "seed": seed, "steps": preset["steps"],
                   "cfg": preset["cfg"], "sampler_name": preset["sampler"],
                   "scheduler": preset["scheduler"], "denoise": 1.0}},
        "9":  {"class_type": "VAEDecode",              "inputs": {"samples": ["8", 0], "vae": vae_src}},
        "10": {"class_type": "SaveImage",              "inputs": {"images": ["9", 0], "filename_prefix": save_prefix}},
    }

    if lora:
        nodes["11"] = {"class_type": "LoraLoader", "inputs": {
            "model": ["4", 0], "clip": ["4", 1],
            "lora_name": lora, "strength_model": lora_strength, "strength_clip": lora_strength,
        }}
        model_src = ["11", 0]
        clip_src  = ["11", 1]
        nodes["8"]["inputs"]["model"] = model_src

    nodes["5"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_src, "text": positive}}
    nodes["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_src, "text": negative}}
    return nodes

def build_comfy_prompt_flux(positive, preset, seed, save_prefix,
                            model=DEFAULT_FLUX_MODEL, lora=None, lora_strength=0.8,
                            width=1024, height=1024):
    guidance  = preset.get("guidance", 3.5)
    clip_src  = ["4", 0]
    model_src = ["7", 0]

    nodes = {
        "4":  {"class_type": "DualCLIPLoaderGGUF", "inputs": {
                   "clip_name1": "clip_l.safetensors",
                   "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux"}},
        "7":  {"class_type": "UNETLoader",         "inputs": {"unet_name": model, "weight_dtype": "fp8_e4m3fn"}},
        "8":  {"class_type": "EmptyLatentImage",   "inputs": {"width": width, "height": height, "batch_size": 1}},
        "10": {"class_type": "VAELoader",          "inputs": {"vae_name": "ae.safetensors"}},
        "11": {"class_type": "VAEDecode",          "inputs": {"samples": ["9", 0], "vae": ["10", 0]}},
        "12": {"class_type": "SaveImage",          "inputs": {"images": ["11", 0], "filename_prefix": save_prefix}},
    }

    if lora:
        nodes["13"] = {"class_type": "LoraLoader", "inputs": {
            "model": ["7", 0], "clip": ["4", 0],
            "lora_name": lora, "strength_model": lora_strength, "strength_clip": lora_strength,
        }}
        model_src = ["13", 0]
        clip_src  = ["13", 1]

    nodes["5"] = {"class_type": "CLIPTextEncodeFlux", "inputs": {
        "clip": clip_src, "clip_l": positive, "t5xxl": positive, "guidance": guidance}}
    nodes["6"] = {"class_type": "CLIPTextEncodeFlux", "inputs": {
        "clip": clip_src, "clip_l": "", "t5xxl": "", "guidance": guidance}}
    nodes["9"] = {"class_type": "KSampler", "inputs": {
        "model": model_src, "positive": ["5", 0], "negative": ["6", 0],
        "latent_image": ["8", 0], "seed": seed, "steps": preset["steps"],
        "cfg": preset["cfg"], "sampler_name": preset["sampler"],
        "scheduler": preset["scheduler"], "denoise": 1.0}}
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
            subfolder = img.get("subfolder", "")
            filename  = img["filename"]
            return (COMFYUI_OUTPUT / subfolder / filename) if subfolder else (COMFYUI_OUTPUT / filename)
    raise RuntimeError("No image output found in history")

# ─── HTML GALLERY ─────────────────────────────────────────────────────────────

def generate_gallery(results, presets):
    preset_ids  = [p["id"] for p in presets]
    subject_ids = [s["id"] for s in SUBJECTS]

    preset_headers = "".join(
        f'<th><div class="pl">{p["id"]}</div>'
        f'<div class="pd">{p["steps"]}st · {p.get("guidance", p.get("cfg","?"))}<br>'
        f'{p["sampler"]}<br>{p["scheduler"]}<br>'
        f'<em>{p["note"]}</em></div></th>'
        for p in presets
    )

    rows = ""
    for sid in subject_ids:
        subject = next(s for s in SUBJECTS if s["id"] == sid)
        cells = ""
        for pid in preset_ids:
            key = f"{sid}__{pid}"
            r = results.get(key)
            if r and r.get("image_file") and Path(r["image_file"]).exists():
                img_rel   = Path(r["image_file"]).name
                exp       = r.get("expanded_prompt", "")
                exp_short = (exp[:300] + "…") if len(exp) > 300 else exp
                gen_time  = f"{r['generation_time_seconds']:.0f}s" if r.get("generation_time_seconds") else "?"
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

    total = len(SUBJECTS) * len(presets)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SDXL Batch — {datetime.now().strftime('%Y-%m-%d')}</title>
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
<h1>SDXL / Illustrious Batch</h1>
<p class="meta">
  {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
  {len([r for r in results.values() if r.get('image_file')])} images &nbsp;·&nbsp;
  {len(SUBJECTS)} subjects × {len(presets)} presets &nbsp;·&nbsp;
  {total} total
</p>
<table>
  <thead><tr><th>Subject / Prompt</th>{preset_headers}</tr></thead>
  <tbody>{rows}</tbody>
</table>
<p style="margin-top:12px;font-size:0.65rem;color:#444">
  Hover image for expanded prompt · Click to open full size · .json sidecar per image
</p>
</body>
</html>"""

# ─── ARG PARSING ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SDXL/Illustrious + FLUX safetensors batch runner")
    p.add_argument("--pipeline",      default="sdxl", choices=["sdxl", "flux"],
                   help="sdxl = Illustrious/Pony (default), flux = FLUX safetensors (Fluxed Up etc)")
    p.add_argument("--model",         default=None,
                   help="Checkpoint filename in ComfyUI model paths (default: WAI Illustrious for sdxl, Fluxed Up for flux)")
    p.add_argument("--lora",          default=None,
                   help="LoRA filename (safetensors, in ComfyUI loras/ dir)")
    p.add_argument("--lora-strength", type=float, default=0.8,
                   help="LoRA strength (default: 0.8)")
    p.add_argument("--negative",      default=DEFAULT_NEGATIVE,
                   help="Negative prompt (sdxl pipeline only)")
    p.add_argument("--width",         type=int, default=1024)
    p.add_argument("--height",        type=int, default=1024)
    p.add_argument("--output-dir",    default=None,
                   help=f"Override output dir (default: {OUTPUT_DIR})")
    return p.parse_args()

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    pipeline = args.pipeline
    model    = args.model or (DEFAULT_FLUX_MODEL if pipeline == "flux" else DEFAULT_SDXL_MODEL)
    presets  = PRESETS_FLUX if pipeline == "flux" else PRESETS_SDXL
    system   = OLLAMA_SYSTEM_FLUX if pipeline == "flux" else OLLAMA_SYSTEM_SDXL
    out_dir  = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    total      = len(SUBJECTS) * len(presets)
    results    = {}
    job_num    = 0
    start_time = time.time()

    lora_label = f" + {args.lora} @ {args.lora_strength}" if args.lora else ""
    print(f"SDXL Batch [{pipeline.upper()}] — {total} images ({len(SUBJECTS)} subjects × {len(presets)} presets)")
    print(f"Model:   {model}{lora_label}")
    print(f"Output:  {out_dir}  |  {args.width}×{args.height}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 65)

    for subject in SUBJECTS:
        sid         = subject["id"]
        seed        = subject["seed"]
        base_prompt = subject["prompt"]

        print(f"\n[{sid}] Expanding prompt via Ollama...")
        try:
            expanded = expand_prompt(base_prompt, system)
            print(f"  → {expanded[:120]}...")
        except Exception as e:
            print(f"  ! Ollama failed ({e}) — using base prompt")
            expanded = base_prompt

        for preset in presets:
            job_num += 1
            pid     = preset["id"]
            fname   = make_filename(sid, preset, seed, model, args.lora)
            key     = f"{sid}__{pid}"

            elapsed   = time.time() - start_time
            avg_secs  = elapsed / (job_num - 1) if job_num > 1 else 40
            eta       = datetime.now() + timedelta(seconds=avg_secs * (total - job_num + 1))

            cfg_disp = preset.get("guidance", preset.get("cfg", "?"))
            print(
                f"\n[{job_num:3d}/{total}] {sid} {pid}  "
                f"{preset['steps']}steps {cfg_disp} "
                f"{preset['sampler']}/{preset['scheduler']}  "
                f"ETA {eta.strftime('%H:%M')}"
            )

            subfolder    = out_dir.name
            save_prefix  = f"{subfolder}/{fname}"

            if pipeline == "flux":
                comfy_prompt = build_comfy_prompt_flux(
                    expanded, preset, seed, save_prefix,
                    model=model, lora=args.lora, lora_strength=args.lora_strength,
                    width=args.width, height=args.height,
                )
            else:
                comfy_prompt = build_comfy_prompt_sdxl(
                    expanded, args.negative, preset, seed, save_prefix,
                    model=model, lora=args.lora, lora_strength=args.lora_strength,
                    width=args.width, height=args.height,
                )

            try:
                t0        = time.time()
                prompt_id = queue_prompt(comfy_prompt)
                print(f"  queued {prompt_id[:8]}...", end=" ", flush=True)

                history  = wait_for_completion(prompt_id)
                gen_time = time.time() - t0
                print(f"done in {gen_time:.0f}s")

                src = get_output_image_path(history)
                dst = out_dir / f"{fname}.png"
                shutil.move(str(src), str(dst))

                meta = {
                    "subject":                 sid,
                    "preset":                  pid,
                    "preset_note":             preset["note"],
                    "pipeline":                pipeline,
                    "steps":                   preset["steps"],
                    "cfg":                     preset.get("cfg"),
                    "guidance":                preset.get("guidance"),
                    "sampler":                 preset["sampler"],
                    "scheduler":               preset["scheduler"],
                    "seed":                    seed,
                    "model":                   model,
                    "lora":                    args.lora,
                    "lora_strength":           args.lora_strength if args.lora else None,
                    "width":                   args.width,
                    "height":                  args.height,
                    "base_prompt":             base_prompt,
                    "expanded_prompt":         expanded,
                    "negative_prompt":         args.negative if pipeline == "sdxl" else None,
                    "generation_time_seconds": round(gen_time, 1),
                    "prompt_id":               prompt_id,
                    "image_file":              str(dst),
                    "generated_at":            datetime.now().isoformat(),
                }
                (out_dir / f"{fname}.json").write_text(json.dumps(meta, indent=2))
                results[key] = meta

            except Exception as e:
                print(f"  FAILED: {e}")
                results[key] = {"subject": sid, "preset": pid, "error": str(e)}

    print("\n" + "=" * 65)
    print("Generating gallery...")
    (out_dir / "index.html").write_text(generate_gallery(results, presets))
    (out_dir / "manifest.json").write_text(
        json.dumps({"runs": list(results.values()), "generated_at": datetime.now().isoformat()}, indent=2)
    )

    total_time = time.time() - start_time
    succeeded  = sum(1 for r in results.values() if "image_file" in r)
    print(f"Done! {succeeded}/{total} images in {total_time/60:.1f} min")
    print(f"Gallery: {out_dir}/index.html")

if __name__ == "__main__":
    main()
