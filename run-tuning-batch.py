#!/usr/bin/env python3
"""
FLUX quality tuning batch runner.
Generates 54 images (9 subjects × 6 presets) with full metadata sidecars
and an HTML gallery for review.
"""

import json
import os
import shutil
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────

COMFYUI       = "http://localhost:8188"
OLLAMA_HOST   = "http://10.100.20.18:11434"
OLLAMA_MODEL  = "llama3.1:8b"
OUTPUT_DIR    = Path("/home/alex/claude/comfyui/output/tuning")
COMFYUI_OUTPUT= Path("/home/alex/claude/comfyui/output")

OLLAMA_SYSTEM = (
    "You are a prompt engineer for AI image generation. Take the user's simple "
    "description and expand it into a rich, detailed visual prompt for a "
    "photorealistic image generator. Add specific details about lighting, "
    "atmosphere, textures, colors, depth of field, and composition. "
    "Reply ONLY with the enhanced prompt — no explanations, no preamble, no quotes."
)

# ─── SUBJECTS ────────────────────────────────────────────────────────────────

SUBJECTS = [
    {"id": "asian-woman",  "seed": 1002, "prompt": "portrait of a young Asian woman in a park"},
    {"id": "bird",         "seed": 1008, "prompt": "red-tailed hawk in mid-flight against a blue sky"},
    {"id": "cat",          "seed": 1001, "prompt": "a tabby cat sitting on a sunny windowsill"},
    {"id": "cityscape",    "seed": 1005, "prompt": "city skyline at dusk seen from across the water"},
    {"id": "food",         "seed": 1006, "prompt": "bowl of tonkotsu ramen with chashu pork and soft boiled egg"},
    {"id": "interior",     "seed": 1007, "prompt": "cozy living room with fireplace and bookshelves"},
    {"id": "landscape",    "seed": 1004, "prompt": "mountain valley at golden hour with a river"},
    {"id": "night-street", "seed": 1009, "prompt": "rainy street in Tokyo at night with neon reflections"},
    {"id": "white-man",    "seed": 1003, "prompt": "portrait of a middle-aged white man in a coffee shop"},
]

# ─── PRESETS ─────────────────────────────────────────────────────────────────

PRESETS = [
    {"id": "A", "steps": 20, "guidance": 3.5, "sampler": "euler",       "scheduler": "beta",   "note": "baseline"},
    {"id": "B", "steps": 30, "guidance": 3.5, "sampler": "euler",       "scheduler": "beta",   "note": "more steps"},
    {"id": "C", "steps": 20, "guidance": 5.0, "sampler": "euler",       "scheduler": "beta",   "note": "higher guidance"},
    {"id": "D", "steps": 35, "guidance": 4.5, "sampler": "euler",       "scheduler": "beta",   "note": "high quality"},
    {"id": "E", "steps": 25, "guidance": 3.5, "sampler": "euler",       "scheduler": "simple", "note": "simple scheduler"},
    {"id": "F", "steps": 25, "guidance": 3.5, "sampler": "dpmpp_2m",    "scheduler": "beta",   "note": "dpmpp_2m sampler"},
]

TOTAL = len(SUBJECTS) * len(PRESETS)

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def make_filename(subject_id, preset):
    return (
        f"{subject_id}"
        f"__{preset['id']}"
        f"__{preset['steps']}steps"
        f"__{preset['guidance']}guid"
        f"__{preset['sampler']}"
        f"__{preset['scheduler']}"
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

def build_comfy_prompt(base_prompt, expanded_prompt, preset, seed, tmp_prefix):
    guidance = preset["guidance"]
    return {
        "4":  {"class_type": "DualCLIPLoaderGGUF",  "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux"}},
        "5":  {"class_type": "CLIPTextEncodeFlux",   "inputs": {"clip": ["4", 0], "clip_l": base_prompt,  "t5xxl": expanded_prompt, "guidance": guidance}},
        "6":  {"class_type": "CLIPTextEncodeFlux",   "inputs": {"clip": ["4", 0], "clip_l": "",            "t5xxl": "",              "guidance": guidance}},
        "7":  {"class_type": "UnetLoaderGGUF",       "inputs": {"unet_name": "flux1-dev-Q4_K.gguf"}},
        "8":  {"class_type": "EmptyLatentImage",     "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "9":  {"class_type": "KSampler",             "inputs": {
                    "model": ["7", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["8", 0],
                    "seed": seed, "steps": preset["steps"], "cfg": 1.0,
                    "sampler_name": preset["sampler"], "scheduler": preset["scheduler"], "denoise": 1.0}},
        "10": {"class_type": "VAELoader",            "inputs": {"vae_name": "ae.safetensors"}},
        "11": {"class_type": "VAEDecode",            "inputs": {"samples": ["9", 0], "vae": ["10", 0]}},
        "12": {"class_type": "SaveImage",            "inputs": {"images": ["11", 0], "filename_prefix": f"tuning/{tmp_prefix}"}},
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
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI error: {msgs}")
        time.sleep(4)
    raise TimeoutError(f"Timed out after {timeout}s waiting for {prompt_id}")

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
    preset_ids = [p["id"] for p in PRESETS]
    subject_ids = [s["id"] for s in SUBJECTS]

    preset_headers = "".join(
        f'<th><div class="preset-label">{p["id"]}</div>'
        f'<div class="preset-detail">{p["steps"]}steps · {p["guidance"]}guid<br>'
        f'{p["sampler"]} · {p["scheduler"]}<br>'
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
                expanded_short = (r["expanded_prompt"][:200] + "…") if len(r.get("expanded_prompt","")) > 200 else r.get("expanded_prompt","")
                gen_time = f"{r['generation_time_seconds']:.0f}s" if r.get("generation_time_seconds") else "?"
                cells += (
                    f'<td><div class="cell">'
                    f'<a href="{img_rel}" target="_blank">'
                    f'<img src="{img_rel}" loading="lazy" title="{expanded_short}"></a>'
                    f'<div class="cell-meta">{gen_time}</div>'
                    f'</div></td>'
                )
            else:
                cells += '<td><div class="cell missing">missing</div></td>'
        rows += f'<tr><th class="subject-label">{sid}<br><small>{subject["prompt"][:40]}…</small></th>{cells}</tr>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FLUX Tuning Batch — {datetime.now().strftime('%Y-%m-%d')}</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }}
  h1 {{ font-size: 1.2rem; margin-bottom: 4px; }}
  p.meta {{ color: #888; font-size: 0.8rem; margin-bottom: 16px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #333; padding: 4px; vertical-align: top; }}
  th.subject-label {{ text-align: left; font-size: 0.75rem; width: 120px; background: #1a1a1a; white-space: nowrap; }}
  th.subject-label small {{ color: #888; font-weight: normal; }}
  th {{ background: #1a1a1a; font-size: 0.75rem; text-align: center; }}
  .preset-label {{ font-size: 1.1rem; font-weight: bold; }}
  .preset-detail {{ font-size: 0.65rem; color: #aaa; line-height: 1.4; }}
  .cell img {{ width: 100%; display: block; border-radius: 2px; }}
  .cell img:hover {{ opacity: 0.85; cursor: pointer; }}
  .cell-meta {{ font-size: 0.65rem; color: #888; text-align: center; padding-top: 2px; }}
  .missing {{ background: #2a1a1a; color: #666; text-align: center; padding: 20px; font-size: 0.75rem; }}
</style>
</head>
<body>
<h1>FLUX Tuning Batch</h1>
<p class="meta">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · {len(results)} images · 9 subjects × 6 presets · 1024×1024 · seed fixed per subject</p>
<table>
  <thead><tr><th>Subject</th>{preset_headers}</tr></thead>
  <tbody>{rows}</tbody>
</table>
<p style="margin-top:16px;font-size:0.75rem;color:#555">
  Hover image for expanded prompt · Click to open full size · Sidecar .json files contain full metadata
</p>
</body>
</html>"""

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    job_num = 0
    start_time = time.time()

    print(f"FLUX Tuning Batch — {TOTAL} images")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    for subject in SUBJECTS:
        sid = subject["id"]
        seed = subject["seed"]
        base_prompt = subject["prompt"]

        # Expand prompt once per subject (same expansion used across all presets)
        print(f"\n[{sid}] Expanding prompt via Ollama...")
        try:
            expanded = expand_prompt(base_prompt)
            print(f"  → {expanded[:100]}...")
        except Exception as e:
            print(f"  ✗ Ollama failed: {e} — using base prompt as fallback")
            expanded = base_prompt

        for preset in PRESETS:
            job_num += 1
            pid_str  = preset["id"]
            fname    = make_filename(sid, preset)
            tmp_pfx  = f"tmp_{job_num:03d}"
            key      = f"{sid}__{pid_str}"

            elapsed  = time.time() - start_time
            avg_secs = elapsed / max(job_num - 1, 1) if job_num > 1 else 90
            remaining = avg_secs * (TOTAL - job_num + 1)
            eta      = datetime.now() + timedelta(seconds=remaining)

            print(
                f"\n[{job_num:2d}/{TOTAL}] {sid} preset {pid_str} "
                f"({preset['steps']}steps, {preset['guidance']}guid, "
                f"{preset['sampler']}/{preset['scheduler']})  "
                f"ETA {eta.strftime('%H:%M')}"
            )

            comfy_prompt = build_comfy_prompt(base_prompt, expanded, preset, seed, tmp_pfx)

            try:
                t0 = time.time()
                prompt_id = queue_prompt(comfy_prompt)
                print(f"  queued {prompt_id[:8]}...", end=" ", flush=True)

                history = wait_for_completion(prompt_id)
                gen_time = time.time() - t0
                print(f"done in {gen_time:.0f}s")

                # Locate and rename output file
                src = get_output_image_path(history)
                dst = OUTPUT_DIR / f"{fname}.png"
                shutil.move(str(src), str(dst))

                # Sidecar JSON
                meta = {
                    "subject":                 sid,
                    "preset":                  pid_str,
                    "preset_note":             preset["note"],
                    "steps":                   preset["steps"],
                    "guidance":                preset["guidance"],
                    "sampler":                 preset["sampler"],
                    "scheduler":               preset["scheduler"],
                    "seed":                    seed,
                    "base_prompt":             base_prompt,
                    "expanded_prompt":         expanded,
                    "generation_time_seconds": round(gen_time, 1),
                    "prompt_id":               prompt_id,
                    "image_file":              str(dst),
                    "generated_at":            datetime.now().isoformat(),
                }
                with open(OUTPUT_DIR / f"{fname}.json", "w") as f:
                    json.dump(meta, f, indent=2)

                results[key] = meta

            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                results[key] = {"subject": sid, "preset": pid_str, "error": str(e)}

    # ── Gallery ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Generating gallery...")
    html = generate_gallery(results)
    gallery_path = OUTPUT_DIR / "index.html"
    gallery_path.write_text(html)

    # manifest.json
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump({"runs": list(results.values()), "generated_at": datetime.now().isoformat()}, f, indent=2)

    total_time = time.time() - start_time
    succeeded  = sum(1 for r in results.values() if "image_file" in r)
    print(f"Done! {succeeded}/{TOTAL} succeeded in {total_time/60:.1f} min")
    print(f"Gallery: {gallery_path}")


if __name__ == "__main__":
    main()
