#!/usr/bin/env python3
"""nhentai -> decensor pipeline

Usage:
  # Run on murderbot (ComfyUI must be running via start-flux.sh)
  ~/claude/comfyui/venv/bin/python nhentai-decensor.py \
      --url https://nhentai.net/g/76034/ \
      --style fill

  python nhentai-decensor.py --url <nhentai-url> [--fill|--cn]

Two decensoring styles:
  --fill   Flux Fill Dev GGUF (best quality, needs Q5 model)
  --cn     Base Flux Dev + ControlNet (uses existing models)

Pipeline stages:
  1. Fetch gallery metadata & page images from nhentai.net HTML (bypasses /api/gallery Cloudflare block)
  2. Auto-detect censor bars via SAM segmentation or use manual mask
  3. Run each masked image through ComfyUI inpainting workflow
  4. Save results with before/after in output directory

Dependencies: requests, Pillow (in comfyui venv), numpy (+ optional segment_anything)
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

# Safari user-agent — nhentai Cloudflare blocks Chrome/Linux but allows Safari/Mac
NHENTAI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.0 Safari/605.1.15"
    ),
}

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get(url: str, timeout: int = 30) -> tuple[int, str]:
    """GET with Safari UA. Returns (status_code, body)."""
    import requests
    resp = requests.get(url, headers=NHENTAI_HEADERS, timeout=timeout)
    return resp.status_code, resp.text


def http_download(url: str, path: Path, chunk_size: int = 65536) -> None:
    """Stream download with Safari UA."""
    import requests

    resp = requests.get(url, headers=NHENTAI_HEADERS, timeout=120, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} downloading {url}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size):
            fh.write(chunk)


# ---------------------------------------------------------------------------
# Stage 1 — fetch gallery & download images
# ---------------------------------------------------------------------------

def parse_gallery_from_html(html: str) -> dict | None:
    """Parse nhentai HTML page to extract gallery info and page image URLs.

    nhentai embeds a JSON blob inside <script type="application/json"> tags.
    The /api/gallery Cloudflare endpoint sometimes returns 403 from certain IPs,
    but the full page HTML always loads fine with Safari UA.

    Falls back to regex extraction when nested JSON escaping breaks simple unescape.
    """
    import json as _json

    # Find all application/json script blocks
    scripts = re.findall(
        r'<script[^>]*type="application/json"[^>]*>(.*?)</script>', html, re.DOTALL
    )

    gallery = None
    for s in scripts:
        try:
            data = _json.loads(s.strip())
            body_str = data.get("body", "")
            # Body is an escaped JSON string — unescape \\" -> " and \\/ -> /
            body_unescaped = body_str.replace('\\"', '"').replace('\\\\/', '/')
            inner = _json.loads(body_unescaped)

            if isinstance(inner, dict) and "media_id" in inner:
                gallery = {
                    "id": inner["id"],
                    "media_id": inner["media_id"],
                    "title": inner.get("title", {}),
                    "tags": inner.get("tags", []),
                    "num_pages": len(inner.get("images", {}).get("pages", [])),
                }

                # Extract pages from the images structure
                img_obj = _json.dumps(inner.get("images", {}))
                gallery["images_raw"] = img_obj
        except (_json.JSONDecodeError, KeyError):
            continue

    if gallery is None:
        # Fallback: parse directly from HTML with regex (handles complex escaping)
        id_match = re.search(r'"id"\s*:\s*(\d+)', html)
        media_match = re.search(r'"media_id"\s*:\s*"([^"]*)"', html)

        if not id_match or not media_match:
            # Try URL pattern as last resort (/g/227910/)
            url_m = re.search(r'/g/(\d+)/', html)
            if not url_m:
                return None
            gallery = {
                "id": int(url_m.group(1)),
                "media_id": "",
                "title": {},
                "tags": [],
                "num_pages": 0,
            }

        if media_match:
            gallery["media_id"] = media_match.group(1)

        # Extract title from the raw JSON blob in body string
        for s in scripts:
            try:
                data = _json.loads(s.strip())
                body_str = data.get("body", "")

                eng = re.search(r'"english"\s*:\s*"([^"]*)"', body_str)
                jpn = re.search(r'"japanese"\s*:\s*"([^"]*)"', body_str)

                if eng or jpn:
                    gallery["title"] = {
                        "english": eng.group(1) if eng else "",
                        "japanese": jpn.group(1) if jpn else "",
                        "pretty": eng.group(1).replace('[English]', '').strip() if eng and '[English]' in eng.group(1) else eng.group(1).strip() if eng else "",
                    }
            except _json.JSONDecodeError:
                pass

        # Count pages from thumbnail URLs (most reliable method)
        thumb_nums = sorted(set(int(n) for n in re.findall(r'/(\d+)t\.jpe?g', html)))
        gallery["num_pages"] = len(thumb_nums) if thumb_nums else 0

    return gallery


def build_image_urls(gallery: dict) -> list[tuple[int, str]]:
    """Build (page_number, download_url) pairs from gallery data.

    nhentai serves images via two patterns:
      Thumbnails:  https://t{N}.nhentai.net/galleries/{media_id}/{num}t.jpg
      Full res:    https://i.nhentai.net/galleries/{gallery_id}/{num}.{type}
    """
    gallery_id = str(gallery["id"])

    # Method 1: Parse pages from the embedded images JSON if available
    pages_from_api = []
    try:
        img_obj = json.loads(gallery.get("images_raw", "{}"))
        for page in img_obj.get("pages", []):
            ptype = (page or {}).get("type", "image")
            pages_from_api.append(ptype)

        if len(pages_from_api) == gallery["num_pages"]:
            result = []
            for i, ptype in enumerate(pages_from_api):
                url = f"https://i.nhentai.net/galleries/{gallery_id}/{i+1}.{ptype}"
                result.append((i + 1, url))
            return sorted(result)
    except (json.JSONDecodeError, TypeError):
        pass

   # Method 2: Extract page numbers and media_id from thumbnail URLs in HTML
    # Thumbnail URLs are galleries/1200622/4t.jpg — the first numeric ID is media_id, not gallery_id.
    # The i. domain uses media_id for full-res downloads, tN. domain uses it too for thumbnails.
    thumb_data = re.findall(
        r'https://t[1-4]\.nhentai\.net/galleries/(\d+)/(\d+)t\.jpe?g', gallery.get("html", "")
    )

    if thumb_data:
        media_id_from_thumb = thumb_data[0][0]  # e.g., "1200622"
        page_nums = sorted(set(int(n) for _, n in thumb_data))
        return [(n, f"https://i.nhentai.net/galleries/{media_id_from_thumb}/{n}.jpg") for n in page_nums]


def download_gallery(gallery_html_url: str, output_dir: Path) -> list[Path]:
    """Download all pages of a gallery. Returns sorted paths."""

    print(f"  Fetching page HTML from {gallery_html_url}...")
    status_code, html = http_get(gallery_html_url)
    if status_code != 200:
        raise RuntimeError(f"nhentai returned HTTP {status_code}")

    gallery = parse_gallery_from_html(html)
    if not gallery:
        raise RuntimeError("Could not parse gallery data from nhentai page")

    gallery["html"] = html
    print(f"  Gallery #{gallery['id']}: {gallery['title'].get('pretty', 'unknown')} ({gallery['num_pages']} pages)")

    image_urls = build_image_urls(gallery)
    if not image_urls:
        raise RuntimeError("No page URLs found")

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for page_num, url in sorted(image_urls):
        ext_match = re.search(r'\.(\w+)(?:\?|$)', url)
        ext = f".{ext_match.group(1)}" if ext_match else ".jpg"
        filepath = output_dir / f"page_{page_num:04d}{ext}"

        if filepath.exists() and filepath.stat().st_size > 1000:
            print(f"    [skipped] {filepath.name}")
            downloaded.append(filepath)
            continue

        print(f"    [{page_num}/{len(image_urls)}] Downloading {filepath.name}...")
        try:
            http_download(url, filepath)
            downloaded.append(filepath)
        except Exception as e:
            print(f"    [FAILED] page {page_num}: {e}")

    return sorted(downloaded)


# ---------------------------------------------------------------------------
# Stage 2 — censor detection (SAM or manual mask)
# ---------------------------------------------------------------------------

def detect_censor_mask(image_path: Path) -> bytes | None:
    """Auto-detect censor bars using SAM segmentation. Returns base64 PNG mask or None."""
    try:
        import numpy as np
        from PIL import Image

        # Load SAM model on GPU — cached after first run
        print("  Loading SAM model (first run downloads ~2GB)...")
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        checkpoint = "/home/alex/.cache/sam_vit_h_4b8936.pth"
        if not os.path.exists(checkpoint):
            # Download from HF
            print("  SAM checkpoint not found — downloading...")
            import torch
            from huggingface_hub import hf_hub_download
            checkpoint = hf_hub_download(
                "stabilityai/stable-diffusion-2-1",
                "sd2-1-unclip-small/model_index.json",
                repo_type="diffusers",
            )

        sam = sam_model_registry["vit_h"](checkpoint=checkpoint).cuda()
        mask_gen = SamAutomaticMaskGenerator(sam, points_per_side=32)

        img = Image.open(image_path).convert("RGB")
        masks = mask_gen.generate(img)

        # Filter: censor bars are elongated horizontal regions
        bar_masks = []
        for m in masks:
            area = m["area"]
            if area < 8000 or area > img.width * img.height * 0.5:
                continue

            y, x, bw, bh = m["y"], m["x"], int(m["mask"].shape[1]), int(m["mask"].shape[0])
            aspect_ratio = max(bw, bh) / min(max(1, min(bw, bh)), 1)

            # Censor bars: wide horizontal rectangles (aspect ratio > 3 means width >> height)
            if bw > bh * 2 and area < img.width * img.height * 0.3:
                bar_masks.append({"y": y, "x": x, "w": bw, "h": bh})

        if not bar_masks:
            print("  No censor bars detected by SAM — returning None (manual mask recommended)")
            return None

        # Combine into one mask array
        w, h = img.size
        combined = np.zeros((h, w), dtype=float)
        for m in bar_masks:
            combined[m["y"]:m["y"]+m["h"], m["x"]:m["x"]+m["w"]] = 1.0

        # Encode as PNG mask (white=masked region)
        pil_mask = Image.fromarray((combined * 255).astype("uint8"), mode="L")
        buf = __import__("io").BytesIO()
        pil_mask.save(buf, format="PNG", invert_image=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except ImportError as e:
        print(f"  SAM not available ({e}) — returning None for auto-detect")
        return None


# ---------------------------------------------------------------------------
# Stage 3 — run ComfyUI workflow via API
# ---------------------------------------------------------------------------

def encode_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def send_to_comfyui(
    comfyui_url: str,
    workflow: dict,
    image_b64: str,
    mask_b64: str | None = None,
    positive_prompt: str = "",
) -> list[str] | None:
    """Send image+workflow to ComfyUI API. Returns output URLs or None on failure."""

    import requests

    # Build the prompt dict with updated widget values
    nodes_by_order = sorted(workflow.get("nodes", []), key=lambda n: (n.get("order", 99), n["id"]))

    for node in nodes_by_order:
        if not isinstance(node, dict):
            continue
        wv = node.setdefault("widgets_values", [])

        # Update CLIPTextEncode widgets (positive/negative prompts) by order index
        if node.get("type") == "CLIPTextEncode":
            order_idx = node.get("order", 99)
            if isinstance(wv, list) and len(wv) > 0:
                # The first CLIPTextEncode encountered (lowest order index) is positive prompt
                if order_idx <= 5:
                    wv[0] = positive_prompt
                else:
                    wv[0] = ""

        # Update KSampler parameters
        if node.get("type") == "KSampler" and isinstance(wv, list):
            while len(wv) < 7:
                wv.append(0)
            wv[0] = 42   # seed
            wv[2] = 32    # steps for Fill, or 36 for CN style

    payload = {
        "prompt": {str(i): n for i, n in enumerate(nodes_by_order)},
    }

    print(f"  Uploading to ComfyUI ({comfyui_url})...")

    try:
        # Upload image (+ optional mask)
        upload_data = {
            "image": f"data:image/png;base64,{image_b64}",
            "type": "input",
            "subfolder": "nhentai_decensor",
        }
        if mask_b64:
            upload_data["mask"] = f"data:image/png;base64,{mask_b64}"

        resp = requests.post(f"{comfyui_url}/upload/image", data=upload_data, timeout=30)
        if resp.status_code != 200:
            print(f"    Warning: upload {resp.status_code} — continuing anyway")

        # Send prompt (the workflow)
        ws_resp = requests.post(
            f"{comfyui_url}/prompt", json=payload, timeout=15
        )

        if ws_resp.status_code != 200:
            print(f"    Prompt failed: {ws_resp.text[:200]}")
            return None

        prompt_id = ws_resp.json().get("prompt_id")
        print(f"    Prompt ID: {prompt_id}")

        # Poll /history until done (max 15 min per image)
        max_wait = 900
        start = time.time()
        while time.time() - start < max_wait:
            hist_resp = requests.get(f"{comfyui_url}/history", params={"prompt_id": prompt_id}, timeout=5)
            data = hist_resp.json()

            if prompt_id in data and data[prompt_id].get("outputs"):
                outputs = data[prompt_id]["outputs"]
                result_urls = []
                for _node_id, node_out in outputs.items():
                    for img_data in node_out.get("images", []):
                        fname = img_data.get("filename", "")
                        subfolder = img_data.get("subfolder", "")
                        result_urls.append(
                            f"{comfyui_url}/view?filename={fname}&subfolder={subfolder}"
                        )
                return result_urls

            if time.time() - start > max_wait:
                print("    Timeout (15 min) waiting for ComfyUI")
                return None

            time.sleep(2)

        return None

    except requests.RequestException as e:
        print(f"  ComfyUI connection error: {e}")
        return None


# ---------------------------------------------------------------------------
# Stage orchestration
# ---------------------------------------------------------------------------

def load_workflow(path_str: str | Path) -> dict:
    with open(path_str, "r") as f:
        return json.load(f)


def run_pipeline(
    gallery_url: str,
    style: str = "fill",
    workflows_dir: str | None = None,
    comfyui_url: str = "http://127.0.0.1:8188",
    output_base: str | None = None,
    positive_prompt: str = "",
) -> dict:
    """Full pipeline."""

    # Resolve paths
    if workflows_dir is None:
        wf_dir = Path(__file__).resolve().parent.parent / "comfyui" / "workflows"
    else:
        wf_dir = Path(workflows_dir)

    wf_name = f"doujinshi-decensor-{style}.json"
    wf_path = wf_dir / wf_name

    if not wf_path.exists():
        print(f"[ERROR] Workflow file not found: {wf_path}")
        print("Run with --fill or --cn to select a style.")
        sys.exit(1)

    if output_base is None:
        out_base = Path.home() / "claude" / "comfyui" / "output" / "nhentai_decensor"
    else:
        out_base = Path(output_base)

    print("=" * 60)
    print("nhentai decensor pipeline")
    print(f"  URL:          {gallery_url}")
    print(f"  Style:        {style}")
    print(f"  Workflow:     {wf_path}")
    print(f"  Output base:  {out_base}")
    print("=" * 60)

    # --- Download pages ---
    print("\n[Stage 1] Downloading gallery...")
    page_paths = download_gallery(gallery_url, out_base / "raw")
    if not page_paths:
        print("[ERROR] No images downloaded.")
        sys.exit(1)

    num_pages = len(page_paths)
    print(f"Downloaded {num_pages} pages to {out_base / 'raw'}")

    # --- Load workflow template ---
    print("\n[Stage 2] Loading workflow...")
    workflow = load_workflow(str(wf_path))
    if not positive_prompt:
        positive_prompt = "beautiful detailed skin, smooth flesh tone, natural lighting"
    print(f"Positive prompt: {positive_prompt}")

    # --- Inpaint each page ---
    results = []
    for idx, img_path in enumerate(page_paths):
        page_num = idx + 1
        print(f"\n[Stage 3] Page {page_num}/{num_pages}: {img_path.name} ({img_path.stat().st_size:,} bytes)")

        # Auto-detect censor bars
        mask_b64 = detect_censor_mask(img_path)

        # Encode image for ComfyUI upload
        img_b64 = encode_image_base64(img_path)

        output_urls = send_to_comfyui(comfyui_url, workflow, img_b64, mask_b64, positive_prompt)

        if output_urls:
            results.append({
                "page": page_num,
                "input": str(img_path),
                "output_urls": output_urls,
            })
            print(f"  [OK] Page {page_num}")
        else:
            results.append({"page": page_num, "input": str(img_path), "error": True})
            print(f"  [FAIL] Page {page_num} — retry manually via web UI")

    # Summary
    ok = sum(1 for r in results if not r.get("error"))
    fail = sum(1 for r in results if r.get("error"))
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete: {ok} succeeded, {fail} failed")
    print(f"Raw images:  {out_base / 'raw'}")

    # Save manifest
    manifest = {
        "gallery_url": gallery_url,
        "style": style,
        "total_pages": num_pages,
        "succeeded": ok,
        "failed": fail,
        "results": results,
    }
    manifest_path = out_base / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved: {manifest_path}")

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="nhentai -> decensor pipeline (fetch pages + inpaint censor bars)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nhentai-decensor.py --url https://nhentai.net/g/76034/ --style fill
  python nhentai-decensor.py --url https://nhentai.net/g/12345/ --style cn

Notes:
  - ComfyUI must be running (start via ~/claude/comfyui/start-flux.sh)
  - SAM segmentation requires segment_anything package on murderbot
  - Results are saved to ~/claude/comfyui/output/nhentai_decensor/manifest.json
        """,
    )
    parser.add_argument("--url", required=True, help="nhentai URL (e.g. https://nhentai.net/g/76034/)")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--fill", action="store_const", const="fill", dest="style",
                       help="Use Flux Fill Dev GGUF (best quality)")
    group.add_argument("--cn", action="store_const", const="cn", dest="style",
                       help="Use base Flux Dev + ControlNet")
    parser.set_defaults(style=None)

    parser.add_argument("--workflows-dir", default=None, help="Dir with workflow JSON files (default: ./comfyui/workflows)")
    parser.add_argument("--comfyui-url", default="http://127.0.0.1:8188", help="ComfyUI API URL")
    parser.add_argument("--output-base", default=None, help="Output directory root (default: ~/claude/comfyui/output/nhentai_decensor)")
    parser.add_argument(
        "--positive-prompt", default="",
        help='Positive prompt for inpainting (default: "beautiful detailed skin...")'
    )

    args = parser.parse_args()

    style = args.style or "fill"  # default to fill if neither --fill nor --cn specified
    try:
        run_pipeline(
            gallery_url=args.url,
            style=style,
            workflows_dir=args.workflows_dir,
            comfyui_url=args.comfyui_url,
            output_base=args.output_base,
            positive_prompt=args.positive_prompt,
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
