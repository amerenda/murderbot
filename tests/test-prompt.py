#!/usr/bin/env python3
"""
Interactive prompt expansion tester.

Sends a description to Ollama and shows exactly what it returns.
Optionally compares multiple prompt strategies side by side.

Usage:
  python3 test-prompt.py "close-up portrait of a young woman, soft window light"
  python3 test-prompt.py --all                      # test all 5 default subjects
  python3 test-prompt.py --compare                  # show generic vs routed vs dual-encode side by side
  python3 test-prompt.py --strategy nsfw "a woman"  # test the NSFW v2 system prompt
  OLLAMA_MODEL=mistral-nemo:12b python3 test-prompt.py "bowl of ramen"
"""

import argparse
import json
import os
import pathlib
import sys
import textwrap
import requests

OLLAMA_HOST  = "http://10.100.20.18:11434"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral-nemo:12b")

# ─── SUBJECTS ────────────────────────────────────────────────────────────────

SUBJECTS = [
    "close-up portrait of a young woman, soft window light",
    "portrait of a middle-aged man in a coffee shop",
    "bowl of tonkotsu ramen with chashu pork and soft boiled egg",
    "rainy street in Tokyo at night with neon reflections",
    "cozy living room with fireplace and bookshelves, evening",
]

# ─── PROMPT STRATEGIES ───────────────────────────────────────────────────────

GENERIC = (
    "You are a prompt engineer for photorealistic AI image generation. "
    "Take the user's simple description and expand it into a rich, detailed visual prompt. "
    "Include: specific lighting setup (golden hour, studio strobes, Rembrandt lighting, soft window light), "
    "camera and lens metadata (e.g. 'shot on Sony A7R V, 85mm f/1.4, shallow depth of field'), "
    "atmosphere, surface textures, colors, and composition. "
    "Add post-processing style where natural (e.g. Lightroom color grade, subtle film grain, RAW finish). "
    "Reply ONLY with the enhanced prompt — no explanations, no preamble, no quotes."
)

_PROSE_RULE = (
    "Write as a single flowing paragraph of 80–130 words. "
    "No bullet points, no numbered lists, no headers, no markdown formatting, no surrounding quotes. "
    "Do not open with an imperative like 'Capture', 'Create', or 'Imagine'. "
    "Start directly with the subject or scene."
)

PORTRAIT = (
    "You are a portrait photographer writing a generation prompt. "
    "Expand the description into a detailed, flowing portrait prompt covering: "
    "subject appearance (expression, skin tone, hair texture); "
    "lighting (key light type and angle, fill ratio, rim or hair light, catchlights); "
    "camera body and prime lens (Sony A7R V 85mm f/1.4, Canon R5 135mm f/2, or similar); "
    "aperture, shutter speed, ISO; shallow bokeh on background; skin pore detail. "
    "End with post-processing style (Lightroom, subtle retouch, color grade). " + _PROSE_RULE
)

FOOD = (
    "You are a food photographer writing a generation prompt. "
    "Expand the description into a detailed, flowing food photography prompt covering: "
    "plating, garnishes, surface material (slate, marble, weathered wood); "
    "lighting direction and quality (window light, overhead hero rig, backlit), specular highlights, steam or condensation; "
    "lens choice (Canon 100mm f/2.8 macro or 85mm), aperture and depth of field; "
    "color temperature, hero composition. "
    "End with post-processing (warm Lightroom grade, clarity, vignette). " + _PROSE_RULE
)

STREET = (
    "You are a street photographer writing a generation prompt. "
    "Expand the description into a detailed, flowing street photography prompt covering: "
    "time of day and weather; specific urban details (signage, puddles, crowds, motion); "
    "practical light sources (neon, streetlamps, headlights) and their color temperatures; "
    "camera and lens (Leica Q2 28mm f/1.7, Fujifilm X-T5 35mm f/2, or similar); "
    "exposure settings (high ISO grain or frozen motion); atmospheric mood. "
    "End with color grade direction (neon-saturated, faded film, or high-contrast). " + _PROSE_RULE
)

INTERIOR = (
    "You are an interior photographer writing a generation prompt. "
    "Expand the description into a detailed, flowing interior photography prompt covering: "
    "room features, furniture, and surface materials (wood grain, fabric, stone, metal); "
    "light sources (window direction, golden hour, practical lamps) and color temperature balance; "
    "camera and lens (Sony 16-35mm f/2.8 or 24mm tilt-shift); "
    "shadow and highlight treatment; styling details (art, plants, books, candles). "
    "End with post-processing (Lightroom architectural, shadow lift, clean highlights). " + _PROSE_RULE
)

DUAL_ENCODE = (
    "You are a prompt engineer for FLUX.1 image generation. "
    "FLUX uses two text encoders: CLIP-L (semantic keywords, max 20 tokens) and T5 (rich prose narrative). "
    "Output EXACTLY two lines — no other text, no blank lines, no markdown, no preamble: "
    "Line 1 must start with exactly 'CLIP: ' then 15-20 comma-separated keywords covering subject, "
    "subject type, lighting quality, camera/lens model, photographic style, mood, and quality tags "
    "(e.g. 'young woman, close-up portrait, soft window light, Rembrandt lighting, Sony A7R V, "
    "85mm f/1.4, shallow depth of field, skin texture, photorealistic, RAW photo'). "
    "Line 2 must start with exactly 'T5: ' then a flowing 80-120 word narrative describing the scene, "
    "lighting setup, camera settings, textures, and post-processing — no imperative verbs, "
    "start with the subject description directly."
)

_NSFW_PROMPT_FILE = pathlib.Path(__file__).parent / "nsfw-image-prompt-v2.txt"
NSFW = _NSFW_PROMPT_FILE.read_text().strip() if _NSFW_PROMPT_FILE.exists() else GENERIC

STRATEGIES = {
    "generic":      GENERIC,
    "portrait":     PORTRAIT,
    "food":         FOOD,
    "street":       STREET,
    "interior":     INTERIOR,
    "dual-encode":  DUAL_ENCODE,
    "nsfw":         NSFW,
}

# ─── ROUTING ─────────────────────────────────────────────────────────────────

ROUTING = [
    (["portrait", "woman", "man", "person", "face", "model", "girl", "boy"], "portrait"),
    (["food", "ramen", "dish", "meal", "coffee", "plate", "burger", "sushi"], "food"),
    (["street", "tokyo", "city", "neon", "urban", "night", "alley", "sidewalk"], "street"),
    (["interior", "room", "living", "bedroom", "office", "kitchen", "studio", "fireplace"], "interior"),
]

def detect_strategy(prompt):
    lower = prompt.lower()
    for keywords, name in ROUTING:
        if any(k in lower for k in keywords):
            return name
    return "generic"

# ─── EXPAND ──────────────────────────────────────────────────────────────────

def warmup():
    """Pre-load the model so the first generate call doesn't cold-start timeout."""
    try:
        requests.post(f"{OLLAMA_HOST}/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": "",
            "keep_alive": "10m",
        }, timeout=120)
    except Exception:
        pass

def expand(prompt, system, timeout=180):
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
    }, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["response"].strip()

def parse_dual(text, fallback):
    clip_line, t5_line = fallback, fallback
    for line in text.splitlines():
        if line.startswith("CLIP:"):
            clip_line = line[5:].strip()
        elif line.startswith("T5:"):
            t5_line = line[3:].strip()
    return clip_line, t5_line

# ─── DISPLAY ─────────────────────────────────────────────────────────────────

W = 88

def hr(char="─"): print(char * W)
def header(text): hr(); print(f"  {text}"); hr()

def show_block(label, text, color=""):
    reset = "\033[0m" if color else ""
    print(f"\n{color}▶ {label}{reset}")
    for line in textwrap.wrap(text, W - 4):
        print(f"  {line}")

def count_words(text): return len(text.split())

# ─── MODES ───────────────────────────────────────────────────────────────────

def test_single(prompt, strategy=None):
    if strategy is None:
        strategy = detect_strategy(prompt)
        print(f"\n  Auto-detected strategy: \033[93m{strategy}\033[0m")

    header(f"Prompt Test — {OLLAMA_MODEL}")
    show_block("Input", prompt, "\033[36m")

    system = STRATEGIES[strategy]
    show_block("System prompt (truncated)", system[:200] + "…", "\033[90m")

    print(f"\n  Expanding via Ollama ({OLLAMA_MODEL})…")
    result = expand(prompt, system)

    show_block(f"Output [{strategy}] — {count_words(result)} words", result, "\033[92m")
    hr()


def test_compare(prompt):
    strategy = detect_strategy(prompt)
    header(f"Prompt Comparison — {OLLAMA_MODEL}")
    show_block("Input", prompt, "\033[36m")
    print(f"\n  Auto-detected type: \033[93m{strategy}\033[0m")
    print(f"  Warming up model…")
    warmup()
    print(f"  Running 3 strategies in sequence…\n")

    # Generic
    hr("·")
    print("  \033[90m[1/3] Generic expansion…\033[0m")
    g = expand(prompt, GENERIC)
    show_block(f"generic — {count_words(g)} words", g, "\033[33m")

    # Routed (subject-type specific)
    hr("·")
    print(f"  \033[90m[2/3] {strategy.capitalize()} routing…\033[0m")
    r = expand(prompt, STRATEGIES[strategy])
    show_block(f"routed ({strategy}) — {count_words(r)} words", r, "\033[92m")

    # Dual-encode
    hr("·")
    print("  \033[90m[3/3] Dual-encode (CLIP-L + T5)…\033[0m")
    raw = expand(prompt, DUAL_ENCODE)
    clip, t5 = parse_dual(raw, prompt)
    print(f"\n  \033[95m▶ CLIP-L [{count_words(clip)} words — goes into clip_l encoder]:\033[0m")
    print(f"  {clip}")
    print(f"\n  \033[95m▶ T5     [{count_words(t5)} words — goes into t5xxl encoder]:\033[0m")
    for line in textwrap.wrap(t5, W - 4):
        print(f"  {line}")

    hr()
    print(f"\n  Word counts:  generic={count_words(g)}  routed={count_words(r)}  T5={count_words(t5)}")
    print()


def test_all(compare=False):
    for i, s in enumerate(SUBJECTS, 1):
        print(f"\n\033[1m{'='*W}\033[0m")
        print(f"  Subject {i}/{len(SUBJECTS)}: {s}")
        if compare:
            test_compare(s)
        else:
            test_single(s)

# ─── MAIN ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Test Ollama prompt expansion")
    p.add_argument("prompt", nargs="?", help="Description to expand")
    p.add_argument("--all",     action="store_true", help="Test all default subjects")
    p.add_argument("--compare", action="store_true", help="Compare all 3 strategies side by side")
    p.add_argument("--strategy", choices=list(STRATEGIES), help="Force a specific strategy")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\033[1m  Ollama: {OLLAMA_HOST}  Model: {OLLAMA_MODEL}\033[0m")
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if OLLAMA_MODEL not in models and not any(OLLAMA_MODEL in m for m in models):
            print(f"\033[91m  WARNING: {OLLAMA_MODEL} not found. Available: {', '.join(models[:5])}\033[0m")
    except Exception as e:
        print(f"\033[91m  WARNING: Ollama unreachable ({e})\033[0m")

    if args.all:
        test_all(compare=args.compare)
    elif args.prompt:
        if args.compare:
            test_compare(args.prompt)
        else:
            test_single(args.prompt, args.strategy)
    else:
        # Interactive mode
        print("\n  Enter a description (or 'quit'):")
        while True:
            try:
                prompt = input("\n  > ").strip()
            except (KeyboardInterrupt, EOFError):
                print(); break
            if not prompt or prompt.lower() in ("quit", "q", "exit"):
                break
            if args.compare:
                test_compare(prompt)
            else:
                test_single(prompt)


if __name__ == "__main__":
    main()
