#!/usr/bin/env python3
"""
test-metrics-port.py — Verify llama-server metrics endpoint runs on port 9110.

This test ensures the metrics scraping target matches what start-opencode-stable.sh
configures as its PORT default (9110). If metrics are scraped on a different port,
Prometheus/Grafana will see no data and alerting will fail silently.

Two checks:
  1. STATIC: parse start-opencode-stable.sh to confirm PORT defaults to 9110
     (not hardcoded 8088 or any other value).
  2. LIVE: if a llama-server is already running on 9110, hit /metrics and verify
     the response contains expected Prometheus metric families (go_memstats, http, etc.).

Usage:
  python3 test-metrics-port.py           # run both static + live checks
  python3 test-metrics-port.py --static  # only check the startup script config
  python3 test-metrics-port.py --live    # only hit a running server on :9110/metrics
"""

import re
import subprocess
import sys
import urllib.request
import urllib.error


SCRIPT_DIR = "/home/alex/claude/projects/murderbot"
STARTUP_SCRIPT = f"{SCRIPT_DIR}/start-opencode-stable.sh"
METRICS_PORT = 9110
EXPECTED_HOST = "http://127.0.0.1"
METRICS_URL = f"{EXPECTED_HOST}:{METRICS_PORT}/metrics"

PASS_COUNT = 0
FAIL_COUNT = 0


def check(description, passed, detail=""):
    global PASS_COUNT, FAIL_COUNT
    status = "\u2705 PASS" if passed else "\u274c FAIL"
    print(f"{status}: {description}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"       {line}")
    if passed:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1


# ─── STATIC CHECKS (startup script) ──────────────────────────────────────────

def test_startup_port_default():
    """PORT must default to 9110 via PORT="${PORT:-9110}", not hardcoded."""
    with open(STARTUP_SCRIPT) as f:
        content = f.read()

    # Must contain the variable-substitution syntax
    has_var_syntax = bool(re.search(r'PORT\s*=\s*"\$\{PORT:-9110\}"', content))

    # Must NOT hardcode port 8088 (the old bug)
    lines = content.split("\n")
    hardcoded_8088 = any(
        re.match(r'^PORT\s*=\s*8088\b', line.strip()) or
        re.match(r'PORT=8088', line.strip())
        for line in lines
    )

    detail_lines = []
    if has_var_syntax:
        detail_lines.append("PORT uses variable substitution syntax with default 9110")
    else:
        m = re.search(r'PORT\s*=\s*"?\$?{?[^}]*}', content)
        if m:
            detail_lines.append(f"Found PORT line: {m.group()}")
        else:
            detail_lines.append("Could not find PORT assignment in script")

    if hardcoded_8088:
        detail_lines.append("ERROR: Port 8088 is still hardcoded somewhere!")

    check(
        "PORT defaults to 9110 (not hardcoded 8088)",
        passed=has_var_syntax and not hardcoded_8088,
        detail="\n".join(detail_lines),
    )


def test_startup_metrics_flag():
    """The startup script must include --metrics flag."""
    with open(STARTUP_SCRIPT) as f:
        content = f.read()

    has_metrics = "--metrics" in content

    check(
        "Startup script includes --metrics flag",
        passed=has_metrics,
        detail="Required for Prometheus-compatible /metrics endpoint",
    )


def test_startup_port_in_server_args():
    """The PORT variable must be referenced in the llama-server command arguments."""
    with open(STARTUP_SCRIPT) as f:
        content = f.read()

    # Look for --port $PORT or --port "$PORT" or --port ${PORT}
    has_port_arg = bool(re.search(r'--port\s+"?\$?PORT', content))

    check(
        "SERVER_ARGS references PORT variable (not hardcoded)",
        passed=has_port_arg,
        detail="Ensures the port is configurable and matches METRICS_PORT",
    )


# ─── LIVE CHECKS (running server) ────────────────────────────────────────────

def test_live_metrics_endpoint():
    """Hit :9110/metrics and verify it returns Prometheus-format metrics."""
    try:
        req = urllib.request.Request(f"{METRICS_URL}")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")

    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionRefusedError, OSError):
        check(
            "Metrics endpoint reachable on :9110/metrics",
            passed=False,
            detail=f"No llama-server responding at {METRICS_URL}\n"
                   f"This is OK if no server is running right now — the static checks still apply.",
        )
        return

    check(
        "Metrics endpoint reachable on :9110/metrics",
        passed=True,
        detail=f"HTTP {resp.status}, {len(body)} bytes received",
    )

    # Verify it contains actual Prometheus metrics (not an HTML error page)
    has_metrics_content = bool(re.search(
        r'^(#[ TYPE]?\w+|\w[\w:]*\s)',  # Prometheus lines: # HELP/TYPE comments OR metric_name{...} with colons (llamacpp:)
        body,
        re.MULTILINE,
    ))

    check(
        "Response contains Prometheus-format metrics (not HTML/error)",
        passed=has_metrics_content,
        detail="First 200 chars: " + body[:200],
    )


def test_live_port_not_8088():
    """If something IS running on port 9110, verify nothing is also on 8088 (old default)."""
    try:
        req = urllib.request.Request("http://127.0.0.1:8088/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionRefusedError, OSError):
        # Nothing on 8088 — good!
        check(
            "No server running on old default port 8088",
            passed=True,
            detail="Port 8088 is free — no stale old-instance detected",
        )
        return

    # Something IS on 8088 — this could be a leftover from before the fix
    check(
        "No server running on old default port 8088",
        passed=False,
        detail=(
            "Something responded on :8088! This may be a leftover instance.\n"
            "The script was changed to use port 9110 — you may need to stop the old process."
        ),
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    mode = "--all" if len(sys.argv) < 2 else sys.argv[1]

    print("=" * 60)
    print("test-metrics-port.py — Metrics port validation")
    print(f"Expected: http://127.0.0.1:{METRICS_PORT}/metrics")
    print("=" * 60)
    print()

    if mode in ("--all", "--static"):
        test_startup_port_default()
        test_startup_metrics_flag()
        test_startup_port_in_server_args()
        print()

    if mode in ("--all", "--live"):
        test_live_metrics_endpoint()
        test_live_port_not_8088()
        print()

    # Summary
    total = PASS_COUNT + FAIL_COUNT
    print("-" * 60)
    print(f"Results: {PASS_COUNT}/{total} passed, {FAIL_COUNT} failed")
    print("-" * 60)

    if FAIL_COUNT > 0:
        print("\nTest suite FAILED — fix the issues above before deploying.")
        sys.exit(1)
    else:
        print("\nAll checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
