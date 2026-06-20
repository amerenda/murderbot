#!/usr/bin/env python3
"""
test-tool-calling.py — Regression tests for qwen3-35b-think tool calling via LiteLLM.

Catches the class of bugs where multi-tool sessions fail with 400 Bad Request,
hang silently after N calls, or have reasoning-budget exhaustion during tool turns.

Tests:
  1  basic_tool_use         — 1-2 tool calls → model synthesizes
  2  extended_tool_use      — 8+ tool calls → model synthesizes without stopping
  3  no_thinking_with_tools — thinking is never enabled during tool turns
  4  orphan_cleanup         — tool result with no matching tool_call_id → 200, not 400
  5  bad_json_cleanup        — tool_call with truncated JSON args → 200, not 400
  6  mixed_orphan            — assistant with 2 tool_calls, only 1 result → group removed
  7  clean_history_passthru  — well-formed history passes through unchanged

Usage:
  python3 tests/test-tool-calling.py
  python3 tests/test-tool-calling.py --verbose
  python3 tests/test-tool-calling.py --test basic_tool_use
  LITELLM_URL=https://litellm.amer.dev python3 tests/test-tool-calling.py
"""

import argparse
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

LITELLM_URL = os.environ.get("LITELLM_URL", "https://litellm.amer.dev")
MODEL       = os.environ.get("MODEL", "qwen3-35b-think")
TIMEOUT     = int(os.environ.get("TIMEOUT", "120"))

# Master key from env (or fetch from k8s in CI)
API_KEY = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")


# ── Tool definitions ──────────────────────────────────────────────────────────

MOCK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information on a topic.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": "Read and return the content of a URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL to read"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fact",
            "description": "Return a fact about a topic.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
    },
]

MOCK_TOOL_RESULTS = {
    "search_web": (
        "Search results: (1) Python is a high-level, general-purpose programming language created by Guido van Rossum "
        "in 1991. (2) Python emphasizes code readability and uses significant indentation. "
        "(3) Python 3 was released in 2008 and is the current major version. "
        "(4) Python is widely used in web development, data science, AI/ML, and scripting."
    ),
    "read_url": (
        "Article: Python's ecosystem includes NumPy, pandas, and scikit-learn for data science; "
        "TensorFlow and PyTorch for deep learning; Django and Flask for web development. "
        "Python consistently ranks as one of the top 3 most popular programming languages worldwide."
    ),
    "get_fact": (
        "Fact: Python was named after Monty Python's Flying Circus, not the snake. "
        "The Python Package Index (PyPI) hosts over 400,000 packages as of 2024."
    ),
}

# Number of tool calls after which mock results signal the model to stop
_SYNTHESIS_SIGNAL_AFTER = 4
# After this many tool calls, stop passing tools so the model must synthesize
_FORCE_SYNTHESIS_AFTER = 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def completion(messages, tools=None, max_tokens=2048, stream=False):
    """Call LiteLLM completions endpoint."""
    payload = {
        "model":      MODEL,
        "messages":   messages,
        "max_tokens": max_tokens,
        "stream":     stream,
    }
    if tools:
        payload["tools"] = tools

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    resp = requests.post(
        f"{LITELLM_URL}/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=TIMEOUT,
    )
    return resp


def execute_mock_tool(name, args, call_number=0):
    """Return a fake tool result for any of our mock tools."""
    result = MOCK_TOOL_RESULTS.get(name, f"Tool {name} returned: mock result for args {args}")
    if call_number >= _SYNTHESIS_SIGNAL_AFTER:
        result += (
            f"\n\n[You have now made {call_number + 1} tool calls and have sufficient information. "
            "Do NOT call any more tools. Write your final synthesized answer now.]"
        )
    return result


def drive_tool_session(user_prompt, max_turns=15, verbose=False):
    """
    Drive a full tool-call session to completion.

    Returns (turns, finish_reason, final_text, error) where:
      turns         — number of model calls made
      finish_reason — 'stop', 'tool_calls', or 'error'
      final_text    — final assistant text response (empty if stopped on tool_calls)
      error         — exception or HTTP error string, None on success
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful research assistant with access to search and reading tools. "
                "Use tools to gather information, then write a comprehensive answer. "
                "Once you have made 3-5 tool calls, stop calling tools and synthesize "
                "everything you have found into a final response. "
                "Do not make more than 8 tool calls total."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    tool_call_count = 0

    for turn in range(max_turns):
        # After enough tool calls, stop offering tools so the model must synthesize
        active_tools = MOCK_TOOLS if tool_call_count < _FORCE_SYNTHESIS_AFTER else None
        resp = completion(messages, tools=active_tools)

        if resp.status_code != 200:
            return turn, "error", "", f"HTTP {resp.status_code}: {resp.text[:300]}"

        data  = resp.json()
        choice = data["choices"][0]
        msg    = choice["message"]
        finish = choice["finish_reason"]

        messages.append(msg)

        if verbose:
            tc_count = len(msg.get("tool_calls") or [])
            print(f"  turn {turn+1}: finish={finish} tool_calls={tc_count} "
                  f"content_len={len(msg.get('content') or '')}")

        if finish == "stop":
            return turn + 1, "stop", msg.get("content", ""), None

        if finish == "tool_calls":
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                fn   = tc["function"]["name"]
                args = tc["function"].get("arguments", "{}")
                result = execute_mock_tool(fn, args, call_number=tool_call_count)
                tool_call_count += 1
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      result,
                })
            continue

        # Unexpected finish reason
        return turn + 1, finish, msg.get("content", ""), None

    return max_turns, "tool_calls", "", "hit max_turns without synthesis"


# ── Test cases ────────────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name):
        self.name    = name
        self.passed  = False
        self.message = ""
        self.elapsed = 0.0

    def ok(self, msg=""):
        self.passed  = True
        self.message = msg
        return self

    def fail(self, msg):
        self.passed  = False
        self.message = msg
        return self


def test_basic_tool_use(verbose=False):
    r = TestResult("basic_tool_use")
    t0 = time.time()

    turns, finish, text, err = drive_tool_session(
        "What is Python? Use the search tool to find information.",
        max_turns=10,
        verbose=verbose,
    )
    r.elapsed = time.time() - t0

    if err:
        return r.fail(f"session error: {err}")
    if finish != "stop":
        return r.fail(f"did not synthesize: finish={finish} after {turns} turns")
    if not text or len(text) < 50:
        return r.fail(f"synthesis too short ({len(text)} chars): {text[:100]}")

    return r.ok(f"{turns} turns, {len(text)} char synthesis")


def test_extended_tool_use(verbose=False):
    """8+ tool calls must still produce a synthesis — the core regression."""
    r = TestResult("extended_tool_use")
    t0 = time.time()

    turns, finish, text, err = drive_tool_session(
        (
            "Research the following and give me a comprehensive report: "
            "the history of the Python programming language, its major versions, "
            "its ecosystem, and its current usage in AI/ML. "
            "Search for each topic separately and read relevant URLs."
        ),
        max_turns=20,
        verbose=verbose,
    )
    r.elapsed = time.time() - t0

    if err and "hit max_turns" not in str(err):
        return r.fail(f"session error: {err}")
    if finish != "stop":
        return r.fail(
            f"session did not synthesize after {turns} turns (finish={finish}). "
            f"This is the multi-tool regression: model hung/stopped instead of writing final answer."
        )
    if not text or len(text) < 100:
        return r.fail(f"synthesis too short ({len(text)} chars)")

    return r.ok(f"{turns} turns → synthesis ({len(text)} chars)")


def test_no_thinking_with_tools(verbose=False):
    """
    Thinking must NOT activate during any tool turn.
    Budget exhaustion (reasoning-budget: budget exhausted) on a tool turn was the
    root cause of truncated tool_call.arguments → 400 on the next request.
    """
    r = TestResult("no_thinking_with_tools")
    t0 = time.time()

    # Send a multi-tool session and check llama-server logs for budget exhaustion
    # We can't read llama-server logs from here, so instead verify indirectly:
    # if thinking fires, the reasoning budget gets exhausted mid-tool-call and
    # the next request returns 400. Run 8 turns and if we get no 400, thinking
    # stayed off.
    turns, finish, text, err = drive_tool_session(
        "Use the search tool to find 8 different facts about space exploration.",
        max_turns=20,
        verbose=verbose,
    )
    r.elapsed = time.time() - t0

    if err and "400" in str(err):
        return r.fail(
            f"Got 400 — likely reasoning-budget exhaustion during tool turn. "
            f"Check that auto_disable_thinking_with_tools removes the < 6 threshold. "
            f"Error: {err}"
        )
    if err and "hit max_turns" not in str(err):
        return r.fail(f"unexpected error: {err}")

    # If we completed without 400, thinking didn't fire destructively
    return r.ok(f"{turns} turns, no 400 errors")


def test_orphan_cleanup(verbose=False):
    """
    Orphaned tool result (tool_call_id with no matching assistant.tool_calls)
    must be silently removed by the LiteLLM hook — not cause a 400.
    """
    r = TestResult("orphan_cleanup")
    t0 = time.time()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Hello, what is 2+2?"},
        # Valid exchange
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_valid", "type": "function",
                            "function": {"name": "get_fact", "arguments": '{"topic": "math"}'}}],
        },
        {"role": "tool", "tool_call_id": "call_valid", "content": "Math is the study of numbers."},
        # Orphaned tool result — no matching assistant.tool_calls
        {"role": "tool", "tool_call_id": "call_ORPHAN_NO_MATCH", "content": "This result has no parent."},
        {"role": "user", "content": "Just answer the math question directly."},
    ]

    resp = completion(messages, tools=MOCK_TOOLS)
    r.elapsed = time.time() - t0

    if resp.status_code == 400:
        return r.fail(
            "Got 400 — LiteLLM hook did NOT clean up the orphaned tool result. "
            "Check _cleanup_orphaned_tool_pairs() in tool-strip-hook-configmap.yaml."
        )
    if resp.status_code != 200:
        return r.fail(f"unexpected HTTP {resp.status_code}: {resp.text[:200]}")

    finish = resp.json()["choices"][0]["finish_reason"]
    return r.ok(f"200 OK, finish={finish} (orphan was silently removed)")


def test_bad_json_cleanup(verbose=False):
    """
    Assistant tool_call with truncated/invalid JSON in arguments (caused by
    reasoning-budget exhaustion mid-generation) must be removed by the hook.
    """
    r = TestResult("bad_json_cleanup")
    t0 = time.time()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Search for Python history."},
        # Malformed tool_call: arguments is truncated JSON (simulates budget exhaust)
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id":   "call_truncated",
                "type": "function",
                "function": {
                    "name":      "search_web",
                    "arguments": '{"query": "Python programming language hist',  # truncated
                },
            }],
        },
        # Corresponding result (which will also be removed atomically with the parent)
        {"role": "tool", "tool_call_id": "call_truncated", "content": "Some search result."},
        {"role": "user", "content": "Never mind the search. Just tell me: what year was Python created?"},
    ]

    resp = completion(messages, tools=MOCK_TOOLS)
    r.elapsed = time.time() - t0

    if resp.status_code == 400:
        return r.fail(
            "Got 400 — LiteLLM hook did NOT strip the truncated-JSON tool_call. "
            "Check _has_invalid_args() logic in _cleanup_orphaned_tool_pairs()."
        )
    if resp.status_code != 200:
        return r.fail(f"unexpected HTTP {resp.status_code}: {resp.text[:200]}")

    finish = resp.json()["choices"][0]["finish_reason"]
    return r.ok(f"200 OK, finish={finish} (bad-JSON tool_call removed)")


def test_mixed_orphan(verbose=False):
    """
    Assistant has 2 tool_calls; only 1 has a matching result.
    The entire group (assistant + existing result) must be removed atomically.
    """
    r = TestResult("mixed_orphan")
    t0 = time.time()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Search for two topics."},
        # Assistant issues 2 tool calls
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_a", "type": "function",
                 "function": {"name": "search_web", "arguments": '{"query": "topic A"}'}},
                {"id": "call_b", "type": "function",
                 "function": {"name": "search_web", "arguments": '{"query": "topic B"}'}},
            ],
        },
        # Only call_a has a result; call_b is missing
        {"role": "tool", "tool_call_id": "call_a", "content": "Result for topic A."},
        {"role": "user", "content": "Just summarize what you know."},
    ]

    resp = completion(messages, tools=MOCK_TOOLS)
    r.elapsed = time.time() - t0

    if resp.status_code == 400:
        return r.fail(
            "Got 400 — incomplete tool_call group (2 calls, 1 result) was not removed atomically. "
            "Check that any-missing-result triggers full group removal."
        )
    if resp.status_code != 200:
        return r.fail(f"unexpected HTTP {resp.status_code}: {resp.text[:200]}")

    finish = resp.json()["choices"][0]["finish_reason"]
    return r.ok(f"200 OK, finish={finish} (partial group removed atomically)")


def test_clean_history_passthru(verbose=False):
    """
    A well-formed conversation with complete tool exchange pairs must pass
    through unchanged and return a valid response.
    """
    r = TestResult("clean_history_passthru")
    t0 = time.time()

    call_id = f"call_{uuid.uuid4().hex[:8]}"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Search for Python facts."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "search_web", "arguments": '{"query": "Python facts"}'},
            }],
        },
        {"role": "tool", "tool_call_id": call_id, "content": "Python was created in 1991."},
        {"role": "user", "content": "Great, summarize what you found."},
    ]

    resp = completion(messages, tools=MOCK_TOOLS)
    r.elapsed = time.time() - t0

    if resp.status_code != 200:
        return r.fail(f"HTTP {resp.status_code} on clean history: {resp.text[:200]}")

    finish = resp.json()["choices"][0]["finish_reason"]
    return r.ok(f"200 OK, finish={finish}")


# ── Stress tests ─────────────────────────────────────────────────────────────

def stress_repeat(n=5, verbose=False):
    """Run the full regression suite N times and report per-test pass rates."""
    print(f"[stress:repeat] Running full suite {n} times\n")
    counts = {name: {"pass": 0, "fail": 0} for name in TESTS}
    t0 = time.time()

    for i in range(n):
        print(f"  Round {i+1}/{n}")
        for name, fn in TESTS.items():
            try:
                r = fn(verbose=False)
            except Exception as e:
                r = TestResult(name)
                r.fail(f"exception: {e}")
            status = "PASS" if r.passed else "FAIL"
            counts[name]["pass" if r.passed else "fail"] += 1
            print(f"    {name}: {status}  {r.message}")
        print()

    elapsed = time.time() - t0
    print(f"Results after {n} rounds ({elapsed:.0f}s total):")
    all_ok = True
    for name, c in counts.items():
        rate = c["pass"] / n * 100
        flag = "" if c["fail"] == 0 else f"  ← {c['fail']} FAILURE(S)"
        print(f"  {name}: {c['pass']}/{n} ({rate:.0f}%){flag}")
        if c["fail"]:
            all_ok = False
    return all_ok


def stress_deep_loop(turns=40, verbose=False):
    """
    Single session: 40 tool calls with no forced synthesis cutoff.
    Verifies no 400/500 fires regardless of history length.
    """
    print(f"[stress:deep-loop] {turns} tool calls, no synthesis cutoff\n")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Keep searching for more information. "
                "Call search_web repeatedly with different queries about Python. "
                "Do not stop until instructed."
            ),
        },
        {"role": "user", "content": "Search for everything you can find about Python."},
    ]

    errors = []
    tool_call_count = 0
    t0 = time.time()

    for turn in range(turns + 5):
        resp = completion(messages, tools=MOCK_TOOLS)
        if resp.status_code != 200:
            errors.append(f"turn {turn+1}: HTTP {resp.status_code}")
            print(f"  turn {turn+1}: HTTP {resp.status_code} ERROR")
            break

        data   = resp.json()
        choice = data["choices"][0]
        msg    = choice["message"]
        finish = choice["finish_reason"]
        messages.append(msg)

        tc_count = len(msg.get("tool_calls") or [])
        if verbose:
            print(f"  turn {turn+1}: finish={finish} tool_calls={tc_count} "
                  f"total_calls={tool_call_count} history_len={len(messages)}")
        else:
            print(f"  turn {turn+1}: finish={finish} tool_calls={tc_count} "
                  f"total_calls={tool_call_count}", flush=True)

        if finish == "stop" or finish != "tool_calls":
            break

        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn   = tc["function"]["name"]
            args = tc["function"].get("arguments", "{}")
            result = MOCK_TOOL_RESULTS.get(fn, "mock result")
            tool_call_count += 1
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": f"[call {tool_call_count}] {result}",
            })

        if tool_call_count >= turns:
            print(f"\n  Reached {turns} tool calls — forcing synthesis turn")
            resp2 = completion(messages, tools=None)
            status = resp2.status_code
            finish2 = resp2.json()["choices"][0]["finish_reason"] if status == 200 else "error"
            print(f"  Synthesis turn: HTTP {status}, finish={finish2}")
            if status != 200:
                errors.append(f"synthesis turn: HTTP {status}")
            break

    elapsed = time.time() - t0
    print(f"\n  Completed {tool_call_count} tool calls in {elapsed:.1f}s")
    if errors:
        print(f"  FAILURES: {errors}")
        return False
    print(f"  No errors — {tool_call_count} calls, zero 400/500s")
    return True


def _run_one_session(session_id, prompt):
    """Worker for concurrent stress test."""
    t0 = time.time()
    turns, finish, text, err = drive_tool_session(prompt, max_turns=12)
    elapsed = time.time() - t0
    ok = finish == "stop" and not err
    return session_id, ok, turns, finish, err, elapsed


def stress_concurrent(n=4, verbose=False):
    """
    Fire N sessions in parallel. Verifies the server handles concurrent
    tool-calling sessions without errors.
    """
    print(f"[stress:concurrent] {n} parallel sessions\n")
    prompts = [
        "What is Python? Search and summarize.",
        "Search for information about machine learning frameworks.",
        "Find facts about Linux operating system history.",
        "Research the history of the internet.",
        "What is Kubernetes? Search for information.",
        "Find information about Rust programming language.",
        "Search for Python web frameworks.",
        "Research GPU computing and CUDA.",
    ][:n]

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {pool.submit(_run_one_session, i, p): i for i, p in enumerate(prompts)}
        for fut in as_completed(futures):
            sid, ok, turns, finish, err, elapsed = fut.result()
            results.append((sid, ok, turns, finish, err, elapsed))
            status = "OK" if ok else "FAIL"
            print(f"  session {sid}: {status}  {turns} turns  finish={finish}  {elapsed:.1f}s"
                  + (f"  err={err}" if err else ""))

    elapsed = time.time() - t0
    passed = sum(1 for _, ok, *_ in results if ok)
    print(f"\n  {passed}/{n} sessions completed successfully in {elapsed:.1f}s total")
    return passed == n


def stress_large_payload(verbose=False):
    """
    Tool responses near max_tool_response_chars (3000 chars) to exercise
    the token-budget stripping code paths in the LiteLLM hook.
    """
    print("[stress:large-payload] Large tool responses + multi-turn\n")

    # Build a big result (~2900 chars) that won't trigger truncation but is large
    big_chunk = "Python information: " + ("x" * 100 + " ") * 28  # ~2900 chars
    # And one that exceeds the limit to force server-side truncation
    huge_chunk = "Python information: " + ("x" * 100 + " ") * 60  # ~6100 chars

    messages = [
        {"role": "system", "content": "You are a research assistant. Use tools, then summarize."},
        {"role": "user", "content": "Search for Python information."},
    ]

    errors = []
    call_id = f"call_{uuid.uuid4().hex[:8]}"
    t0 = time.time()

    # Turn 1: send a large (but under-limit) result
    resp = completion(messages, tools=MOCK_TOOLS)
    if resp.status_code != 200:
        print(f"  Turn 1 req: HTTP {resp.status_code}")
        return False
    msg = resp.json()["choices"][0]["message"]
    messages.append(msg)
    if msg.get("tool_calls"):
        tc = msg["tool_calls"][0]
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": big_chunk})
        print(f"  Turn 1: tool call → {len(big_chunk)}-char response sent")
    else:
        print(f"  Turn 1: model skipped tool call (finish={resp.json()['choices'][0]['finish_reason']})")

    # Turn 2: send a huge result (exceeds max_tool_response_chars=3000 — template truncates it)
    resp2 = completion(messages, tools=MOCK_TOOLS)
    if resp2.status_code != 200:
        print(f"  Turn 2 req: HTTP {resp2.status_code}")
        errors.append(f"turn 2: HTTP {resp2.status_code}")
    else:
        msg2 = resp2.json()["choices"][0]["message"]
        messages.append(msg2)
        if msg2.get("tool_calls"):
            tc2 = msg2["tool_calls"][0]
            messages.append({"role": "tool", "tool_call_id": tc2["id"], "content": huge_chunk})
            print(f"  Turn 2: tool call → {len(huge_chunk)}-char response sent (will be truncated by template)")
        else:
            print(f"  Turn 2: model synthesized early (finish={resp2.json()['choices'][0]['finish_reason']})")

    # Turn 3: force synthesis (no tools)
    resp3 = completion(messages, tools=None)
    elapsed = time.time() - t0
    if resp3.status_code != 200:
        errors.append(f"synthesis turn: HTTP {resp3.status_code}")
        print(f"  Turn 3 (synthesis): HTTP {resp3.status_code}")
    else:
        finish3 = resp3.json()["choices"][0]["finish_reason"]
        content3 = resp3.json()["choices"][0]["message"].get("content", "")
        print(f"  Turn 3 (synthesis): HTTP 200, finish={finish3}, {len(content3)} chars")

    print(f"\n  Completed in {elapsed:.1f}s")
    if errors:
        print(f"  FAILURES: {errors}")
        return False
    print("  No errors — large payloads handled correctly")
    return True


STRESS_TESTS = {
    "repeat":       stress_repeat,
    "deep-loop":    stress_deep_loop,
    "concurrent":   stress_concurrent,
    "large-payload": stress_large_payload,
}


# ── Runner ────────────────────────────────────────────────────────────────────

TESTS = {
    "basic_tool_use":        test_basic_tool_use,
    "extended_tool_use":     test_extended_tool_use,
    "no_thinking_with_tools": test_no_thinking_with_tools,
    "orphan_cleanup":        test_orphan_cleanup,
    "bad_json_cleanup":      test_bad_json_cleanup,
    "mixed_orphan":          test_mixed_orphan,
    "clean_history_passthru": test_clean_history_passthru,
}


def main():
    parser = argparse.ArgumentParser(description="qwen3-35b-think tool calling regression tests")
    parser.add_argument("--test",   help="Run a single regression test by name")
    parser.add_argument("--stress", help=(
        "Run a stress scenario: repeat, deep-loop, concurrent, large-payload  "
        "(or 'all' for all four)"
    ))
    parser.add_argument("--repeat-n",    type=int, default=5,  metavar="N",
                        help="Rounds for --stress repeat (default 5)")
    parser.add_argument("--loop-turns",  type=int, default=40, metavar="N",
                        help="Tool calls for --stress deep-loop (default 40)")
    parser.add_argument("--concurrent-n", type=int, default=4, metavar="N",
                        help="Parallel sessions for --stress concurrent (default 4)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--list", action="store_true", help="List test/stress names")
    args = parser.parse_args()

    if args.list:
        print("Regression tests:")
        for name in TESTS:
            print(f"  {name}")
        print("\nStress scenarios:")
        for name in STRESS_TESTS:
            print(f"  {name}")
        return

    # ── Stress mode ──
    if args.stress:
        which = list(STRESS_TESTS.keys()) if args.stress == "all" else [args.stress]
        for s in which:
            if s not in STRESS_TESTS:
                print(f"Unknown stress scenario: {s}. Available: {', '.join(STRESS_TESTS)}")
                sys.exit(1)

        print(f"Stress testing against {LITELLM_URL} model={MODEL}\n")
        ok = True
        for s in which:
            fn = STRESS_TESTS[s]
            print(f"{'='*60}")
            kwargs = {}
            if s == "repeat":
                kwargs["n"] = args.repeat_n
            elif s == "deep-loop":
                kwargs["turns"] = args.loop_turns
            elif s == "concurrent":
                kwargs["n"] = args.concurrent_n
            try:
                result = fn(verbose=args.verbose, **kwargs)
            except Exception as e:
                print(f"  EXCEPTION: {e}")
                result = False
            ok = ok and result
            print()

        sys.exit(0 if ok else 1)

    # ── Regression mode ──
    if args.test and args.test not in TESTS:
        print(f"Unknown test: {args.test}. Available: {', '.join(TESTS)}")
        sys.exit(1)

    tests_to_run = {args.test: TESTS[args.test]} if args.test else TESTS

    print(f"Running {len(tests_to_run)} test(s) against {LITELLM_URL} model={MODEL}\n")

    results = []
    for name, fn in tests_to_run.items():
        print(f"  {name} ... ", end="", flush=True)
        try:
            r = fn(verbose=args.verbose)
        except Exception as e:
            r = TestResult(name)
            r.fail(f"exception: {e}")
        r.elapsed = getattr(r, "elapsed", 0.0)
        results.append(r)

        status = "PASS" if r.passed else "FAIL"
        print(f"{status}  ({r.elapsed:.1f}s)  {r.message}")

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print(f"\n{passed}/{len(results)} passed", end="")
    if failed:
        print(f", {failed} FAILED")
        sys.exit(1)
    else:
        print(" — all good")


if __name__ == "__main__":
    main()
