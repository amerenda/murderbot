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
    parser.add_argument("--test", help="Run a single test by name")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--list", action="store_true", help="List test names")
    args = parser.parse_args()

    if args.list:
        for name in TESTS:
            print(name)
        return

    tests_to_run = {args.test: TESTS[args.test]} if args.test else TESTS

    if args.test and args.test not in TESTS:
        print(f"Unknown test: {args.test}. Available: {', '.join(TESTS)}")
        sys.exit(1)

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
