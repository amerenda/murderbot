# Tool Calling Tuning Log

Tracking incremental improvements to llama.cpp + Qwen3.6-35B-A3B tool calling reliability.
Each change is tested in isolation before the next is applied.

Model: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`  
Engine: llama.cpp (llama-server)  
Client: opencode (build agent mode)

---

## Completed

### 1. froggeric v20 chat template
**Commit:** `f21ca85`  
**Date:** 2026-06-05  
**Change:** Replaced the GGUF-embedded Jinja template with `froggeric-v20.jinja`.

**Why:** The official Qwen3.6 template has two llama.cpp-incompatible Jinja filters (`|items`, `|safe`) that are Python-only. In llama.cpp's minja engine these silently corrupt tool call rendering. froggeric v20 rewrites those patterns with portable equivalents and adds three features:
- `auto_disable_thinking_with_tools: true` — suppresses `<think>` blocks during tool calls only, preserving thinking for non-tool responses
- `preserve_thinking: false` — strips past `<think>` blocks from history, preventing empty-think poisoning (model aborting tool calls on seeing `<think></think>` in context)
- `max_tool_response_chars` — truncates tool responses at Jinja render time before they hit the context window

Also fixed: KV cache invalidation caused by a spacing mismatch in the old template (was forcing full re-prompt every turn; f_keep went from ~0% to 98.2% after fix).

**Notes:** Smoke test passed immediately — `finish_reason: tool_calls`, correct name/args, no stray text, no reasoning leak during tool call. KV cache reuse confirmed at f_keep=0.982.

---

### 2. Reduce max_tool_response_chars 8000 → 3000
**Commit:** `2924276`  
**Date:** 2026-06-05  
**Change:** `--chat-template-kwargs` updated in `start-opencode-stable.sh`.

**Why:** A single large file read (e.g. 47KB) would previously expand to 8000 chars in the tool response, filling ~6K tokens per tool call. Over a long session this was the primary cause of HTTP 400 context overflow errors. 3000 chars still gives enough content for most reads while halving the per-call token cost.

**Notes:** Takes effect on next server restart. Not yet tested in a full session.

---

### 3. Template renamed to froggeric-v20.jinja
**Commit:** `2924276`  
**Date:** 2026-06-05  
**Change:** `git mv froggeric-chat-template.jinja froggeric-v20.jinja`, script updated to match.

**Why:** Clarity — the filename now encodes the version so future upgrades are unambiguous.

**Notes:** Cosmetic only. No functional change.

---

## Pending

Changes to apply and test one at a time, in order.

---

### 4. Qwen3 recommended sampling params
**Status:** Pending  
**Planned change:** Add to `SERVER_ARGS` in `start-opencode-stable.sh`:
```
--temp 0.6 --top-k 20 --top-p 0.95 --min-p 0.05
```

**Why:** Qwen3's published non-thinking agentic mode recommendation. Currently no sampling params are set explicitly — llama.cpp defaults apply (temp=0.8, top_k=40, top_p=0.95). Lower temp + top_k=20 makes the model more deterministic, which reduces the format drift that produces malformed tool calls (`<tool_call>write>` instead of `<tool_call>\n<function=write>`). The `min_p=0.05` cutoff eliminates very-low-probability tokens that are the source of one-off format corruption.

**Expected effect:** Fewer malformed tool call outputs; no change in speed.

**Notes:** *(fill in after testing)*

---

### 5. Larger ubatch for long-context prompt processing
**Status:** Pending  
**Planned change:** Add `--ubatch 1024` to `SERVER_ARGS` (current default is 512).

**Why:** At 131K context, prompt re-evaluation after KV cache misses processes thousands of tokens. The micro-batch size controls throughput of that re-eval. Doubling it from 512 → 1024 should halve prompt eval time at long context with no effect on generation quality.

**Expected effect:** Faster prompt eval (pp tokens/s), no change in tool call format.

**Notes:** *(fill in after testing)*

---

## Reference

### Current server args (as of commit 2924276)
```bash
--jinja
--chat-template-file "$SCRIPT_DIR/froggeric-v20.jinja"
--chat-template-kwargs '{"auto_disable_thinking_with_tools": true, "max_tool_response_chars": 3000, "preserve_thinking": false}'
--repeat-penalty 1.1
-fa 1
--ctk q4_0 --ctv q4_0
```
No explicit temp/top-k/top-p/min-p (using llama.cpp defaults).

### Known non-issues
- **`<tool_call>write>` model drift** — intermittent, ~1 occurrence per 200 steps in long sessions. Not fixable at template level (both official and froggeric use same format). Addressed by sampling params (change #4).
- **HTTP 400 overflow** — was triggered by 8000-char tool responses in long sessions. Addressed by change #2. Proxy re-enable is last resort if it recurs.
- **peg-native format** — confirmed correct for this model. Qwen3.6 uses `<tool_call>/<function=name>` XML format, not JSON. peg-native auto-detects from template.
