# CLAUDE.md — murderbot local LLM workspace

## Context efficiency — READ THIS FIRST

This session runs on **llama.cpp with a fixed context window**. Every token is permanent
until compaction. A single large file read can crash the session with a 400 Bad Request.

### Hard rules on file reading — NO EXCEPTIONS

- **Never `cat` or read a file > 150 lines.** Use targeted reads instead.
- **Large files must be read in chunks**: use `sed -n '1,100p' file` then `sed -n '101,200p'` etc.
- **Before reading ANY file**, run `wc -l file` to check its size.
- **For config/code files**: read only the function or section you need — use `grep -n "pattern" file` to find line numbers first, then `sed -n 'N,Mp'` to read just that block.
- **Log files**: `tail -50`, `grep -n ERROR`, or `grep -n "pattern" | head -20` only. Never full reads.
- **Never dump directory trees with recursive ls** — use `ls -1` and descend one level at a time.

### Why this matters

A 47KB file read = ~13,000 Qwen3 tokens. The entire session budget is ~85,000 tokens.
One large read consumes 15% of the session and can push the next request over the limit.

### Tool call discipline

- One tool call at a time. Wait for result, then decide next step.
- If a command output is long, filter it: `cmd | head -50` or `cmd | grep pattern`.
- Do NOT speculatively read files you might need — read them only when you actually need them.
- Prefer `grep -rn "pattern" dir` over reading multiple files looking for something.

### Writing / editing

- Prefer **targeted edits** — replace a specific function or block, not the whole file.
- If writing a large file, write sections incrementally, not all at once.
- Verify changes with `diff` or targeted `sed -n` reads, not full file dumps.

### Response length

- Keep prose responses short — if it fits in 3 sentences, use 3 sentences.
- Don't repeat file content back — reference it as `filename:line_range`.
- For status updates ("done", "fixed", "running"), one line is enough.

## Environment

- **Host:** murderbot — NVIDIA RTX 4000 Blackwell, 24 GB VRAM, Linux
- **Inference:** llama.cpp llama-server on `http://127.0.0.1:8088`
- **Model:** Qwen3.6-35B-A3B (--reasoning off, no thinking mode)
- **Context window:** 32768 tokens (opencode limit; server supports 131072)
- **Max output:** 4096 tokens per turn
