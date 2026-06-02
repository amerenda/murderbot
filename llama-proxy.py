#!/usr/bin/env python3
"""
llama-proxy.py — context-overflow recovery proxy for llama-server / opencode

Problem: opencode uses the GPT tiktoken tokenizer for all @ai-sdk/openai-compatible
providers. Qwen3's XML tool-call format tokenizes 2-4x heavier than GPT estimates,
AND the Jinja chat template injects the full tools array into the system message at
render time — those tokens are invisible to opencode. Result: opencode sends requests
that exceed the server's CTX limit → instant 400 Bad Request.

Solution: this proxy sits between opencode (port LISTEN_PORT) and llama-server
(port UPSTREAM_PORT). When a /v1/chat/completions POST returns 400, it strips the
oldest/largest messages and retries transparently, up to MAX_RETRIES times.

Performance design:
  - Happy path (no overflow): zero JSON parsing. Raw bytes forwarded directly.
  - Overflow path: JSON parsed once on first 400, stripped in-place, re-serialized.
  - Streaming: reads up to STREAM_CHUNK bytes; for SSE (small events) this returns
    immediately with whatever is available — no buffering penalty.
  - ThreadingHTTPServer: each connection gets its own thread, no head-of-line blocking.

Metrics:
  GET /metrics on the proxy port returns Prometheus text-format metrics covering
  overflow recovery activity, request latency, upstream errors, and context size.
  Add a scrape target in prometheus.yml pointing at this host:port/metrics.

Usage: python3 llama-proxy.py [--upstream http://127.0.0.1:8088] [--port 8089]
       (also auto-started by start-opencode-stable.sh)
"""

import json
import sys
import http.client
import argparse
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_UPSTREAM    = 'http://127.0.0.1:8088'
DEFAULT_PORT        = 8089
MAX_RETRIES         = 200    # strip rounds before giving up
STREAM_CHUNK        = 4096   # bytes per read
MAX_TOOL_CHARS      = 2000   # truncate tool message content to this many chars
                             # Research sessions accumulate 100+ tool results; at 112KB each they
                             # push prompt eval to 40-60s/request. 2KB ≈ 500 tokens is enough for
                             # the model to understand what the tool returned without burning context.


# ── Metrics ───────────────────────────────────────────────────────────────────

class _Metrics:
    """Thread-safe Prometheus-format metrics for the overflow-recovery proxy."""

    # Latency histogram bucket boundaries (seconds, end-to-end per chat request)
    LATENCY_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)

    # Overflow depth histogram (strip rounds needed per overflowing request)
    DEPTH_BUCKETS = (1, 2, 3, 5, 7, 10, 15, 25)

    def __init__(self):
        self._lock = threading.Lock()

        # ── Counters ──────────────────────────────────────────────────────────
        # Request outcomes (chat completions only)
        self.requests_success            = 0  # passed through with no overflow
        self.requests_overflow_recovered = 0  # had ≥1 strip round, then succeeded
        self.requests_error              = 0  # ended in upstream error / broken pipe
        self.requests_non_chat           = 0  # passthrough for non-chat paths

        # Overflow strip activity
        self.overflow_events_total   = 0  # total 400s received (one per strip round)
        self.messages_stripped_total = 0  # total messages removed
        self.bytes_stripped_total    = 0  # total bytes of content removed

        # Upstream errors by type
        self.errors_broken_pipe  = 0
        self.errors_conn_refused = 0
        self.errors_other        = 0

        # ── Gauges ────────────────────────────────────────────────────────────
        self.active_requests         = 0
        self.last_messages_remaining = 0   # post-strip message count from last overflow request
        self.last_strip_rounds       = 0   # strip rounds from last overflow request

        # ── Latency histogram ─────────────────────────────────────────────────
        # _lat_inc[i] = requests with duration in (BUCKETS[i-1], BUCKETS[i]]
        self._lat_inc   = [0] * len(self.LATENCY_BUCKETS)
        self._lat_above = 0   # above last bucket
        self._lat_sum   = 0.0
        self._lat_count = 0

        # ── Overflow depth histogram ───────────────────────────────────────────
        self._dep_inc   = [0] * len(self.DEPTH_BUCKETS)
        self._dep_above = 0
        self._dep_count = 0

    # ── Mutation helpers (called from request threads) ────────────────────────

    def inc_active(self):
        with self._lock:
            self.active_requests += 1

    def dec_active(self):
        with self._lock:
            self.active_requests -= 1

    def record_overflow_strip(self, n_stripped, n_bytes, msgs_remaining):
        """Call once per strip round."""
        with self._lock:
            self.overflow_events_total   += 1
            self.messages_stripped_total += n_stripped
            self.bytes_stripped_total    += n_bytes
            self.last_messages_remaining  = msgs_remaining

    def record_request(self, *, overflow_rounds, duration_s, error_type=None):
        """Call once per completed (or errored) chat completions request."""
        with self._lock:
            # Outcome
            if error_type == 'broken_pipe':
                self.errors_broken_pipe += 1
                self.requests_error += 1
            elif error_type == 'conn_refused':
                self.errors_conn_refused += 1
                self.requests_error += 1
            elif error_type:
                self.errors_other += 1
                self.requests_error += 1
            elif overflow_rounds == 0:
                self.requests_success += 1
            else:
                self.requests_overflow_recovered += 1
                self.last_strip_rounds = overflow_rounds

            # Latency histogram
            self._lat_sum   += duration_s
            self._lat_count += 1
            placed = False
            for i, b in enumerate(self.LATENCY_BUCKETS):
                if duration_s <= b:
                    self._lat_inc[i] += 1
                    placed = True
                    break
            if not placed:
                self._lat_above += 1

            # Depth histogram (only for overflow requests)
            if overflow_rounds > 0:
                self._dep_count += 1
                placed = False
                for i, b in enumerate(self.DEPTH_BUCKETS):
                    if overflow_rounds <= b:
                        self._dep_inc[i] += 1
                        placed = True
                        break
                if not placed:
                    self._dep_above += 1

    def record_non_chat(self):
        with self._lock:
            self.requests_non_chat += 1

    # ── Prometheus text rendering ─────────────────────────────────────────────

    def render(self):
        with self._lock:
            lines = []

            def metric(name, help_text, type_, *samples):
                lines.append(f'# HELP {name} {help_text}')
                lines.append(f'# TYPE {name} {type_}')
                for lbl, val in samples:
                    if lbl:
                        lines.append(f'{name}{{{lbl}}} {val}')
                    else:
                        lines.append(f'{name} {val}')

            # Active requests
            metric('proxy_active_requests',
                   'Number of in-flight proxy requests',
                   'gauge', ('', self.active_requests))

            # Request outcomes
            lines.append('# HELP proxy_requests_total Total proxy requests completed, by outcome')
            lines.append('# TYPE proxy_requests_total counter')
            lines.append(f'proxy_requests_total{{outcome="success"}} {self.requests_success}')
            lines.append(f'proxy_requests_total{{outcome="overflow_recovered"}} {self.requests_overflow_recovered}')
            lines.append(f'proxy_requests_total{{outcome="error"}} {self.requests_error}')
            lines.append(f'proxy_requests_total{{outcome="non_chat"}} {self.requests_non_chat}')

            # Overflow strip counters
            metric('proxy_overflow_events_total',
                   'Total 400 overflow responses from upstream (one per strip round)',
                   'counter', ('', self.overflow_events_total))
            metric('proxy_messages_stripped_total',
                   'Total messages removed by overflow strip-and-retry',
                   'counter', ('', self.messages_stripped_total))
            metric('proxy_bytes_stripped_total',
                   'Total content bytes removed by overflow strip-and-retry',
                   'counter', ('', self.bytes_stripped_total))

            # Last-seen gauges (useful for alerting on stuck sessions)
            metric('proxy_last_messages_remaining',
                   'Message count after stripping in the most recent overflow request',
                   'gauge', ('', self.last_messages_remaining))
            metric('proxy_last_strip_rounds',
                   'Strip rounds needed by the most recent overflow request',
                   'gauge', ('', self.last_strip_rounds))

            # Upstream errors
            lines.append('# HELP proxy_upstream_errors_total Upstream errors by type')
            lines.append('# TYPE proxy_upstream_errors_total counter')
            lines.append(f'proxy_upstream_errors_total{{type="broken_pipe"}} {self.errors_broken_pipe}')
            lines.append(f'proxy_upstream_errors_total{{type="conn_refused"}} {self.errors_conn_refused}')
            lines.append(f'proxy_upstream_errors_total{{type="other"}} {self.errors_other}')

            # Latency histogram
            lines.append('# HELP proxy_request_duration_seconds End-to-end chat completions latency (s)')
            lines.append('# TYPE proxy_request_duration_seconds histogram')
            cum = 0
            for i, b in enumerate(self.LATENCY_BUCKETS):
                cum += self._lat_inc[i]
                lines.append(f'proxy_request_duration_seconds_bucket{{le="{b}"}} {cum}')
            cum += self._lat_above
            lines.append(f'proxy_request_duration_seconds_bucket{{le="+Inf"}} {cum}')
            lines.append(f'proxy_request_duration_seconds_sum {self._lat_sum:.3f}')
            lines.append(f'proxy_request_duration_seconds_count {self._lat_count}')

            # Strip-round depth histogram
            lines.append('# HELP proxy_overflow_strip_rounds Strip rounds needed per request that triggered overflow')
            lines.append('# TYPE proxy_overflow_strip_rounds histogram')
            cum = 0
            for i, b in enumerate(self.DEPTH_BUCKETS):
                cum += self._dep_inc[i]
                lines.append(f'proxy_overflow_strip_rounds_bucket{{le="{b}"}} {cum}')
            cum += self._dep_above
            lines.append(f'proxy_overflow_strip_rounds_bucket{{le="+Inf"}} {cum}')
            lines.append(f'proxy_overflow_strip_rounds_count {self._dep_count}')
            if self._dep_count > 0:
                dep_sum = sum(b * self._dep_inc[i] for i, b in enumerate(self.DEPTH_BUCKETS))
                # approximate sum using bucket midpoints; last bucket uses boundary
                lines.append(f'proxy_overflow_strip_rounds_sum {dep_sum}')
            else:
                lines.append('proxy_overflow_strip_rounds_sum 0')

            return '\n'.join(lines) + '\n'


METRICS = _Metrics()


# ── Tool message truncation ───────────────────────────────────────────────────

def truncate_tool_messages(data, max_chars=MAX_TOOL_CHARS):
    """
    Cap tool message content at max_chars characters in-place.
    Returns (n_truncated, bytes_saved) — both 0 if nothing changed.

    Large tool results (URL fetches, search dumps) are the #1 cause of slow prompt
    eval (40-60s/request at 131K tokens). Capping at 2KB keeps the context lean
    while preserving enough for the model to understand what the tool returned.
    Applied on every request, before overflow stripping, so it runs on the happy path.
    """
    messages = data.get('messages', [])
    n_truncated = 0
    bytes_saved = 0

    for msg in messages:
        if msg.get('role') != 'tool':
            continue
        content = msg.get('content')
        if not content:
            continue

        if isinstance(content, str):
            if len(content) > max_chars:
                bytes_saved += len(content) - max_chars
                msg['content'] = content[:max_chars] + f'…[truncated, {len(content):,} chars total]'
                n_truncated += 1
        elif isinstance(content, list):
            # OpenAI content-parts format: [{type: "text", text: "..."}, ...]
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'text':
                    text = part.get('text', '')
                    if len(text) > max_chars:
                        bytes_saved += len(text) - max_chars
                        part['text'] = text[:max_chars] + f'…[truncated, {len(text):,} chars total]'
                        n_truncated += 1

    return n_truncated, bytes_saved


# ── Message stripping ─────────────────────────────────────────────────────────

def _msg_size(msg):
    """Approximate JSON byte size of a message's content fields."""
    c  = msg.get('content') or ''
    tc = msg.get('tool_calls') or []
    c_len  = len(c) if isinstance(c, str) else len(json.dumps(c))
    tc_len = len(json.dumps(tc)) if tc else 0
    return c_len + tc_len


def strip_messages(messages, strip_n=2):
    """
    Remove strip_n messages from the strippable window (index 1 through len-5).
    Strategy:
      1. Remove the LARGEST tool messages + their preceding assistant calls.
      2. Fall back to oldest non-system messages if more stripping is needed.
    Preserves: messages[0] (system) and the last 4 messages (current turn).
    Returns: (removed_count, bytes_saved)
    """
    protected_tail = 4
    limit = len(messages) - protected_tail  # strippable window: [1, limit)
    saved = 0

    # Pre-sort strippable tool messages by size (largest first) — one O(n log n) pass.
    tool_by_size = sorted(
        (i for i in range(1, limit) if messages[i].get('role') == 'tool'),
        key=lambda i: _msg_size(messages[i]),
        reverse=True,
    )

    # Collect indices to remove: largest tools + their preceding assistant calls.
    to_remove = set()
    for i in tool_by_size:
        if len(to_remove) >= strip_n:
            break
        saved += _msg_size(messages[i])
        to_remove.add(i)
        prev = i - 1
        if (len(to_remove) < strip_n
                and prev >= 1
                and prev < limit
                and messages[prev].get('role') == 'assistant'):
            saved += _msg_size(messages[prev])
            to_remove.add(prev)

    # Pass 2: oldest non-system messages if more stripping needed.
    j = 1
    while len(to_remove) < strip_n and j < limit:
        if j not in to_remove and messages[j].get('role') in ('user', 'assistant', 'tool'):
            saved += _msg_size(messages[j])
            to_remove.add(j)
        j += 1

    if not to_remove:
        return 0, 0

    # Pop in descending index order — no index drift.
    for i in sorted(to_remove, reverse=True):
        messages.pop(i)

    return len(to_remove), saved


# ── Proxy handler ─────────────────────────────────────────────────────────────

class ProxyHandler(BaseHTTPRequestHandler):
    _quiet         = False
    _upstream      = {'host': '127.0.0.1', 'port': 8088}
    _max_tool_chars = MAX_TOOL_CHARS

    def log_message(self, fmt, *args):
        pass  # silence BaseHTTPServer's default access log

    def _log(self, msg):
        if not ProxyHandler._quiet:
            print(f'[proxy] {msg}', file=sys.stderr, flush=True)

    # ─── HTTP verb handlers ───────────────────────────────────────────────────

    def do_GET(self):
        if self.path.rstrip('/') == '/metrics':
            body = METRICS.render().encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Connection', 'close')
            self.close_connection = True
            self.end_headers()
            self.wfile.write(body)
            return
        self._handle(None)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        self._handle(self.rfile.read(n) if n else b'')

    def do_DELETE(self):
        self._handle(None)

    # ─── Core proxy logic ─────────────────────────────────────────────────────

    def _handle(self, body):
        ct = self.headers.get('Content-Type', '')
        is_chat = (
            self.path.startswith('/v1/chat/completions')
            and body
            and 'application/json' in ct
        )

        if not is_chat:
            METRICS.record_non_chat()

        t0 = time.time()
        METRICS.inc_active()

        # ── Pre-processing: truncate large tool messages ─────────────────────
        # Always applied before sending, not just on overflow. Tool results from
        # web searches / URL fetches can be 50-100KB each. With 100+ messages
        # accumulated, prompt eval hits 40-60s/request. Capping at MAX_TOOL_CHARS
        # keeps context small and prompt eval fast without losing useful info.
        if is_chat and ProxyHandler._max_tool_chars > 0:
            try:
                pre_data = json.loads(body)
                n_trunc, saved = truncate_tool_messages(pre_data, ProxyHandler._max_tool_chars)
                if n_trunc:
                    body = json.dumps(pre_data).encode('utf-8')
                    self._log(f'truncated {n_trunc} tool msgs ({saved:,} bytes saved)')
            except (json.JSONDecodeError, Exception):
                pass  # malformed body — let it through, overflow handler will catch it

        # ── Lazy parse: data stays None until we actually get a 400 ──────────
        # Happy path forwards raw bytes with zero JSON overhead (after truncation).
        data        = None
        send_body   = body
        attempt     = 0   # overflow strip rounds
        conn_retries = 0  # upstream connection reset retries (max 1)

        # Pre-build the header dict once (same for all attempts)
        hdrs = {k: v for k, v in self.headers.items()
                if k.lower() not in ('host', 'connection', 'content-length')}
        hdrs['Connection'] = 'close'

        upstream = ProxyHandler._upstream
        host     = upstream['host']
        port     = upstream['port']

        try:
            while True:
                hdrs['Content-Length'] = str(len(send_body)) if send_body else '0'

                conn = http.client.HTTPConnection(host, port, timeout=300)
                try:
                    conn.request(self.command, self.path, body=send_body, headers=hdrs)
                    resp = conn.getresponse()

                    # ── 400 overflow: parse (lazily), strip, retry ────────────
                    if resp.status == 400 and is_chat and attempt < MAX_RETRIES:
                        resp.read()
                        conn.close()

                        # First 400: parse the body for the first time
                        if data is None:
                            try:
                                data = json.loads(body)
                            except (json.JSONDecodeError, Exception):
                                self._send_err(400, 'context overflow and request body is not valid JSON')
                                if is_chat:
                                    METRICS.record_request(overflow_rounds=attempt,
                                                           duration_s=time.time() - t0,
                                                           error_type='other')
                                return

                        msgs    = data.get('messages', [])
                        strip_n = max(4, len(msgs) // 20)  # ~5% of remaining; min 4 to cut retry round-trips
                        n_stripped, n_bytes = strip_messages(msgs, strip_n=strip_n)

                        if n_stripped == 0:
                            self._log('400 — nothing left to strip, forwarding error')
                            self._send_err(400, 'context overflow: no strippable messages remain')
                            if is_chat:
                                METRICS.record_request(overflow_rounds=attempt,
                                                       duration_s=time.time() - t0,
                                                       error_type='other')
                            return

                        attempt   += 1
                        send_body  = json.dumps(data).encode('utf-8')
                        METRICS.record_overflow_strip(n_stripped, n_bytes, len(msgs))
                        self._log(
                            f'400 overflow → stripped {n_stripped} msgs '
                            f'({n_bytes:,} bytes, retry {attempt}/{MAX_RETRIES}, '
                            f'{len(msgs)} msgs remain)'
                        )
                        continue

                    # ── Forward response ──────────────────────────────────────
                    try:
                        self.send_response(resp.status)
                        self.close_connection = True
                        self.send_header('Connection', 'close')

                        for k, v in resp.getheaders():
                            if k.lower() not in ('connection', 'transfer-encoding', 'keep-alive'):
                                self.send_header(k, v)
                        self.end_headers()

                        while True:
                            chunk = resp.read(STREAM_CHUNK)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        # Client disconnected mid-response — normal cancellation, not an error.
                        pass

                    conn.close()
                    if is_chat:
                        METRICS.record_request(overflow_rounds=attempt,
                                               duration_s=time.time() - t0)
                    return

                except ConnectionRefusedError:
                    self._log(f'upstream {host}:{port} refused — is llama-server running?')
                    conn.close()
                    self._send_err(503, f'upstream {host}:{port} refused connection')
                    if is_chat:
                        METRICS.record_request(overflow_rounds=attempt,
                                               duration_s=time.time() - t0,
                                               error_type='conn_refused')
                    return

                except (BrokenPipeError, ConnectionResetError) as e:
                    # Server closed the connection — retry once before giving up.
                    conn.close()
                    if conn_retries == 0:
                        conn_retries += 1
                        self._log(f'upstream reset connection, retrying: {e}')
                        time.sleep(0.3)
                        continue
                    self._log(f'upstream reset connection after retry: {e}')
                    self._send_err(503, str(e))
                    if is_chat:
                        METRICS.record_request(overflow_rounds=attempt,
                                               duration_s=time.time() - t0,
                                               error_type='conn_reset')
                    return

                except Exception as e:
                    err = str(e)
                    self._log(f'upstream error: {e}')
                    conn.close()
                    self._send_err(503, err)
                    if is_chat:
                        METRICS.record_request(overflow_rounds=attempt,
                                               duration_s=time.time() - t0,
                                               error_type='other')
                    return

        finally:
            METRICS.dec_active()

    def _send_err(self, code, message):
        body = json.dumps({'error': {'message': message, 'type': 'proxy_error'}}).encode()
        try:
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Connection', 'close')
            self.close_connection = True
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client already gone


# ── Threaded server ───────────────────────────────────────────────────────────

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """One thread per connection — no head-of-line blocking."""
    daemon_threads = True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    from urllib.parse import urlparse
    up = urlparse(args.upstream)
    ProxyHandler._upstream = {
        'host': up.hostname or '127.0.0.1',
        'port': up.port or 8088,
    }
    ProxyHandler._quiet          = args.quiet
    ProxyHandler._max_tool_chars = args.max_tool_chars

    server = ThreadedHTTPServer(('0.0.0.0', args.port), ProxyHandler)
    trunc_info = f'tool truncation: {args.max_tool_chars} chars' if args.max_tool_chars > 0 else 'tool truncation: disabled'
    print(
        f'[proxy] :{args.port} → {args.upstream}  '
        f'(overflow recovery: strip+retry up to {MAX_RETRIES}×, {trunc_info}, metrics at :{args.port}/metrics)',
        file=sys.stderr, flush=True
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('[proxy] shutting down', file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(description='llama-proxy: overflow-recovery proxy')
    p.add_argument('--upstream', default=DEFAULT_UPSTREAM,
                   help=f'llama-server base URL (default: {DEFAULT_UPSTREAM})')
    p.add_argument('--port', type=int, default=DEFAULT_PORT,
                   help=f'Port to listen on (default: {DEFAULT_PORT})')
    p.add_argument('--quiet', action='store_true',
                   help='Suppress all proxy log output')
    p.add_argument('--max-tool-chars', type=int, default=MAX_TOOL_CHARS,
                   help=f'Truncate tool message content to this many chars (0=disable, default: {MAX_TOOL_CHARS})')
    return p.parse_args()


if __name__ == '__main__':
    main()
