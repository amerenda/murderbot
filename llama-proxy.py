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

With this proxy, opencode can use the full server context (131072 or 200K+) without
artificial reduction. The proxy is the overflow safety net.

Usage: python3 llama-proxy.py [--upstream http://127.0.0.1:8088] [--port 8089]
       (also auto-started by start-opencode-stable.sh)
"""

import json
import sys
import http.client
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_UPSTREAM = 'http://127.0.0.1:8088'
DEFAULT_PORT     = 8089
MAX_RETRIES      = 10     # strip rounds before giving up
STREAM_CHUNK     = 512    # bytes per read on streaming responses (small = low latency)


def parse_args():
    p = argparse.ArgumentParser(description='llama-proxy: overflow-recovery proxy')
    p.add_argument('--upstream', default=DEFAULT_UPSTREAM,
                   help=f'llama-server base URL (default: {DEFAULT_UPSTREAM})')
    p.add_argument('--port', type=int, default=DEFAULT_PORT,
                   help=f'Port to listen on (default: {DEFAULT_PORT})')
    p.add_argument('--quiet', action='store_true',
                   help='Suppress all proxy log output')
    return p.parse_args()


# ── Message stripping ─────────────────────────────────────────────────────────

def _msg_size(msg):
    """Rough byte size of a message's content."""
    c = msg.get('content') or ''
    tc = msg.get('tool_calls') or []
    return len(str(c)) + len(str(tc))


def strip_messages(messages, strip_n=2):
    """
    Remove strip_n messages from the strippable window (index 1 through len-5).
    Strategy:
      1. First pass: find and remove the LARGEST tool messages (file reads, etc.)
         and the assistant message that called them.
      2. Second pass: remove oldest non-system messages in order.
    Preserves: messages[0] (system) and the last 4 messages (current turn).
    Returns: (removed_count, bytes_saved)
    """
    protected_tail = 4
    removed = 0
    saved = 0

    # Pass 1: strip largest tool messages + their preceding assistant
    while removed < strip_n:
        best_i  = -1
        best_sz = 0
        # Find the largest tool message in the strippable window
        for i in range(1, len(messages) - protected_tail):
            if messages[i].get('role') == 'tool':
                sz = _msg_size(messages[i])
                if sz > best_sz:
                    best_sz = sz
                    best_i  = i

        if best_i == -1:
            break  # no more tool messages

        # Remove the tool message
        saved += best_sz
        messages.pop(best_i)
        removed += 1

        # Remove the preceding assistant message if it's in the strippable window
        prev = best_i - 1  # index shifted down by the pop above
        if (removed < strip_n and
                prev >= 1 and prev < len(messages) - protected_tail and
                messages[prev].get('role') == 'assistant'):
            saved += _msg_size(messages[prev])
            messages.pop(prev)
            removed += 1

    # Pass 2: if we still need to strip more, remove oldest messages
    while removed < strip_n and len(messages) > protected_tail + 1:
        if messages[1].get('role') in ('user', 'assistant', 'tool'):
            saved += _msg_size(messages[1])
            messages.pop(1)
            removed += 1
        else:
            break

    return removed, saved


# ── Proxy handler ─────────────────────────────────────────────────────────────

class ProxyHandler(BaseHTTPRequestHandler):
    _quiet = False

    def log_message(self, fmt, *args):
        pass  # silence BaseHTTPServer's default access log

    def _log(self, msg):
        if not ProxyHandler._quiet:
            print(f'[proxy] {msg}', file=sys.stderr, flush=True)

    # ─── HTTP verb handlers ───────────────────────────────────────────────────

    def do_GET(self):
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

        data = None
        if is_chat:
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                is_chat = False

        send_body = body
        attempt   = 0

        while True:
            if data is not None:
                send_body = json.dumps(data).encode('utf-8')

            # Build upstream request headers
            hdrs = {k: v for k, v in self.headers.items()
                    if k.lower() not in ('host', 'connection', 'content-length')}
            hdrs['Connection']     = 'close'
            hdrs['Content-Length'] = str(len(send_body)) if send_body else '0'

            # Parse upstream host/port from args
            upstream = ProxyHandler._upstream
            host = upstream['host']
            port = upstream['port']

            conn = http.client.HTTPConnection(host, port, timeout=300)
            try:
                conn.request(self.command, self.path, body=send_body, headers=hdrs)
                resp = conn.getresponse()

                # ── 400 overflow: strip + retry ───────────────────────────
                if resp.status == 400 and is_chat and attempt < MAX_RETRIES:
                    resp.read()
                    conn.close()
                    msgs = data.get('messages', [])
                    n_stripped, n_bytes = strip_messages(msgs, strip_n=2)
                    if n_stripped == 0:
                        self._log('400 — nothing left to strip, forwarding error')
                        self._send_err(400, 'context overflow: no strippable messages remain')
                        return
                    attempt += 1
                    self._log(
                        f'400 overflow → stripped {n_stripped} msgs '
                        f'({n_bytes:,} bytes, retry {attempt}/{MAX_RETRIES}, '
                        f'{len(msgs)} msgs remain)'
                    )
                    continue

                # ── Forward response ──────────────────────────────────────
                self.send_response(resp.status)
                self.close_connection = True
                self.send_header('Connection', 'close')

                for k, v in resp.getheaders():
                    if k.lower() not in ('connection', 'transfer-encoding', 'keep-alive'):
                        self.send_header(k, v)
                self.end_headers()

                # Stream body (small chunks for SSE latency)
                while True:
                    chunk = resp.read(STREAM_CHUNK)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()

                conn.close()
                return

            except ConnectionRefusedError:
                self._log(f'upstream {host}:{port} refused connection — is llama-server running?')
                conn.close()
                self._send_err(503, f'upstream {host}:{port} refused connection')
                return

            except Exception as e:
                self._log(f'upstream error: {e}')
                conn.close()
                self._send_err(503, str(e))
                return

    def _send_err(self, code, message):
        body = json.dumps({'error': {'message': message, 'type': 'proxy_error'}}).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.close_connection = True
        self.end_headers()
        self.wfile.write(body)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Parse upstream URL
    from urllib.parse import urlparse
    up = urlparse(args.upstream)
    ProxyHandler._upstream = {
        'host': up.hostname or '127.0.0.1',
        'port': up.port or 8088,
    }
    ProxyHandler._quiet = args.quiet

    server = HTTPServer(('127.0.0.1', args.port), ProxyHandler)
    print(
        f'[proxy] :{args.port} → {args.upstream}  '
        f'(overflow recovery: strip+retry up to {MAX_RETRIES}×)',
        file=sys.stderr, flush=True
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('[proxy] shutting down', file=sys.stderr)


if __name__ == '__main__':
    main()
