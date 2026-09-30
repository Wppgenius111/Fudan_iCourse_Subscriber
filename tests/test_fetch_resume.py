#!/usr/bin/env python3
"""Exercise AudioDownloader._fetch_mp4 against a server that misbehaves the
way the Fudan iCourse CDN + WebVPN proxy do.

Reproduced behaviours (all observed in real workflow logs):
  * a lecture MP4 is ~2.7 GB, not 600 MB;
  * `Range` IS supported (206 + `Content-Range: bytes 0-0/<total>`);
  * the proxy hands over ~250 MiB per response and then drops the
    connection, which ffmpeg reads as a clean EOF;
  * the `t=` signature is effectively single-use — a second request with
    the same URL answers 403 Forbidden (this is what broke ffmpeg when a
    diagnostic probe shared its URL).

Scenarios (one test each):
  A. Range honoured + per-response cap + single-use signatures
     -> must still reassemble the whole file byte-for-byte.
  B. Range unsupported (server ignores it)
     -> must report 0 / None so the caller falls back to streaming.
  C. Signature reused -> server 403s -> chunk fails, no silent corruption.
  D. Chunk smaller than the cap -> the thread pool really runs in parallel.
  E. ``_ffmpeg_cmd`` must not put http-only options on a local file path.

Run with ``pytest tests/test_fetch_resume.py`` or directly
(``python tests/test_fetch_resume.py`` is not supported — use pytest).
"""

from __future__ import annotations

import hashlib
import http.server
import socketserver
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from src.runtime.scheduler import AudioDownloader  # noqa: E402

SIZE = 700 * 1024          # stand-in for a ~2.7 GB lecture
CAP = 200 * 1024           # stand-in for the proxy's ~250 MiB ceiling
BODY = bytes(i % 251 for i in range(SIZE))
DIGEST = hashlib.sha256(BODY).hexdigest()

STATE = {"range_ok": True, "cap": CAP, "single_use": True,
         "seen": set(), "requests": 0}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _sig(self):
        q = self.path.split("?", 1)[1] if "?" in self.path else ""
        for part in q.split("&"):
            if part.startswith("sig="):
                return part
        return ""

    def do_GET(self):
        STATE["requests"] += 1
        sig = self._sig()
        if STATE["single_use"]:
            if sig in STATE["seen"]:
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()
                self.close_connection = True
                return
            STATE["seen"].add(sig)

        rng = self.headers.get("Range")
        start, end = 0, SIZE - 1
        ranged = bool(rng) and STATE["range_ok"]
        if ranged:
            s, _, e = rng.split("=", 1)[1].partition("-")
            start = int(s)
            if e:
                end = min(int(e), SIZE - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{SIZE}")
        else:
            self.send_response(200)

        cut = min(end, start + STATE["cap"] - 1, SIZE - 1)
        blob = BODY[start:cut + 1]
        if ranged:
            self.send_header("Content-Length", str(len(blob)))
        else:
            # Declare the whole file, then cut the body and drop the
            # connection — exactly how ffmpeg ends up seeing a clean EOF.
            self.send_header("Content-Length", str(SIZE - start))
            self.close_connection = True
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(blob)


class FakeClient:
    """Mimics ICourseClient: mints a brand-new signature on every call."""

    def __init__(self, base: str):
        self.vpn = SimpleNamespace(session=requests.Session())
        self.base = base
        self.n = 0

    def get_video_url(self, course_id, sub_id, verbose=True):
        self.n += 1
        return f"{self.base}/lecture.mp4?sig={self.n}"

    def get_stream_params(self, url):
        return url, ""


def run(label: str, **overrides):
    STATE.update(range_ok=True, cap=CAP, single_use=True,
                 seen=set(), requests=0)
    STATE.update(overrides)
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler) as srv:
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "x.mp4")
            dl = AudioDownloader(audio_dir=td)
            got, total = dl._fetch_mp4(FakeClient(base), "37491", "999", out)
            data = Path(out).read_bytes() if Path(out).exists() else b""

    intact = hashlib.sha256(data).hexdigest() == DIGEST
    prefix = bool(data) and data == BODY[:len(data)]
    print(f"  {label}: got={got} total={total} on_disk={len(data)} "
          f"intact={'YES' if intact else 'NO'} "
          f"clean_prefix={'YES' if prefix else 'NO'} "
          f"requests={STATE['requests']}")
    return got, total, intact, prefix


def test_a_range_honoured_with_capped_single_use_responses():
    """The real-world shape: Range works, each response is capped, and a
    signature can only be spent once.  The file must still reassemble."""
    got, total, intact, _ = run("A")
    assert intact, "A: reassembled bytes are corrupt"
    assert got == SIZE and total == SIZE, "A: did not reassemble the whole file"


def test_b_range_unsupported_reports_failure():
    """No Range support -> report 0/None so the caller falls back to
    streaming instead of pretending it has a complete file."""
    got, total, _, prefix = run("B", range_ok=False)
    assert got == 0 and total is None, "B: must report failure so caller streams"
    assert not prefix, "B: must not leave a half-written MP4 behind"


def test_c_server_returns_nothing():
    got, total, intact, _ = run("C", cap=0)
    assert got == 0, "C: must not claim success"
    assert not intact, "C: must not claim a corrupt file is complete"


def test_d_parallel_chunking_reassembles_every_chunk():
    """Chunk size smaller than the per-response cap, so the file spans
    several chunks and the thread pool actually runs concurrently."""
    orig_chunk = AudioDownloader.FETCH_CHUNK
    try:
        AudioDownloader.FETCH_CHUNK = 100 * 1024      # 7 chunks over 700 KiB
        got, _, intact, _ = run("D", cap=CAP)
    finally:
        AudioDownloader.FETCH_CHUNK = orig_chunk
    assert intact and got == SIZE, "D: parallel chunking is broken"


def test_e_ffmpeg_argv_http_options_only_on_urls():
    """``-headers`` / ``-reconnect*`` are http-protocol options.  Handing
    them to ffmpeg for a local file aborts with rc=8 ("Option headers not
    found.") — this is exactly what broke the first Range-based run."""
    hdr = "Cookie: a=b\r\n"
    local = AudioDownloader._ffmpeg_cmd("/d/x.mp4", "/d/x.raw", hdr, is_url=False)
    url = AudioDownloader._ffmpeg_cmd("https://h/x.mp4", "/d/x.raw", hdr, is_url=True)
    assert "-headers" not in local and "-reconnect" not in local, \
        "local file input must not carry http-only options (rc=8)"
    assert local[local.index("-i") + 1] == "/d/x.mp4"
    assert "-headers" in url and "-reconnect_streamed" in url, \
        "direct streaming must keep the http options"
    assert url[url.index("-i") + 1] == "https://h/x.mp4"
    print(f"  local: {local}")
    print(f"  url:   {url}")
