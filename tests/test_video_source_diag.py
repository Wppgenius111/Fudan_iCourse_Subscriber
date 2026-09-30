#!/usr/bin/env python3
"""Tests for the video-source diagnostics in ``ICourseClient.get_video_url``.

Why this exists
---------------
A lecture download can come back *complete* — full length, full byte count —
and still carry a silent audio track, because iCourse publishes a lecture's
media entry the moment the class starts.  The only cheap way to tell that
apart from a healthy download is to know **which** of the four URL sources
supplied the URL:

  1. ``video_list[*].preview_url``   — the school published a finished recording
  2. ``playurl[*]``                  — healthy alternate
  3. ``content.playback.url``        — review-gated, i.e. still being generated
  4. ``sub_detail.content.playback.url``

The diagnostics that report this used to be gated on ``verbose``, but the
download path calls ``get_video_url(..., verbose=False)`` — once per signed
HTTP request — so they never fired where the answer actually mattered.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.api import icourse
from src.api.icourse import ICourseClient


@pytest.fixture(autouse=True)
def _reset_reported():
    """The once-per-lecture guard is module state; isolate every test."""
    icourse._VIDEO_SOURCE_REPORTED.clear()
    yield
    icourse._VIDEO_SOURCE_REPORTED.clear()


def make_client(info: dict, detail: dict | None = None) -> ICourseClient:
    """A client with the network stripped out.

    Instance-attribute lambdas are *not* bound as methods, so each one is
    called with exactly the arguments spelled out here — no implicit self.
    """
    c = ICourseClient.__new__(ICourseClient)
    c.get_sub_info = lambda course_id, sub_id: info
    c.get_sub_detail = lambda course_id, sub_id: detail or {}
    c.sign_video_url = lambda base_url, now=None: base_url
    return c


def healthy_info(previews: list[str]) -> dict:
    return {
        "now": 1700000000,
        "video_list": {
            str(i + 1): {"preview_url": u} for i, u in enumerate(previews)
        },
    }


# ── 1. Quiet on the hot path for an unambiguous lecture ──────────────────

def test_single_candidate_skips_the_landscape_dump(capsys):
    """One candidate, verbose=False → the multi-line dump is suppressed.

    This method runs once per signed HTTP request during a download, so the
    landscape dump (which can be several lines) must stay opt-in.  The
    one-line source summary is still emitted, exactly once.
    """
    c = make_client(healthy_info(["https://cdn.example/a.mp4"]))
    for _ in range(3):
        c.get_video_url("1", "111", verbose=False)
    out = capsys.readouterr().out
    assert "[VideoDiag]" not in out
    assert out == "    [VideoSource] 111: video_list[1].preview_url\n"


def test_single_candidate_is_reported_when_verbose(capsys):
    """The original manual-diagnostic behaviour is preserved."""
    c = make_client(healthy_info(["https://cdn.example/a.mp4"]))
    c.get_video_url("1", "111", verbose=True)
    out = capsys.readouterr().out
    assert "[VideoDiag] 111: video_list exposes 1 .mp4 candidate(s)" in out
    assert "[VideoSource] 111: video_list[1].preview_url" in out


# ── 2. The ambiguous case fires even with verbose=False ──────────────────

def test_two_candidates_are_reported_without_verbose(capsys):
    """Upstream issue #44: several MP4s per lecture, first is not always right."""
    c = make_client(healthy_info([
        "https://cdn.example/live/0.mp4",
        "https://cdn.example/rec/0.mp4",
    ]))
    c.get_video_url("1", "222", verbose=False)
    out = capsys.readouterr().out
    assert "exposes 2 .mp4 candidate(s)" in out
    assert "cdn.example/live/0.mp4" in out
    assert "cdn.example/rec/0.mp4" in out


def test_ambiguous_report_is_emitted_only_once(capsys):
    """Once per lecture, not once per signed HTTP request."""
    c = make_client(healthy_info([
        "https://cdn.example/live/0.mp4",
        "https://cdn.example/rec/0.mp4",
    ]))
    for _ in range(5):
        c.get_video_url("1", "222", verbose=False)
    assert capsys.readouterr().out.count("exposes 2 .mp4") == 1


def test_query_string_is_never_logged(capsys):
    """Signed URLs carry a token in the query string — log host+path only."""
    c = make_client(healthy_info([
        "https://cdn.example/live/0.mp4?t=1700000000&sign=SUPERSECRET",
        "https://cdn.example/rec/0.mp4?t=1700000000&sign=ALSOSECRET",
    ]))
    c.get_video_url("1", "222", verbose=False)
    out = capsys.readouterr().out
    assert "SUPERSECRET" not in out
    assert "ALSOSECRET" not in out
    assert "sign=" not in out


# ── 3. The review-gated fallback names itself ────────────────────────────

def test_review_gated_fallback_is_named(capsys):
    """Zero candidates → we are on the pre-release path, say so."""
    info = {
        "now": 1700000000,
        "video_list": {},
        "content": {"playback": {"url": "https://cdn.example/gated/0.mp4"}},
    }
    c = make_client(info)
    url = c.get_video_url("1", "333", verbose=False)
    out = capsys.readouterr().out
    assert url == "https://cdn.example/gated/0.mp4"
    assert "exposes 0 .mp4 candidate(s)" in out
    assert "content.playback.url (review-gated)" in out


def test_gated_source_is_named_only_once(capsys):
    c = make_client({
        "now": 1700000000,
        "video_list": {},
        "content": {"playback": {"url": "https://cdn.example/gated/0.mp4"}},
    })
    for _ in range(4):
        c.get_video_url("1", "333", verbose=False)
    assert capsys.readouterr().out.count("review-gated") == 1


def test_sub_detail_fallback_is_named(capsys):
    c = make_client(
        {"now": 1700000000, "video_list": {}},
        detail={"content": {"playback": {"url": "https://cdn.example/d/0.mp4"}}},
    )
    url = c.get_video_url("1", "444", verbose=False)
    assert url == "https://cdn.example/d/0.mp4"
    assert "sub_detail.content.playback.url" in capsys.readouterr().out


def test_playurl_fallback_is_named(capsys):
    c = make_client({
        "now": 1700000000,
        "video_list": {},
        "playurl": {"0": "https://cdn.example/p/0.mp4"},
    })
    url = c.get_video_url("1", "555", verbose=False)
    assert url == "https://cdn.example/p/0.mp4"
    assert "playurl[0]" in capsys.readouterr().out


# ── 4. The guard is per-lecture, not global ─────────────────────────────

def test_two_lectures_are_reported_separately(capsys):
    c = make_client(healthy_info([
        "https://cdn.example/a.mp4",
        "https://cdn.example/b.mp4",
    ]))
    c.get_video_url("1", "666", verbose=False)
    c.get_video_url("1", "777", verbose=False)
    out = capsys.readouterr().out
    assert "666" in out and "777" in out
    assert out.count("exposes 2 .mp4") == 2


def test_no_url_at_all_reports_and_returns_none(capsys):
    c = make_client({"now": 1700000000, "video_list": {}})
    assert c.get_video_url("1", "888", verbose=False) is None
    assert "No video URL found for 888" in capsys.readouterr().out
