"""A complete download can still be an empty recording.

iCourse publishes a lecture's video entry the moment the class starts, so a
run that lands mid-class downloads the whole ~2 GB file and transcribes
silence.  The resulting transcript is short but non-empty, so nothing
downstream complains: it gets stored, the lecture is marked processed, and
the real summary is never produced.  Observed live on 2026-09-30 (lecture
671279): 9760 s of media, 11 segments, 144 chars — of which ~110 were the
synthetic silence marker, i.e. ~34 chars of actual speech.

``_check_speech_density`` closes that hole.  Its counterpart
``_check_completeness`` already catches the truncated-download case; these
tests pin down the new one, including the cases it must NOT fire on.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.transcriber import (  # noqa: E402
    SPEECH_DENSITY_MIN_MEDIA_SEC,
    SparseAudioError,
    Transcriber,
)


def check(media_duration_s, speech_chars):
    """Run the guard on a bare Transcriber — no models, no audio needed."""
    t = Transcriber.__new__(Transcriber)
    t._media_duration = media_duration_s
    t._last_speech_chars = speech_chars
    t._check_speech_density()


# ── must fire ──────────────────────────────────────────────────────────

def test_still_empty_recording_is_rejected():
    """The real 2026-09-30 case: 162.7 min of media, ~34 chars of speech."""
    with pytest.raises(SparseAudioError) as e:
        check(9760, 34)
    assert e.value.chars == 34
    assert e.value.duration_s == 9760
    assert "still being generated" in str(e.value)


def test_zero_speech_in_a_long_recording_is_rejected():
    with pytest.raises(SparseAudioError):
        check(7200, 0)


def test_barely_above_zero_is_rejected():
    # 120 min * 5 chars/min = 600 chars is the floor; 300 is below it.
    with pytest.raises(SparseAudioError):
        check(7200, 300)


# ── must not fire ──────────────────────────────────────────────────────

def test_normal_lecture_passes():
    # 10045 s / 167 min with 18840 chars ≈ 113 chars/min.
    check(10045, 18840)


def test_sparse_but_real_lecture_passes():
    """A mostly-silent problem-solving session: 20 min of speech in 160 min."""
    check(9600, 3200)          # 20 chars/min


def test_exactly_at_the_floor_passes():
    # 120 min, 600 chars -> exactly 5.0 chars/min; the test is strict '<'.
    check(7200, 600)


def test_short_recording_is_not_judged():
    """Below the duration floor a sparse clip is plausible (a 5-min intro)."""
    assert SPEECH_DENSITY_MIN_MEDIA_SEC == 600.0
    check(599, 0)


def test_unknown_media_duration_is_not_judged():
    """No ``Duration:`` line in ffmpeg's stderr -> nothing to compare against."""
    check(None, 0)
    check(0, 0)


def test_zero_duration_is_not_judged():
    check(0.0, 0)
