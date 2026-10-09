#!/usr/bin/env python3
"""Phase D must retry a silent recording against the lecture's other MP4s.

Why this exists
---------------
Lecture 671279 (2026-09-30 第3-5节, 复杂系统理论及其在脑科学中的应用) exposed
two ``video_list`` entries.  The code took the first, which downloaded
*completely* — 2130 MiB, 17/17 chunks, full 9760 s of media — and whose audio
track was empty.  The same numbers came back on nine consecutive nights.

``SparseAudioError`` is the right guard (it stops the near-empty transcript
from being stored and marking the lecture processed), but on its own it just
retries the same wrong file forever.  Phase D now advances to the next
candidate before recording the failure.

These tests drive ``LectureRunner.run`` with ``_get_transcript`` and
``_summarize`` stubbed out, so no network, no ffmpeg and no ASR models are
needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.ai.transcriber import SparseAudioError
from src.pipeline.lecture_runner import MAX_VIDEO_CANDIDATE_TRIES, LectureRunner

SUB_ID = "671279"


def silent(chars: int = 34, duration_s: float = 9760.0) -> SparseAudioError:
    return SparseAudioError(
        f"{duration_s:.0f}s of audio contains only {chars} chars of speech",
        chars=chars, duration_s=duration_s,
    )


# ── Fakes ────────────────────────────────────────────────────────────────

class FakeReporter:
    def __init__(self):
        self.lines: list[str] = []

    def info(self, message: str) -> None:
        self.lines.append(message)

    def __getattr__(self, _name):
        # lecture_start / lecture_done / lecture_skip_no_video / ...
        return lambda *a, **kw: None


class FakeDB:
    def __init__(self):
        self.errors: list[tuple[str, str]] = []
        self.processed: list[str] = []
        self.transcripts: list[str] = []
        self.summaries: list[str] = []

    def get_lecture(self, sub_id):
        return None

    def update_error(self, sub_id, stage, message):
        self.errors.append((sub_id, stage))

    def clear_error(self, sub_id):
        pass

    def mark_processed(self, sub_id):
        self.processed.append(sub_id)

    def update_transcript(self, sub_id, text):
        self.transcripts.append(text)

    def update_summary(self, sub_id, summary):
        self.summaries.append(summary)


class FakeHandle:
    def __init__(self):
        self.drains = 0

    def drain(self):
        self.drains += 1
        return {}


class FakePPT:
    def __init__(self):
        self.handle = FakeHandle()

    def submit(self, *a, **kw):
        return self.handle

    def prefetch_and_ocr(self, *a, **kw):
        pass


class _FakeDownloader:
    def schedule(self, *a, **kw):
        pass

    def release(self, sub_id):
        pass


class FakeScheduler:
    def __init__(self):
        self.audio_downloader = _FakeDownloader()


class FakeClient:
    """Mirrors ICourseClient.try_next_video_candidate for ``n`` candidates."""

    def __init__(self, n: int):
        self.n = n
        self.pref = 0
        self.advances = 0

    def try_next_video_candidate(self, sub_id) -> bool:
        if self.pref + 1 >= self.n:
            return False
        self.pref += 1
        self.advances += 1
        return True


def make_runner(n_candidates: int, transcripts: list):
    """Build a LectureRunner whose transcript phase is scripted.

    ``transcripts`` is consumed one entry per attempt: a ``SparseAudioError``
    instance is raised, ``None`` means "skip for some other reason" (the
    ``(None, None)`` return), anything else is returned as the transcript.
    """
    r = LectureRunner.__new__(LectureRunner)
    r._client = FakeClient(n_candidates)
    r._db = FakeDB()
    r._scheduler = FakeScheduler()
    r._transcriber = None
    r._summarizer = None
    r._reporter = FakeReporter()
    r._ppt = FakePPT()

    remaining = list(transcripts)
    r.attempts = []

    def fake_get_transcript(existing, course_id, sub_id):
        r.attempts.append(sub_id)
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is None:
            return None, None
        return outcome, [{"text": outcome}]

    r._get_transcript = fake_get_transcript
    r._summarize = lambda *a, **kw: "摘要"
    return r


def run(runner):
    return runner.run(
        "38678", "复杂系统理论及其在脑科学中的应用",
        {"sub_id": SUB_ID, "sub_title": "2026-09-30第3-5节", "date": "2026-09-30"},
        None,
    )


# ── 1. The retry that fixes 671279 ───────────────────────────────────────

def test_silent_first_candidate_falls_through_to_the_second():
    """The headline case: candidate #0 is silent, candidate #1 is not."""
    r = make_runner(2, [silent(), "真正的转录文本"])
    assert run(r) == "摘要"
    assert r.attempts == [SUB_ID, SUB_ID]
    assert r._client.advances == 1
    # No failure recorded — the lecture genuinely succeeded.
    assert r._db.errors == []
    assert any("[Retry]" in line for line in r._reporter.lines)


def test_the_retry_is_visible_in_the_log():
    r = make_runner(2, [silent(), "文本"])
    run(r)
    retry = [ln for ln in r._reporter.lines if "[Retry]" in ln]
    assert len(retry) == 1
    assert "candidate #0" in retry[0]


# ── 2. Giving up is still correct when every candidate is silent ─────────

def test_single_candidate_still_skips_without_retrying():
    r = make_runner(1, [silent()])
    assert run(r) is None
    assert r.attempts == [SUB_ID]
    assert r._client.advances == 0
    assert r._db.errors == [(SUB_ID, "transcribe")]
    assert any("[SKIP]" in line for line in r._reporter.lines)


def test_all_candidates_silent_gives_up_after_the_last_one():
    r = make_runner(2, [silent(), silent(chars=40)])
    assert run(r) is None
    assert r.attempts == [SUB_ID, SUB_ID]
    assert r._client.advances == 1
    # Exactly one error recorded for the whole lecture, not one per attempt.
    assert r._db.errors == [(SUB_ID, "transcribe")]


def test_the_loop_is_bounded():
    """Even if the client kept saying yes, Phase D stops at the cap."""
    r = make_runner(99, [silent() for _ in range(MAX_VIDEO_CANDIDATE_TRIES)])
    assert run(r) is None
    assert len(r.attempts) == MAX_VIDEO_CANDIDATE_TRIES


# ── 3. Other skip paths are untouched ────────────────────────────────────

def test_non_sparse_skip_is_not_retried():
    """A None return (no video, timeout, ...) must not consume candidates."""
    r = make_runner(3, [None])
    assert run(r) is None
    assert r.attempts == [SUB_ID]
    assert r._client.advances == 0
    # _get_transcript is responsible for its own error bookkeeping there.
    assert r._db.errors == []


def test_silent_then_other_skip_records_only_the_other_reason():
    """If the retry ends in a *different* skip, don't also log the silence.

    ``_get_transcript`` persists its own reason for every non-sparse outcome
    (no_video, timeout, truncated download); recording a second 'transcribe'
    error on top would overwrite that stage, and the frontend keys its gray
    "无视频" hint off it.
    """
    r = make_runner(2, [silent(), None])
    assert run(r) is None
    assert r.attempts == [SUB_ID, SUB_ID]
    assert r._client.advances == 1
    assert r._db.errors == []
    assert not any("Recording not generated yet" in ln
                   for ln in r._reporter.lines)


def test_ppt_handle_is_drained_on_every_give_up_path():
    """Skipping must not leave OCR pages 'pending' forever."""
    for transcripts in ([silent()], [None]):
        r = make_runner(1, transcripts)
        run(r)
        assert r._ppt.handle.drains == 1


def test_first_attempt_success_needs_no_retry():
    r = make_runner(2, ["一次就成功"])
    assert run(r) == "摘要"
    assert r.attempts == [SUB_ID]
    assert r._client.advances == 0
    assert r._db.errors == []


# ── 4. The real client's bookkeeping ─────────────────────────────────────

def test_client_advances_and_then_reports_exhaustion():
    from src.api.icourse import ICourseClient

    c = ICourseClient.__new__(ICourseClient)
    c._video_candidates = {"111": ["a.mp4", "b.mp4"]}
    c._video_pref = {}

    assert c.video_candidate_count("111") == 2
    assert c.try_next_video_candidate("111") is True
    assert c._video_pref["111"] == 1
    assert c.try_next_video_candidate("111") is False
    assert c._video_pref["111"] == 1     # unchanged when exhausted


def test_client_handles_a_lecture_it_has_never_looked_up():
    from src.api.icourse import ICourseClient

    c = ICourseClient.__new__(ICourseClient)
    c._video_candidates = {}
    c._video_pref = {}
    assert c.try_next_video_candidate("999") is False
    assert c.video_candidate_count("999") == 0


@pytest.mark.parametrize("n,expected", [(0, 0), (1, 0), (2, 1), (3, 2)])
def test_advance_count_matches_candidate_count(n, expected):
    from src.api.icourse import ICourseClient

    c = ICourseClient.__new__(ICourseClient)
    c._video_candidates = {"111": [f"{i}.mp4" for i in range(n)]}
    c._video_pref = {}
    advances = 0
    while c.try_next_video_candidate("111"):
        advances += 1
    assert advances == expected
