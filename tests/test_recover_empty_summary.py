"""``get_unprocessed_lectures`` must re-queue a lecture whose summary was
lost, without re-queuing the lectures that legitimately have no summary.

Background: ``_call_llm`` can return an empty string (a reasoning model
burning its whole budget in ``reasoning_content``), and ``_summarize``
stored whatever it got.  The lecture then had ``processed_at`` set with an
empty ``summary``, so ``get_unprocessed_lectures`` — which filtered on
``processed_at IS NULL`` — never looked at it again and the transcript was
unreachable.  The fix widens the predicate to also match "has a transcript
but no summary".

The interesting half is the *negative* cases: ``LectureRunner`` marks a
lecture processed with no summary on two deliberate skip paths (empty
transcript, video with no audio stream).  Those must stay out of the retry
set, or they would be re-queued on every single run forever.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.database import Database  # noqa: E402


@pytest.fixture()
def db():
    with tempfile.TemporaryDirectory() as tmp:
        d = Database(os.path.join(tmp, "test.db"))
        yield d
        d.conn.close()


def _add(db: Database, sub_id: str, *, transcript, summary,
         processed_at="2026-09-24T03:00:00", error_count=0):
    with db.conn:
        db.conn.execute(
            "INSERT INTO lectures (sub_id, course_id, sub_title, date,"
            " transcript, summary, processed_at, error_count)"
            " VALUES (?, '39596', ?, '2026-09-24', ?, ?, ?, ?)",
            (sub_id, sub_id, transcript, summary, processed_at, error_count),
        )


def _ids(db: Database) -> set[str]:
    return {r["sub_id"] for r in db.get_unprocessed_lectures()}


# ── positive: the lost-summary case must come back ─────────────────────

def test_empty_summary_with_transcript_is_requeued(db):
    _add(db, "668227", transcript="x" * 18840, summary="")
    assert _ids(db) == {"668227"}


def test_null_summary_with_transcript_is_requeued(db):
    _add(db, "668227", transcript="x" * 18840, summary=None)
    assert _ids(db) == {"668227"}


def test_whitespace_only_summary_is_requeued(db):
    _add(db, "668227", transcript="x" * 18840, summary="   \n\t ")
    assert _ids(db) == {"668227"}


def test_never_processed_lecture_is_still_returned(db):
    _add(db, "100001", transcript=None, summary=None, processed_at=None)
    assert _ids(db) == {"100001"}


# ── negative: deliberate no-summary paths must stay out ────────────────

def test_completed_lecture_is_not_requeued(db):
    _add(db, "100002", transcript="x" * 100, summary="### 摘要\n正文")
    assert _ids(db) == set()


def test_empty_transcript_skip_is_not_requeued(db):
    """``run`` Phase F: empty transcript → mark_processed, no summary."""
    _add(db, "100003", transcript="", summary=None)
    assert _ids(db) == set()


def test_missing_transcript_skip_is_not_requeued(db):
    """``_get_transcript``: video with no audio stream → mark_processed."""
    _add(db, "100004", transcript=None, summary=None)
    assert _ids(db) == set()


def test_whitespace_transcript_skip_is_not_requeued(db):
    _add(db, "100005", transcript="  \n ", summary="")
    assert _ids(db) == set()


# ── the retry ceiling still bounds the recovery ────────────────────────

def test_lost_summary_respects_error_ceiling(db):
    _add(db, "100006", transcript="x" * 100, summary="", error_count=30)
    assert _ids(db) == set()


def test_lost_summary_below_error_ceiling_is_requeued(db):
    _add(db, "100007", transcript="x" * 100, summary="", error_count=29)
    assert _ids(db) == {"100007"}


# ── course filter still applies to both branches ───────────────────────

def test_course_filter_applies_to_recovered_lecture(db):
    _add(db, "100008", transcript="x" * 100, summary="")
    with db.conn:
        db.conn.execute(
            "INSERT INTO lectures (sub_id, course_id, transcript, summary,"
            " processed_at) VALUES ('100009', '37491', ?, '', '2026-09-24')",
            ("y" * 100,),
        )
    assert {r["sub_id"] for r in db.get_unprocessed_lectures("37491")} == {"100009"}
    assert {r["sub_id"] for r in db.get_unprocessed_lectures("39596")} == {"100008"}
