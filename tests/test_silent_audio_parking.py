#!/usr/bin/env python3
"""A silent recording gets its own, much shorter retry ceiling.

Background: ``SparseAudioError`` stops a near-empty transcript from being
stored, but on its own it just re-runs the same lecture forever.  Lecture
671279 (2026-09-30 第3-5节) came back byte-identical on nine consecutive
nights — 2130 MiB downloaded, 9760 s of media, 144 chars of speech — and
both of its ``video_list`` candidates turned out to hold no audio at all.
Every one of those attempts is a full download.

So the failure is counted twice: once in ``error_count`` (the global
ceiling, 30) and once in ``silent_count``, which has its own much lower
ceiling (7 by default, ``FICS_MAX_SILENT_ERRORS``).  A week is long enough
for a genuinely late recording to appear; after that the lecture is parked
instead of costing a download every night.
"""

from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.database import Database  # noqa: E402
from src.data.schema import SCHEMA_SQL  # noqa: E402

# ``scripts/`` is not a package, so load merge_db by path.
_spec = importlib.util.spec_from_file_location(
    "merge_db",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "merge_db.py",
    ),
)
merge_db = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge_db)


@pytest.fixture()
def db():
    with tempfile.TemporaryDirectory() as tmp:
        d = Database(os.path.join(tmp, "test.db"))
        yield d
        d.conn.close()


def _add(db: Database, sub_id: str, *, error_count=0, silent_count=0,
         error_stage=None, processed_at=None):
    with db.conn:
        db.conn.execute(
            "INSERT INTO lectures (sub_id, course_id, sub_title, date,"
            " error_count, silent_count, error_stage, processed_at)"
            " VALUES (?, '38678', ?, '2026-09-30', ?, ?, ?, ?)",
            (sub_id, sub_id, error_count, silent_count, error_stage,
             processed_at),
        )


def _ids(db: Database, **kw) -> set[str]:
    return {r["sub_id"] for r in db.get_unprocessed_lectures(**kw)}


# ── the ceiling itself ───────────────────────────────────────────────────

def test_default_ceiling_is_a_week(db):
    assert Database.DEFAULT_MAX_SILENT_ERRORS == 7


def test_below_the_ceiling_is_still_retried(db):
    _add(db, "671279", silent_count=6, error_stage="silent_audio")
    assert _ids(db) == {"671279"}


def test_at_the_ceiling_is_parked(db):
    _add(db, "671279", silent_count=7, error_stage="silent_audio")
    assert _ids(db) == set()


def test_well_past_the_ceiling_stays_parked(db):
    _add(db, "671279", silent_count=30, error_stage="silent_audio")
    assert _ids(db) == set()


def test_the_ceiling_is_independent_of_the_global_one(db):
    """silent_count = 7 parks the lecture even at error_count = 7 < 30."""
    _add(db, "671279", error_count=7, silent_count=7,
         error_stage="silent_audio")
    assert _ids(db) == set()


def test_a_non_silent_failure_is_not_parked_by_silent_count(db):
    """A truncated download (transcribe stage) still gets the full 30 runs."""
    _add(db, "671279", error_count=7, silent_count=0,
         error_stage="transcribe")
    assert _ids(db) == {"671279"}


def test_max_silent_can_be_overridden(db):
    _add(db, "671279", silent_count=3, error_stage="silent_audio")
    assert _ids(db) == {"671279"}
    assert _ids(db, max_silent=3) == set()


def test_zero_disables_the_extra_ceiling(db):
    _add(db, "671279", silent_count=99, error_stage="silent_audio")
    assert _ids(db, max_silent=999) == {"671279"}


# ── recording the failure ────────────────────────────────────────────────

def test_update_silent_audio_error_bumps_both_counters(db):
    _add(db, "671279")
    db.update_silent_audio_error("671279", "9760s of audio, 144 chars")
    row = db.conn.execute(
        "SELECT error_stage, error_msg, error_count, silent_count"
        " FROM lectures WHERE sub_id = '671279'"
    ).fetchone()
    assert row["error_stage"] == Database.SILENT_AUDIO_STAGE
    assert row["error_msg"] == "9760s of audio, 144 chars"
    assert row["error_count"] == 1
    assert row["silent_count"] == 1


def test_repeated_silent_failures_accumulate(db):
    _add(db, "671279")
    for i in range(7):
        db.update_silent_audio_error("671279", f"attempt {i}")
    assert _ids(db) == set()


def test_clear_error_resets_the_silent_counter(db):
    """A lecture that finally succeeds must not stay half-parked."""
    _add(db, "671279", error_count=6, silent_count=6,
         error_stage="silent_audio")
    assert _ids(db) == {"671279"}
    db.clear_error("671279")
    row = db.conn.execute(
        "SELECT error_count, silent_count FROM lectures WHERE sub_id = '671279'"
    ).fetchone()
    assert row["silent_count"] == 0
    assert row["error_count"] == 0


def test_a_parked_lecture_that_gains_a_summary_is_not_returned(db):
    _add(db, "671279", silent_count=99, error_stage="silent_audio")
    db.update_summary("671279", "### 摘要\n正文", "deepseek-v4-flash")
    db.mark_processed("671279")
    db.clear_error("671279")
    assert _ids(db) == set()


# ── schema migration ─────────────────────────────────────────────────────

def test_column_is_added_to_a_pre_existing_database():
    """Existing deployments get silent_count via ALTER TABLE on open."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "old.db")
        # A DB shaped like the one already on the data branch: no column.
        raw = sqlite3.connect(path)
        raw.execute(
            "CREATE TABLE lectures (sub_id TEXT PRIMARY KEY,"
            " course_id TEXT NOT NULL, sub_title TEXT, date TEXT,"
            " transcript TEXT, summary TEXT, processed_at TEXT,"
            " emailed_at TEXT, error_msg TEXT,"
            " error_count INTEGER DEFAULT 0, error_stage TEXT,"
            " summary_model TEXT)"
        )
        raw.execute(
            "INSERT INTO lectures (sub_id, course_id, error_count)"
            " VALUES ('671279', '38678', 5)"
        )
        raw.commit()
        raw.close()

        d = Database(path)
        cols = {r[1] for r in d.conn.execute("PRAGMA table_info(lectures)")}
        assert "silent_count" in cols
        # Existing rows default to 0, so nothing is parked retroactively.
        assert d.get_unprocessed_lectures()[0]["silent_count"] == 0
        d.conn.close()


def test_the_stage_name_matches_the_frontend_contract(db):
    """frontend/js/db.js maps this exact string to the gray 'No Audio' badge."""
    assert Database.SILENT_AUDIO_STAGE == "silent_audio"


# ── one-time backfill for the database already on the data branch ────────

_SPARSE_MSG = ("9760s of audio (163 min) contains only 144 chars of speech "
               "(0.9 chars/min, floor is 5). The recording is probably still "
               "being generated.")


def _legacy_db(tmp: str, rows: list[tuple]) -> str:
    """A database shaped like the one on the data branch before this change."""
    path = os.path.join(tmp, "legacy.db")
    raw = sqlite3.connect(path)
    raw.execute(
        "CREATE TABLE lectures (sub_id TEXT PRIMARY KEY,"
        " course_id TEXT NOT NULL, sub_title TEXT, date TEXT,"
        " transcript TEXT, summary TEXT, processed_at TEXT,"
        " emailed_at TEXT, error_msg TEXT,"
        " error_count INTEGER DEFAULT 0, error_stage TEXT,"
        " summary_model TEXT)"
    )
    raw.executemany(
        "INSERT INTO lectures (sub_id, course_id, error_count, error_stage,"
        " error_msg) VALUES (?, '38678', ?, ?, ?)",
        rows,
    )
    raw.commit()
    raw.close()
    return path


def test_backfill_parks_a_lecture_that_already_burned_ten_runs():
    """The 671279 case: ten failures on record, all of them silent."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Database(_legacy_db(tmp, [("671279", 10, "transcribe", _SPARSE_MSG)]))
        row = d.conn.execute(
            "SELECT silent_count FROM lectures WHERE sub_id = '671279'"
        ).fetchone()
        assert row["silent_count"] == 10
        assert d.get_unprocessed_lectures() == []
        d.conn.close()


def test_backfill_leaves_other_failures_alone():
    """A truncated download is not a silent recording; keep its full budget."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Database(_legacy_db(tmp, [
            ("100001", 10, "transcribe", "download came up short at 41%"),
            ("100002", 10, "no_video", "no playable video URL"),
            ("100003", 10, "transcribe", _SPARSE_MSG),
        ]))
        rows = {r["sub_id"]: r for r in d.get_unprocessed_lectures()}
        # The two non-sparse rows keep their full 30-run budget...
        assert rows["100001"]["silent_count"] == 0
        assert rows["100002"]["silent_count"] == 0
        # ...and the sparse one is parked, so it is not in the list at all.
        assert "100003" not in rows
        assert d.conn.execute(
            "SELECT silent_count FROM lectures WHERE sub_id = '100003'"
        ).fetchone()["silent_count"] == 10
        d.conn.close()


def test_backfill_runs_only_once():
    """Reopening the database must not re-copy error_count over a reset."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _legacy_db(tmp, [("671279", 10, "transcribe", _SPARSE_MSG)])
        d = Database(path)
        assert d.conn.execute(
            "SELECT silent_count FROM lectures WHERE sub_id = '671279'"
        ).fetchone()["silent_count"] == 10
        d.conn.close()

        # A later run retries once more and records it on the silent counter;
        # reopening must not snap silent_count back up to error_count.
        d = Database(path)
        d.update_silent_audio_error("671279", _SPARSE_MSG)
        assert d.conn.execute(
            "SELECT silent_count FROM lectures WHERE sub_id = '671279'"
        ).fetchone()["silent_count"] == 11
        d.conn.close()


def _modern_db(tmp: str, rows: list[tuple]) -> str:
    """A database built from the current ``SCHEMA_SQL``.

    This is what the runner actually opens: ``sharder.reassemble_database``
    creates the database from ``SCHEMA_SQL``, so every migration column —
    ``silent_count`` included — is *already present* before ``Database``
    ever sees it.  The ``ALTER TABLE`` branch in ``_init_tables`` therefore
    never fires in CI.
    """
    path = os.path.join(tmp, "modern.db")
    raw = sqlite3.connect(path)
    raw.executescript(SCHEMA_SQL)
    raw.executemany(
        "INSERT INTO lectures (sub_id, course_id, error_count, silent_count,"
        " error_stage, error_msg) VALUES (?, '38678', ?, ?, ?, ?)",
        rows,
    )
    raw.commit()
    raw.close()
    return path


def test_backfill_runs_even_though_the_column_already_exists():
    """The real CI shape: column present, backfill still needed.

    Regression test for a bug that shipped: the backfill hung off the
    ``ALTER TABLE`` branch, which never runs in CI, so ``silent_count`` sat at
    1 while ``error_count`` was 12 and the lecture kept being retried every
    night — a full 3.4 GiB download each time.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Database(_modern_db(
            tmp, [("671279", 12, 0, "silent_audio", _SPARSE_MSG)],
        ))
        assert d.conn.execute(
            "SELECT silent_count FROM lectures WHERE sub_id = '671279'"
        ).fetchone()["silent_count"] == 12
        assert d.get_unprocessed_lectures() == []
        d.conn.close()


def test_backfill_raises_a_partially_counted_row_too():
    """671279's exact shape: one silent attempt counted, eleven uncounted.

    The previous attempt at this guard used ``COALESCE(silent_count, 0) = 0``
    and so excluded this row — which is the one it existed for.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Database(_modern_db(
            tmp, [("671279", 12, 1, "silent_audio", _SPARSE_MSG)],
        ))
        assert d.conn.execute(
            "SELECT silent_count FROM lectures WHERE sub_id = '671279'"
        ).fetchone()["silent_count"] == 12
        assert d.get_unprocessed_lectures() == []
        d.conn.close()


def test_backfill_ignores_a_row_with_no_attempts():
    """A fresh row must not be given a budget it never spent."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Database(_modern_db(
            tmp, [("671279", 0, 0, "silent_audio", _SPARSE_MSG)],
        ))
        assert d.conn.execute(
            "SELECT silent_count FROM lectures WHERE sub_id = '671279'"
        ).fetchone()["silent_count"] == 0
        d.conn.close()


def test_backfill_does_not_run_twice():
    """The flag in ``meta`` is what makes it one-time, not a counter guard.

    After the transition run the two counters evolve independently, so a later
    ``error_count`` must not drag ``silent_count`` up with it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = _modern_db(
            tmp, [("100001", 12, 1, "silent_audio", _SPARSE_MSG)],
        )
        d = Database(path)
        assert d.conn.execute(
            "SELECT silent_count FROM lectures WHERE sub_id = '100001'"
        ).fetchone()["silent_count"] == 12
        d.conn.close()

        # A row added later starts its own count; reopening must not raise it
        # to its error_count just because the numbers happen to differ.
        raw = sqlite3.connect(path)
        raw.execute(
            "INSERT INTO lectures (sub_id, course_id, error_count, silent_count,"
            " error_stage, error_msg) VALUES ('100002', '38678', 9, 2,"
            " 'silent_audio', ?)", (_SPARSE_MSG,),
        )
        raw.commit()
        raw.close()

        d = Database(path)
        assert d.conn.execute(
            "SELECT silent_count FROM lectures WHERE sub_id = '100002'"
        ).fetchone()["silent_count"] == 2
        d.conn.close()


def test_merge_db_carries_the_backfill_flag_across():
    """The flag has to survive the shard round-trip or the backfill repeats.

    ``check.yml`` merges the local database into a freshly reassembled copy of
    the remote one and shards *that*, so a ``meta`` key written only locally
    would be dropped — and the next run would inflate the counters again.
    """
    with tempfile.TemporaryDirectory() as tmp:
        local_path = _modern_db(
            tmp, [("100001", 12, 1, "silent_audio", _SPARSE_MSG)],
        )
        local = Database(local_path)          # applies the backfill + flag
        assert local.read_meta(Database.BACKFILL_FLAG_KEY) == "1"
        local.conn.close()

        remote_path = _modern_db(tmp, [])
        merge_db.merge(local_path, remote_path)

        remote = sqlite3.connect(remote_path)
        try:
            assert remote.execute(
                "SELECT value FROM meta WHERE key = ?",
                (Database.BACKFILL_FLAG_KEY,),
            ).fetchone() == ("1",)
        finally:
            remote.close()
