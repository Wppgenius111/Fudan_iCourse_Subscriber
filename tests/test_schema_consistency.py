"""Guard: every migration column must also exist in ``SCHEMA_SQL``.

``SCHEMA_SQL`` builds *new* databases — a fresh install, and every shard that
``sharder._build_shard_db`` materializes.  ``*_MIGRATION_COLUMNS`` only
migrates databases that already exist.  A column that is in the migration list
but missing from ``SCHEMA_SQL`` therefore works fine for the runner's own
database and then explodes at deploy time, because the sharder copies columns
straight off the source rows into a freshly created shard::

    sqlite3.OperationalError: table lectures has no column named silent_count

That is exactly what happened when ``silent_count`` was added — and because the
deploy step is what publishes the data branch, the failure is self-sustaining:
every later run hits it again.  These tests pin the invariant down.
"""

from __future__ import annotations

import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.schema import (  # noqa: E402
    LECTURES_MIGRATION_COLUMNS,
    PPT_PAGES_MIGRATION_COLUMNS,
    SCHEMA_SQL,
)
from src.data.sharder import _build_meta_shard, _build_shard_db  # noqa: E402


def _fresh_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_SQL)
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


@pytest.mark.parametrize("table,cols", [
    ("lectures", LECTURES_MIGRATION_COLUMNS),
    ("ppt_pages", PPT_PAGES_MIGRATION_COLUMNS),
])
def test_migration_columns_present_in_fresh_schema(tmp_path, table, cols):
    """A brand-new DB must already carry every migration column."""
    conn = _fresh_db(str(tmp_path / "fresh.db"))
    try:
        present = _columns(conn, table)
        missing = [c for c, _ in cols if c not in present]
        assert not missing, (
            f"{table}: {missing} are in the migration list but not in "
            f"SCHEMA_SQL — shard builds will fail at deploy time"
        )
    finally:
        conn.close()


def _seed_source(path: str) -> None:
    """A source DB shaped exactly like the runner's, migration columns included."""
    conn = _fresh_db(path)
    with conn:
        conn.execute(
            "INSERT INTO courses (course_id, title, teacher) VALUES (?, ?, ?)",
            ("38678", "复杂系统理论", "纪鹏"),
        )
        conn.execute(
            "INSERT INTO lectures (sub_id, course_id, sub_title, date, transcript,"
            " summary, processed_at, error_count, error_stage, silent_count)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("671279", "38678", "2026-09-30第3-5节", "2026-09-30", "t", "s",
             "2026-10-09T00:00:00", 3, "silent_audio", 3),
        )
    conn.close()


def test_build_shard_db_round_trips_a_migrated_source(tmp_path):
    """The sharder copies source columns verbatim — the shard must accept them."""
    src = str(tmp_path / "src.db")
    out = str(tmp_path / "shard.db")
    _seed_source(src)

    _build_shard_db(src, ["38678"], out)

    conn = sqlite3.connect(out)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM lectures WHERE sub_id = '671279'"
        ).fetchone()
        assert row is not None
        assert row["silent_count"] == 3
        assert row["error_stage"] == "silent_audio"
    finally:
        conn.close()


def test_build_meta_shard_round_trips(tmp_path):
    src = str(tmp_path / "src.db")
    out = str(tmp_path / "meta.db")
    _seed_source(src)
    conn = sqlite3.connect(src)
    with conn:
        conn.execute("INSERT INTO meta (key, value) VALUES ('course_ids', '38678')")
    conn.close()

    _build_meta_shard(src, out)

    conn = sqlite3.connect(out)
    try:
        assert conn.execute(
            "SELECT value FROM meta WHERE key = 'course_ids'"
        ).fetchone()[0] == "38678"
    finally:
        conn.close()
