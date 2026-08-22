"""SQLite access layer.

Every stage checkpoints here and every stage is idempotent on resume, so the
connection helpers are deliberately thin: no ORM, no session state, no lazy
loading that could hide a partial write.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open the store, creating it and applying the schema if absent."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def insert(conn: sqlite3.Connection, table: str, **values: Any) -> int:
    cols = ", ".join(values)
    marks = ", ".join("?" * len(values))
    cur = conn.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(values.values())
    )
    return int(cur.lastrowid)


def insert_many(
    conn: sqlite3.Connection, table: str, columns: list[str], rows: Iterable[tuple]
) -> None:
    marks = ", ".join("?" * len(columns))
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({marks})", rows
    )


def one(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> sqlite3.Row | None:
    return conn.execute(sql, params).fetchone()


def all_rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    return conn.execute(sql, params).fetchall()


def audit(
    conn: sqlite3.Connection,
    action: str,
    entity: str,
    entity_id: int | None = None,
    detail: str | None = None,
    actor: str = "system",
) -> None:
    insert(
        conn,
        "audit_entry",
        action=action,
        entity=entity,
        entity_id=entity_id,
        detail=detail,
        actor=actor,
    )
