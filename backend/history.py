"""Tiny SQLite-backed query log."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path("data/finagent.db")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    config TEXT NOT NULL,
    market TEXT NOT NULL,
    answer TEXT,
    latency REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


_conn: Optional[sqlite3.Connection] = None


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = _connect()
    return _conn


def log_query(question: str, config: str, market: str,
              answer: str, latency: float) -> int:
    c = conn()
    cur = c.execute(
        "INSERT INTO query_log (question, config, market, answer, latency) "
        "VALUES (?, ?, ?, ?, ?)",
        (question, config, market, answer, latency),
    )
    c.commit()
    return cur.lastrowid or 0


def list_recent(limit: int = 50) -> list[dict]:
    c = conn()
    rows = c.execute(
        "SELECT id, question, config, market, answer, latency, created_at "
        "FROM query_log ORDER BY id DESC LIMIT ?", (limit,),
    ).fetchall()
    return [dict(r) for r in rows]
