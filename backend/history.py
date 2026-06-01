"""SQLite-backed query log.

Stores enough per row to fully re-render a past conversation: the question,
the model's answer, AND the retrieved chunks + run metadata. That way the UI
can show a historical chat without re-running the LLM.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path("data/finagent.db")

# Idempotent migration: older DBs created before we stored chunks/metadata
# pick up the new columns via ALTER on import. SQLite ignores `IF NOT EXISTS`
# on ADD COLUMN, so we check `PRAGMA table_info` first.
_BASE_SCHEMA = """
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
_NEW_COLUMNS = {
    "chunks_json": "TEXT",
    "metadata_json": "TEXT",
}


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_BASE_SCHEMA)
    existing = {r[1] for r in conn.execute("PRAGMA table_info(query_log)").fetchall()}
    for col, typ in _NEW_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE query_log ADD COLUMN {col} {typ}")
    conn.commit()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


_conn: Optional[sqlite3.Connection] = None


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = _connect()
    return _conn


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #

def log_query(
    question: str,
    config: str,
    market: str,
    answer: str,
    latency: float,
    chunks: Optional[list[dict]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> int:
    c = conn()
    cur = c.execute(
        "INSERT INTO query_log "
        "(question, config, market, answer, latency, chunks_json, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            question, config, market, answer, latency,
            json.dumps(chunks or [], default=str, ensure_ascii=False),
            json.dumps(metadata or {}, default=str, ensure_ascii=False),
        ),
    )
    c.commit()
    return cur.lastrowid or 0


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #

def _row_to_summary(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "question": r["question"],
        "config": r["config"],
        "market": r["market"],
        "answer": r["answer"] or "",
        "latency": r["latency"] or 0.0,
        "created_at": r["created_at"],
    }


def _row_to_full(r: sqlite3.Row) -> dict:
    out = _row_to_summary(r)
    out["chunks"] = json.loads(r["chunks_json"]) if r["chunks_json"] else []
    out["metadata"] = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
    return out


def list_recent(limit: int = 50) -> list[dict]:
    """Summaries only (no chunks/metadata) — keeps the sidebar payload light."""
    c = conn()
    rows = c.execute(
        "SELECT id, question, config, market, answer, latency, created_at "
        "FROM query_log ORDER BY id DESC LIMIT ?", (limit,),
    ).fetchall()
    return [_row_to_summary(r) for r in rows]


def get_one(item_id: int) -> Optional[dict]:
    """Full record including chunks + metadata, for `view past chat`."""
    c = conn()
    row = c.execute(
        "SELECT id, question, config, market, answer, latency, created_at, "
        "       chunks_json, metadata_json "
        "FROM query_log WHERE id = ?", (item_id,),
    ).fetchone()
    return _row_to_full(row) if row else None


# --------------------------------------------------------------------------- #
# Delete
# --------------------------------------------------------------------------- #

def delete_one(item_id: int) -> bool:
    c = conn()
    cur = c.execute("DELETE FROM query_log WHERE id = ?", (item_id,))
    c.commit()
    return cur.rowcount > 0


def delete_all() -> int:
    c = conn()
    cur = c.execute("DELETE FROM query_log")
    c.commit()
    return cur.rowcount
