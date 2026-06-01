"""SQLite-backed chat store.

Two tables:
  `chats`      — one row per chat thread (title, market, timestamps)
  `messages`   — one row per turn (user or assistant) belonging to a chat

The previous schema (`query_log`) is kept for backward-compat; on first run
under the new code we migrate each existing query_log row into a one-turn
chat so historical conversations don't vanish.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path("data/finagent.db")


# --------------------------------------------------------------------------- #
# Schema + migration
# --------------------------------------------------------------------------- #

_LEGACY_SCHEMA = """
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
_LEGACY_COLUMNS = {"chunks_json": "TEXT", "metadata_json": "TEXT"}

_NEW_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL DEFAULT 'New chat',
    market TEXT NOT NULL DEFAULT 'us',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    chunks_json TEXT,
    charts_json TEXT,
    metadata_json TEXT,
    latency REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    # Legacy table (kept so old rows can be migrated) + missing columns.
    conn.executescript(_LEGACY_SCHEMA)
    existing = {r[1] for r in conn.execute("PRAGMA table_info(query_log)").fetchall()}
    for col, typ in _LEGACY_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE query_log ADD COLUMN {col} {typ}")

    conn.executescript(_NEW_SCHEMA)
    conn.commit()
    _migrate_query_log(conn)


def _migrate_query_log(conn: sqlite3.Connection) -> None:
    """One-shot: turn every legacy query_log row into a one-turn chat.

    Idempotent — runs only when the `chats` table is empty AND query_log has
    rows. Once the user has any new chats, this is skipped forever.
    """
    have_chats = conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
    if have_chats > 0:
        return
    legacy = conn.execute(
        "SELECT id, question, market, answer, latency, chunks_json, "
        "       metadata_json, created_at "
        "FROM query_log ORDER BY id"
    ).fetchall()
    if not legacy:
        return

    for r in legacy:
        title = ((r["question"] or "").strip()[:60].rstrip(" .?!,")) or "Imported chat"
        cur = conn.execute(
            "INSERT INTO chats (title, market, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (title, r["market"] or "us", r["created_at"], r["created_at"]),
        )
        chat_id = cur.lastrowid
        # user turn
        conn.execute(
            "INSERT INTO messages "
            "  (chat_id, role, content, chunks_json, charts_json, metadata_json, "
            "   latency, created_at) "
            "VALUES (?, 'user', ?, NULL, '[]', '{}', NULL, ?)",
            (chat_id, r["question"] or "", r["created_at"]),
        )
        # assistant turn — carry over chunks/metadata if present
        conn.execute(
            "INSERT INTO messages "
            "  (chat_id, role, content, chunks_json, charts_json, metadata_json, "
            "   latency, created_at) "
            "VALUES (?, 'assistant', ?, ?, '[]', ?, ?, ?)",
            (
                chat_id, r["answer"] or "",
                r["chunks_json"] or "[]",
                r["metadata_json"] or "{}",
                r["latency"], r["created_at"],
            ),
        )
    conn.commit()


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #

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
# Chats — CRUD
# --------------------------------------------------------------------------- #

def create_chat(title: str = "New chat", market: str = "us") -> dict:
    c = conn()
    cur = c.execute(
        "INSERT INTO chats (title, market) VALUES (?, ?)", (title, market),
    )
    c.commit()
    return get_chat(cur.lastrowid or 0) or {}


def get_chat(chat_id: int) -> Optional[dict]:
    c = conn()
    row = c.execute(
        "SELECT id, title, market, created_at, updated_at FROM chats WHERE id = ?",
        (chat_id,),
    ).fetchone()
    return dict(row) if row else None


def list_chats(limit: int = 200) -> list[dict]:
    c = conn()
    rows = c.execute(
        """
        SELECT c.id, c.title, c.market, c.created_at, c.updated_at,
               (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS message_count,
               (SELECT m.content FROM messages m
                WHERE m.chat_id = c.id AND m.role = 'user'
                ORDER BY m.id ASC LIMIT 1) AS preview
        FROM chats c
        ORDER BY c.updated_at DESC, c.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def rename_chat(chat_id: int, title: str) -> bool:
    c = conn()
    cur = c.execute(
        "UPDATE chats SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (title, chat_id),
    )
    c.commit()
    return cur.rowcount > 0


def touch_chat(chat_id: int) -> None:
    c = conn()
    c.execute(
        "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (chat_id,),
    )
    c.commit()


def delete_chat(chat_id: int) -> bool:
    c = conn()
    c.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    cur = c.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    c.commit()
    return cur.rowcount > 0


def delete_all_chats() -> int:
    c = conn()
    c.execute("DELETE FROM messages")
    cur = c.execute("DELETE FROM chats")
    c.commit()
    return cur.rowcount


def auto_title_if_default(chat_id: int, question: str) -> None:
    """If the chat is still titled 'New chat', derive a title from the first question."""
    c = conn()
    row = c.execute("SELECT title FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if not row or row["title"] != "New chat":
        return
    snippet = (question or "").strip().split("\n")[0][:60].rstrip(" .?!,")
    if snippet:
        rename_chat(chat_id, snippet)


# --------------------------------------------------------------------------- #
# Messages — append + read
# --------------------------------------------------------------------------- #

def add_message(
    chat_id: int,
    role: str,
    content: str,
    chunks: Optional[list[dict]] = None,
    charts: Optional[list[dict]] = None,
    metadata: Optional[dict[str, Any]] = None,
    latency: Optional[float] = None,
) -> int:
    c = conn()
    cur = c.execute(
        "INSERT INTO messages "
        " (chat_id, role, content, chunks_json, charts_json, metadata_json, latency) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            chat_id, role, content,
            json.dumps(chunks or [], default=str, ensure_ascii=False),
            json.dumps(charts or [], default=str, ensure_ascii=False),
            json.dumps(metadata or {}, default=str, ensure_ascii=False),
            latency,
        ),
    )
    touch_chat(chat_id)
    c.commit()
    return cur.lastrowid or 0


def list_messages(chat_id: int) -> list[dict]:
    c = conn()
    rows = c.execute(
        "SELECT id, chat_id, role, content, chunks_json, charts_json, "
        "       metadata_json, latency, created_at "
        "FROM messages WHERE chat_id = ? ORDER BY id ASC",
        (chat_id,),
    ).fetchall()
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        d["chunks"] = json.loads(d.pop("chunks_json") or "[]")
        d["charts"] = json.loads(d.pop("charts_json") or "[]")
        d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
        out.append(d)
    return out


def recent_turns(chat_id: int, k: int = 6) -> list[dict]:
    """Last K messages (oldest → newest), trimmed for the agent's context window."""
    msgs = list_messages(chat_id)
    return msgs[-k:]
