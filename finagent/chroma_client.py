"""
chroma_client.py  ·  finagent/chroma_client.py

One place that knows where the on-disk Chroma store lives. The same prebuilt
`data/chroma` directory is read in local development and in the deployed
container (where it's baked into the image), so behaviour is identical.

Environment
-----------
    CHROMA_DIR   — Chroma directory (default "data/chroma"; the image sets it
                   to the baked-in path).
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Union

from dotenv import load_dotenv

load_dotenv()


_lock = Lock()
# One PersistentClient PER on-disk path, shared across every Chroma store on that
# path. This is critical for correctness, not just efficiency: chromadb's
# hnswlib/sqlite backend is NOT safe for two separate client handles touching the
# same collection concurrently — the dynamic-fetch ingest WRITES `us_filings`
# while the agent's retriever holds it OPEN for reads, and two native handles on
# one segment SIGSEGVs the process. Sharing a single client routes both through
# chromadb's own locking instead.
_clients: dict[str, Any] = {}


def _resolve_dir(persist_dir: Union[str, Path, None]) -> str:
    if persist_dir is None:
        persist_dir = os.getenv("CHROMA_DIR", "data/chroma")
    return str(Path(persist_dir).resolve())


def get_chroma_client(persist_dir: Union[str, Path, None] = None) -> Any:
    """Return the shared Chroma PersistentClient for `persist_dir`, built once."""
    key = _resolve_dir(persist_dir)
    client = _clients.get(key)
    if client is None:
        with _lock:
            client = _clients.get(key)
            if client is None:
                import chromadb

                Path(key).mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=key)
                _clients[key] = client
    return client


def chroma_kwargs_for_langchain(persist_dir: Union[str, Path, None] = None) -> dict:
    """Kwargs for `langchain_chroma.Chroma(...)`.

    Returns the SHARED client (not a bare `persist_directory`) so every store on
    a given path reuses one chromadb handle — see `_clients` above for why this
    prevents a concurrent read/write segfault.
    """
    return {"client": get_chroma_client(persist_dir)}


def describe() -> dict:
    """Used by /api/health to confirm where data lives."""
    info: dict[str, Optional[str]] = {"mode": "local"}
    info["dir"] = os.getenv("CHROMA_DIR", "data/chroma")
    return info
