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
_client: Any = None


def _build_client() -> Any:
    import chromadb

    persist_dir = Path(os.getenv("CHROMA_DIR", "data/chroma"))
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_chroma_client() -> Any:
    """Return the shared Chroma PersistentClient, built once."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = _build_client()
    return _client


def chroma_kwargs_for_langchain(persist_dir: Union[str, Path, None] = None) -> dict:
    """Kwargs for `langchain_chroma.Chroma(...)`."""
    if persist_dir is None:
        persist_dir = os.getenv("CHROMA_DIR", "data/chroma")
    return {"persist_directory": str(persist_dir)}


def describe() -> dict:
    """Used by /api/health to confirm where data lives."""
    info: dict[str, Optional[str]] = {"mode": "local"}
    info["dir"] = os.getenv("CHROMA_DIR", "data/chroma")
    return info
