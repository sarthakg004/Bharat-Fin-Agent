"""
chroma_client.py  ·  finagent/chroma_client.py

One place that knows where Chroma lives. The project is **local-first**: both
local development and the deployed HF Space read the same on-disk Chroma
`PersistentClient` directory (668 MB of prebuilt vectors), so behaviour is
identical in both environments.

Chroma Cloud remains available as an explicit opt-in — set
`CHROMA_BACKEND=cloud` (plus the cloud creds). Note we deliberately do NOT
switch to cloud just because `CHROMA_API_KEY` happens to be in your `.env`;
that key may linger from earlier experiments and silently pointing local dev
at a half-populated cloud DB is exactly the inconsistency we want to avoid.

Environment
-----------
    CHROMA_DIR          — Local directory (default: "data/chroma";
                          the Space sets it to the /data mount).
    CHROMA_BACKEND      — "cloud" to use Chroma Cloud; anything else = local.
    CHROMA_API_KEY      — Chroma Cloud API key  (only when backend=cloud)
    CHROMA_TENANT       — Cloud tenant          (default: "default_tenant")
    CHROMA_DATABASE     — Cloud database        (default: "default_database")
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
    """Construct the Chroma client. Local by default; cloud only on opt-in."""
    import chromadb

    if is_cloud():
        return chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT", "default_tenant"),
            database=os.getenv("CHROMA_DATABASE", "default_database"),
        )

    persist_dir = Path(os.getenv("CHROMA_DIR", "data/chroma"))
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_chroma_client() -> Any:
    """Return the shared Chroma client (cloud or local), built once."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = _build_client()
    return _client


def is_cloud() -> bool:
    """Cloud only when explicitly opted in *and* a key is present."""
    return (
        os.getenv("CHROMA_BACKEND", "").strip().lower() == "cloud"
        and bool(os.getenv("CHROMA_API_KEY"))
    )


def chroma_kwargs_for_langchain(persist_dir: Union[str, Path, None] = None) -> dict:
    """Kwargs to pass into `langchain_chroma.Chroma(...)`.

    Local mode keeps `persist_directory` for backward-compat (so we don't
    break callers that already pass a path). Cloud mode passes a `client`
    instance instead, and langchain_chroma will use it transparently.
    """
    if is_cloud():
        return {"client": get_chroma_client()}
    if persist_dir is None:
        persist_dir = os.getenv("CHROMA_DIR", "data/chroma")
    return {"persist_directory": str(persist_dir)}


# --------------------------------------------------------------------------- #
# Diagnostic
# --------------------------------------------------------------------------- #

def describe() -> dict:
    """Useful at startup / on the `/api/health` route to confirm where data lives."""
    mode = "cloud" if is_cloud() else "local"
    info: dict[str, Optional[str]] = {"mode": mode}
    if mode == "cloud":
        info["tenant"] = os.getenv("CHROMA_TENANT", "default_tenant")
        info["database"] = os.getenv("CHROMA_DATABASE", "default_database")
    else:
        info["dir"] = os.getenv("CHROMA_DIR", "data/chroma")
    return info
