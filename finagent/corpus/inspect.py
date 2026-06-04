"""
inspect.py  ·  finagent/corpus/inspect.py

Pull a single real chunk (document text + metadata) out of a Chroma collection,
so the notebook can show exactly what one stored record looks like.
"""

from __future__ import annotations

import os
from typing import Optional

from finagent.vectorstore import build_store


def get_sample_chunk(
    collection: str = "us_filings",
    chroma_dir: Optional[str] = None,
    index: int = 0,
) -> dict:
    """Return ``{"document": <text>, "metadata": {...}}`` for one chunk."""
    if chroma_dir is None:
        chroma_dir = os.getenv("CHROMA_DIR", "data/chroma")

    store = build_store(collection, chroma_dir=chroma_dir)
    batch = store._collection.get(
        include=["documents", "metadatas"], limit=index + 1
    )
    docs = batch.get("documents") or []
    metas = batch.get("metadatas") or []
    if not docs:
        raise ValueError(f"Collection {collection!r} is empty — nothing to sample.")
    i = min(index, len(docs) - 1)
    return {"document": docs[i], "metadata": metas[i]}
