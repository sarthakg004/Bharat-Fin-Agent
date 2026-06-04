"""
inspect.py  ·  finagent/corpus/inspect.py

Pull a single real chunk (document text + metadata) out of a Chroma collection,
so the notebook can show exactly what one stored record looks like.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from finagent.vectorstore import build_store


def sample_filing(
    corpus_dir: Union[str, Path] = "data/us/pdfs",
    suffixes: tuple = (".htm", ".html", ".pdf"),
) -> Optional[Path]:
    """Return the path to one real source filing under `corpus_dir`, or None.

    Used by the reference notebook so the parsing demo picks whatever filing was
    actually fetched (filenames carry SEC accession numbers, so they can't be
    hardcoded). Skips the downloader's nested `sec-edgar-filings/` tree in favour
    of the flattened per-ticker copies.
    """
    root = Path(corpus_dir)
    if not root.exists():
        return None
    candidates = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in suffixes
        and p.is_file()
        and "sec-edgar-filings" not in p.parts
    )
    return candidates[0] if candidates else None


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
