"""
stats.py  ·  finagent/corpus/stats.py

Read-only corpus statistics for a Chroma collection: how many companies,
documents, and chunks are indexed, the year range, and the on-disk size. Used by
the reference notebook to answer "what's actually in the corpus?".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from finagent.vectorstore import build_store


def _all_metadatas(store) -> list[dict]:
    """Page through a collection and return every chunk's metadata."""
    col = store._collection
    metas, offset = [], 0
    while True:
        batch = col.get(include=["metadatas"], limit=5000, offset=offset)
        got = batch.get("metadatas") or []
        if not got:
            break
        metas.extend(got)
        offset += len(got)
        if len(got) < 5000:
            break
    return metas


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1e6


def corpus_summary(
    collection: str = "us_filings",
    chroma_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Return a one-row-per-metric summary DataFrame for `collection`.

    Columns: metric, value. Designed to `display()` cleanly in a notebook.
    """
    if chroma_dir is None:
        chroma_dir = os.getenv("CHROMA_DIR", "data/chroma")

    store = build_store(collection, chroma_dir=chroma_dir)
    metas = _all_metadatas(store)

    companies = {m.get("company") or m.get("ticker", "") for m in metas} - {""}
    docs = {m.get("source_url") or m.get("local_path", "") for m in metas} - {""}
    years = sorted(
        {str(m.get("year", "")) for m in metas if str(m.get("year", "")).isdigit()}
    )

    rows = [
        ("collection", collection),
        ("companies", len(companies)),
        ("documents", len(docs)),
        ("chunks", len(metas)),
        ("year_range", f"{years[0]}–{years[-1]}" if years else "—"),
        ("chroma_dir_size_mb", round(_dir_size_mb(Path(chroma_dir)), 1)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def corpus_by_company(
    collection: str = "us_filings",
    chroma_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Chunk counts per company (descending) — a fuller view of the corpus."""
    if chroma_dir is None:
        chroma_dir = os.getenv("CHROMA_DIR", "data/chroma")
    store = build_store(collection, chroma_dir=chroma_dir)
    metas = _all_metadatas(store)
    df = pd.DataFrame([
        {"company": m.get("company") or m.get("ticker", "?"),
         "year": m.get("year", "?")}
        for m in metas
    ])
    return (
        df.groupby("company").size()
        .sort_values(ascending=False)
        .rename("chunks").reset_index()
    )
