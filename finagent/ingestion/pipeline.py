"""
pipeline.py  ·  finagent/ingestion/pipeline.py

The end-to-end ingestion orchestrator: parse → chunk → embed → store. This is
the public entry point to the ingestion layer.

The orchestration is implemented by `CorpusIngester` (in `ingest.py`), which
walks a manifest, extracts per-page text (PDF via pypdf, HTML via unstructured),
chunks with the production splitter, embeds with BGE, and writes to Chroma. This
module exposes it under the clean `pipeline` name and adds a one-call helper.
"""

from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

from finagent.config import settings
from finagent.ingestion.ingest import CorpusIngester, IngestionStats


def ingest_corpus(
    manifest_path: Union[str, Path],
    collection_name: str,
    market: str = "us",
    chroma_dir: Optional[str] = None,
    reset: bool = False,
) -> IngestionStats:
    """Run the full ingestion pipeline for one manifest into one collection.

    Args:
        manifest_path: JSON manifest of source files (from the fetch step).
        collection_name: target Chroma collection.
        market: "us" or "india" — stored as chunk metadata.
        chroma_dir: Chroma directory; defaults to `settings.chroma_dir`.
        reset: drop the collection before ingesting.

    Returns:
        IngestionStats for the run.
    """
    ing = CorpusIngester(
        chroma_dir=chroma_dir or settings.chroma_dir,
        collection_name=collection_name,
        market=market,
    )
    if reset:
        ing.reset_collection()
    return ing.ingest_all(manifest_path=str(manifest_path))


__all__ = ["CorpusIngester", "IngestionStats", "ingest_corpus"]
