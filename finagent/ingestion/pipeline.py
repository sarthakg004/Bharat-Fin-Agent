"""
pipeline.py  ·  finagent/ingestion/pipeline.py

The end-to-end ingestion orchestrator: parse → chunk → embed → store. This is
the public entry point to the ingestion layer.

The orchestration is implemented by `CorpusIngester` (in `ingest.py`), which
walks a manifest, extracts per-page text (PDF via pypdf, HTML via unstructured),
chunks with the production splitter, embeds with BGE, and writes to Qdrant. This
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
    reset: bool = False,
) -> IngestionStats:
    """Run the full ingestion pipeline for one manifest into one collection.

    Args:
        manifest_path: JSON manifest of source files (from the fetch step).
        collection_name: target Qdrant collection.
        market: stored as chunk metadata (default "us").
        reset: drop the collection before ingesting.

    Returns:
        IngestionStats for the run.
    """
    ing = CorpusIngester(
        collection_name=collection_name,
        market=market,
    )
    if reset:
        ing.reset_collection()
    return ing.ingest_all(manifest_path=str(manifest_path))


__all__ = ["CorpusIngester", "IngestionStats", "ingest_corpus"]
