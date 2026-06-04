"""Data ingestion pipeline: raw filings → parsed elements → chunks → embeddings → Chroma.

Public exports:
    ingest_corpus    — one-call pipeline entry point (parse → chunk → embed → store).
    CorpusIngester   — the orchestrator class (also the `ingest.py` CLI).
    parse_document   — PDF/HTML → unstructured Elements.
    chunk_text       — text → overlapping chunks (production splitter config).
    embed_text       — text → normalized vector.

Depends on: config, corpus, the low-level vectorstore helpers.
"""

from finagent.ingestion.pipeline import ingest_corpus, CorpusIngester, IngestionStats
from finagent.ingestion.parser import parse_document, element_summary
from finagent.ingestion.chunker import chunk_text
from finagent.ingestion.embedder import embed_text, cosine

__all__ = [
    "ingest_corpus",
    "CorpusIngester",
    "IngestionStats",
    "parse_document",
    "element_summary",
    "chunk_text",
    "embed_text",
    "cosine",
]
