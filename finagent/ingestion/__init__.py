"""Data ingestion pipeline: raw filings → parsed text → chunks → embeddings → Chroma.

PDFs are parsed with pypdf (reliable on born-digital annual reports); SEC HTML
filings with unstructured's partition_html + chunk_by_title. User-uploaded
documents go through `upload.parse_upload` (Docling) instead.

Public exports:
    ingest_corpus    — one-call pipeline entry point (parse → chunk → embed → store).
    CorpusIngester   — the orchestrator class (also the `ingest.py` CLI).
"""

from finagent.ingestion.pipeline import ingest_corpus, CorpusIngester, IngestionStats

__all__ = [
    "ingest_corpus",
    "CorpusIngester",
    "IngestionStats",
]
