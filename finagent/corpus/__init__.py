"""Vector-store (corpus) management.

Read/inspect views over the on-disk Chroma store used across the app and the
reference notebook.

Public exports:
    ChromaClient     — typed handle over the Chroma store (counts, stores, fetch).
    corpus_summary   — one-row-per-metric summary for a collection.
    corpus_by_company— chunk counts per company.
    get_sample_chunk — one real stored chunk (document + metadata).

Depends on: config, the low-level `finagent.chroma_client` / `finagent.vectorstore`.
"""

from finagent.corpus.client import ChromaClient
from finagent.corpus.stats import corpus_summary, corpus_by_company
from finagent.corpus.inspect import get_sample_chunk

__all__ = ["ChromaClient", "corpus_summary", "corpus_by_company", "get_sample_chunk"]
