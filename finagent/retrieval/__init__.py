"""Retrieval pipeline (canonical home; was mixed into `finagent.graph`).

The retrieval stack — BM25, dense, hybrid fusion, and cross-encoder reranking —
behind a `BaseRetriever` interface.

Public exports:
    BaseRetriever        — abstract retriever interface.
    HybridRetriever      — BM25 ∪ dense + rerank (the retriever the agent uses).
    BM25Retriever        — standalone lexical retriever.
    DenseRetriever       — standalone semantic retriever.
    CrossEncoderReranker — OOP wrapper over the shared cross-encoder.

`finagent.graph.corrective` re-exports `HybridRetriever`/`_get_shared_reranker`
for backwards compatibility during migration.

Depends on: config, corpus, the low-level vectorstore helpers.
"""

from finagent.retrieval.base import BaseRetriever
from finagent.retrieval.hybrid import HybridRetriever
from finagent.retrieval.bm25 import BM25Retriever
from finagent.retrieval.dense import DenseRetriever
from finagent.retrieval.reranker import CrossEncoderReranker, _get_shared_reranker

__all__ = [
    "BaseRetriever",
    "HybridRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "CrossEncoderReranker",
    "_get_shared_reranker",
]
