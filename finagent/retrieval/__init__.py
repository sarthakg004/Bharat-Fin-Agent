"""Retrieval pipeline (canonical home; was mixed into `finagent.graph`).

BM25 ∪ dense hybrid fusion with cross-encoder reranking, behind
`HybridRetriever` — the retriever the agent uses.

Public exports:
    HybridRetriever      — BM25 ∪ dense + rerank.

`finagent.graph.corrective` re-exports `HybridRetriever`/`_get_shared_reranker`
for backwards compatibility during migration.
"""

from finagent.retrieval.hybrid import HybridRetriever
from finagent.retrieval.reranker import _get_shared_reranker

__all__ = [
    "HybridRetriever",
    "_get_shared_reranker",
]
