"""
bm25.py  ·  finagent/retrieval/bm25.py

Standalone BM25 (lexical) retriever over a Chroma collection. BM25 ranks by
exact term overlap — strong for tickers, account names, and regulation numbers.

This is a focused, single-responsibility implementation of `BaseRetriever`. The
agent currently uses `HybridRetriever` (which builds its own BM25 index
internally); this class exists so BM25 can be used or evaluated in isolation.
"""

from __future__ import annotations

import re
from typing import Optional

from finagent.retrieval.base import BaseRetriever

# Same tokenizer as HybridRetriever: lowercase alphanumeric tokens, so
# punctuation never glues to terms ("revenue," ≠ "revenue").
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever(BaseRetriever):
    """Lexical retriever backed by `rank_bm25.BM25Okapi`.

    Args:
        chroma_client: a `finagent.corpus.ChromaClient`.
        collection: collection name to index.
    """

    def __init__(self, chroma_client, collection: str):
        self._client = chroma_client
        self._collection = collection
        self._bm25 = None
        self._docs: list[tuple[str, dict]] = []

    def _ensure_index(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            data = self._client.fetch_all(self._collection)
            self._docs = list(zip(data.get("documents", []), data.get("metadatas", [])))
            self._bm25 = BM25Okapi([_tokenize(t) for t, _ in self._docs])
        return self._bm25

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, dict, float]]:
        """Top-k `(text, metadata, score)` by BM25 score."""
        bm25 = self._ensure_index()
        scores = bm25.get_scores(_tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self._docs[i][0], self._docs[i][1], float(scores[i])) for i in top]
