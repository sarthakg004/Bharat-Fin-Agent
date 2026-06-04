"""
dense.py  ·  finagent/retrieval/dense.py

Standalone dense (semantic) retriever over a Chroma collection. Embeds the query
and returns the nearest chunk vectors by cosine similarity — catches paraphrases
that lexical search misses.

A focused, single-responsibility implementation of `BaseRetriever`. The agent
currently uses `HybridRetriever`; this class exists so dense retrieval can be
used or evaluated in isolation.
"""

from __future__ import annotations

from finagent.retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Embedding-similarity retriever backed by a Chroma store.

    Args:
        chroma_client: a `finagent.corpus.ChromaClient`.
        collection: collection name to query.
    """

    def __init__(self, chroma_client, collection: str):
        self._store = chroma_client.get_store(collection)

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, dict, float]]:
        """Top-k `(text, metadata, relevance_score)` by embedding similarity."""
        hits = self._store.similarity_search_with_relevance_scores(query, k=k)
        return [(d.page_content, d.metadata, float(s)) for d, s in hits]
