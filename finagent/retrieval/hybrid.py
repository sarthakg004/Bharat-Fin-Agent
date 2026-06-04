"""
hybrid.py  ·  finagent/retrieval/hybrid.py

`HybridRetriever` — fuse BM25 (lexical) and dense (semantic) retrieval, then
rerank the union with a cross-encoder. This is the retriever the agent uses.

Migrated verbatim from `finagent.graph.corrective` during the layout
restructure; `finagent.graph.corrective` keeps a re-export shim so existing
import paths (`from finagent.graph.corrective import HybridRetriever`) still work.
"""

from __future__ import annotations

from typing import Optional

from finagent.retrieval.base import BaseRetriever
from finagent.retrieval.reranker import _get_shared_reranker


class HybridRetriever(BaseRetriever):
    """Fuse BM25 and dense retrieval, then rerank with a cross-encoder.

    Builds the BM25 index lazily over every chunk in the Chroma collection
    (batched to avoid SQLite's "too many SQL variables" limit). The reranker
    is also loaded lazily — first use downloads/loads the model.
    """

    DEFAULT_RERANKER = "BAAI/bge-reranker-large"  # spec'd model; ~1.3 GB
    # Use "BAAI/bge-reranker-base" (~280 MB) for a much faster but slightly
    # weaker reranker — pass reranker_model="BAAI/bge-reranker-base".

    def __init__(
        self,
        chroma_store,
        reranker_model: str = DEFAULT_RERANKER,
        bm25_top_k: int = 10,
        dense_top_k: int = 10,
        final_top_k: int = 5,
        fetch_batch_size: int = 1000,
        use_mmr: bool = True,
    ):
        self.store = chroma_store
        self.reranker_model = reranker_model
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.final_top_k = final_top_k
        self.fetch_batch_size = fetch_batch_size
        # Phase 10: select dense candidates with Maximal Marginal Relevance so
        # the pool is diverse, not five near-identical passages. Falls back to
        # plain similarity if the store doesn't support MMR.
        self.use_mmr = use_mmr

        self._bm25 = None
        self._bm25_empty = False          # True once we've seen an empty collection
        self._all_docs: Optional[list[tuple[str, dict]]] = None
        self._reranker = None

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def search(self, query: str) -> list[tuple[str, dict]]:
        """Return up to `final_top_k` (text, metadata) pairs for the query."""
        union = self._bm25_then_dense(query)
        if not union:
            return []
        return self._rerank(query, union)[: self.final_top_k]

    # ------------------------------------------------------------------ #
    # Demonstration / inspection API (used by the reference notebook).
    # These expose each retrieval stage separately so a reader can see what
    # BM25, dense, fusion, and the reranker each contribute. They reuse the
    # same internals as `search()`.
    # ------------------------------------------------------------------ #

    @classmethod
    def from_collection(
        cls,
        collection: str,
        chroma_dir: Optional[str] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        **kwargs,
    ) -> "HybridRetriever":
        """Build a retriever directly from a Chroma collection name."""
        from finagent.vectorstore import build_store

        store = build_store(collection, embedding_model, chroma_dir)
        return cls(store, **kwargs)

    def bm25_search(self, query: str, k: int = 5) -> list[tuple[str, dict, float]]:
        """Top-k lexical (BM25) hits as (text, metadata, score)."""
        bm25, docs = self._ensure_bm25()
        if not docs:
            return []
        scores = bm25.get_scores(self._tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(docs[i][0], docs[i][1], float(scores[i])) for i in top]

    def dense_search(self, query: str, k: int = 5) -> list[tuple[str, dict, float]]:
        """Top-k dense (embedding) hits as (text, metadata, relevance score)."""
        hits = self.store.similarity_search_with_relevance_scores(query, k=k)
        return [(d.page_content, d.metadata, float(s)) for d, s in hits]

    def get_pool(self, query: str) -> list[tuple[str, dict]]:
        """The deduped BM25 ∪ dense candidate pool, before reranking."""
        return self._bm25_then_dense(query)

    def rerank(
        self, query: str, candidates: list[tuple[str, dict]], top_k: int = 5
    ) -> list[tuple[str, dict]]:
        """Cross-encoder rerank a candidate pool; return the top_k."""
        return self._rerank(query, candidates)[:top_k]

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, dict]]:
        """Full hybrid retrieval (pool → rerank) returning the top-k."""
        union = self._bm25_then_dense(query)
        if not union:
            return []
        return self._rerank(query, union)[:k]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _bm25_then_dense(self, query: str) -> list[tuple[str, dict]]:
        bm25, docs = self._ensure_bm25()
        # Empty collection (e.g. a not-yet-ingested corpus): nothing to retrieve.
        # Guard here so BM25Okapi is never built over an empty corpus (which
        # divides by zero) and downstream merges just see no candidates.
        if not docs:
            return []

        scores = bm25.get_scores(self._tokenize(query))
        # argsort descending for top-k
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.bm25_top_k]
        bm25_hits = [docs[i] for i in top_idx]

        dense_hits = [
            (d.page_content, d.metadata) for d in self._dense_docs(query)
        ]

        # Dedupe by (local_path, page, prefix) — same key the rest of the
        # graph uses elsewhere.
        seen, union = set(), []
        for text, meta in bm25_hits + dense_hits:
            key = (meta.get("local_path", ""), meta.get("page", ""), text[:80])
            if key in seen:
                continue
            seen.add(key)
            union.append((text, meta))
        return union

    def _dense_docs(self, query: str):
        """Dense candidates — MMR (diverse) when enabled, else plain similarity.

        MMR over-fetches `fetch_k` and greedily picks `dense_top_k` that balance
        relevance with novelty, so the pool isn't five paraphrases of one
        passage. Any error (older Chroma, embedding quirk) falls back cleanly.
        """
        if self.use_mmr:
            try:
                return self.store.max_marginal_relevance_search(
                    query, k=self.dense_top_k,
                    fetch_k=max(20, self.dense_top_k * 4), lambda_mult=0.5,
                )
            except Exception:
                pass
        return self.store.similarity_search(query, k=self.dense_top_k)

    def _rerank(self, query: str, candidates: list[tuple[str, dict]]):
        reranker = self._ensure_reranker()
        pairs = [(query, text) for text, _ in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [c for c, _ in ranked]

    def _ensure_bm25(self):
        if self._bm25 is None and not self._bm25_empty:
            from rank_bm25 import BM25Okapi

            self._all_docs = self._fetch_all_chunks()
            if not self._all_docs:
                # Empty collection — don't build BM25 (BM25Okapi([]) raises
                # ZeroDivisionError). Mark it so we don't re-fetch every query.
                self._bm25_empty = True
                return None, []
            tokenized = [self._tokenize(t) for t, _ in self._all_docs]
            self._bm25 = BM25Okapi(tokenized)
        return self._bm25, (self._all_docs or [])

    def _ensure_reranker(self):
        if self._reranker is None:
            # Share ONE CrossEncoder per model across every HybridRetriever in
            # the process. The agent builds one retriever per collection, so
            # without this each collection loaded its own ~1.1 GB reranker —
            # two collections blew past Cloud Run's 4 GiB and OOM-killed the
            # container (exit 137) mid-query.
            self._reranker = _get_shared_reranker(self.reranker_model)
        return self._reranker

    def _fetch_all_chunks(self) -> list[tuple[str, dict]]:
        """Page through the Chroma collection. Doing it in one call hits
        SQLite's parameter cap on large collections."""
        col = self.store._collection
        out: list[tuple[str, dict]] = []
        offset = 0
        while True:
            batch = col.get(
                include=["documents", "metadatas"],
                limit=self.fetch_batch_size,
                offset=offset,
            )
            docs = batch.get("documents") or []
            metas = batch.get("metadatas") or []
            if not docs:
                break
            out.extend(zip(docs, metas))
            offset += len(docs)
            if len(docs) < self.fetch_batch_size:
                break
        return out

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # Minimal BM25 tokenizer — lowercased whitespace split. Sufficient for
        # financial-prose retrieval; swap in a real tokenizer if you want.
        return text.lower().split()
