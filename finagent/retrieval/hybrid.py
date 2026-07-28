"""
hybrid.py  ·  finagent/retrieval/hybrid.py

`HybridRetriever` — hybrid (lexical + semantic) retrieval over a Qdrant
collection, reranked with a cross-encoder. This is the retriever the agent uses.

Both halves now run inside the database. Qdrant scores a BM25 *sparse* vector
and a dense vector for the same query and fuses the two rankings with Reciprocal
Rank Fusion, returning one list.

This used to be an in-process `rank_bm25` index built by paging the entire
collection into RAM at startup. That is why the corpus had to be baked into the
container image, and it had two defects a writable corpus makes fatal: the index
froze its IDF statistics at build time, so any filing added afterwards was
invisible to lexical search; and rebuilding it meant re-reading every chunk.
Server-side sparse vectors keep lexical scoring correct as documents arrive.

RRF also replaces a hardcoded 50/50 pool split (24 lexical + 24 dense) with a
per-query mix: a question naming an exact account line leans lexical, a
"how is the company positioned" question leans semantic.
"""

from __future__ import annotations

from typing import Optional

from finagent.retrieval.reranker import _get_shared_reranker
from finagent.vectorstore import DEFAULT_EMBED_MODEL


class HybridRetriever:
    """Fuse lexical and semantic retrieval in Qdrant, then cross-encoder rerank.

    The reranker is loaded lazily and shared process-wide — first use
    downloads/loads the model.
    """

    # Measured best on FinanceBench (see results/RETRIEVAL_EXPERIMENTS.md):
    # hit@8 0.4000 -> 0.5467 vs bge-reranker-base on an identical pool.
    # Its 8192-token window also removes the parent-size ceiling that capped
    # base at 512 tokens (~2000 chars).
    DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
    # "BAAI/bge-reranker-base" is ~3.4x faster per pair on CPU and measurably
    # weaker; pass it explicitly if latency matters more than recall.

    def __init__(
        self,
        store,
        reranker_model: str = DEFAULT_RERANKER,
        # Candidate pool depth. The reranker can only reorder what the pool
        # contains, and 10+10 measurably capped recall; the fused pool is one
        # list now, so this is the total rather than a per-branch count.
        pool_top_k: int = 48,
        final_top_k: int = 5,
        use_mmr: bool = False,
        auto_filter: bool = True,
        parent_doc: bool = True,
        # Append the underlying statement line items when the query names a
        # derived metric ("quick ratio" -> "total current assets ..."). See
        # `finagent.retrieval.expansion`; measured in §13.
        expand_queries: bool = True,
    ):
        self.store = store
        self.reranker_model = reranker_model
        self.pool_top_k = pool_top_k
        self.final_top_k = final_top_k
        # Maximal Marginal Relevance for pool diversity. OFF, because
        # LangChain's MMR path is dense-only: it issues a NearestQuery with
        # `using="dense"` and no prefetch, so the sparse vector and the RRF
        # fusion below never run. Leaving this on silently reduced the whole
        # hybrid retriever to a single dense search — BM25 vectors were written
        # for every point at ingest and never read at query time. Do not flip it
        # back without re-measuring: `tests/test_hybrid_is_hybrid.py` asserts the
        # query that actually reaches Qdrant is the fused one.
        self.use_mmr = use_mmr
        # Company/year metadata filtering inferred from the query text (no LLM).
        # Without it, one company's question competes against every other
        # filing's near-identical accounting language.
        self.auto_filter = auto_filter
        # Parent-document retrieval: chunks are embedded as small children; after
        # matching, collapse each child back to its parent (bigger context) and
        # rerank the parents. Degrades gracefully — a chunk with no `parent_text`
        # is kept as-is.
        self.parent_doc = parent_doc
        self.expand_queries = expand_queries

        self._reranker = None
        self._company_vocab: Optional[dict] = None
        self._years_by_co: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def search(self, query: str,
               top_k: Optional[int] = None) -> list[tuple[str, dict]]:
        """Return up to `top_k` (default `final_top_k`) (text, metadata) pairs.

        This is `retrieve()` plus the query→filter inference and the relax
        fallback; the actual pool→collapse→rerank engine lives in `retrieve`.

        The derived-metric expansion happens HERE rather than inside `_pool`,
        so the filter inference, the candidate pool and the cross-encoder all
        see the same query text — that is the combination §13 measured.
        """
        if self.expand_queries:
            from finagent.retrieval.expansion import expand_query

            query = expand_query(query)
        top_k = self.final_top_k if top_k is None else top_k
        flt = self.infer_filter(query) if self.auto_filter else None
        hits = self.retrieve(query, top_k, flt)
        if not hits and flt and (flt.get("years") or flt.get("items")):
            # The YEAR (fiscal-calendar labelling) or the ITEM clause (a
            # collection ingested before section tagging) can over-narrow —
            # relax both back to company-only. The COMPANY clause is never
            # relaxed: chunks from a different company are noise that misleads
            # the synthesizer, and an empty result correctly hands off to the
            # fetch/web fallbacks. (retrieve() returns [] iff the pool is empty,
            # so this fires on exactly the over-narrowed case the old code did.)
            hits = self.retrieve(query, top_k, {"companies": flt["companies"]})
        return hits

    def infer_filter(self, query: str) -> Optional[dict]:
        """Company/year filter inferred from the query against the collection's
        own metadata vocabulary. None when no indexed company is named."""
        from finagent.retrieval.filters import build_company_vocab, infer_filter

        if self._company_vocab is None:
            self._company_vocab, self._years_by_co = build_company_vocab(
                self._all_metadata())
        return infer_filter(query, self._company_vocab, self._years_by_co)

    # ------------------------------------------------------------------ #
    # Retrieval stages, exposed as named steps. `search()` composes them
    # (get_pool → collapse → rerank); a notebook can also call each one
    # standalone to see what each contributes. `flt` defaults to off, so a
    # bare `get_pool(query)` / `retrieve(query)` gives the raw, unfiltered
    # view while `search()` passes the inferred company/year filter through.
    # ------------------------------------------------------------------ #

    @classmethod
    def from_collection(
        cls,
        collection: str,
        embedding_model: str = DEFAULT_EMBED_MODEL,
        **kwargs,
    ) -> "HybridRetriever":
        """Build a retriever directly from a Qdrant collection name."""
        from finagent.vectorstore import build_store

        return cls(build_store(collection, embedding_model), **kwargs)

    def get_pool(self, query: str,
                 flt: Optional[dict] = None) -> list[tuple[str, dict]]:
        """The fused (sparse+dense, RRF) candidate pool, before reranking."""
        return self._pool(query, flt)

    def rerank(
        self, query: str, candidates: list[tuple[str, dict]], top_k: int = 5
    ) -> list[tuple[str, dict]]:
        """Cross-encoder rerank a candidate pool; return the top_k."""
        return self._rerank(query, candidates)[:top_k]

    def retrieve(self, query: str, k: int = 5,
                 flt: Optional[dict] = None) -> list[tuple[str, dict]]:
        """The retrieval engine: pool → collapse-to-parents → rerank → top-k.
        `search()` wraps this with filter inference and the relax fallback."""
        pool = self.get_pool(query, flt)
        if not pool:
            return []
        pool = self._collapse_to_parents(pool)
        return self.rerank(query, pool, k)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _pool(self, query: str,
              flt: Optional[dict] = None) -> list[tuple[str, dict]]:
        """Fused (sparse + dense, RRF) candidates from Qdrant."""
        from finagent.retrieval.filters import qdrant_filter

        qf = qdrant_filter(flt)
        try:
            if self.use_mmr:
                docs = self.store.max_marginal_relevance_search(
                    query, k=self.pool_top_k,
                    fetch_k=max(20, self.pool_top_k * 2), lambda_mult=0.5,
                    filter=qf)
            else:
                docs = self.store.similarity_search(
                    query, k=self.pool_top_k, filter=qf)
        except Exception:
            # Collection missing/empty, or a transient cluster error: an empty
            # pool correctly hands off to the fetch/web fallbacks.
            return []

        # Dedupe by (source, page, prefix) — the same key the rest of the graph
        # uses. Fusion can surface one chunk from both branches.
        seen, out = set(), []
        for d in docs:
            meta = d.metadata or {}
            key = (meta.get("local_path", ""), meta.get("page", ""),
                   d.page_content[:80])
            if key in seen:
                continue
            seen.add(key)
            out.append((d.page_content, meta))
        return out

    def _all_metadata(self):
        """Synthetic (company, ticker, year) rows for the vocabulary builder —
        sourced from the cheap facet index rather than a full-payload scroll of
        every point (see `vectorstore.company_facet_index`).
        `build_company_vocab` reads only these three fields, so this feeds it
        exactly what a real scroll would — `tests/test_qdrant_writes` pins that
        the two paths build an identical vocabulary."""
        from finagent.vectorstore import company_facet_index

        idx = company_facet_index(self.store.collection_name)
        rows = []
        for c, f in idx.items():
            years = sorted(f.get("years") or [""]) or [""]
            tickers = sorted(f.get("tickers") or [""]) or [""]
            for y in years:
                for t in tickers:
                    rows.append({"company": c, "ticker": t, "year": y})
        return rows

    def _collapse_to_parents(
        self, hits: list[tuple[str, dict]]
    ) -> list[tuple[str, dict]]:
        """Parent-document retrieval: replace each matched child with its parent
        text (bigger context), deduped by (local_path, parent_id) so several
        children of one parent collapse to a single candidate. Order is
        preserved (best-ranked child first) — the reranker re-sorts anyway.
        Chunks without parent metadata pass through.
        """
        if not self.parent_doc:
            return hits
        seen, out = set(), []
        for text, meta in hits:
            pid = meta.get("parent_id")
            ptext = meta.get("parent_text")
            if pid is None or not ptext:
                key = ("__child__", meta.get("local_path", ""),
                       meta.get("element_index", ""), text[:80])
                parent_text = text
            else:
                key = ("__parent__", meta.get("local_path", ""), pid)
                parent_text = ptext
            if key in seen:
                continue
            seen.add(key)
            out.append((parent_text, meta))
        return out

    def _rerank(self, query: str, candidates: list[tuple[str, dict]]):
        reranker = self._ensure_reranker()
        pairs = [(query, text) for text, _ in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        # The score is deliberately NOT attached to the chunk. §11 propagated it
        # so the graph's global cap could rank on it instead of re-scoring the
        # merged pool against the question; §13 measured that the re-score is
        # the better rule (58 -> 61 of 99), so nothing downstream reads it.
        return [c for c, _ in ranked]

    def _ensure_reranker(self):
        if self._reranker is None:
            # Share ONE CrossEncoder per model across every HybridRetriever in
            # the process. The agent builds one retriever per collection, so
            # without this each collection loaded its own ~1.1 GB reranker —
            # two collections blew past Cloud Run's 4 GiB and OOM-killed the
            # container (exit 137) mid-query.
            self._reranker = _get_shared_reranker(self.reranker_model)
        return self._reranker
