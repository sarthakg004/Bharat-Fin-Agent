"""`HybridRetriever`: search a Qdrant collection by meaning and by exact words,
then rerank.

    query -> add line items for derived metrics (expansion.py)
          -> company/year filter inferred from the query (filters.py)
          -> Qdrant hybrid search: dense + BM25, fused by Reciprocal Rank Fusion
          -> swap each small child chunk for its larger parent passage
          -> rerank the parents, keep the top k
"""

from __future__ import annotations

from typing import Optional

from finagent.retrieval.reranker import get_reranker
from finagent.vectorstore import DEFAULT_EMBED_MODEL


class HybridRetriever:
    DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"

    def __init__(self, store, reranker_model: str = DEFAULT_RERANKER,
                 pool_top_k: int = 48, final_top_k: int = 5):
        self.store = store
        self.reranker_model = reranker_model
        # How many fused candidates to pull. The reranker can only reorder what
        # the pool contains; 48 was the measured knee (deeper costs more than it gains).
        self.pool_top_k = pool_top_k
        self.final_top_k = final_top_k
        self._company_vocab: Optional[dict] = None
        self._years_by_co: Optional[dict] = None

    @classmethod
    def from_collection(cls, collection: str, embedding_model: str = DEFAULT_EMBED_MODEL,
                        **kwargs) -> "HybridRetriever":
        from finagent.vectorstore import build_store

        return cls(build_store(collection, embedding_model), **kwargs)

    def search(self, query: str, top_k: Optional[int] = None) -> list[tuple[str, dict]]:
        """Up to `top_k` (text, metadata) pairs for `query`."""
        from finagent.retrieval.expansion import expand_query

        query = expand_query(query)
        top_k = self.final_top_k if top_k is None else top_k
        flt = self.infer_filter(query)
        hits = self.retrieve(query, top_k, flt)
        if not hits and flt and (flt.get("years") or flt.get("items")):
            # The year or section clause matched nothing: retry on the company
            # alone. The company clause is never dropped, because another
            # company's filing is worse than no result.
            hits = self.retrieve(query, top_k, {"companies": flt["companies"]})
        return hits

    def infer_filter(self, query: str) -> Optional[dict]:
        """Company/year filter from the query text. None when no indexed company is named."""
        from finagent.retrieval.filters import build_company_vocab, infer_filter

        if self._company_vocab is None:
            self._company_vocab, self._years_by_co = build_company_vocab(self._all_metadata())
        return infer_filter(query, self._company_vocab, self._years_by_co)

    def retrieve(self, query: str, k: int = 5, flt: Optional[dict] = None) -> list[tuple[str, dict]]:
        """pool -> parents -> rerank -> top k."""
        pool = self.get_pool(query, flt)
        if not pool:
            return []
        return self.rerank(query, self.collapse_to_parents(pool), k)

    def get_pool(self, query: str, flt: Optional[dict] = None) -> list[tuple[str, dict]]:
        """The fused (dense + BM25) candidate pool, before reranking."""
        from finagent.retrieval.filters import qdrant_filter

        try:
            # Must be `similarity_search`: LangChain's MMR search is dense-only
            # and would silently skip the BM25 half.
            docs = self.store.similarity_search(query, k=self.pool_top_k,
                                                filter=qdrant_filter(flt))
        except Exception:
            return []                   # missing collection or cluster error
        seen, out = set(), []
        for d in docs:
            meta = d.metadata or {}
            key = (meta.get("local_path", ""), meta.get("page", ""), d.page_content[:80])
            if key not in seen:
                seen.add(key)
                out.append((d.page_content, meta))
        return out

    @staticmethod
    def collapse_to_parents(hits: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
        """Replace each matched child with its parent text; several children of
        one parent become a single candidate."""
        seen, out = set(), []
        for text, meta in hits:
            pid, ptext = meta.get("parent_id"), meta.get("parent_text")
            if pid is None or not ptext:
                key = ("child", meta.get("local_path", ""), meta.get("element_index", ""), text[:80])
            else:
                key, text = ("parent", meta.get("local_path", ""), pid), ptext
            if key not in seen:
                seen.add(key)
                out.append((text, meta))
        return out

    def rerank(self, query: str, candidates: list[tuple[str, dict]],
               top_k: int = 5) -> list[tuple[str, dict]]:
        scores = get_reranker(self.reranker_model).predict(
            [(query, text) for text, _ in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [c for c, _ in ranked][:top_k]

    def _all_metadata(self) -> list[dict]:
        """(company, ticker, year) rows for the filter vocabulary, from Qdrant facets."""
        from finagent.vectorstore import company_facet_index

        rows = []
        for c, f in company_facet_index(self.store.collection_name).items():
            for y in sorted(f.get("years") or [""]) or [""]:
                for t in sorted(f.get("tickers") or [""]) or [""]:
                    rows.append({"company": c, "ticker": t, "year": y})
        return rows
