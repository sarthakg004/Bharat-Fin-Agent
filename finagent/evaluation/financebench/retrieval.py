"""
retrieval.py  ·  finagent/evaluation/financebench/retrieval.py

The **decomposed retrieval eval**. It splits retrieval quality into the two
stages that fail for different reasons, so a regression points at the right fix:

    pool_recall@{20,50,100}   — is the gold chunk anywhere in the BM25 ∪ dense
                                candidate pool *before* reranking? A miss here is
                                a first-stage recall problem (the reranker never
                                gets a chance).
    Hit@{1,3,5} / MRR         — *given* the gold chunk made it into the pool,
                                does the cross-encoder reranker surface it near
                                the top? These are conditional on pool recall, so
                                they isolate the reranker's contribution.

BM25 and dense are fused with Reciprocal Rank Fusion (RRF) into one ranked pool,
which makes pool_recall@K well-defined and monotonic in K. The reranker (same
cross-encoder as production, `bge-reranker-base`) then reorders the top of that
pool. Metrics are reported overall and broken down by question type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from finagent.vectorstore import build_store

from .indexing import EVAL_COLLECTION

RRF_K = 60          # RRF damping constant (standard default)
POOL_DEPTH = 100    # how many candidates to pull from each retriever
DEFAULT_RERANKER = "BAAI/bge-reranker-base"  # matches the deployed agent


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


@dataclass
class RetrievalResult:
    pool_recall: dict = field(default_factory=dict)      # {20: .., 50: .., 100: ..}
    hit: dict = field(default_factory=dict)              # {1: .., 3: .., 5: ..}
    mrr: float = 0.0
    n_questions: int = 0
    n_in_pool: int = 0                                   # gold reached the pool
    by_type: dict = field(default_factory=dict)          # qtype -> sub-metrics

    def as_dict(self) -> dict:
        flat = {
            f"pool_recall@{k}": round(v, 4) for k, v in self.pool_recall.items()
        }
        flat.update({f"hit@{k}": round(v, 4) for k, v in self.hit.items()})
        flat["mrr"] = round(self.mrr, 4)
        flat["n_questions"] = self.n_questions
        flat["n_in_pool"] = self.n_in_pool
        return flat


class RetrievalEvaluator:
    """Score retrieval over `financebench_eval` against a gold-chunk map."""

    def __init__(
        self,
        collection_name: str = EVAL_COLLECTION,
        chroma_dir: Optional[str] = None,
        reranker_model: str = DEFAULT_RERANKER,
        pool_ks: tuple = (20, 50, 100),
        hit_ks: tuple = (1, 3, 5),
    ):
        self.store = build_store(collection_name, chroma_dir=chroma_dir)
        self.reranker_model = reranker_model
        self.pool_ks = pool_ks
        self.hit_ks = hit_ks
        self._bm25 = None
        self._all_texts: list[str] = []
        self._all_metas: list[dict] = []
        self._vocab: dict = {}
        self._years_by_co: dict = {}
        self._reranker = None

    # ------------------------------------------------------------------ #
    # Lazy resources
    # ------------------------------------------------------------------ #

    def _ensure_bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            col = self.store._collection
            texts, metas, offset = [], [], 0
            while True:
                batch = col.get(include=["documents", "metadatas"],
                                limit=2000, offset=offset)
                docs = batch.get("documents") or []
                if not docs:
                    break
                texts.extend(docs)
                metas.extend(batch.get("metadatas") or [{}] * len(docs))
                offset += len(docs)
            self._all_texts = texts
            self._all_metas = metas
            from finagent.retrieval.filters import build_company_vocab
            self._vocab, self._years_by_co = build_company_vocab(metas)
            self._bm25 = BM25Okapi([_tokenize(t) for t in texts])
        return self._bm25

    def _ensure_reranker(self):
        if self._reranker is None:
            from finagent.graph.corrective import _get_shared_reranker

            self._reranker = _get_shared_reranker(self.reranker_model)
        return self._reranker

    # ------------------------------------------------------------------ #
    # One query
    # ------------------------------------------------------------------ #

    def _fused_pool(self, query: str) -> list[str]:
        """RRF-fused BM25 ∪ dense ranked list of chunk texts (deduped).

        Mirrors production: the candidate pool is restricted to the company /
        year the question names (inferred from the collection's own metadata,
        no LLM) so one filing's question doesn't compete with 83 other
        filings' boilerplate. Questions naming no indexed company search the
        whole collection, exactly as the deployed retriever does.
        """
        from finagent.retrieval.filters import (
            chroma_where, infer_filter, metadata_matches)

        bm25 = self._ensure_bm25()
        flt = infer_filter(query, self._vocab, self._years_by_co)

        scores = bm25.get_scores(_tokenize(query))
        idx_pool = (range(len(scores)) if not flt else
                    [i for i in range(len(self._all_texts))
                     if metadata_matches(self._all_metas[i], flt)])
        bm25_rank = sorted(idx_pool, key=lambda i: -scores[i])[:POOL_DEPTH]
        bm25_texts = [self._all_texts[i] for i in bm25_rank]

        dense_texts = [
            d.page_content
            for d in self.store.similarity_search(query, k=POOL_DEPTH,
                                                  filter=chroma_where(flt))
        ]

        # Reciprocal Rank Fusion across the two ranked lists.
        rrf: dict[str, float] = {}
        for ranked in (bm25_texts, dense_texts):
            for rank, text in enumerate(ranked, start=1):
                rrf[text] = rrf.get(text, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(rrf, key=lambda t: -rrf[t])

    @staticmethod
    def _first_gold_rank(ranked: list[str], gold: list[str]) -> Optional[int]:
        gold_set = set(gold)
        for i, text in enumerate(ranked, start=1):
            if text in gold_set:
                return i
        return None

    # ------------------------------------------------------------------ #
    # Full eval
    # ------------------------------------------------------------------ #

    def evaluate(self, gold_map: dict, subset_ids: Optional[set] = None) -> RetrievalResult:
        """Run the decomposed eval over every question that has a gold chunk."""
        reranker = self._ensure_reranker()
        max_pool = max(self.pool_ks)

        # Accumulators, overall and per qtype.
        def _blank():
            return {
                "pool_hits": {k: 0 for k in self.pool_ks},
                "in_pool": 0,
                "rerank_ranks": [],   # gold rank after rerank (in-pool only)
                "n": 0,
            }

        overall = _blank()
        by_type: dict[str, dict] = {}

        for fb_id, entry in gold_map.items():
            if subset_ids is not None and fb_id not in subset_ids:
                continue
            gold = entry.get("gold_chunks") or []
            if not gold:
                continue
            qtype = entry.get("qtype", "unknown")
            bucket = by_type.setdefault(qtype, _blank())
            overall["n"] += 1
            bucket["n"] += 1

            fused = self._fused_pool(entry["question"])

            # Stage 1 — pool recall at each depth.
            gold_rank_in_pool = self._first_gold_rank(fused, gold)
            for k in self.pool_ks:
                present = gold_rank_in_pool is not None and gold_rank_in_pool <= k
                if present:
                    overall["pool_hits"][k] += 1
                    bucket["pool_hits"][k] += 1

            # Stage 2 — conditional rerank metrics (gold must be in pool@max).
            if gold_rank_in_pool is None or gold_rank_in_pool > max_pool:
                continue
            overall["in_pool"] += 1
            bucket["in_pool"] += 1

            pool = fused[:max_pool]
            pairs = [(entry["question"], t) for t in pool]
            r_scores = reranker.predict(pairs)
            reranked = [pool[i] for i in sorted(range(len(pool)), key=lambda i: -r_scores[i])]
            rank = self._first_gold_rank(reranked, gold)
            if rank:
                overall["rerank_ranks"].append(rank)
                bucket["rerank_ranks"].append(rank)

        return self._finalize(overall, by_type)

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #

    def _summarize(self, acc: dict) -> RetrievalResult:
        n = max(1, acc["n"])
        ranks = acc["rerank_ranks"]
        in_pool = max(1, acc["in_pool"])
        res = RetrievalResult(
            pool_recall={k: acc["pool_hits"][k] / n for k in self.pool_ks},
            hit={k: sum(1 for r in ranks if r <= k) / in_pool for k in self.hit_ks},
            mrr=(sum(1.0 / r for r in ranks) / in_pool) if ranks else 0.0,
            n_questions=acc["n"],
            n_in_pool=acc["in_pool"],
        )
        return res

    def _finalize(self, overall: dict, by_type: dict) -> RetrievalResult:
        res = self._summarize(overall)
        res.by_type = {
            qtype: self._summarize(acc).as_dict() for qtype, acc in by_type.items()
        }
        return res
