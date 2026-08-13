"""The decomposed retrieval eval. It splits retrieval quality into the two stages
that fail for different reasons, so a regression points at the right fix:

    pool_recall@{20,50,100}  is the gold chunk anywhere in the fused candidate
                             pool BEFORE reranking? A miss here is a first-stage
                             recall problem — the reranker never gets a chance.
    Hit@{1,3,5} / MRR        GIVEN it made the pool, does the reranker surface
                             it near the top? Conditional on pool recall, so
                             they isolate the reranker's contribution.

BM25 and dense are fused with RRF into one ranked pool, which makes pool_recall@K
well-defined and monotonic in K. Reported overall and by question type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from finagent.vectorstore import build_store

from .indexing import EVAL_COLLECTION

POOL_DEPTH = 100    # candidates pulled from the fused (sparse+dense) search
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"  # matches the deployed agent


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
        reranker_model: str = DEFAULT_RERANKER,
        pool_ks: tuple = (20, 50, 100),
        hit_ks: tuple = (1, 3, 5),
    ):
        self.collection_name = collection_name
        self.store = build_store(collection_name)
        self.reranker_model = reranker_model
        self.pool_ks = pool_ks
        self.hit_ks = hit_ks
        self._vocab: dict = {}
        self._years_by_co: dict = {}
        self._reranker = None

    # ------------------------------------------------------------------ #
    # Lazy resources
    # ------------------------------------------------------------------ #

    def _ensure_vocab(self):
        """Company/year vocabulary from the collection's own metadata."""
        if not self._vocab:
            from finagent.retrieval.filters import build_company_vocab
            from finagent.vectorstore import scroll_payloads

            self._vocab, self._years_by_co = build_company_vocab(
                scroll_payloads(self.collection_name))
        return self._vocab

    def _ensure_reranker(self):
        if self._reranker is None:
            from finagent.retrieval.reranker import _get_shared_reranker

            self._reranker = _get_shared_reranker(self.reranker_model)
        return self._reranker

    # ------------------------------------------------------------------ #
    # One query
    # ------------------------------------------------------------------ #

    def _fused_pool(self, query: str) -> list[str]:
        """The production candidate pool: sparse (BM25) and dense fused by RRF.

        Fusion now happens inside Qdrant, so this calls exactly what the
        deployed retriever calls. The pool is restricted to the company / year
        the question names (inferred from the collection's own metadata, no
        LLM) so one filing's question doesn't compete with 83 other filings'
        boilerplate. Questions naming no indexed company search the whole
        collection, exactly as the deployed retriever does.
        """
        from finagent.retrieval.filters import infer_filter, qdrant_filter

        self._ensure_vocab()
        flt = infer_filter(query, self._vocab, self._years_by_co)

        def _pool(f):
            try:
                return [d.page_content for d in self.store.similarity_search(
                    query, k=POOL_DEPTH, filter=qdrant_filter(f))]
            except Exception:
                return []

        texts = _pool(flt)
        # Mirror production: relax an over-narrow year/item clause (e.g. this
        # collection predates section tagging) back to company-only.
        if not texts and flt and (flt.get("years") or flt.get("items")):
            texts = _pool({"companies": flt["companies"]})
        return texts

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


def rerank_ablation(gold_map: Optional[dict] = None,
                    hit_ks: tuple = (1, 3, 5),
                    limit: Optional[int] = None) -> dict:
    """Before/after cross-encoder ablation: gold-chunk Hit@k / MRR of the
    RRF-fused pool order vs the reranked order, over the SAME candidates —
    the delta isolates the reranker's contribution. Local compute only
    (no LLM quota); strict exact-gold-chunk matching, so absolute numbers
    read low — compare relatively.
    """
    from finagent.evaluation.financebench.gold import build_gold_map

    gold_map = gold_map or build_gold_map(reuse=True)
    evaluator = RetrievalEvaluator(hit_ks=hit_ks)
    reranker = evaluator._ensure_reranker()

    base_ranks: list[int] = []
    reranked_ranks: list[Optional[int]] = []
    n = 0
    for entry in gold_map.values():
        gold = entry.get("gold_chunks") or []
        if not gold:
            continue
        if limit is not None and n >= limit:
            break
        n += 1
        fused = evaluator._fused_pool(entry["question"])
        pool_rank = evaluator._first_gold_rank(fused, gold)
        if pool_rank is None:
            continue                 # first-stage miss; reranker never sees it
        base_ranks.append(pool_rank)
        candidates = fused[:POOL_DEPTH]
        scores = reranker.predict([(entry["question"], t) for t in candidates])
        reranked = [t for _, t in
                    sorted(zip(scores, candidates), key=lambda x: -x[0])]
        reranked_ranks.append(evaluator._first_gold_rank(reranked, gold))

    def _metrics(ranks):
        found = [r for r in ranks if r]
        if not found:
            return {"mrr": 0.0, **{f"hit@{k}": 0.0 for k in hit_ks}}
        out = {f"hit@{k}": round(sum(1 for r in found if r <= k) / len(ranks), 4)
               for k in hit_ks}
        out["mrr"] = round(sum(1.0 / r for r in found) / len(ranks), 4)
        return out

    in_pool = len(base_ranks)
    return {
        "n_questions": n,
        "gold_in_pool": in_pool,
        f"pool_recall@{POOL_DEPTH}": round(in_pool / n, 4) if n else 0.0,
        "before_rerank": _metrics(base_ranks),
        "after_rerank": _metrics(reranked_ranks),
    }
