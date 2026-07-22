"""Retrieval funnel sweep: pool depth x retrieval mode, measured end to end.

Answers three things the existing results/ files do not:

  1. How many chunks each stage actually passes on (Qdrant -> collapse ->
     rerank -> LLM), measured rather than read off the config.
  2. Whether the gold evidence survives all the way into the FINAL chunks the
     synthesizer sees (recall@5 / @8) -- every prior number stopped at the pool.
  3. Whether raising pool_top_k moves that final number, and where it flattens.

Matching is at PARENT identity, not text equality: production collapses each
matched child to its parent, so "did the LLM get the evidence" means "did any
returned candidate share (local_path, parent_id) with the gold child".

Modes:
  mmr    - what production runs today (use_mmr=True): dense-only + MMR.
  hybrid - dense + BM25-sparse fused with RRF server-side (use_mmr=False).
"""
from __future__ import annotations

import json
import os
import sys
import time

# run from the repo root regardless of cwd (this file lives in scripts/)
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from finagent.evaluation.financebench.indexing import EVAL_COLLECTION
from finagent.retrieval.filters import build_company_vocab, infer_filter
from finagent.retrieval.hybrid import HybridRetriever
from finagent.vectorstore import CONTENT_KEY, META_KEY, build_store, get_client

GOLD = "results/financebench_gold_parentdoc.json"
OUT = "results/funnel_sweep.json"
DEPTHS = (24, 48, 96, 192, 384)
MODES = ("mmr", "hybrid")
FINAL_KS = (5, 8)           # final_top_k (per sub-query) and retrieve_cap
RERANKER = "BAAI/bge-reranker-base"     # the deployed model


def parent_index(collection: str) -> dict:
    """text -> (local_path, parent_id) for every indexed child."""
    client, offset, out = get_client(), None, {}
    while True:
        pts, offset = client.scroll(collection_name=collection, limit=2000,
                                    with_payload=True, with_vectors=False,
                                    offset=offset)
        if not pts:
            break
        for p in pts:
            pay = p.payload or {}
            m = pay.get(META_KEY, {}) or {}
            out.setdefault(pay.get(CONTENT_KEY, ""),
                           (m.get("local_path", ""), m.get("parent_id")))
        if offset is None:
            break
    return out


def main() -> None:
    key_by_text = parent_index(EVAL_COLLECTION)
    gold = json.loads(open(GOLD).read())

    retr = HybridRetriever(build_store(EVAL_COLLECTION), reranker_model=RERANKER,
                           final_top_k=5)
    vocab, years = build_company_vocab(
        (lambda: __import__("finagent.vectorstore", fromlist=["scroll_payloads"])
         .scroll_payloads(EVAL_COLLECTION))())

    # Per-question fixtures computed once: the filter and the gold parent keys.
    qs = []
    for entry in gold.values():
        chunks = entry.get("gold_chunks") or []
        if not chunks:
            continue
        keys = {key_by_text.get(t) for t in chunks}
        keys = {k for k in keys if k and k[1] is not None}
        if not keys:
            continue
        qs.append({"question": entry["question"], "qtype": entry.get("qtype", "?"),
                   "keys": keys, "flt": infer_filter(entry["question"], vocab, years)})
    print(f"{len(qs)} questions with a resolvable gold parent", flush=True)

    results = {}
    for mode in MODES:
        retr.use_mmr = (mode == "mmr")
        for depth in DEPTHS:
            retr.pool_top_k = depth
            t0 = time.time()
            acc = {"pool_hit": 0, "n": 0, "pool_size": 0, "collapsed": 0,
                   "final": {k: 0 for k in FINAL_KS}, "ranks": []}
            by_type: dict = {}
            for q in qs:
                pool = retr._pool(q["question"], q["flt"])
                if not pool and q["flt"] and (q["flt"].get("years")
                                              or q["flt"].get("items")):
                    pool = retr._pool(q["question"],
                                      {"companies": q["flt"]["companies"]})
                collapsed = retr._collapse_to_parents(pool)
                ranked = retr._rerank(q["question"], collapsed) if collapsed else []

                def hit(cands) -> bool:
                    return any((m.get("local_path", ""), m.get("parent_id")) in q["keys"]
                               for _, m in cands)

                rank = next((i for i, (_, m) in enumerate(ranked, 1)
                             if (m.get("local_path", ""), m.get("parent_id")) in q["keys"]),
                            None)
                t = by_type.setdefault(q["qtype"], {"n": 0, "pool_hit": 0,
                                                    "final": {k: 0 for k in FINAL_KS}})
                acc["n"] += 1; t["n"] += 1
                acc["pool_size"] += len(pool); acc["collapsed"] += len(collapsed)
                if hit(collapsed):
                    acc["pool_hit"] += 1; t["pool_hit"] += 1
                if rank:
                    acc["ranks"].append(rank)
                for k in FINAL_KS:
                    if rank and rank <= k:
                        acc["final"][k] += 1; t["final"][k] += 1

            n = acc["n"]
            row = {
                "pool_recall": round(acc["pool_hit"] / n, 4),
                **{f"final_recall@{k}": round(acc["final"][k] / n, 4) for k in FINAL_KS},
                "mean_pool_size": round(acc["pool_size"] / n, 1),
                "mean_after_collapse": round(acc["collapsed"] / n, 1),
                "mean_gold_rank": (round(sum(acc["ranks"]) / len(acc["ranks"]), 2)
                                   if acc["ranks"] else None),
                "elapsed_s": round(time.time() - t0, 1),
                "by_type": {k: {"n": v["n"],
                                "pool_recall": round(v["pool_hit"] / v["n"], 4),
                                **{f"final_recall@{j}": round(v["final"][j] / v["n"], 4)
                                   for j in FINAL_KS}}
                            for k, v in sorted(by_type.items())},
            }
            results[f"{mode}@{depth}"] = row
            print(f"{mode:6} depth={depth:4}  pool_recall={row['pool_recall']:.4f}  "
                  f"final@5={row['final_recall@5']:.4f}  final@8={row['final_recall@8']:.4f}  "
                  f"pool={row['mean_pool_size']:.0f}->{row['mean_after_collapse']:.0f}  "
                  f"({row['elapsed_s']}s)", flush=True)

    json.dump({"n_questions": len(qs), "reranker": RERANKER,
               "gold": GOLD, "results": results},
              open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
