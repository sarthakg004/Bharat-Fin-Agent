"""Reranker A/B: bge-reranker-base vs bge-reranker-v2-m3, end to end.

The pool sweep showed the reranker is the bottleneck -- pool_recall climbs from
0.44 to 0.85 as depth goes 24 -> 384 while final@5 stays pinned at 0.30. So the
question is not "more candidates" but "a reranker that can pick from them".

Reports, at fixed hybrid retrieval:
  pool_recall     gold parent anywhere in the pool (identical across rerankers
                  at the same depth -- the control)
  final@5/@8      gold parent in what the synthesizer actually receives
  retention       final@k / pool_recall -- of the questions whose evidence WAS
                  retrievable, what fraction survives reranking. This isolates
                  the reranker from first-stage recall.

max_length is set per model: base caps at 512, v2-m3 at 8192. Current parents
are ~1500 chars (~375 tokens) so neither truncates today -- meaning any gain
here is ranking quality, NOT truncation relief. Truncation relief only arrives
if parents are re-ingested at 2500+, which v2-m3 is a prerequisite for.
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
from finagent.vectorstore import (
    CONTENT_KEY, META_KEY, build_store, get_client, scroll_payloads,
)

GOLD = "results/financebench_gold_parentdoc.json"
OUT = __import__("os").getenv("OUT", "results/reranker_ab.json")
DEPTHS = tuple(int(x) for x in __import__("os").getenv("DEPTHS","24,48").split(","))
FINAL_KS = (5, 8)
MODELS = ({"BAAI/bge-reranker-v2-m3": 1024} if __import__("os").getenv("M3_ONLY")
          else {"BAAI/bge-reranker-base": 512, "BAAI/bge-reranker-v2-m3": 1024})


def parent_index(collection: str) -> dict:
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
    vocab, years = build_company_vocab(scroll_payloads(EVAL_COLLECTION))

    qs = []
    for e in gold.values():
        keys = {key_by_text.get(t) for t in (e.get("gold_chunks") or [])}
        keys = {k for k in keys if k and k[1] is not None}
        if keys:
            qs.append({"q": e["question"], "qtype": e.get("qtype", "?"), "keys": keys,
                       "flt": infer_filter(e["question"], vocab, years)})
    print(f"{len(qs)} questions\n", flush=True)

    store = build_store(EVAL_COLLECTION)
    results = {}
    for model, maxlen in MODELS.items():
        from sentence_transformers import CrossEncoder

        from finagent.device import get_device
        ce = CrossEncoder(model, device=get_device(), max_length=maxlen)
        retr = HybridRetriever(store, reranker_model=model)   # use_mmr=False now
        retr._reranker = ce                                   # pin max_length

        for depth in DEPTHS:
            retr.pool_top_k = depth
            t0 = time.time()
            acc = {"pool": 0, "n": 0, "final": {k: 0 for k in FINAL_KS}}
            by_type: dict = {}
            for q in qs:
                pool = retr._pool(q["q"], q["flt"])
                if not pool and q["flt"] and (q["flt"].get("years")
                                              or q["flt"].get("items")):
                    pool = retr._pool(q["q"], {"companies": q["flt"]["companies"]})
                coll = retr._collapse_to_parents(pool)
                ranked = retr._rerank(q["q"], coll) if coll else []
                in_pool = any((m.get("local_path", ""), m.get("parent_id")) in q["keys"]
                              for _, m in coll)
                rank = next((i for i, (_, m) in enumerate(ranked, 1)
                             if (m.get("local_path", ""), m.get("parent_id")) in q["keys"]),
                            None)
                t = by_type.setdefault(q["qtype"],
                                       {"n": 0, "pool": 0, "final": {k: 0 for k in FINAL_KS}})
                acc["n"] += 1; t["n"] += 1
                if in_pool:
                    acc["pool"] += 1; t["pool"] += 1
                for k in FINAL_KS:
                    if rank and rank <= k:
                        acc["final"][k] += 1; t["final"][k] += 1

            n, pr = acc["n"], acc["pool"] / acc["n"]
            row = {"pool_recall": round(pr, 4),
                   **{f"final@{k}": round(acc["final"][k] / n, 4) for k in FINAL_KS},
                   **{f"retention@{k}": (round(acc["final"][k] / acc["pool"], 4)
                                         if acc["pool"] else None) for k in FINAL_KS},
                   "elapsed_s": round(time.time() - t0, 1),
                   "by_type": {k: {"n": v["n"], "pool_recall": round(v["pool"] / v["n"], 4),
                                   **{f"final@{j}": round(v["final"][j] / v["n"], 4)
                                      for j in FINAL_KS}}
                               for k, v in sorted(by_type.items())}}
            results[f"{model.split('/')[-1]}@{depth}"] = row
            print(f"{model.split('/')[-1]:22} depth={depth:3}  pool={row['pool_recall']:.4f}  "
                  f"final@5={row['final@5']:.4f}  final@8={row['final@8']:.4f}  "
                  f"retention@8={row['retention@8']}  ({row['elapsed_s']}s)", flush=True)
        del ce, retr

    json.dump({"n_questions": len(qs), "gold": GOLD, "models": MODELS,
               "results": results}, open(OUT, "w"), indent=2)
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
