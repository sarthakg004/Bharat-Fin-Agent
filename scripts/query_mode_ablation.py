"""Does decomposing the question actually help retrieval?

Every retrieval experiment in `results/RETRIEVAL_EXPERIMENTS.md` was run on the
planner's sub-queries. The harness has always carried a `--mode question`
control and it was never run on the shipped configuration, so the value of
decomposition itself has never been measured. This script measures it.

Three arms, ONE index, one candidate-pool engine, one reranker — only the query
text changes:

    subquery   the served path: planner decomposition -> retrieve per sub-query
               -> merge -> global cap. Plans cached in
               results/financebench_subqueries.json.
    question   retrieve on the user's question, unchanged.
    rewrite    retrieve on ONE query restated in the filing's vocabulary
               (company + fiscal year + statement caption + the line items a
               filing actually prints). Cached in
               results/financebench_rewrites.json.

All three are LLM-free at measurement time: the decomposition and the rewrite
are both authored once, upstream, and held fixed — the same discipline §10
applied to the plans, for the same reason (an LLM inside the loop would fold
its nondeterminism into the deltas we are trying to read).

`expand_query` runs in every arm, because it is part of `HybridRetriever.search`
and removing it from one arm would confound the comparison. So `question` is
"question + line-item expansion", not a naked question.

Usage (needs the local eval Qdrant and the shipped index already built):

    export QDRANT_EVAL_URL=http://localhost:6333
    python scripts/query_mode_ablation.py
    python scripts/query_mode_ablation.py --collection qx_p2500_c600_...
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("QDRANT_EVAL_URL", "http://localhost:6333")

from finagent.evaluation import evaluate_retrieval as E

OUT_JSON = Path("results/query_mode_ablation.json")
OUT_MD = Path("results/query_mode_ablation.md")
MODES = ("subquery", "question", "rewrite")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--collection", default=f"qx_{E.Config().tag}",
                    help="an ALREADY-BUILT index at the shipped geometry")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3",
                    help="production reranker; the arms differ only in query text")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    args = ap.parse_args()

    from finagent.vectorstore import get_client
    if not get_client().collection_exists(args.collection):
        raise SystemExit(f"{args.collection} not on {os.environ['QDRANT_EVAL_URL']} — "
                         f"build it first with evaluate_retrieval.py")

    # One reranker arm: this ablation is about the QUERY, and every extra arm is
    # a full cross-encoder pass over 99 questions.
    E.RERANKERS = (args.reranker,)

    questions, dropped = E.load_eval_questions(None, E.HIT_THRESHOLD)
    print(f"Questions: {len(questions)} scored, {dropped} dropped (parsing ceiling)")
    E.attach_subqueries(questions)
    missing = E.attach_rewrites(questions)
    if missing:
        raise SystemExit(f"{missing} questions have no cached rewrite")
    print(f"Plans: {statistics.mean(len(q.subs) for q in questions):.2f} "
          f"sub-queries/question · rewrites: 1/question "
          f"({statistics.mean(len(q.rewrite.split()) for q in questions):.0f} words mean)")

    cfg = E.Config()
    rows = {}
    for mode in args.modes:
        print(f"\n=== {mode} ===", flush=True)
        t0 = time.time()
        got = E.evaluate_index(cfg, args.collection, questions, mode=mode)
        for r in got:
            r["wall_s"] = round(time.time() - t0, 1)
            rows[mode] = r
            print(f"  pool={r['pool_recall']:.4f} cov@8={r['cov@8']:.4f} "
                  f"hit@8={r['hit@8']:.4f} hit@5={r['hit@5']:.4f} "
                  f"ctx={r['ctx_chars@8']} queries/q={r['subqueries']} "
                  f"{r['wall_s']}s", flush=True)

    OUT_JSON.write_text(json.dumps(
        {"collection": args.collection, "reranker": args.reranker,
         "n": len(questions), "rows": rows}, indent=1))
    _write_md(rows, len(questions), args)
    print(f"\nwrote {OUT_JSON} and {OUT_MD}")


def _write_md(rows: dict, n: int, args) -> None:
    hdr = ("| arm | queries/question | pool_recall | cov@5 | hit@5 | cov@8 | "
           "hit@8 | retention | ctx_chars@8 | rerank_ms |")
    lines = [
        "# Query mode ablation — does decomposition help?",
        "",
        f"One index (`{args.collection}`), one reranker (`{args.reranker}`), "
        f"n={n}. Only the query text differs between arms.",
        "",
        "`hit@k` = the verified evidence span is ≥50% covered by the top-k "
        "parents handed to the synthesizer. `retention` = hit@8 / pool_recall — "
        "how much of what the pool found actually survives selection.",
        "",
        hdr, "|" + "---|" * 10,
    ]
    for mode in MODES:
        r = rows.get(mode)
        if not r:
            continue
        lines.append(
            f"| {mode} | {r['subqueries']} | {r['pool_recall']:.4f} | "
            f"{r['cov@5']:.4f} | {r['hit@5']:.4f} | {r['cov@8']:.4f} | "
            f"{r['hit@8']:.4f} | {r['retention']:.4f} | {r['ctx_chars@8']} | "
            f"{r['rerank_ms']} |")
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
