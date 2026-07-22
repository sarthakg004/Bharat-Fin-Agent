"""Full retrieval metrics + measured failure analysis on the real cluster.

Run against `financebench_eval` with the SHIPPING config (hybrid RRF, v2-m3,
pool 48) so the numbers describe the deployed system, not a lab harness. The
geometry study indexed one document per collection and searched only that
document -- structurally impossible to make a cross-company error, so its
coverage numbers are optimistic. This does not do that.

Metrics
-------
hit@k        any gold parent in the top-k                  (answerability)
recall@k     |retrieved gold| / |all gold|, micro-averaged (completeness)
precision@k  |retrieved gold| / k                          (noise the LLM eats)
MRR          1 / rank of the first gold parent
nDCG@k       rank-discounted gain, binary relevance
coverage@k   fraction of FinanceBench evidence LINES reaching the LLM
pool_recall  gold parent anywhere in the pool, before reranking
retention    hit@k / pool_recall -- what the reranker keeps

Failure analysis
----------------
Every question failing at k=8 is assigned ONE reason by the first matching
rule. Each rule is a measurement, not a guess; the rule is printed with the
result so the classification can be audited or disputed.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections import Counter

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
OUT_JSON = "results/retrieval_report.json"
OUT_MD = "results/retrieval_report.md"
DEPTH = 48
KS = (1, 3, 5, 8, 20)
MAIN_K = 8
RERANKER = "BAAI/bge-reranker-v2-m3"
MIN_LINE = 25
COVER_HIT = 0.5


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def ev_units(evidence) -> list[str]:
    out = []
    for e in evidence or []:
        t = e.get("evidence_text") if isinstance(e, dict) else str(e)
        out += [n for n in (norm(l) for l in (t or "").split("\n")) if len(n) >= MIN_LINE]
    return out


def numeric_density(units: list[str]) -> float:
    """Share of characters that are digits/currency/separators — a table tell."""
    blob = " ".join(units)
    if not blob:
        return 0.0
    return sum(c.isdigit() or c in "$%,.()-" for c in blob) / len(blob)


def ndcg(ranks: list[int], k: int, n_gold: int) -> float:
    dcg = sum(1 / math.log2(r + 1) for r in ranks if r <= k)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n_gold, k)))
    return dcg / idcg if idcg else 0.0


def main() -> None:
    from sentence_transformers import CrossEncoder

    from finagent.device import get_device

    client = get_client()
    # text -> (local_path, parent_id); and every parent's text per filing, so we
    # can ask "does the evidence survive chunking anywhere in this filing?"
    key_by_text, parents_by_doc = {}, {}
    offset = None
    while True:
        pts, offset = client.scroll(EVAL_COLLECTION, limit=2000, with_payload=True,
                                    with_vectors=False, offset=offset)
        if not pts:
            break
        for p in pts:
            pay = p.payload or {}
            m = pay.get(META_KEY, {}) or {}
            lp, pid = m.get("local_path", ""), m.get("parent_id")
            key_by_text.setdefault(pay.get(CONTENT_KEY, ""), (lp, pid))
            ptext = m.get("parent_text") or pay.get(CONTENT_KEY, "")
            parents_by_doc.setdefault(lp, {}).setdefault(pid, ptext)
        if offset is None:
            break
    print(f"indexed: {len(key_by_text):,} chunks across {len(parents_by_doc)} filings",
          flush=True)

    from finagent.evaluation.dataset import load_eval_dataset
    df = load_eval_dataset()
    ev_by_id = {r["financebench_id"]: r["evidence"] for _, r in df.iterrows()}

    gold_map = json.loads(open(GOLD).read())
    vocab, years = build_company_vocab(scroll_payloads(EVAL_COLLECTION))

    store = build_store(EVAL_COLLECTION)
    retr = HybridRetriever(store, reranker_model=RERANKER, pool_top_k=DEPTH)
    retr._reranker = CrossEncoder(RERANKER, device=get_device(), max_length=1024)

    agg = {f"hit@{k}": 0 for k in KS}
    agg.update({f"recall@{k}": 0.0 for k in KS})
    agg.update({f"precision@{k}": 0.0 for k in KS})
    agg.update({f"ndcg@{k}": 0.0 for k in KS})
    agg.update({f"coverage@{k}": 0.0 for k in KS})
    agg["mrr"] = 0.0
    agg["pool_recall"] = 0
    by_type: dict = {}
    failures: list[dict] = []
    n = 0
    t0 = time.time()

    for fb_id, entry in gold_map.items():
        gold_keys = {key_by_text.get(t) for t in (entry.get("gold_chunks") or [])}
        gold_keys = {k for k in gold_keys if k and k[1] is not None}
        if not gold_keys:
            continue
        n += 1
        q, qtype = entry["question"], entry.get("qtype", "?")
        units = ev_units(ev_by_id.get(fb_id))
        flt = infer_filter(q, vocab, years)

        pool = retr._pool(q, flt)
        if not pool and flt and (flt.get("years") or flt.get("items")):
            flt = {"companies": flt["companies"]}
            pool = retr._pool(q, flt)
        coll = retr._collapse_to_parents(pool)
        ranked = retr._rerank(q, coll) if coll else []

        def keyof(m):
            return (m.get("local_path", ""), m.get("parent_id"))

        ranks = [i for i, (_, m) in enumerate(ranked, 1) if keyof(m) in gold_keys]
        in_pool = any(keyof(m) in gold_keys for _, m in coll)
        agg["pool_recall"] += in_pool
        agg["mrr"] += 1 / ranks[0] if ranks else 0.0

        t = by_type.setdefault(qtype, {"n": 0, f"hit@{MAIN_K}": 0, "mrr": 0.0,
                                       f"recall@{MAIN_K}": 0.0})
        t["n"] += 1
        t["mrr"] += 1 / ranks[0] if ranks else 0.0
        for k in KS:
            got = sum(1 for r in ranks if r <= k)
            agg[f"hit@{k}"] += bool(got)
            agg[f"recall@{k}"] += got / len(gold_keys)
            agg[f"precision@{k}"] += got / k
            agg[f"ndcg@{k}"] += ndcg(ranks, k, len(gold_keys))
            if units:
                blob = " || ".join(norm(x) for x, _ in ranked[:k])
                agg[f"coverage@{k}"] += sum(1 for u in units if u in blob) / len(units)
            if k == MAIN_K:
                t[f"hit@{k}"] += bool(got)
                t[f"recall@{k}"] += got / len(gold_keys)

        # ---------------- failure classification ----------------
        if not ranks or min(ranks) > MAIN_K:
            gold_doc = next(iter(gold_keys))[0]
            top_docs = {m.get("local_path", "") for _, m in ranked[:MAIN_K]}
            # does the evidence survive chunking anywhere in the right filing?
            best_single = 0.0
            if units:
                for ptext in (parents_by_doc.get(gold_doc) or {}).values():
                    pn = norm(ptext)
                    c = sum(1 for u in units if u in pn) / len(units)
                    best_single = max(best_single, c)
            reason = classify(in_pool, ranks, best_single, gold_doc, top_docs,
                              numeric_density(units), len(units), flt)
            failures.append({"id": fb_id, "qtype": qtype, "reason": reason,
                             "gold_rank": min(ranks) if ranks else None,
                             "in_pool": in_pool,
                             "best_single_parent_coverage": round(best_single, 3),
                             "numeric_density": round(numeric_density(units), 3),
                             "filtered_to_company": bool(flt and flt.get("companies"))})
        if n % 25 == 0:
            print(f"  {n}/{len(gold_map)}  ({time.time()-t0:.0f}s)", flush=True)

    write_report(agg, by_type, failures, n, time.time() - t0)


def classify(in_pool, ranks, best_single, gold_doc, top_docs, ndens,
             n_units, flt) -> str:
    """First matching rule wins. Each is a measurement on this question."""
    # The evidence does not fit inside ANY single parent of the right filing:
    # no retriever can return it whole. This is the chunking ceiling.
    if best_single < COVER_HIT:
        return "Chunk boundary split evidence"
    # It was retrievable and the reranker had it in hand, but ranked it out.
    if in_pool and (not ranks or min(ranks) > MAIN_K):
        return "Reranker removed relevant chunk"
    # Everything returned came from other filings — the company filter failed
    # to constrain (or matched nothing) and other issuers' boilerplate won.
    if top_docs and gold_doc not in top_docs and not (flt or {}).get("companies"):
        return "Cross-company retrieval"
    if top_docs and gold_doc not in top_docs:
        return "Cross-document retrieval (same company, wrong filing)"
    # Dense/sparse never surfaced it, and the evidence is mostly figures:
    # embeddings are weak on dense numeric tables.
    if ndens > 0.35:
        return "Table retrieval failure"
    # Too little text to identify anything reliably — the annotation itself is
    # thin, so a miss here is not clearly the retriever's fault.
    if n_units <= 2:
        return "Gold annotation ambiguity"
    return "Embedding semantic miss"


def write_report(agg, by_type, failures, n, elapsed) -> None:
    m = {k: (v / n if isinstance(v, float) else v / n) for k, v in agg.items()}
    reasons = Counter(f["reason"] for f in failures)
    total_f = sum(reasons.values())

    payload = {"n_questions": n, "collection": EVAL_COLLECTION, "pool_depth": DEPTH,
               "reranker": RERANKER, "elapsed_s": round(elapsed, 1),
               "metrics": {k: round(v, 4) for k, v in m.items()},
               "by_type": {t: {"n": v["n"],
                               f"hit@{MAIN_K}": round(v[f"hit@{MAIN_K}"] / v["n"], 4),
                               f"recall@{MAIN_K}": round(v[f"recall@{MAIN_K}"] / v["n"], 4),
                               "mrr": round(v["mrr"] / v["n"], 4)}
                           for t, v in sorted(by_type.items())},
               "failures": {"n": total_f,
                            "share_of_all_questions": round(total_f / n, 4),
                            "breakdown": {r: {"count": c, "pct_of_failures":
                                              round(100 * c / total_f, 1)}
                                          for r, c in reasons.most_common()}},
               "failure_detail": failures}
    json.dump(payload, open(OUT_JSON, "w"), indent=2)

    L = []
    L.append("# Retrieval report — FinanceBench eval collection\n")
    L.append(f"`{EVAL_COLLECTION}` · {n} questions · hybrid RRF (dense + BM25 sparse) "
             f"· pool {DEPTH} · reranker `{RERANKER}` · parent-document retrieval\n")
    L.append("Measured on the live cluster, so cross-filing distractors are present "
             "(the geometry harness searched one filing at a time and cannot show them).\n")
    L.append("\n## Ranking metrics\n")
    L.append("| k | hit@k | recall@k | precision@k | nDCG@k | coverage@k |")
    L.append("|---|---|---|---|---|---|")
    for k in KS:
        L.append(f"| {k} | {m[f'hit@{k}']:.4f} | {m[f'recall@{k}']:.4f} | "
                 f"{m[f'precision@{k}']:.4f} | {m[f'ndcg@{k}']:.4f} | "
                 f"{m[f'coverage@{k}']:.4f} |")
    L.append(f"\n**MRR** {m['mrr']:.4f}  ·  **pool_recall** {m['pool_recall']:.4f} "
             f"(gold in the pool before reranking)  ·  "
             f"**retention@{MAIN_K}** {m[f'hit@{MAIN_K}']/m['pool_recall']:.4f} "
             f"(share of retrievable questions the reranker delivers)\n")
    L.append("\n### What each metric means here\n")
    L.append("- **hit@k** — the question is *answerable*: at least one gold parent "
             "reached the LLM. Most questions have one gold parent (median 1), so "
             "this is the headline number.")
    L.append("- **recall@k** — of all gold parents, how many arrived. Differs from "
             "hit@k only for the 35 questions with 2–3 gold spans.")
    L.append("- **precision@k** — share of delivered chunks that are gold. It is "
             "*mechanically low*: with 1 gold parent, precision@8 cannot exceed "
             "0.125. Read it as noise ratio, not quality.")
    L.append("- **nDCG@k** — rewards ranking gold higher, not merely including it. "
             "This is what `_cap_pool` and the grader actually consume.")
    L.append("- **coverage@k** — fraction of FinanceBench's verified evidence *lines* "
             "present in the top-k. Independent of chunk geometry, so it is the "
             "metric that survives a reindex.\n")
    L.append("\n## By question type\n")
    L.append("| type | n | hit@8 | recall@8 | MRR |")
    L.append("|---|---|---|---|---|")
    for t, v in payload["by_type"].items():
        L.append(f"| {t} | {v['n']} | {v['hit@8']:.4f} | {v['recall@8']:.4f} | "
                 f"{v['mrr']:.4f} |")
    L.append(f"\n## Failure analysis\n")
    L.append(f"{total_f} of {n} questions ({100*total_f/n:.1f}%) fail at k={MAIN_K}: "
             f"no gold parent reached the synthesizer. Each is assigned ONE reason by "
             f"the first matching rule below — every rule is a measurement on that "
             f"question, not a judgement call.\n")
    L.append("| Failure reason | Count | % of failures | % of all questions |")
    L.append("|---|---|---|---|")
    for r, c in reasons.most_common():
        L.append(f"| {r} | {c} | {100*c/total_f:.1f}% | {100*c/n:.1f}% |")
    L.append("\n### Classification rules (in order)\n")
    L.append("1. **Chunk boundary split evidence** — no single parent in the correct "
             "filing contains ≥50% of the evidence lines. The chunker destroyed it; "
             "no retriever could return it whole. *Fixed by a larger parent window.*")
    L.append("2. **Reranker removed relevant chunk** — a gold parent WAS in the pool, "
             "and the cross-encoder ranked it below k. *Fixed by a stronger reranker.*")
    L.append("3. **Cross-company retrieval** — every returned parent came from another "
             "filing AND no company filter was applied (`infer_filter` matched no "
             "indexed company). *Fixed by company-vocabulary coverage.*")
    L.append("4. **Cross-document retrieval** — right company, wrong filing/year.")
    L.append("5. **Table retrieval failure** — never entered the pool and the evidence "
             "is >35% digits/currency characters, i.e. a dense numeric table, where "
             "embeddings carry little lexical signal.")
    L.append("6. **Gold annotation ambiguity** — the evidence span has ≤2 content "
             "lines; too thin to attribute the miss to retrieval.")
    L.append("7. **Embedding semantic miss** — residual: retrievable, in the right "
             "filing, prose, and still never surfaced.\n")
    open(OUT_MD, "w").write("\n".join(L) + "\n")
    print(f"\n→ {OUT_MD}\n→ {OUT_JSON}")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
