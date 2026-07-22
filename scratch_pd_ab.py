"""Parent-level A/B for parent-document retrieval — the fair lens.

child_recall@k : the exact gold CHILD is in the top-k fused pool (pessimistic —
                 what the standard eval measures).
parent_recall@k: the top-k fused pool contains ANY child of the gold child's
                 parent — i.e. production's parent-swap returns the parent that
                 contains the gold evidence. This is what parent-doc delivers.

Same fused pool for both, so the gap isolates parent-doc's contribution.
"""
import json
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from finagent.evaluation.financebench.retrieval import POOL_DEPTH, RetrievalEvaluator
from finagent.retrieval.filters import infer_filter, qdrant_filter
from finagent.vectorstore import scroll_payloads

GOLD = "results/financebench_gold_parentdoc.json"
KS = (20, 50, 100)


def fused_with_meta(ev, query):
    """The production pool, carrying metadata so we can test parent identity."""
    ev._ensure_vocab()
    flt = infer_filter(query, ev._vocab, ev._years_by_co)

    def pool(f):
        try:
            return [(d.page_content, d.metadata) for d in ev.store.similarity_search(
                query, k=POOL_DEPTH, filter=qdrant_filter(f))]
        except Exception:
            return []

    hits = pool(flt)
    if not hits and flt and (flt.get("years") or flt.get("items")):
        hits = pool({"companies": flt["companies"]})
    return hits


def main():
    ev = RetrievalEvaluator()
    # text -> (local_path, parent_id) for every indexed child
    key_by_text = {}
    for m in scroll_payloads(ev.collection_name):
        pass  # metadata only; text keys come from the gold map below
    from finagent.vectorstore import get_client, CONTENT_KEY, META_KEY
    client, offset = get_client(), None
    while True:
        pts, offset = client.scroll(collection_name=ev.collection_name, limit=2000,
                                    with_payload=True, with_vectors=False, offset=offset)
        if not pts:
            break
        for p in pts:
            pay = p.payload or {}
            m = pay.get(META_KEY, {}) or {}
            key_by_text.setdefault(pay.get(CONTENT_KEY, ""),
                                   (m.get("local_path", ""), m.get("parent_id")))
        if offset is None:
            break

    gold = json.loads(open(GOLD).read())
    child_hits = {k: 0 for k in KS}
    parent_hits = {k: 0 for k in KS}
    n = 0
    for entry in gold.values():
        gold_chunks = entry.get("gold_chunks") or []
        if not gold_chunks:
            continue
        n += 1
        gold_texts = set(gold_chunks)
        gold_parent_keys = {key_by_text.get(t) for t in gold_chunks if key_by_text.get(t)}
        gold_parent_keys = {k for k in gold_parent_keys if k and k[1] is not None}

        fused = fused_with_meta(ev, entry["question"])
        child_rank = parent_rank = None
        for i, (text, meta) in enumerate(fused, 1):
            if child_rank is None and text in gold_texts:
                child_rank = i
            if parent_rank is None:
                pk = (meta.get("local_path", ""), meta.get("parent_id"))
                if pk in gold_parent_keys:
                    parent_rank = i
            if child_rank and parent_rank:
                break
        for k in KS:
            if child_rank and child_rank <= k:
                child_hits[k] += 1
            if parent_rank and parent_rank <= k:
                parent_hits[k] += 1

    out = {
        "n_questions": n,
        "child_recall": {k: round(child_hits[k] / n, 4) for k in KS},
        "parent_recall": {k: round(parent_hits[k] / n, 4) for k in KS},
    }
    json.dump(out, open("results/parentdoc_ab_measurement.json", "w"), indent=2)
    print("PARENT-LEVEL A/B (n=%d)" % n)
    for k in KS:
        print(f"  @{k}: child_recall={out['child_recall'][k]}  "
              f"parent_recall={out['parent_recall'][k]}"
              f"  (+{round(out['parent_recall'][k] - out['child_recall'][k], 4)})")


if __name__ == "__main__":
    main()
