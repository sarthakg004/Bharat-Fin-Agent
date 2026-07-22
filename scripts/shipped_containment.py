"""Gold containment at the sizes we ACTUALLY ship.

results/chunk_ablation.md reports parent_doc=0.8007, but it measured a
2500/300 window. Production splits parents at CorpusIngester.PARENT_CHUNK_SIZE
= 1500/200 and embeds children at 600/100. This re-runs the SAME metric (same
12-doc sample, same seed, same macro-average over per-doc fractions) at the
shipped numbers, so the figure we quote describes the code we run.

containment = fraction of gold passages that fit entirely inside ONE chunk.
It is the ceiling on retrieval: evidence split across two chunks can never be
returned whole, however good the retriever is.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys

# run from the repo root regardless of cwd (this file lives in scripts/)
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from finagent.evaluation.parsing.chunk_ablation import _fixed_chunks
from finagent.evaluation.parsing.compare import (
    GOLD_PATH, PDF_DIR, _normalize, _parse_pypdf,
)
from finagent.ingestion.ingest import CorpusIngester as CI

OUT = "results/shipped_containment.json"
SAMPLE, SEED = 12, 0

STRATEGIES = {
    "child_600_100 (what we embed)":
        lambda t: _fixed_chunks(t, CI.CHILD_CHUNK_SIZE, CI.CHILD_CHUNK_OVERLAP),
    "fixed_1000_200 (old default)":
        lambda t: _fixed_chunks(t, 1000, 200),
    "parent_1500_200 (SHIPPED)":
        lambda t: _fixed_chunks(t, CI.PARENT_CHUNK_SIZE, CI.PARENT_CHUNK_OVERLAP),
    "parent_2500_300 (what 0.8007 measured)":
        lambda t: _fixed_chunks(t, 2500, 300),
}


def main() -> None:
    from tqdm import tqdm

    from finagent.evaluation.dataset import load_eval_dataset

    gold = json.loads(GOLD_PATH.read_text())
    df = load_eval_dataset()
    by_doc: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        chunks = (gold.get(row["financebench_id"]) or {}).get("gold_chunks") or []
        by_doc.setdefault(row["doc_name"], []).extend(c for c in chunks if c.strip())

    docs = sorted(d for d in by_doc if (PDF_DIR / f"{d}.pdf").exists())
    random.Random(SEED).shuffle(docs)
    docs = docs[:SAMPLE]

    hits: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    n_chunks: dict[str, list[int]] = {s: [] for s in STRATEGIES}
    total_gold = 0
    for doc in tqdm(docs, desc="documents", unit="doc"):
        text = _parse_pypdf(PDF_DIR / f"{doc}.pdf")["text"]
        golds = [g for g in (_normalize(x) for x in by_doc[doc]) if g]
        total_gold += len(golds)
        if not golds:
            continue
        for name, chunker in STRATEGIES.items():
            cn = [_normalize(c) for c in chunker(text)]
            n_chunks[name].append(len(cn))
            hits[name].append(sum(1 for g in golds if any(g in c for c in cn)) / len(golds))

    out = {name: {"containment": round(statistics.mean(hits[name]), 4),
                  "mean_chunks_per_doc": round(statistics.mean(n_chunks[name]), 1)}
           for name in STRATEGIES if hits[name]}
    for name, row in out.items():
        print(f"{name:40} containment={row['containment']:.4f}  "
              f"chunks/doc={row['mean_chunks_per_doc']:.0f}")

    json.dump({"docs": len(docs), "gold_passages": total_gold,
               "sample": SAMPLE, "seed": SEED, "results": out},
              open(OUT, "w"), indent=2)
    print(f"\n{len(docs)} docs · {total_gold} gold passages → {OUT}")


if __name__ == "__main__":
    main()
