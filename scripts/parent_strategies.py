"""Which parent unit should we return? Containment vs. the two ceilings.

Containment alone argues for ever-bigger parents. Two things cap it:

  reranker window — `_collapse_to_parents` runs BEFORE `_rerank`, so the
    cross-encoder scores the PARENT. bge-reranker-base truncates at 512 tokens
    (~2000 chars of financial text). A parent longer than that is scored on its
    head only; gold evidence in the tail is invisible to ranking.

  LLM context — retrieve_cap=8 parents go to the synthesizer. Groq's per-request
    cap makes mean_parent_chars x 8 a real budget, not a rounding error.

So we report containment, the fraction of parents that overflow the reranker,
and the context cost, for fixed windows AND for structure-aware units
(paragraph, page) — which is what "return the whole paragraph instead" means
for a PDF.

Same 12-doc sample / seed / macro-average as results/chunk_ablation.md.
"""
from __future__ import annotations

import json
import os
import random
import re
import statistics
import sys

# run from the repo root regardless of cwd (this file lives in scripts/)
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from finagent.evaluation.parsing.chunk_ablation import _fixed_chunks
from finagent.evaluation.parsing.compare import GOLD_PATH, PDF_DIR, _normalize
from finagent.ingestion.ingest import CorpusIngester as CI

OUT = "results/parent_strategies.json"
SAMPLE, SEED = 12, 0
RERANK_CHAR_BUDGET = 2000       # ~512 tokens for bge-reranker-base
RETRIEVE_CAP = 8                # parents handed to the synthesizer


def _pages(pdf):
    from pypdf import PdfReader
    return [(p.extract_text() or "") for p in PdfReader(str(pdf)).pages]


def _paragraphs(pages: list[str]) -> list[str]:
    """Natural paragraphs: blank-line blocks, falling back to single newlines
    when a PDF has none (pypdf often joins with '\\n' only)."""
    out = []
    for pg in pages:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", pg) if b.strip()]
        if len(blocks) <= 1 and pg.strip():
            blocks = [b.strip() for b in pg.split("\n") if b.strip()]
        out.extend(blocks)
    return out


def _merge_to(units: list[str], target: int) -> list[str]:
    """Greedy merge of whole units up to `target` chars — never splits a unit,
    so a paragraph/table row stays intact."""
    out, buf = [], ""
    for u in units:
        if buf and len(buf) + len(u) + 1 > target:
            out.append(buf); buf = ""
        buf = f"{buf}\n{u}" if buf else u
    if buf:
        out.append(buf)
    return out


STRATEGIES = {
    "child_600_100 (embedded unit)":   lambda pg, t: _fixed_chunks(t, CI.CHILD_CHUNK_SIZE, CI.CHILD_CHUNK_OVERLAP),
    "fixed_1000_200 (old flat)":       lambda pg, t: _fixed_chunks(t, 1000, 200),
    "parent_1500_200 (SHIPPED)":       lambda pg, t: _fixed_chunks(t, CI.PARENT_CHUNK_SIZE, CI.PARENT_CHUNK_OVERLAP),
    "parent_2500_300":                 lambda pg, t: _fixed_chunks(t, 2500, 300),
    "parent_4000_400":                 lambda pg, t: _fixed_chunks(t, 4000, 400),
    "paragraph (raw)":                 lambda pg, t: _paragraphs(pg),
    "paragraph_merged_1500":           lambda pg, t: _merge_to(_paragraphs(pg), 1500),
    "paragraph_merged_2500":           lambda pg, t: _merge_to(_paragraphs(pg), 2500),
    "page (whole page)":               lambda pg, t: [p for p in pg if p.strip()],
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

    hits = {s: [] for s in STRATEGIES}
    nchunks = {s: [] for s in STRATEGIES}
    sizes = {s: [] for s in STRATEGIES}
    overflow = {s: [] for s in STRATEGIES}

    for doc in tqdm(docs, desc="documents", unit="doc"):
        pages = _pages(PDF_DIR / f"{doc}.pdf")
        text = "\n".join(pages)
        golds = [g for g in (_normalize(x) for x in by_doc[doc]) if g]
        if not golds:
            continue
        for name, fn in STRATEGIES.items():
            raw = [c for c in fn(pages, text) if c and c.strip()]
            if not raw:
                continue
            cn = [_normalize(c) for c in raw]
            hits[name].append(sum(1 for g in golds if any(g in c for c in cn)) / len(golds))
            nchunks[name].append(len(raw))
            sizes[name].extend(len(c) for c in raw)
            overflow[name].append(
                sum(1 for c in raw if len(c) > RERANK_CHAR_BUDGET) / len(raw))

    out = {}
    for name in STRATEGIES:
        if not hits[name]:
            continue
        mean_size = statistics.mean(sizes[name])
        out[name] = {
            "containment": round(statistics.mean(hits[name]), 4),
            "chunks_per_doc": round(statistics.mean(nchunks[name]), 1),
            "mean_chars": round(mean_size),
            "median_chars": round(statistics.median(sizes[name])),
            "pct_over_reranker_window": round(statistics.mean(overflow[name]), 4),
            "ctx_chars_at_cap8": round(mean_size * RETRIEVE_CAP),
        }

    hdr = f"{'strategy':34} {'contain':>8} {'chunks':>7} {'mean_ch':>8} {'>window':>8} {'ctx@8':>7}"
    print("\n" + hdr); print("-" * len(hdr))
    for name, r in out.items():
        print(f"{name:34} {r['containment']:8.4f} {r['chunks_per_doc']:7.0f} "
              f"{r['mean_chars']:8} {r['pct_over_reranker_window']:8.1%} "
              f"{r['ctx_chars_at_cap8']:7}")

    json.dump({"docs": len(docs), "sample": SAMPLE, "seed": SEED,
               "reranker_char_budget": RERANK_CHAR_BUDGET,
               "retrieve_cap": RETRIEVE_CAP, "results": out},
              open(OUT, "w"), indent=2)
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
