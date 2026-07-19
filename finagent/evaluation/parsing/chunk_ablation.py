"""
chunk_ablation.py  ·  finagent/evaluation/parsing/chunk_ablation.py

"Is 1000 actually optimal?" — sweep chunk strategies against gold-evidence
*containment*, the number the fixed 1000/200 default was never measured against.

Why containment, not the parsing_eval shingle recall: `compare.shingle_recall`
scores the gold passage against the WHOLE parsed document, so it is identical for
every chunk size (chunking happens after). The retrieval-relevant question is
different — does a gold evidence passage land INSIDE a single retrievable chunk,
or is it sliced across a boundary so no one chunk can answer? That is
`containment@strategy` = fraction of gold passages whose (normalised) text is a
substring of at least one chunk. Larger/semantic/parent windows raise it at the
cost of retrieval precision; this settles the tradeoff with a number.

No models, no embedding, no LLM — reuses the pypdf parse and the gold map from
`compare.py`. CPU string work only (safe to run alongside other jobs).

Usage:
    python -m finagent.evaluation.parsing.chunk_ablation --sample 10
    → results/chunk_ablation.json + results/chunk_ablation.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from finagent.evaluation.parsing.compare import (
    GOLD_PATH, PDF_DIR, _normalize, _parse_pypdf)

OUT_JSON = Path("results/chunk_ablation.json")
OUT_MD = Path("results/chunk_ablation.md")


def _fixed_chunks(text: str, size: int, overlap: int) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_text(text)


def _semantic_chunks(text: str, target: int) -> list[str]:
    """Structure-aware greedy merge to ~target chars: split on line boundaries
    (pypdf joins pages with single '\\n', so paragraph '\\n\\n' breaks are rare),
    merge adjacent lines up to `target`, and hard-split any single line longer
    than target so no chunk balloons past the embedder window. A dependency-free
    stand-in for embedding-based semantic chunking (SemanticChunker isn't
    installed). ponytail: line-merge proxy; swap in SemanticChunker if it
    undersells."""
    units: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        while len(line) > target:                 # force-split an oversized line
            units.append(line[:target])
            line = line[target:]
        if line:
            units.append(line)

    out, buf = [], ""
    for u in units:
        if buf and len(buf) + len(u) + 1 > target:
            out.append(buf)
            buf = ""
        buf = f"{buf} {u}" if buf else u
    if buf:
        out.append(buf)
    return out


# name -> callable(text) -> list[chunk_str]. Parent-doc is the containment of
# the PARENT window (children are matched, the parent is what's returned/read).
STRATEGIES = {
    "fixed_500": lambda t: _fixed_chunks(t, 500, 100),
    "fixed_1000": lambda t: _fixed_chunks(t, 1000, 200),   # current default
    "fixed_1500": lambda t: _fixed_chunks(t, 1500, 300),
    "semantic": lambda t: _semantic_chunks(t, 1000),
    "parent_doc": lambda t: _fixed_chunks(t, 2500, 300),
}


def _contained(gold_norm: str, chunks_norm: list[str]) -> bool:
    return bool(gold_norm) and any(gold_norm in c for c in chunks_norm)


def run(sample: int, seed: int = 0) -> dict:
    from tqdm import tqdm

    from finagent.evaluation.dataset import load_eval_dataset

    gold = json.loads(GOLD_PATH.read_text())
    df = load_eval_dataset()
    by_doc: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        chunks = (gold.get(row["financebench_id"]) or {}).get("gold_chunks") or []
        by_doc.setdefault(row["doc_name"], []).extend(c for c in chunks if c.strip())
    docs = sorted(d for d in by_doc if (PDF_DIR / f"{d}.pdf").exists())
    import random
    random.Random(seed).shuffle(docs)
    docs = docs[:sample]

    # {strategy: [per-doc containment fraction]} and chunk-count stats.
    hits: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    n_chunks: dict[str, list[int]] = {s: [] for s in STRATEGIES}
    total_gold = 0
    for doc in tqdm(docs, desc="documents", unit="doc"):
        text = _parse_pypdf(PDF_DIR / f"{doc}.pdf")["text"]
        golds = [_normalize(g) for g in by_doc[doc]]
        golds = [g for g in golds if g]
        total_gold += len(golds)
        if not golds:
            continue
        for name, chunker in STRATEGIES.items():
            chunks_norm = [_normalize(c) for c in chunker(text)]
            n_chunks[name].append(len(chunks_norm))
            captured = sum(1 for g in golds if _contained(g, chunks_norm))
            hits[name].append(captured / len(golds))

    summary = {
        name: {
            "containment": round(statistics.mean(hits[name]), 4) if hits[name] else None,
            "mean_chunks_per_doc": round(statistics.mean(n_chunks[name]), 1)
            if n_chunks[name] else None,
        }
        for name in STRATEGIES
    }
    report = {"sample_docs": len(docs), "total_gold_passages": total_gold,
              "metric": "fraction of gold passages fully contained in a single chunk",
              "summary": summary}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))

    lines = ["# Chunk-strategy ablation — single-chunk gold containment", "",
             f"{len(docs)} FinanceBench docs · {total_gold} gold passages · "
             "containment = fraction of gold passages that fit inside one chunk.",
             "", "| strategy | containment | mean chunks/doc |", "|---|---|---|"]
    for name, agg in summary.items():
        lines.append(f"| {name} | {agg['containment']} | {agg['mean_chunks_per_doc']} |")
    lines += [
        "",
        "Notes: `parent_doc` = containment of a 2500-char parent window (children "
        "are matched, the parent is returned). `semantic` is a dependency-free "
        "line-merge proxy (SemanticChunker isn't installed) — it under-represents "
        "true embedding-based semantic chunking; read it as a floor, not a verdict.",
        "",
        "Caveat: containment measures whether gold evidence fits in ONE chunk. "
        "Larger chunks raise it but exceed bge-small's ~2048-char (512-token) embed "
        "window and dilute retrieval precision — the tension parent-document "
        "retrieval resolves (embed small children, return the large parent).",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"[chunk-ablation] report -> {OUT_JSON} / {OUT_MD}")
    for name, agg in summary.items():
        print(f"  {name:12s} containment={agg['containment']} "
              f"chunks/doc={agg['mean_chunks_per_doc']}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.sample, args.seed)


if __name__ == "__main__":
    main()
