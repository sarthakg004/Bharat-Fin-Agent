"""
compare.py  ·  finagent/evaluation/parsing/compare.py

Offline parser-quality eval: run Docling (the document-upload parser) and the
pypdf production baseline over a sample of FinanceBench PDFs and measure how
faithfully each extracts the text that actually matters.

Ground truth is free: `results/financebench_gold.json` carries the human-
annotated evidence passages for every question, so **evidence recall** — the
fraction of a gold passage's 10-word shingles found in the parsed text — is a
judge-free fidelity metric. Alongside it: extraction yield (chars/page), page
coverage, chunk-size stats under the production splitter, tables detected
(Docling), and parse time.

Usage:
    python -m finagent.evaluation.parsing.compare --sample 10
    → results/parsing_eval.json + results/parsing_eval.md
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

PDF_DIR = Path("data/us/eval/financebench/pdfs")
GOLD_PATH = Path("results/financebench_gold.json")
OUT_JSON = Path("results/parsing_eval.json")
OUT_MD = Path("results/parsing_eval.md")

_WS = re.compile(r"\s+")
SHINGLE_WORDS = 10


def _normalize(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def shingle_recall(gold: str, parsed: str, n: int = SHINGLE_WORDS) -> float:
    """Fraction of `gold`'s n-word shingles present verbatim in `parsed`
    (whitespace/case-insensitive). 1.0 when gold is shorter than one shingle
    and fully contained."""
    gold_n, parsed_n = _normalize(gold), _normalize(parsed)
    words = gold_n.split()
    if len(words) < n:
        return 1.0 if gold_n and gold_n in parsed_n else 0.0
    shingles = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    hits = sum(1 for s in shingles if s in parsed_n)
    return hits / len(shingles)


def _parse_pypdf(pdf: Path) -> dict:
    """Production-baseline extraction (CorpusIngester's per-page pypdf path)."""
    from pypdf import PdfReader

    t0 = time.time()
    reader = PdfReader(str(pdf))
    pages = {i + 1: (p.extract_text() or "") for i, p in enumerate(reader.pages)}
    text = "\n".join(pages.values())
    return {
        "text": text,
        "total_pages": len(reader.pages),
        "covered_pages": sum(1 for t in pages.values() if len(t.strip()) > 50),
        "tables": None,                       # pypdf has no table detection
        "parse_s": round(time.time() - t0, 2),
    }


def _parse_docling(pdf: Path) -> dict:
    """The upload parser (Docling, tables as markdown, no OCR)."""
    from pypdf import PdfReader

    from finagent.ingestion.upload import parse_upload

    total_pages = len(PdfReader(str(pdf)).pages)
    t0 = time.time()
    res = parse_upload(pdf.read_bytes(), pdf.name)
    return {
        "text": "\n".join(c["text"] for c in res["chunks"]),
        "total_pages": total_pages,
        "covered_pages": res["pages"],
        "tables": res["tables"],
        "parse_s": round(time.time() - t0, 2),
    }


PARSERS = {"pypdf": _parse_pypdf, "docling": _parse_docling}


def _chunk_stats(text: str) -> dict:
    from finagent.ingestion.upload import _get_splitter

    sizes = [len(c) for c in _get_splitter().split_text(text)] or [0]
    return {"n_chunks": len(sizes),
            "mean_chars": round(statistics.mean(sizes), 1),
            "p50_chars": statistics.median(sizes)}


def run(sample: int, seed: int = 0) -> dict:
    from tqdm import tqdm

    gold = json.loads(GOLD_PATH.read_text())
    from finagent.evaluation.dataset import load_eval_dataset
    df = load_eval_dataset()

    # Gold evidence passages grouped by source PDF.
    by_doc: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        chunks = (gold.get(row["financebench_id"]) or {}).get("gold_chunks") or []
        by_doc.setdefault(row["doc_name"], []).extend(c for c in chunks if c.strip())
    docs = sorted(d for d in by_doc if (PDF_DIR / f"{d}.pdf").exists())
    import random
    random.Random(seed).shuffle(docs)
    docs = docs[:sample]

    per_doc: list[dict] = []
    for doc in tqdm(docs, desc="documents", unit="doc"):
        pdf = PDF_DIR / f"{doc}.pdf"
        entry: dict = {"doc": doc, "gold_passages": len(by_doc[doc])}
        for name, parse in PARSERS.items():
            try:
                p = parse(pdf)
            except Exception as e:
                entry[name] = {"error": f"{type(e).__name__}: {e}"}
                continue
            recalls = [shingle_recall(g, p["text"]) for g in by_doc[doc]]
            entry[name] = {
                "evidence_recall": round(statistics.mean(recalls), 4) if recalls else None,
                "chars_per_page": round(len(p["text"]) / max(1, p["total_pages"]), 1),
                "page_coverage": round(p["covered_pages"] / max(1, p["total_pages"]), 4),
                "tables": p["tables"],
                "parse_s": p["parse_s"],
                **_chunk_stats(p["text"]),
            }
        per_doc.append(entry)

    # Aggregate means across docs (skip errored parses).
    summary: dict = {}
    for name in PARSERS:
        rows = [d[name] for d in per_doc if "error" not in d.get(name, {"error": 1})]
        if not rows:
            summary[name] = {"docs": 0}
            continue
        agg = {"docs": len(rows)}
        for key in ("evidence_recall", "chars_per_page", "page_coverage",
                    "parse_s", "n_chunks", "mean_chars"):
            vals = [r[key] for r in rows if r.get(key) is not None]
            agg[key] = round(statistics.mean(vals), 4) if vals else None
        if name == "docling":
            agg["tables_per_doc"] = round(
                statistics.mean([r["tables"] for r in rows if r["tables"] is not None]), 1)
        summary[name] = agg

    report = {"sample_docs": len(per_doc), "shingle_words": SHINGLE_WORDS,
              "summary": summary, "per_doc": per_doc}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))

    keys = ["evidence_recall", "chars_per_page", "page_coverage",
            "n_chunks", "parse_s"]
    lines = ["# Parsing evaluation — Docling (upload parser) vs pypdf (baseline)",
             "",
             f"{len(per_doc)} FinanceBench PDFs · evidence recall = fraction of each "
             f"gold passage's {SHINGLE_WORDS}-word shingles found in the parsed text.",
             "",
             "| parser | " + " | ".join(keys) + " | tables/doc |",
             "|---|" + "---|" * (len(keys) + 1)]
    for name, agg in summary.items():
        cells = [str(agg.get(k, "—")) for k in keys]
        lines.append(f"| {name} | " + " | ".join(cells) +
                     f" | {agg.get('tables_per_doc', '—')} |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"[parsing] report → {OUT_JSON} / {OUT_MD}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=10,
                    help="How many PDFs to evaluate (Docling on CPU takes "
                         "~1-3 min per annual report).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.sample, args.seed)


if __name__ == "__main__":
    main()
