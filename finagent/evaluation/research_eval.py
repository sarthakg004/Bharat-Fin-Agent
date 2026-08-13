"""Deep Research evaluation — judge-free structural metrics for the generated
reports, benchmarked against the single-agent baseline (production `run_agentic`
answering the same question in one pass).

Per question, for both the report and the baseline:

    completed          a non-empty answer was produced (no exception)
    sections           number of `##` sections
    required_coverage  fraction of required sections present (Executive Summary,
                       Investment Thesis, Bull/Bear/Base, Risks). This is where
                       the baseline is expected to lose: one pass answers the
                       question but does not produce a structured thesis.
    citation_coverage  fraction of substantive paragraphs carrying a [N]
    citation_validity  fraction of cited [N] pointing inside the evidence list
                       (a dangling number is a hallucinated citation)
    source_diversity   distinct evidence kinds used (xbrl/calc/text/web/…)
    latency / tokens   cost of the run

Research-only: agent_success_rate (specialists done vs planned) and the count of
cross-check contradictions flagged.

Resumable — already-evaluated questions are skipped, so re-running after a change
and diffing the .md doubles as a regression harness.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

# The benchmark set: general theses, a specific question (custom agent), a
# comparison, and a risk-angled request — one per major scoping shape.
QUESTIONS = [
    {"question": "Should I invest in Apple for the long term?", "company": "Apple"},
    {"question": "Is Microsoft a good investment right now?", "company": "Microsoft"},
    {"question": "How will AI demand affect Nvidia's growth, and is the stock overvalued?", "company": "Nvidia"},
    {"question": "Compare Coca-Cola and PepsiCo as long-term holdings.", "company": "Coca-Cola"},
    {"question": "What are the biggest risks of holding Tesla stock?", "company": "Tesla"},
    {"question": "Assess Amazon's cash-flow quality and capital intensity.", "company": "Amazon"},
    {"question": "Is Intel's turnaround investable?", "company": "Intel"},
    {"question": "Should I buy JPMorgan for dividend income?", "company": "JPMorgan"},
]

REQUIRED_SECTIONS = ("executive summary", "investment thesis", "bull case",
                     "bear case", "base case", "risk")

_CITE_RE = re.compile(r"\[(\d+)(?:\s*,\s*\d+)*\]")
_HEADING_RE = re.compile(r"^##\s", re.MULTILINE)


def score_output(text: str, chunks: list[dict]) -> dict:
    """Judge-free structural scores for one answer/report."""
    text = text or ""
    lower = text.lower()
    paragraphs = [p for p in text.split("\n\n")
                  if len(p.strip()) > 120 and not p.lstrip().startswith("#")]
    cited = [p for p in paragraphs if _CITE_RE.search(p)]
    cite_ids = [int(m.group(1)) for m in _CITE_RE.finditer(text)]
    n_chunks = len(chunks or [])
    valid = [i for i in cite_ids if 1 <= i <= n_chunks]
    kinds = {c.get("kind") or "?" for c in (chunks or [])}

    return {
        "completed": bool(text.strip()),
        "words": len(text.split()),
        "sections": len(_HEADING_RE.findall(text)),
        "required_coverage": round(
            sum(1 for s in REQUIRED_SECTIONS if s in lower) / len(REQUIRED_SECTIONS), 3),
        "citation_coverage": round(len(cited) / len(paragraphs), 3) if paragraphs else 0.0,
        "citation_validity": round(len(valid) / len(cite_ids), 3) if cite_ids else None,
        "citations": len(cite_ids),
        "evidence_count": n_chunks,
        "source_diversity": len(kinds),
    }


def _evaluate_one(q: dict, skip_baseline: bool) -> dict:
    from finagent.api.rag_service import run_agentic
    from finagent.research import DeepResearch

    row: dict = {"question": q["question"], "company": q["company"]}

    t0 = time.time()
    try:
        research = DeepResearch(run_fn=lambda question: run_agentic(question))
        out = research.run(q["question"])
        meta = out["metadata"]
        row["research"] = {
            **score_output(out["report"], out["chunks"]),
            "latency_s": round(time.time() - t0, 1),
            "input_tokens": meta.get("input_tokens"),
            "output_tokens": meta.get("output_tokens"),
            "agents_planned": len(meta.get("tasks") or []),
            "agents_run": meta.get("agents_run"),
            "agent_success_rate": round(
                meta.get("agents_run", 0)
                / max(1, meta.get("agents_run", 0) + meta.get("agents_failed", 0)), 3),
            "contradictions": len(meta.get("contradictions") or []),
        }
    except Exception as e:
        row["research"] = {"completed": False, "error": f"{type(e).__name__}: {e}"}

    if not skip_baseline:
        t0 = time.time()
        try:
            base = run_agentic(q["question"])
            bmeta = base.get("metadata") or {}
            row["baseline"] = {
                **score_output(base.get("answer") or "", base.get("chunks") or []),
                "latency_s": round(time.time() - t0, 1),
                "input_tokens": bmeta.get("input_tokens"),
                "output_tokens": bmeta.get("output_tokens"),
            }
        except Exception as e:
            row["baseline"] = {"completed": False, "error": f"{type(e).__name__}: {e}"}
    return row


def _mean(rows: list[dict], side: str, key: str):
    vals = [r[side][key] for r in rows
            if r.get(side, {}).get(key) is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize(rows: list[dict]) -> dict:
    keys = ("sections", "required_coverage", "citation_coverage",
            "citation_validity", "source_diversity", "evidence_count",
            "words", "latency_s", "input_tokens", "output_tokens")
    summary = {"n": len(rows)}
    for side in ("research", "baseline"):
        if not any(side in r for r in rows):
            continue
        summary[side] = {k: _mean(rows, side, k) for k in keys}
    if any("research" in r for r in rows):
        summary["research"]["agent_success_rate"] = _mean(rows, "research", "agent_success_rate")
        summary["research"]["contradictions"] = _mean(rows, "research", "contradictions")
    return summary


def write_markdown(summary: dict, path: Path) -> None:
    res, base = summary.get("research") or {}, summary.get("baseline") or {}
    lines = [
        "# Deep Research evaluation",
        "",
        f"{summary['n']} questions. Research = multi-specialist report; "
        "baseline = single-agent answer to the same question.",
        "",
        "| Metric | Deep Research | Baseline |",
        "|---|---|---|",
    ]
    for k in ("required_coverage", "citation_coverage", "citation_validity",
              "sections", "source_diversity", "evidence_count", "words",
              "latency_s", "input_tokens", "output_tokens"):
        lines.append(f"| {k.replace('_', ' ')} | {res.get(k, '—')} | {base.get(k, '—')} |")
    if "agent_success_rate" in res:
        lines.append(f"| agent success rate | {res['agent_success_rate']} | — |")
        lines.append(f"| contradictions flagged | {res['contradictions']} | — |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate Deep Research Mode.")
    p.add_argument("--sample", type=int, default=len(QUESTIONS),
                   help="How many benchmark questions to run.")
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip the single-agent baseline runs.")
    p.add_argument("--output", default="results/research_eval.json")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if out_path.exists():
        rows = json.loads(out_path.read_text()).get("rows", [])
        print(f"Resuming: {len(rows)} already evaluated")
    done = {r["question"] for r in rows}

    for q in QUESTIONS[: args.sample]:
        if q["question"] in done:
            continue
        print(f"→ {q['question']}")
        rows.append(_evaluate_one(q, args.skip_baseline))
        out_path.write_text(json.dumps(
            {"rows": rows, "summary": summarize(rows)}, indent=2))

    summary = summarize(rows)
    out_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    write_markdown(summary, out_path.with_suffix(".md"))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved → {out_path} and {out_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
