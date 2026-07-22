"""
custom.py  ·  finagent/evaluation/custom.py

Eval harness for YOUR OWN question set — the failure modes FinanceBench doesn't
cover (relative periods like "year-over-year", quarterly filings, multi-part
driver questions). One JSONL file in, one scored JSON + summary out.

Dataset row (data/eval/custom_questions.jsonl), only `question` is required:
    {
      "id": "now-q1-2026-opmargin",
      "question": "By how many percentage points did ServiceNow's operating margin improve year-over-year?",
      "gold_answer": "Improved ~1.5pp, from 9.9% (Q1 2025) to 11.4% (Q1 2026)",
      "expected_periods": ["Q1 2026", "Q1 2025"],
      "forbidden_periods": ["FY2022", "FY2023"],
      "gold_evidence": ["sales and marketing", "research and development"]
    }

Metrics per question (all deterministic, judge-free):
    numeric_accuracy — gold figure appears in the answer (reuses
                       financebench.answer_match; None if gold has no number)
    period_match     — EVERY expected period is named in the answer
    stale_period     — ANY forbidden period is named (the wrong-year failure)
    evidence_recall  — fraction of gold_evidence strings found in the
                       retrieved chunks (retrieval quality without gold chunks)

Outputs are written in the FinanceBench runner's row shape, so the existing
RAGAS judge scores them unchanged:
    python -m finagent.evaluation.custom --dataset data/eval/custom_questions.jsonl
    python -m finagent.evaluation.custom --ragas          # add judge metrics
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional, Union

DEFAULT_DATASET = "data/eval/custom_questions.jsonl"
DEFAULT_OUTPUTS = "results/custom_eval/outputs.json"
DEFAULT_SUMMARY = "results/custom_eval/summary.json"


def _period_in(period: str, text: str) -> bool:
    """'Q1 2026' / 'FY2025' named in `text`, tolerant of 'Q1 FY2026' /
    'first quarter of 2026' / '2026 Q1' orderings."""
    t = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "
    p = period.lower().strip()
    m = re.match(r"(q[1-4]|fy)\s*(\d{4})$", re.sub(r"[^a-z0-9]+", "", p))
    if not m:
        return p in t
    tag, year = m.groups()
    if tag == "fy":
        return bool(re.search(rf"\b(fy\s*)?{year}\b", t))
    ordinal = {"q1": "first", "q2": "second", "q3": "third", "q4": "fourth"}[tag]
    return bool(re.search(rf"{tag}\s*(fy\s*)?{year}", t)
                or re.search(rf"{year}\s*{tag}", t)
                or (ordinal in t and re.search(rf"\b{year}\b", t)))


def score_row(row: dict, spec: dict) -> dict:
    """Deterministic scores for one answered row against its dataset spec."""
    from finagent.evaluation.financebench.answer_match import numeric_match

    answer = row.get("answer") or ""
    chunks_text = "\n".join(row.get("retrieved_chunks") or []).lower()

    expected = spec.get("expected_periods") or []
    forbidden = spec.get("forbidden_periods") or []
    gold_ev = spec.get("gold_evidence") or []
    return {
        "numeric_accuracy": numeric_match(spec.get("gold_answer") or "", answer),
        "period_match": (all(_period_in(p, answer) for p in expected)
                         if expected else None),
        "stale_period": (any(_period_in(p, answer) for p in forbidden)
                         if forbidden else None),
        "evidence_recall": (sum(e.lower() in chunks_text for e in gold_ev) / len(gold_ev)
                            if gold_ev else None),
    }


def run_custom_eval(
    dataset: Union[str, Path] = DEFAULT_DATASET,
    outputs_path: Union[str, Path] = DEFAULT_OUTPUTS,
    summary_path: Union[str, Path] = DEFAULT_SUMMARY,
    provider: str = "groq",
    collection: Optional[str] = None,
    resume: bool = True,
    ragas: bool = False,
) -> dict:
    """Run the production agent over the custom set; score retrieval + answers."""
    from finagent.api.rag_service import run_agentic

    specs = [json.loads(l) for l in Path(dataset).read_text().splitlines()
             if l.strip()]
    outputs_path, summary_path = Path(outputs_path), Path(summary_path)
    outputs_path.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, dict] = {}
    if resume and outputs_path.exists():
        for r in json.loads(outputs_path.read_text()):
            if not r.get("error"):
                done[r["financebench_id"]] = r

    rows: list[dict] = []
    for spec in specs:
        qid = spec.get("id") or spec["question"]
        if qid in done:
            rows.append(done[qid])
            continue
        t0 = time.time()
        try:
            res = run_agentic(question=spec["question"],
                              provider=provider, collection=collection)
            row = {
                "financebench_id": qid,
                "question": spec["question"],
                "answer": res.get("answer", ""),
                "gold": spec.get("gold_answer", ""),
                "retrieved_chunks": [c.get("text", "") for c in res.get("chunks", [])],
                "qtype": spec.get("qtype", "custom"),
                "latency_s": round(time.time() - t0, 2),
                "error": None,
            }
        except Exception as e:
            row = {"financebench_id": qid, "question": spec["question"],
                   "answer": "", "gold": spec.get("gold_answer", ""),
                   "retrieved_chunks": [], "qtype": spec.get("qtype", "custom"),
                   "error": f"{type(e).__name__}: {e}"}
        row["scores"] = score_row(row, spec)
        rows.append(row)
        outputs_path.write_text(json.dumps(rows, indent=2))

    # Re-score resumed rows too (the spec may have gained gold fields).
    by_id = {s.get("id") or s["question"]: s for s in specs}
    for r in rows:
        r["scores"] = score_row(r, by_id.get(r["financebench_id"], {}))
    outputs_path.write_text(json.dumps(rows, indent=2))

    def _mean(key):
        vals = [r["scores"][key] for r in rows
                if r["scores"].get(key) is not None]
        return round(sum(float(v) for v in vals) / len(vals), 4) if vals else None

    summary = {
        "n": len(rows),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "numeric_accuracy": _mean("numeric_accuracy"),
        "period_match": _mean("period_match"),
        "stale_period_rate": _mean("stale_period"),
        "evidence_recall": _mean("evidence_recall"),
        "mean_latency_s": _mean_latency(rows),
    }
    if ragas:
        from finagent.evaluation.financebench.runner import score_answers
        summary["ragas"] = score_answers(
            outputs_path=outputs_path,
            scores_csv=outputs_path.with_name("ragas_scores.csv"))
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def _mean_latency(rows: list[dict]) -> Optional[float]:
    vals = [r["latency_s"] for r in rows if r.get("latency_s")]
    return round(sum(vals) / len(vals), 2) if vals else None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--ragas", action="store_true",
                    help="also run the RAGAS judge over the outputs")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    out = run_custom_eval(dataset=args.dataset, provider=args.provider,
                          collection=args.collection, ragas=args.ragas,
                          resume=not args.no_resume)
    print(json.dumps(out, indent=2))
