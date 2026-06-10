"""
runner.py  ·  finagent/evaluation/financebench/runner.py

Run the *current production agent* over a set of FinanceBench questions and
score the answers with RAGAS. This produces the `baseline_v0` answer-quality
numbers — the "where we are today" measurement every later phase is compared
against.

Two steps:
    run_agent_outputs(...)  — call the deployed entrypoint
                              (`rag_service.run_agentic`) per question and save
                              outputs in the shape the RAGAS harness consumes.
    score_answers(...)      — run `RAGASEvaluator` over those outputs and return
                              the four metrics overall *and* per question type.

We deliberately call the real `run_agentic`, so the answers are the ones a user
would actually get — not a notebook re-implementation.

Output-row key convention (matches `finagent.evaluation.ragas.RAGASEvaluator`):
    "answer"  -> the model's generated answer   (RAGAS `response`)
    "gold"    -> FinanceBench verified answer    (RAGAS `reference`)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from tqdm import tqdm

DEFAULT_OUTPUTS = "results/financebench_baseline_outputs.json"
DEFAULT_SCORES = "results/financebench_baseline_ragas.csv"


def run_agent_outputs(
    questions: pd.DataFrame,
    output_path: Union[str, Path] = DEFAULT_OUTPUTS,
    market: str = "us",
    provider: str = "groq",
    synth_model: Optional[str] = None,
    top_k: int = 5,
    resume: bool = True,
) -> list[dict]:
    """Answer every question with the production agent; save incrementally.

    Incremental + resumable: results are flushed after every question, so a
    rate-limit or crash never loses completed work.
    """
    # Imported lazily — pulls in the whole graph stack.
    from finagent.api.rag_service import run_agentic

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, dict] = {}
    if resume and output_path.exists():
        for row in json.loads(output_path.read_text()):
            done[row.get("financebench_id", row.get("question"))] = row

    outputs: list[dict] = []
    for _, q in tqdm(questions.iterrows(), total=len(questions), desc="agent"):
        fb_id = q.get("financebench_id", q["question"])
        if fb_id in done:
            outputs.append(done[fb_id])
            continue
        try:
            res = run_agentic(
                market=market,
                question=q["question"],
                top_k=top_k,
                provider=provider,
                synth_model=synth_model,
            )
            meta = res.get("metadata") or {}
            answer = res.get("answer", "")
            # When the confidence gate withheld a draft ("I'm not confident
            # enough…"), evaluate the draft itself — a low-confidence answer
            # scores what the system actually produced; the withhold notice
            # scores as no answer at all.
            suppressed = (meta.get("suppressed_answer") or "").strip()
            if suppressed:
                answer = suppressed
            row = {
                "financebench_id": fb_id,
                "question": q["question"],
                "answer": answer,                                # generated (RAGAS response)
                "gold": q.get("answer", ""),                     # gold (RAGAS reference)
                "retrieved_chunks": [c.get("text", "") for c in res.get("chunks", [])],
                "qtype": q.get("qtype", ""),
                "company": q.get("company", ""),
                "confidence": meta.get("confidence"),
                "answer_status": meta.get("answer_status"),
                "used_suppressed_draft": bool(suppressed),
                "error": None,
            }
        except Exception as e:  # keep going; record the failure
            row = {
                "financebench_id": fb_id,
                "question": q["question"],
                "answer": "",
                "gold": q.get("answer", ""),
                "retrieved_chunks": [],
                "qtype": q.get("qtype", ""),
                "company": q.get("company", ""),
                "error": f"{type(e).__name__}: {e}",
            }
        outputs.append(row)
        output_path.write_text(json.dumps(outputs, indent=2))

    return outputs


def score_answers(
    outputs_path: Union[str, Path] = DEFAULT_OUTPUTS,
    scores_csv: Union[str, Path] = DEFAULT_SCORES,
    judge_provider: str = "groq",
    judge_model: Optional[str] = None,
    sample: Optional[int] = None,
    max_workers: int = 4,
    timeout: int = 300,
    batch_size: int = 10,
) -> dict:
    """RAGAS the saved outputs; return overall + per-qtype metric means.

    Reuses the production `RAGASEvaluator` (the reference is taken from the
    `"gold"` key). Returns:
        {
          "overall": {faithfulness, answer_relevancy, context_precision,
                      context_recall},
          "by_type": {qtype: {...same four...}, ...},
        }

    On free-tier judges (Groq/Gemini) the default parallelism can trip
    `TimeoutError`s; pass `max_workers=1` (serialize), a larger `timeout`, and
    a smaller `batch_size` to keep the run stable.
    """
    from finagent.evaluation.ragas import RAGASEvaluator

    ev = RAGASEvaluator(judge_provider=judge_provider, judge_model=judge_model,
                        max_workers=max_workers, timeout=timeout)
    scores_df = ev.evaluate(
        outputs_path=str(outputs_path),
        output_csv=str(scores_csv),
        ground_truth_col="gold",
        sample=sample,
        batch_size=batch_size,
    )

    metric_cols = [
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    ]
    per_q = scores_df[scores_df["question"] != "*** MEAN ***"].copy()

    # Attach qtype by joining back on the question text.
    outputs = json.loads(Path(outputs_path).read_text())
    qtype_by_q = {o["question"]: o.get("qtype", "") for o in outputs}
    per_q["qtype"] = per_q["question"].map(qtype_by_q)

    def _means(frame):
        return {
            c: round(float(pd.to_numeric(frame[c], errors="coerce").mean()), 4)
            for c in metric_cols if c in frame.columns
        }

    by_type = {
        qt: _means(grp) for qt, grp in per_q.groupby("qtype") if qt
    }
    return {"overall": _means(per_q), "by_type": by_type}
