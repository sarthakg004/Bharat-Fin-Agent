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
import os
import time
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from tqdm import tqdm

DEFAULT_OUTPUTS = "results/financebench_baseline_outputs.json"
DEFAULT_SCORES = "results/financebench_baseline_ragas.csv"


def run_agent_outputs(
    questions: pd.DataFrame,
    output_path: Union[str, Path] = DEFAULT_OUTPUTS,
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
    from finagent.config import settings

    # The eval must search the dedicated `financebench_eval` collection — the one
    # that actually holds the benchmark's filings at the asked-about historical
    # years. Production serves `us_filings`; pointing the eval there (the old
    # default) made ~⅓ of questions find no filing and fall through to web noise.
    eval_collection = settings.financebench_collection

    # The eval corpus already contains every benchmark filing, so on-the-fly
    # EDGAR fetch is pure waste here — and the gate would misfire anyway, since
    # financebench_eval stores `ticker` as the company name (e.g. "Microsoft"),
    # not the symbol the resolver returns ("MSFT"). Turn it off for the eval; the
    # served path (us_filings + fetch) is unaffected.
    os.environ["DISABLE_DYNAMIC_FETCH"] = "1"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, dict] = {}
    if resume and output_path.exists():
        for row in json.loads(output_path.read_text()):
            # Errored rows are NOT done — a resume retries them (a transient
            # failure like a rate limit or a dead key shouldn't stick).
            if not row.get("error"):
                done[row.get("financebench_id", row.get("question"))] = row

    from finagent.llm import is_daily_quota_error, is_rate_limit_error

    outputs: list[dict] = []
    stop_reason = ""
    # Quiet under the parallel orchestrator: the parent draws the single
    # progress bar, so a worker's own bar would just interleave garbage.
    quiet = os.getenv("FINAGENT_EVAL_QUIET") == "1"
    for _, q in tqdm(questions.iterrows(), total=len(questions), desc="agent",
                     disable=quiet):
        if stop_reason:
            break
        fb_id = q.get("financebench_id", q["question"])
        if fb_id in done:
            outputs.append(done[fb_id])
            continue
        try:
            # One in-place retry for a per-minute exhaustion: every key being
            # limited usually clears within the cooldown window. A second
            # exhaustion (or a daily quota) stops the whole run — grinding the
            # remaining questions into error rows helps nobody; a resume picks
            # up exactly here once the limits reset.
            # FinanceBench is open-book over a KNOWN company's filing, but a few
            # questions never name it ("Was there any drop in Cash & Cash
            # equivalents…"). Append the company so retrieval can filter to it;
            # the output row and RAGAS still record/score the original question.
            question = q["question"]
            company = q.get("company") or ""
            if company and company.split()[0].lower() not in question.lower():
                question = f"{question} (Company: {company})"
            question_started = time.time()
            for attempt in (1, 2):
                try:
                    res = run_agentic(
                        question=question,
                        top_k=top_k,
                        provider=provider,
                        synth_model=synth_model,
                        collection=eval_collection,
                    )
                    break
                except Exception as e:
                    if (attempt == 1 and is_rate_limit_error(e)
                            and not is_daily_quota_error(e)):
                        tqdm.write("all keys rate-limited; waiting 65s before retrying…")
                        time.sleep(65)
                        continue
                    raise
            meta = res.get("metadata") or {}
            answer = res.get("answer", "")
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
                # Per-question cost/latency (aggregated into final_metrics):
                # wall-clock end-to-end, and token totals across every LLM call
                # in the run (from the usage callback).
                "latency_s": round(time.time() - question_started, 2),
                "input_tokens": meta.get("input_tokens"),
                "output_tokens": meta.get("output_tokens"),
                "error": None,
            }
        except Exception as e:  # record the failure
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
            if is_rate_limit_error(e):
                stop_reason = ("daily quota exhausted on every key"
                               if is_daily_quota_error(e)
                               else "every key rate-limited twice in a row")
        outputs.append(row)
        output_path.write_text(json.dumps(outputs, indent=2))

    if stop_reason:
        print(f"\nLIMIT EXHAUSTED — stopping the run ({stop_reason}). "
              f"Completed answers are saved; re-run the same command to resume "
              f"once the limits reset.")
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
    from finagent.llm import AllKeysExhaustedError

    ev = RAGASEvaluator(judge_provider=judge_provider, judge_model=judge_model,
                        max_workers=max_workers, timeout=timeout)
    try:
        scores_df = ev.evaluate(
            outputs_path=str(outputs_path),
            output_csv=str(scores_csv),
            ground_truth_col="gold",
            sample=sample,
            batch_size=batch_size,
        )
    except AllKeysExhaustedError:
        # Scorer stopped mid-run — read back whatever was flushed so far and
        # return partial metrics rather than crashing the whole pipeline.
        print("[score_answers] Keys exhausted mid-run; returning partial metrics.")
        scores_csv_path = Path(str(scores_csv))
        scores_df = pd.read_csv(str(scores_csv_path)) if scores_csv_path.exists() else pd.DataFrame()

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

    # Answered-subset view. RAGAS scores refusals as ~0 by construction
    # (a non-answer has no relevant claims), so the overall means conflate
    # "answers badly" with "abstains". Splitting them shows answer quality and
    # refusal cost separately — the honest headline pair.
    status_by_q = {o["question"]: o.get("answer_status", "") for o in outputs}
    answered_mask = per_q["question"].map(status_by_q) == "answered"
    answered_only = _means(per_q[answered_mask]) if answered_mask.any() else {}

    return {"overall": _means(per_q), "by_type": by_type,
            "answered_only": answered_only,
            "n_answered": int(answered_mask.sum()), "n_scored": len(per_q)}
