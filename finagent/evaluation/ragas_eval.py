"""
ragas_eval.py  ·  finagent/evaluation/ragas_eval.py

Notebook-facing RAGAS answer evaluation. Runs the production agent over a slice
of the eval set and scores the answers (faithfulness, answer relevancy, context
precision/recall) overall and per question type.

Delegates to `finagent.evaluation.financebench.runner`.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from finagent.evaluation.financebench.runner import run_agent_outputs, score_answers
# Canonical re-export: the RAGAS judge harness lives in `evaluation.ragas`
# (kept there for its CLI `python -m finagent.evaluation.ragas`); expose it from
# the canonical module name too.
from finagent.evaluation.ragas import RAGASEvaluator  # noqa: F401


def run_ragas_eval(
    eval_dataset: pd.DataFrame,
    judge_provider: str = "groq",
    judge_model: Optional[str] = None,
    sample: Optional[int] = None,
    market: str = "us",
    provider: str = "groq",
) -> dict:
    """Run the agent over `eval_dataset`, then RAGAS-score the answers.

    Returns ``{"overall": {...}, "by_type": {qtype: {...}}}``.
    """
    run_agent_outputs(eval_dataset, market=market, provider=provider)
    return score_answers(
        judge_provider=judge_provider, judge_model=judge_model, sample=sample,
    )


def ragas_frame(scores: dict) -> pd.DataFrame:
    """RAGAS overall + per-type scores as a DataFrame for display."""
    rows = {"overall": scores.get("overall", {}), **scores.get("by_type", {})}
    return pd.DataFrame(rows).T


__all__ = ["run_ragas_eval", "ragas_frame", "RAGASEvaluator"]
