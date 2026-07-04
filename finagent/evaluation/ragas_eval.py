"""
ragas_eval.py  ·  finagent/evaluation/ragas_eval.py

Notebook-facing RAGAS answer evaluation. Runs the production agent over a slice
of the eval set and scores the answers (faithfulness, answer relevancy, context
precision/recall) overall and per question type.

Delegates to `finagent.evaluation.financebench.runner`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from finagent.evaluation.financebench.runner import (
    DEFAULT_OUTPUTS,
    DEFAULT_SCORES,
    run_agent_outputs,
    score_answers,
)
# Canonical re-export: the RAGAS judge harness lives in `evaluation.ragas`
# (kept there for its CLI `python -m finagent.evaluation.ragas`); expose it from
# the canonical module name too.
from finagent.evaluation.ragas import RAGASEvaluator  # noqa: F401


def run_ragas_eval(
    eval_dataset: pd.DataFrame,
    judge_provider: str = "groq",
    judge_model: Optional[str] = None,
    sample: Optional[int] = None,
    provider: str = "groq",
    synth_model: Optional[str] = None,
    output_path: Union[str, Path] = DEFAULT_OUTPUTS,
    scores_csv: Union[str, Path] = DEFAULT_SCORES,
    max_workers: int = 4,
    timeout: int = 300,
    batch_size: int = 10,
) -> dict:
    """Run the agent over `eval_dataset`, then RAGAS-score the answers.

    Returns ``{"overall": {...}, "by_type": {qtype: {...}}}``.

    Model control:
      * `synth_model` sets the model that writes the final answer (e.g.
        ``"qwen/qwen3.6-27b"``); ``None`` keeps the agent's default.
      * `judge_model` sets the RAGAS judge (e.g. ``"openai/gpt-oss-120b"``).

    `output_path` is the agent-answer cache. Generation is resumable, so a
    fresh path forces a clean run with the new `synth_model` instead of reusing
    answers written by a previous model.

    Free-tier stability: pass `max_workers=1`, a larger `timeout`, and a smaller
    `batch_size` if the judge trips `TimeoutError`s under parallelism.
    """
    run_agent_outputs(eval_dataset, provider=provider,
                      synth_model=synth_model, output_path=output_path)
    return score_answers(
        outputs_path=output_path, scores_csv=scores_csv,
        judge_provider=judge_provider, judge_model=judge_model, sample=sample,
        max_workers=max_workers, timeout=timeout, batch_size=batch_size,
    )


def ragas_frame(scores: dict) -> pd.DataFrame:
    """RAGAS overall + per-type scores as a DataFrame for display."""
    rows = {"overall": scores.get("overall", {}), **scores.get("by_type", {})}
    return pd.DataFrame(rows).T


__all__ = ["run_ragas_eval", "ragas_frame", "RAGASEvaluator"]
