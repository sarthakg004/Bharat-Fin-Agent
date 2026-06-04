"""
ledger.py  ·  finagent/evaluation/ledger.py

Notebook-facing view of the experiment ledger: load it, and plot how the headline
metrics move across phases. Append/write logic lives in
`finagent.evaluation.financebench.ledger` (re-exported here so the notebook has a
single import).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from finagent.evaluation.financebench.ledger import (
    DEFAULT_LEDGER,
    append_row,
    load_ledger,
)


def plot_metrics(
    ledger: Optional[pd.DataFrame] = None,
    metrics: Optional[list[str]] = None,
    ledger_path: Union[str, Path] = DEFAULT_LEDGER,
    title: str = "FinAgent — metric progression by phase",
):
    """Line chart of selected metrics across phases (one line per metric).

    `metrics` are column names in the ledger (e.g.
    "ragas.overall.faithfulness", "retrieval.hit@3"). Missing columns are
    skipped with a note. Returns the matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    if ledger is None:
        ledger = load_ledger(ledger_path)
    if ledger.empty:
        raise ValueError("Ledger is empty — record at least one phase first.")

    ledger = ledger.sort_values("timestamp")
    metrics = metrics or [
        c for c in (
            "ragas.overall.faithfulness", "ragas.overall.answer_relevancy",
            "ragas.overall.context_recall", "retrieval.hit@3",
        ) if c in ledger.columns
    ]
    present = [m for m in metrics if m in ledger.columns]
    missing = [m for m in metrics if m not in ledger.columns]
    if missing:
        print(f"(skipping metrics not in ledger: {missing})")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(ledger))
    for m in present:
        y = pd.to_numeric(ledger[m], errors="coerce")
        ax.plot(list(x), y, marker="o", label=m)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ledger["phase"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return ax


def style_ledger(ledger: Optional[pd.DataFrame] = None,
                 ledger_path: Union[str, Path] = DEFAULT_LEDGER):
    """Return a Styler: phases as rows, metrics green→red. For notebook display."""
    if ledger is None:
        ledger = load_ledger(ledger_path)
    if ledger.empty:
        return ledger
    metric_cols = [
        c for c in ledger.columns
        if c not in ("phase", "timestamp", "notes", "config")
        and pd.api.types.is_numeric_dtype(pd.to_numeric(ledger[c], errors="coerce"))
    ]
    view = ledger.drop(columns=["config"], errors="ignore")
    return view.style.background_gradient(
        subset=[c for c in metric_cols if c in view.columns], cmap="RdYlGn"
    ).format(precision=3)


__all__ = ["append_row", "load_ledger", "plot_metrics", "style_ledger", "DEFAULT_LEDGER"]
