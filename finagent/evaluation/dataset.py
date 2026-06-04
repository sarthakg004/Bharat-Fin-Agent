"""
dataset.py  ·  finagent/evaluation/dataset.py

Notebook-facing loader for the evaluation dataset. Delegates to
`finagent.evaluation.financebench.dataset`, returning the questions already
tagged by type so the notebook can show the distribution.
"""

from __future__ import annotations

from typing import Union
from pathlib import Path

import pandas as pd

from finagent.evaluation.financebench.dataset import (
    DEFAULT_QUESTIONS,
    load_questions,
    tag_question_types,
    split_heldout,
    type_breakdown,
)


def load_eval_dataset(path: Union[str, Path] = DEFAULT_QUESTIONS) -> pd.DataFrame:
    """Load FinanceBench questions tagged with a `qtype` column."""
    return tag_question_types(load_questions(path))


__all__ = [
    "load_eval_dataset",
    "load_questions",
    "tag_question_types",
    "split_heldout",
    "type_breakdown",
]
