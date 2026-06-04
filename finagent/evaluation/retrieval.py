"""
retrieval.py  ·  finagent/evaluation/retrieval.py

Notebook-facing retrieval evaluation. Builds (or reuses) the gold-chunk map and
runs the decomposed retrieval eval — `pool_recall@{20,50,100}` plus conditional
`Hit@{1,3,5}`/`MRR` — over the `financebench_eval` collection.

Delegates to `finagent.evaluation.financebench`.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from finagent.evaluation.financebench.gold import build_gold_map
from finagent.evaluation.financebench.retrieval import (
    RetrievalEvaluator,
    RetrievalResult,
)
from finagent.evaluation.financebench.indexing import EVAL_COLLECTION


def run_retrieval_eval(
    eval_dataset: Optional[pd.DataFrame] = None,
    collection: str = EVAL_COLLECTION,
    chroma_dir: Optional[str] = None,
    reranker_model: str = "BAAI/bge-reranker-base",
) -> RetrievalResult:
    """Run the decomposed retrieval eval; return a `RetrievalResult`.

    `eval_dataset` is optional — the gold map is derived from the full
    FinanceBench question set when omitted.
    """
    gold_map = build_gold_map(questions=eval_dataset, collection_name=collection,
                              chroma_dir=chroma_dir)
    evaluator = RetrievalEvaluator(
        collection_name=collection, chroma_dir=chroma_dir,
        reranker_model=reranker_model,
    )
    return evaluator.evaluate(gold_map)


def retrieval_result_frame(result: RetrievalResult) -> pd.DataFrame:
    """Per-question-type retrieval metrics as a tidy DataFrame for display."""
    rows = {"overall": result.as_dict(), **result.by_type}
    return pd.DataFrame(rows).T
