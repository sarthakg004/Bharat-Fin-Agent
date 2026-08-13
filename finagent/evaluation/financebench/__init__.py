"""FinanceBench evaluation harness.

    dataset    — load questions, tag by type, carve a stable held-out slice.
    indexing   — index the referenced filings into the eval Qdrant collection.
    gold       — map each question to the chunk(s) holding its evidence.
    retrieval  — pool_recall@{20,50,100} + conditional Hit@{1,3,5}/MRR.
    runner     — run the agent over the held-out slice and score it.
    parallel   — sharded runner + the final metrics report.
"""

from .dataset import (
    load_questions,
    tag_question_types,
    classify_question,
    split_heldout,
    type_breakdown,
)
from .indexing import build_manifest, index_eval_corpus, EVAL_COLLECTION
from .gold import build_gold_map
from .retrieval import RetrievalEvaluator, RetrievalResult
from .runner import run_agent_outputs, score_answers
from .parallel import run_parallel, final_report, summarize_outputs

__all__ = [
    "load_questions",
    "tag_question_types",
    "classify_question",
    "split_heldout",
    "type_breakdown",
    "build_manifest",
    "index_eval_corpus",
    "EVAL_COLLECTION",
    "build_gold_map",
    "RetrievalEvaluator",
    "RetrievalResult",
    "run_agent_outputs",
    "score_answers",
    "run_parallel",
    "final_report",
    "summarize_outputs",
]
