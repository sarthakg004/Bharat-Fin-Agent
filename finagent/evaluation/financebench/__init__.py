"""
FinanceBench evaluation harness (Phase 0 — eval foundation).

A small, script-based toolkit the notebook calls into (rather than redefining
functions inline) to measure the system end-to-end on FinanceBench:

    dataset    — load questions, tag by type, carve a stable held-out slice.
    indexing   — index the referenced filings into a dedicated
                 `financebench_eval` Chroma collection via the production
                 ingestion pipeline.
    gold       — map each question to the chunk(s) holding its verified evidence.
    retrieval  — the decomposed retrieval eval: pool_recall@{20,50,100} plus
                 conditional Hit@{1,3,5}/MRR after reranking.
    runner     — run the production agent over the held-out slice and RAGAS it.
    ledger     — the append-only experiment ledger (results/ledger.csv).
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
from .ledger import append_row, load_ledger

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
    "append_row",
    "load_ledger",
]
