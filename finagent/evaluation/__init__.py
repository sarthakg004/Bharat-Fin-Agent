"""Evaluation harness. Dev-time only — not imported by the serve path.

Canonical modules:
    dataset      — load_eval_dataset(), question-type tagging, held-out split.
    retrieval    — run_retrieval_eval(): pool_recall@k + conditional Hit@k/MRR.
    ragas_eval   — run_ragas_eval() + RAGASEvaluator (faithfulness, relevancy, …).
    ledger       — append_row(), load_ledger(), plot_metrics(), style_ledger().

Underlying implementations / CLIs (kept for their `python -m` entry points):
    ragas        — RAGASEvaluator class + CLI.
    reranker_ab  — reranker A/B CLI.
    financebench/ — the FinanceBench-specific harness the wrappers delegate to.

Imports are intentionally NOT done at package import time (eval pulls heavy
deps); import the specific module you need.

Depends on: config, retrieval, agents.
"""
