"""Evaluation harness. Dev-time only — never imported by the serve path.

    dataset            — load_eval_dataset(), question-type tagging, held-out split.
    ragas              — RAGASEvaluator: faithfulness, relevancy, precision, recall.
    evaluate_retrieval — the retrieval sweep (index geometry × embedder × reranker).
    financebench/      — the FinanceBench end-to-end harness.
    custom             — small hand-written question set.
    research_eval      — scores deep-research reports against a baseline answer.

Nothing is imported at package-import time: eval pulls heavy deps, so import the
module you need directly.
"""
