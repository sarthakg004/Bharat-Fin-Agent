"""
builder.py  ·  finagent/graph/builder.py

One entrypoint for getting the deployed agent and its compiled LangGraph, so the
notebook can render the graph and run traces without knowing the construction
details. This returns the *exact* agent the API serves (`AgenticRAGv4` via
`rag_service`), so the diagram and traces match production.
"""

from __future__ import annotations

from typing import Optional


def build_agent(market: str = "us", provider: str = "groq"):
    """Return the singleton production agent (`AgenticRAGv4`)."""
    from finagent.api.rag_service import get_agentic

    return get_agentic(market=market, provider=provider)


def build_graph(market: str = "us", provider: str = "groq"):
    """Return the compiled LangGraph for the production agent.

    The result supports `.get_graph().draw_mermaid_png()` for visualization and
    `.invoke()` / `.stream()` for execution.
    """
    return build_agent(market=market, provider=provider).graph


# Node reference table — name → (reads, does, writes, llm). Surfaced so the
# notebook's node-by-node walkthrough is generated from one source of truth.
NODE_REFERENCE = [
    ("planner", "question, chat_history", "Fused plan+route: decomposes into 1-8 sub-queries AND tags each lane in ONE structured call", "sub_queries, query_routes", "router tier (gpt-oss-120b)"),
    ("router", "sub_queries, query_routes", "No-op when the plan carries aligned routes; re-classifies after a rewrite", "query_routes", "router tier (gpt-oss-120b, only on re-route)"),
    ("fetch_filing", "question", "Corpus gate: fetches a missing company's 10-K(s) from EDGAR (in-memory on cloud), year-aware depth", "fetch_status, fetched_chunks", "router tier (gpt-oss-120b)"),
    ("retrieve", "sub_queries", "Hybrid BM25∪dense retrieval + metadata filter + cross-encoder rerank per sub-query", "retrieved_chunks", "—"),
    ("grader", "retrieved_chunks", "Scores each chunk 1-5 for relevance; drops low ones", "grades, avg_grade", "grader (llama-3.3-70b)"),
    ("rewrite", "question, avg_grade", "Reformulates the query when retrieval graded poorly (loops back to router)", "sub_queries, rewrite_history", "grader (llama-3.3-70b)"),
    ("xbrl", "numeric sub-queries", "Looks up the exact reported figure from SEC XBRL company-facts", "xbrl_facts", "router tier (gpt-oss-120b, batched extraction)"),
    ("calculator", "numeric sub-queries", "Deterministically computes derived metrics (margins/ratios/growth/CAGR) over XBRL inputs", "calc_results", "router tier (gpt-oss-120b, batched extraction)"),
    ("table_agent", "unanswered numeric sub-queries", "Retrieves tables and runs generated pandas code (numeric fallback)", "table_results", "code (gpt-oss-120b)"),
    ("market_data", "market sub-queries", "Calls yfinance tools for live price/history/news (parallel lane)", "market_data, charts", "planner tier"),
    ("web_search", "external sub-queries + escalations", "Tavily web search; fires on news intent, empty retrieval, or an 'I can't answer' draft (parallel lane)", "web_results", "—"),
    ("edgar_search", "cross_document sub-queries", "EDGAR full-text search across all filers (parallel lane)", "edgar_results", "router tier (gpt-oss-120b)"),
    ("evidence_builder", "all lane outputs", "Normalises every lane into one evidence list; one-shot corpus fallback when all lanes are empty", "evidence", "—"),
    ("synthesize", "evidence, chat_history", "Writes the cited draft answer from all evidence (analyst voice)", "draft_answer, citations", "synth (gpt-oss-120b)"),
    ("critic", "draft_answer, evidence", "Checks every claim; routes to a focused re-draft or web escalation", "grading_score, needs_retry, critic_feedback", "critic (gpt-oss-120b)"),
    ("verify_numbers", "draft_answer, evidence", "Deterministically grounds every figure (LLM verifier only as rescue)", "numeric_verification", "verifier (gpt-oss-120b, rescue only)"),
    ("confidence", "sub-scores", "Blends retrieval/verification/citation/critic into one confidence + band", "confidence, confidence_band", "—"),
    ("refuse", "numeric_verification", "Declines to answer when most figures can't be grounded", "final_answer, refused", "—"),
]
