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
    ("planner", "question", "Decomposes the question into 1-3 sub-queries", "sub_queries", "planner (llama-3.1-8b)"),
    ("router", "sub_queries", "Classifies each sub-query: narrative/numeric/market/external", "query_routes", "planner (llama-3.1-8b)"),
    ("retrieve", "sub_queries", "Hybrid BM25∪dense retrieval + cross-encoder rerank per sub-query", "retrieved_chunks", "—"),
    ("grader", "retrieved_chunks", "Scores each chunk 1-5 for relevance; drops low ones", "grades, avg_grade", "grader (llama-3.1-8b)"),
    ("rewrite", "question, avg_grade", "Reformulates the query when retrieval graded poorly (loops back to router)", "sub_queries, rewrite_history", "planner (llama-3.1-8b)"),
    ("table_agent", "numeric sub-queries", "Retrieves tables and runs generated pandas code for numeric answers", "table_results", "synth (gpt-oss-120b)"),
    ("market_data", "market sub-queries", "Calls yfinance tools for live price/valuation", "market_data, charts", "planner (llama-3.1-8b)"),
    ("web_search", "external/low-confidence sub-queries", "Tavily web search (news fallback) for current/out-of-corpus info", "web_results", "—"),
    ("synthesize", "retrieved_chunks, table/web/market results", "Writes the cited draft answer from all evidence", "draft_answer, citations", "synth (gpt-oss-120b)"),
    ("critic", "draft_answer, retrieved_chunks", "Checks every claim against context; may schedule a retrieve loop", "grading_score, needs_retry", "critic (gpt-oss-120b)"),
    ("verify_numbers", "draft_answer, evidence", "Extracts each figure and verifies it appears in the evidence", "numeric_verification, final_answer", "critic (gpt-oss-120b)"),
    ("refuse", "numeric_verification", "Declines to answer when figures can't be grounded", "final_answer, refused", "—"),
]
