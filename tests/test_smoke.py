"""Smoke tests — fast, no network, no API keys.

They verify the package wiring stays intact after refactors: the public
imports resolve, the FastAPI app builds, and the deployed agent's LangGraph
compiles with the expected nodes (graph construction is lazy w.r.t. LLMs and
retrievers, so this needs neither a key nor the vector store).
"""

from __future__ import annotations


def test_package_imports():
    import finagent
    from finagent.graph import AgenticRAGv4, FinAgent

    assert finagent.__version__
    assert FinAgent is AgenticRAGv4


def test_api_app_builds():
    from finagent.api.main import app

    assert app.title == "FinAgent API"


def test_health_reports_agentic_only():
    from finagent.api import rag_service

    health = rag_service.health()
    assert health["status"] == "ok"
    assert health["configs"] == ["agentic"]  # naive mode removed


def _build_agent():
    from finagent.graph import AgenticRAGv4

    return AgenticRAGv4(
        collection_name="us_filings", market="us", provider="groq",
        reranker_model="BAAI/bge-reranker-base",
        bm25_top_k=8, dense_top_k=8, final_top_k=5,
        max_rewrites=2, max_critic_retries=1,
        table_collection="tables", news_collection="news",
        web_top_k=10, table_top_k=3,
    )


def test_agent_graph_has_expected_nodes():
    agent = _build_agent()
    nodes = {n for n in agent.graph.get_graph().nodes if not n.startswith("__")}
    expected = {
        "planner", "router", "retrieve", "grader", "rewrite", "table_agent",
        "market_data", "web_search", "synthesize", "critic", "verify_numbers",
        "refuse",
    }
    assert expected <= nodes, f"missing nodes: {expected - nodes}"
    # English-only: the bilingual translation nodes must be gone.
    assert not ({"detect_lang", "translate_in", "translate_out"} & nodes)


def test_device_selection_returns_valid_value():
    from finagent.device import get_device

    assert get_device() in {"cpu", "cuda", "mps"}
