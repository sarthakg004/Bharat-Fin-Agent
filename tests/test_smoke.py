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


def test_confidence_gate_is_wired():
    """The confidence framework (#8/9) sits after verification: verify_numbers
    feeds `confidence`, which gates to answer / warning / refuse."""
    agent = _build_agent()
    g = agent.graph.get_graph()
    nodes = {n for n in g.nodes if not n.startswith("__")}
    assert {"confidence", "answer_with_warning", "low_confidence"} <= nodes
    targets = {(e.source, e.target) for e in g.edges}
    assert ("verify_numbers", "confidence") in targets
    assert ("confidence", "answer_with_warning") in targets
    # Low band withholds the draft (for opt-in reveal) rather than hard-refusing.
    assert ("confidence", "low_confidence") in targets


def test_confidence_score_blends_and_renormalises():
    """A fully-grounded pure-XBRL answer (no retrieval) must not be penalised for
    the missing retrieval component — weights renormalise over what applies."""
    agent = _build_agent()
    out = agent.confidence_node({
        "draft_answer": "Net income was $99.8 billion [1].",
        "grades": [], "avg_grade": None, "grading_score": 1.0,
        "numeric_verification": {"numbers_total": 1, "numbers_grounded": 1, "score": 1.0},
        "xbrl_facts": [{"value": 99.8e9}],
    })
    assert out["retrieval_score"] is None          # not applicable
    assert out["confidence"] == 1.0
    assert out["confidence_band"] == "answer"


def test_xbrl_derivation_is_not_falsely_refused():
    """Regression: a correct XBRL-grounded growth answer that shows its working
    (scale-free figures inside a formula, ×100 percent conversion) and uses
    fullwidth 【N】 citations must verify clean — not get refused as ungrounded.
    """
    agent = _build_agent()
    state = {
        "draft_answer": (
            "Amazon revenue grew +30.8% YoY 【3】.\n"
            "FY2016 $135.987 billion 【1】; FY2017 $177.866 billion 【2】.\n"
            "YoY = (177.866 - 135.987) / 135.987 × 100 = 30.8% 【3】."
        ),
        "xbrl_facts": [{"value": 135987000000}, {"value": 177866000000}],
        "calc_results": [{"value": 0.30796, "inputs": [
            {"value": 135987000000}, {"value": 177866000000}]}],
    }
    mags = agent._evidence_numbers(state)
    figures = agent._extract_numbers(state["draft_answer"])
    ungrounded = [d["raw"] for d in figures if not agent._grounded(d["magnitudes"], mags)]
    assert not ungrounded, f"falsely ungrounded: {ungrounded}"
    # 【N】 citation markers must not be mistaken for figures 1/2/3.
    assert not any(d["raw"] in ("1", "2", "3") for d in figures)
    # Fullwidth citations still count toward citation coverage.
    assert agent._citation_score(state) == 1.0


def test_device_selection_returns_valid_value():
    from finagent.device import get_device

    assert get_device() in {"cpu", "cuda", "mps"}
