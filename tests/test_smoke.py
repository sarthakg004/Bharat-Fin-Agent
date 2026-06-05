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
        "market_data", "web_search", "evidence_builder", "synthesize", "critic",
        "verify_numbers", "refuse",
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


def test_evidence_builder_normalises_all_lanes():
    """#3: every lane projects into one {kind,fact,value,unit,source,citation,
    confidence,sub_query} shape, with per-source confidence ordering preserved
    (XBRL > web) and filing items refined by their grader score."""
    agent = _build_agent()
    state = {
        "xbrl_facts": [{"entity": "AMAZON", "concept": "revenue",
                        "period_label": "FY2017", "value_str": "$177.866B",
                        "value": 177866000000, "unit": "USD", "tag": "Revenues",
                        "source": "SEC XBRL", "sub_query": "amzn rev"}],
        "calc_results": [{"metric": "growth", "ticker": "AMZN", "value": 0.308,
                          "value_str": "+30.8%", "is_percent": True,
                          "source": "computed", "sub_query": "amzn growth"}],
        "retrieved_chunks": [{"text": "Risk factors include competition.",
                              "source": "[AMZN 10-K 2017, p.5]", "sub_query": "risks"}],
        "grades": [5],
        "web_results": [{"title": "Amazon hits new high", "source": "Reuters",
                         "url": "http://x", "sub_query": "news"}],
    }
    out = agent.evidence_builder_node(state)
    ev = out["evidence"]
    kinds = {e["kind"] for e in ev}
    assert {"xbrl", "calc", "filing", "web"} <= kinds
    # Required shape on every item.
    for e in ev:
        assert set(e) >= {"kind", "fact", "value", "unit", "source",
                          "citation", "confidence", "sub_query"}
    by_kind = {e["kind"]: e for e in ev}
    assert by_kind["xbrl"]["value"] == 177866000000
    assert by_kind["calc"]["unit"] == "%"
    assert by_kind["filing"]["confidence"] == 1.0          # grade 5 → (5-1)/4
    assert by_kind["xbrl"]["confidence"] > by_kind["web"]["confidence"]


def test_market_label_tokens_are_not_verified_as_figures():
    """#5 regression: window/year-range labels ("52-wk high", "FY2023-24",
    "200-day"), incl. with Unicode hyphens, must not be extracted as claimed
    figures — that was falsely refusing correct market answers."""
    agent = _build_agent()
    text = ("FY 2023‑24 Stock – 52‑wk high $316.94 ; 52‑wk low $194.30 ; "
            "200-day moving average over 2023-2024")
    raws = [d["raw"] for d in agent._extract_numbers(text)]
    assert not any(x in ("24", "52", "200", "2023", "2024") for x in raws), raws
    assert any("316.94" in x for x in raws) and any("194.30" in x for x in raws)


def test_verification_report_cross_source_and_units():
    """#5: the report attributes each grounded figure to the lanes that
    corroborate it, flags a stated-scale/unit mismatch, and reports citation
    coverage of numeric evidence — without driving refusals itself."""
    agent = _build_agent()
    # Revenue corroborated by BOTH an XBRL fact and a web snippet; plus a draft
    # figure stated in the WRONG unit ("$136 million" for a $136bn value).
    state = {
        "draft_answer": "Revenue was $135.987 billion and also $136 million somewhere.",
        "xbrl_facts": [{"value": 135987000000, "sub_query": "rev"}],
        "web_results": [{"content": "Amazon revenue reached $135.987 billion.",
                         "source": "Reuters"}],
        "evidence": [{"kind": "xbrl", "value": 135987000000, "citation": "us-gaap:Revenues"}],
    }
    draft_nums = agent._extract_numbers(state["draft_answer"])
    by_kind = agent._evidence_numbers_by_kind(state)
    ungrounded = {d["raw"] for d in draft_nums
                  if not agent._grounded(d["magnitudes"], agent._evidence_numbers(state))}
    rep = agent._build_verification_report(state, draft_nums, ungrounded, by_kind)
    # The billions figure is corroborated by ≥2 lanes (xbrl + web).
    assert rep["cross_source"]["corroborated"] >= 1
    # The "$136 million" form grounds only via a scale restatement → unit flag.
    assert rep["units"]["scale_shifted_figures"] >= 1
    # Citation coverage of numeric evidence is reported.
    assert rep["sources"]["numeric_evidence_items"] == 1
    assert rep["sources"]["with_citation"] == 1


def test_device_selection_returns_valid_value():
    from finagent.device import get_device

    assert get_device() in {"cpu", "cuda", "mps"}
