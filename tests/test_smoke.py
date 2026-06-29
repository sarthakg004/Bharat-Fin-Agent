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


def test_audit_trail_has_all_required_sections():
    """#13: every answer carries an audit trail — sources used, calculations
    performed, verification status, and the confidence score."""
    from finagent.api.rag_service import _build_audit
    state = {
        "status": "answered", "query_routes": ["numeric"],
        "evidence": [{"kind": "xbrl", "source": "SEC XBRL",
                      "citation": "us-gaap:Revenues", "confidence": 0.98}],
        "calc_results": [{"metric": "growth", "ticker": "AMZN", "value_str": "+30.8%",
                          "formula": "(b-a)/a",
                          "inputs": [{"concept": "revenue", "value_str": "$136B", "fy": 2016}]}],
        "numeric_verification": {"score": 1.0, "numbers_total": 6, "numbers_grounded": 6,
                                 "unverified": []},
        "verification_report": {"cross_source": {"corroborated": 1}},
        "confidence": 0.889, "confidence_band": "answer",
        "citation_score": 0.667, "critic_score": 1.0,
    }
    a = _build_audit(state)
    assert a["sources_used"] and a["sources_used"][0]["kind"] == "xbrl"
    assert a["calculations"] and a["calculations"][0]["result"] == "+30.8%"
    assert a["verification"]["numeric_score"] == 1.0
    assert a["confidence"]["score"] == 0.889 and a["confidence"]["band"] == "answer"


def test_observability_summaries():
    """#11: token usage flattens across models, and tool health tallies per lane."""
    from types import SimpleNamespace
    from finagent.api.rag_service import _sum_usage, _tool_health
    cb = SimpleNamespace(usage_metadata={
        "llama-3.1-8b-instant": {"input_tokens": 100, "output_tokens": 20},
        "openai/gpt-oss-120b": {"input_tokens": 400, "output_tokens": 80},
    })
    u = _sum_usage(cb)
    assert u["input_tokens"] == 500 and u["output_tokens"] == 100 and u["total_tokens"] == 600
    assert _sum_usage(None) == {}

    th = _tool_health({
        "xbrl_facts": [{"ok": True}, {"ok": False}],
        "market_data": [{"ok": True}],
        "table_results": [{"error": "boom"}, {"answer": "x"}],
        "web_results": [{"title": "a"}],
    })
    assert th["xbrl"] == {"calls": 2, "ok": 1, "failed": 1}
    assert th["market"]["ok"] == 1
    assert th["table"] == {"calls": 2, "ok": 1, "failed": 1}
    assert th["web"]["ok"] == 1


def test_market_cache_is_bounded_and_ttl():
    """#12: the market tool cache serves hits, never caches failures, stays hard
    bounded (LRU), and expires by TTL — so it can't grow unbounded on a small box."""
    import time as _time
    import finagent.tools.market as M
    calls = {"n": 0}
    M.TOOLS["__t__"] = {"fn": lambda symbol: (
        calls.__setitem__("n", calls["n"] + 1)
        or {"ok": True, "data": {"symbol": symbol}, "error": None})}
    M._cache.clear()
    r1 = M.call_tool("__t__", symbol="AAPL")
    r2 = M.call_tool("__t__", symbol="AAPL")
    assert calls["n"] == 1 and r1 is r2            # 2nd request served from cache

    M.TOOLS["__e__"] = {"fn": lambda symbol: {"ok": False, "data": None, "error": "x"}}
    M.call_tool("__e__", symbol="X"); M.call_tool("__e__", symbol="X")
    assert "__e__|" + repr([("symbol", "X")]) not in M._cache   # failures not cached

    M._cache.clear()
    for i in range(M._CACHE_MAX + 25):
        M.call_tool("__t__", symbol=f"T{i}")
    assert len(M._cache) <= M._CACHE_MAX           # hard bound

    old = M._CACHE_TTL; M._CACHE_TTL = 0.05; calls["n"] = 0
    M._cache.clear()
    M.call_tool("__t__", symbol="Z"); _time.sleep(0.08); M.call_tool("__t__", symbol="Z")
    M._CACHE_TTL = old
    assert calls["n"] == 2                          # expired → refetched
    M.TOOLS.pop("__t__", None); M.TOOLS.pop("__e__", None)


def test_retrieval_status_caps_and_defers_refusal():
    """#7: retrieval status reports attempts vs cap + retrieval confidence and
    flags exhaustion, but defers the refuse decision to the confidence gate."""
    from finagent.api.rag_service import _retrieval_status
    ex = _retrieval_status({"iteration_count": 2, "avg_grade": 1.5}, 2, 3.0)
    assert ex["exhausted"] is True and ex["refusal_handled_by"] == "confidence_gate"
    ok = _retrieval_status({"iteration_count": 0, "avg_grade": 4.5}, 2, 3.0)
    assert ok["exhausted"] is False
    # No grades at all (pure tool path) → not exhausted-with-grade, just capped check.
    none = _retrieval_status({"iteration_count": 0, "avg_grade": None}, 2, 3.0)
    assert none["exhausted"] is False


def test_new_nodes_degrade_gracefully_on_bad_state(monkeypatch=None):
    """#14: the evidence_builder / confidence / verification-report additions
    must never crash the run on a malformed state — they degrade to a safe dict."""
    agent = _build_agent()
    # Malformed calc_results (inputs is not a list of dicts) must not raise.
    bad = {"calc_results": [{"value": 0.3, "inputs": "oops"}],
           "xbrl_facts": [{"value": None}], "draft_answer": "x"}
    out = agent.evidence_builder_node(bad)
    assert "evidence" in out                      # returned, didn't raise

    # Confidence scoring on a malformed verification dict fails OPEN (answers).
    out2 = agent.confidence_node({"numeric_verification": "not-a-dict",
                                  "draft_answer": "x", "grading_score": 1.0})
    assert "confidence_band" in out2              # returned a band, didn't raise


def test_audit_degraded_summary():
    """#14: the audit flags lanes that ran but returned nothing usable."""
    from finagent.api.rag_service import _build_audit
    a = _build_audit({"market_data": [{"ok": False}, {"ok": False}],
                      "xbrl_facts": [{"ok": True}]})
    assert a["degraded"]["is_degraded"] is True
    assert "market" in a["degraded"]["lanes"] and "xbrl" not in a["degraded"]["lanes"]


def test_active_critic_router_severity_and_bound():
    """#6: critic routes to a re-draft only when there are unsupported claims AND
    evidence is rich; thin evidence defers to verify (heavy re-gather); the flag
    disables it. The loop is bounded by the existing critic_iterations cap."""
    agent = _build_agent()
    agent._log = lambda s, m: None
    # unsupported + rich evidence → light re-draft
    assert agent._critic_router({"needs_retry": True, "xbrl_facts": [{"ok": True}]}) == "resynthesize"
    # unsupported + thin evidence → defer to verify (heavy path)
    assert agent._critic_router({"needs_retry": True}) == "verify"
    # nothing unsupported → proceed
    assert agent._critic_router({"needs_retry": False, "xbrl_facts": [1]}) == "verify"
    # flag off → always proceed
    agent.active_critic = False
    assert agent._critic_router({"needs_retry": True, "xbrl_facts": [1]}) == "verify"
    agent.active_critic = True
    # critic edge offers both targets (bound comes from critic_iterations cap).
    targets = {(e.source, e.target) for e in agent.graph.get_graph().edges}
    assert ("critic", "synthesize") in targets and ("critic", "verify_numbers") in targets


def test_xbrl_tag_relevance_guard_blocks_concept_mismatch():
    """Regression: an unreported concept must NOT be silently mapped onto a
    loosely-related tag (the bug that surfaced AES 'restructuring costs' as
    AssetImpairmentCharges = $1.715B). The relevance guard rejects a fallback
    tag that shares no salient word with the concept."""
    from finagent.tools.xbrl import XBRLClient as X, CONCEPT_TAGS
    # The exact mismatch that caused the hallucination.
    assert X._tag_relevant("restructuring costs", "AssetImpairmentCharges") is False
    assert X._tag_relevant("restructuring costs", "CostsAndExpenses") is False
    # Correct mappings still pass.
    assert X._tag_relevant("restructuring costs", "RestructuringCharges") is True
    assert X._tag_relevant("asset impairment", "AssetImpairmentCharges") is True
    assert X._tag_relevant("research and development", "ResearchAndDevelopmentExpense") is True
    # Restructuring is now in the curated map, so reporters resolve correctly.
    assert "restructuring" in CONCEPT_TAGS


def test_quick_ratio_metric_is_supported():
    """Quick (acid-test) ratio must be a recognised derived metric with a
    composite numerator (current assets − inventory) / current liabilities."""
    from finagent.tools.calculator import (
        _canonical_metric, COMPOSITE_RATIOS, RATIOS)
    assert _canonical_metric("quick ratio") == "quick_ratio"
    assert _canonical_metric("acid-test ratio") == "quick_ratio"
    assert "quick_ratio" in COMPOSITE_RATIOS
    spec = COMPOSITE_RATIOS["quick_ratio"]
    assert spec["add"] == ["current_assets"] and spec["sub"] == ["inventory"]
    assert spec["den"] == "current_liabilities"
    # cash ratio is a recognised simple ratio too.
    assert _canonical_metric("cash ratio") == "cash_ratio" and "cash_ratio" in RATIOS


def test_router_uses_strong_tool_tier():
    """Tool selection / structured extraction must default to the strong synth
    tier, not the fast planner model (mis-routing was sending M&A to filings)."""
    agent = _build_agent()
    assert agent.router_model == agent.synth_model


def test_web_search_historical_vs_news():
    """A historical/factual lookup (past fiscal years, acquisitions) must search
    all-time by relevance, not as recency-filtered 'news' — that was returning
    recent articles about unrelated companies for an FY2021-2023 M&A question."""
    from finagent.tools.web_search import is_historical
    assert is_historical("major acquisitions Amcor did in FY2023, FY2022, FY2021") is True
    assert is_historical("AES restructuring history") is True
    assert is_historical("AAPL premarket price right now") is False
    assert is_historical("latest news on Tesla") is False


def test_device_selection_returns_valid_value():
    from finagent.device import get_device

    assert get_device() in {"cpu", "cuda", "mps"}


def test_derived_figures_are_not_falsely_refused():
    """Regression (FinanceBench refusal sweep): figures the draft legitimately
    COMPUTES from evidence values — a D&A margin, a 3-year average via chained
    per-year ratios, a 365-day working-capital metric — must ground, while a
    figure with no arithmetic path to the evidence must stay ungrounded."""
    agent = _build_agent()
    by_kind = {"xbrl": [167e6, 3.991e9], "const": list(agent._MATH_CONSTANTS)}
    base = agent._derivation_base(by_kind)
    # margin = a / b (×100 percent form)
    assert agent._derivable({4.2, 0.042}, base)
    # DSO-style: 365 × a / b
    ar, rev = 1.679e9, 16.865e9
    assert agent._derivable({36.34}, agent._derivation_base({"xbrl": [ar, rev]}))
    # two-period average, then a chained ratio over it (worked CCC steps)
    inv18, inv19, cogs = 1.642e9, 1.560e9, 11.108e9
    b = agent._derivation_base({"xbrl": [inv18, inv19, cogs]})
    avg = (inv18 + inv19) / 2
    assert agent._derivable({avg}, b)
    assert agent._derivable({round(365 * avg / cogs, 2)}, b + [avg])
    # a hallucinated figure must NOT be rescued
    assert not agent._derivable({7.77, 0.0777},
                                agent._derivation_base({"xbrl": [ar, rev]}))


def test_label_tokens_are_not_extracted_as_figures():
    """Regression: '8k'/'10K' form names, 'FY2015 - FY2017' ranges, and
    '3 year average' period labels were extracted as numeric claims and caused
    wrongful refusals ('unverified figures: 8k')."""
    agent = _build_agent()
    texts = [
        "The key agenda of AMCOR's 8k filing dated 1 July 2022 was the merger.",
        "Per the 10-K, 10K and 10 Q filings.",
        "The FY2015 - FY2017 3 year average margin.",
        "A 5-year CAGR per the 8-K.",
    ]
    for t in texts:
        raws = [d["raw"] for d in agent._extract_numbers(t)
                if d["raw"] not in ("1",)]          # bare day-of-month is a const
        assert not raws, f"label tokens extracted as figures: {raws} in {t!r}"


def test_mostly_grounded_answer_is_not_hard_refused():
    """Partial ungrounding goes to the confidence gate (warn/withhold keeps the
    draft); only a substantially fabricated answer (< refuse_below_grounding
    share grounded) still hard-refuses at the retry cap."""
    agent = _build_agent()
    out_of_retries = {"critic_iterations": 99, "verify_iterations": 99}
    mostly = {"numeric_verification": {
        "numbers_total": 10, "unverified": [{"number": "7"}], "score": 0.9},
        **out_of_retries}
    assert agent._verify_router(mostly) == "end"
    fabricated = {"numeric_verification": {
        "numbers_total": 10, "unverified": [{"number": str(i)} for i in range(6)],
        "score": 0.4}, **out_of_retries}
    assert agent._verify_router(fabricated) == "refuse"


def test_working_capital_days_metrics_are_supported():
    """dio/dso/dpo/ccc must be recognised derived metrics, and the LAST period
    named is year t (periods[0] is the prior averaging year, not the target)."""
    from finagent.tools.calculator import WC_DAYS_METRICS, _canonical_metric
    assert _canonical_metric("cash conversion cycle") == "ccc"
    assert _canonical_metric("days inventory outstanding") == "dio"
    assert _canonical_metric("days payables outstanding") == "dpo"
    assert WC_DAYS_METRICS == {"dio", "dso", "dpo", "ccc"}


def test_tools_lane_corpus_fallback_is_wired():
    """A tools-path run whose lanes all return empty must route back through
    fetch_filing → retrieve once (and only once) instead of synthesising from
    nothing."""
    agent = _build_agent()
    empty_tools_state = {
        "question": "q", "query_routes": ["numeric"],
        "retrieved_chunks": [], "xbrl_facts": [], "calc_results": [],
        "table_results": [], "web_results": [], "market_data": [],
        "edgar_results": [],
    }
    out = agent.evidence_builder_node(dict(empty_tools_state))
    assert out.get("corpus_fallback_pending") is True
    assert agent._evidence_router({**empty_tools_state, **out}) == "retrieve"
    # second pass: the latch prevents a loop even if retrieval found nothing
    out2 = agent.evidence_builder_node({**empty_tools_state, **out})
    assert out2.get("corpus_fallback_pending") is False
    assert agent._evidence_router({**empty_tools_state, **out, **out2}) == "synthesize"


def test_dated_form_request_detection():
    """A question naming an event filing + date must be detected (and routed to
    the retrieval path so fetch_filing can pull the document from EDGAR by
    date); ordinary numeric questions must not."""
    from datetime import date
    from finagent.graph.agent import AgenticRAGv4 as A
    assert A._dated_form_request(
        "What was the key agenda of the AMCOR's 8k filing dated 1st July 2022?"
    ) == ("8-K", date(2022, 7, 1))
    assert A._dated_form_request(
        "Summarize Apple's 10-Q for July 1, 2022") == ("10-Q", date(2022, 7, 1))
    assert A._dated_form_request("What is AMD's FY2015 D&A margin?") is None
    assert A._dated_form_request("Best Buy 8-K filings in general") is None


def test_insufficient_draft_escalates_to_web_once():
    """A draft admitting the evidence can't answer must trigger the one-shot
    web escalation; an answered draft, a run with web results, or a spent latch
    must not."""
    agent = _build_agent()
    insufficient = {"draft_answer": "The key agenda is not explicitly stated "
                                    "in the provided evidence."}
    sig = agent._web_fallback_signal(insufficient)
    assert sig == {"web_fallback_pending": True, "web_fallback_used": True}
    assert agent._critic_router({**insufficient, **sig}) == "websearch"
    # latch spent → no second escalation
    assert agent._web_fallback_signal(
        {**insufficient, "web_fallback_used": True}
    ) == {"web_fallback_pending": False}
    # web already ran → no escalation
    assert agent._web_fallback_signal(
        {**insufficient, "web_results": [{"content": "x"}]}
    ) == {"web_fallback_pending": False}
    # confident draft → no escalation
    assert agent._web_fallback_signal(
        {"draft_answer": "Revenue was $10 billion [1]."}
    ) == {"web_fallback_pending": False}


def test_all_keys_exhausted_circuit_breaker():
    """When every key is rate-limited, the rotator raises a fast, classified
    AllKeysExhaustedError and latches a cooldown so subsequent calls fail
    instantly (no network) — the UI then reports 'limit exhausted' in seconds
    instead of grinding every node through the whole pool again."""
    import finagent.llm as L

    # Classified as a rate-limit error (message hint), so the API layer shows
    # the friendly limit message rather than a raw traceback.
    assert L.is_rate_limit_error(L.AllKeysExhaustedError("keys hit their rate limit"))

    m = L.RotatingChatModel(provider="groq", chat_model="llama-3.1-8b-instant",
                            keys=["k1", "k2"], chat_kwargs={"temperature": 0})
    # A full failed rotation cycle latches the provider cooldown…
    L._EXHAUSTED_UNTIL.pop("groq", None)
    exc = m._exhausted(Exception("429 rate limit"))
    assert isinstance(exc, L.AllKeysExhaustedError)
    assert L._EXHAUSTED_UNTIL["groq"] > __import__("time").time()
    # …and while latched, calls fail fast without touching the network.
    try:
        m._retry(lambda llm: (_ for _ in ()).throw(AssertionError("network hit")))
        assert False, "expected AllKeysExhaustedError"
    except L.AllKeysExhaustedError:
        pass
    finally:
        L._EXHAUSTED_UNTIL.pop("groq", None)   # don't leak the latch to other tests


def test_numeric_accuracy_metric():
    """The deterministic numeric-correctness metric matches gold within tolerance,
    handles the percent/ratio dual form, and excludes non-numeric golds."""
    from finagent.evaluation.financebench.answer_match import (
        numeric_accuracy, numeric_match)

    assert numeric_match("$1577.00", "Capex was **$1,577 million** [1].") is True
    assert numeric_match("$1577.00", "Capex was $1,600 million [1].") is False
    # Gold 0.40 ↔ answer "40%" (ratio/percent dual form).
    assert numeric_match("0.40", "The ratio was 40% [1].") is True
    # Non-numeric gold → not scored (None).
    assert numeric_match("Yes, it increased.", "Revenue rose [1].") is None

    rows = [
        {"qtype": "numeric", "gold": "$100.00", "answer": "It was $100 [1]."},
        {"qtype": "numeric", "gold": "$200.00", "answer": "It was $250 [1]."},
        {"qtype": "narrative", "gold": "Strong growth", "answer": "Grew a lot."},
    ]
    res = numeric_accuracy(rows)
    assert res["overall"]["n"] == 2          # narrative excluded
    assert res["overall"]["correct"] == 1
    assert res["overall"]["accuracy"] == 0.5
    assert len(res["misses"]) == 1


def test_explicit_abstention_detection():
    """Soft-refusal phrasing is detected, real cited answers are not, and the
    metrics layer counts an explicit abstention as a refusal."""
    from finagent.graph.agent import _SOFT_REFUSAL_RE
    from finagent.evaluation.financebench.parallel import _is_refusal

    assert _SOFT_REFUSAL_RE.search("the figure cannot be calculated")
    assert _SOFT_REFUSAL_RE.search("Apple does not disclose segment margins")
    assert not _SOFT_REFUSAL_RE.search("Revenue was $394.3 billion (FY2022) [1].")
    # v2 failure phrasings that previously slipped through and scored conf=1.0:
    assert _SOFT_REFUSAL_RE.search("No relevant evidence provided to calculate the ratio.")
    assert _SOFT_REFUSAL_RE.search("No available data on FY2015 revenue is present.")
    assert _SOFT_REFUSAL_RE.search("No information is available in the provided sources.")

    assert _is_refusal("**Insufficient evidence to answer.** The filings ...")
    assert _is_refusal("I don't have enough information to answer this from ...")
    assert not _is_refusal("Revenue was $394.3 billion [1].")


def test_turnover_ratio_averages_denominator():
    """Flow÷stock ratios average the balance-sheet denominator over (t-1, t);
    the numerator stays single-period. Stubs XBRL so no network is needed."""
    from finagent.tools.calculator import FinancialCalculator, AVG_DENOMINATOR_RATIOS

    assert "fixed_asset_turnover" in AVG_DENOMINATOR_RATIOS
    assert "inventory_turnover" in AVG_DENOMINATOR_RATIOS

    c = FinancialCalculator.__new__(FinancialCalculator)   # skip XBRLClient init
    facts = {
        ("revenue", "FY2019"): 6489.0,
        ("ppe_net", "FY2019"): 253.0,
        ("ppe_net", "FY2018"): 282.0,                       # avg = 267.5
    }

    def fake_input(ticker, concept, period):
        if (concept, period) in facts:
            v = facts[(concept, period)]
            return {"ok": True, "concept": concept, "value": v,
                    "value_str": f"{v}", "tag": "X", "fy": 2019, "form": "10-K",
                    "source": "stub", "ticker": ticker}
        return {"ok": False, "concept": concept, "period": period, "error": "miss"}

    c._input = fake_input
    r = c.ratio("ATVI", "fixed_asset_turnover", "FY2019")
    assert r["ok"]
    # 6489 / avg(253, 282) = 6489 / 267.5 = 24.26 — NOT 6489/253 = 25.65.
    assert round(r["value"], 2) == 24.26
    assert "avg(ppe_net)" in r["formula"]

    # Prior year missing → graceful fall back to the year-end value.
    del facts[("ppe_net", "FY2018")]
    r2 = c.ratio("ATVI", "fixed_asset_turnover", "FY2019")
    assert r2["ok"] and round(r2["value"], 2) == 25.65


def test_ratio_from_spec_planned_formula():
    """A planned formula computes deterministically from stubbed XBRL facts —
    the LLM supplies structure, the numbers stay exact. No network."""
    from finagent.tools.calculator import FinancialCalculator

    c = FinancialCalculator.__new__(FinancialCalculator)
    facts = {
        ("operating_income", "FY2022"): 1000.0,
        ("depreciation_amortization", "FY2022"): 200.0,
        ("revenue", "FY2022"): 5000.0,
    }

    def fake_input(ticker, concept, period):
        if (concept, period) in facts:
            v = facts[(concept, period)]
            return {"ok": True, "concept": concept, "value": v,
                    "value_str": str(v), "fy": 2022, "ticker": ticker}
        return {"ok": False, "concept": concept, "period": period, "error": "miss"}

    c._input = fake_input

    # Ratio: EBITDA margin = (operating_income + D&A) / revenue = 1200/5000 = 24%.
    spec = {"numerator_add": ["operating_income", "depreciation_amortization"],
            "denominator_add": ["revenue"], "is_percent": True}
    r = c.ratio_from_spec("X", spec, "FY2022", "ebitda_margin")
    assert r["ok"] and round(r["value"], 4) == 0.24 and r["value_str"] == "24.0%"

    # Dollar amount: empty denominator → numerator only (unadjusted EBITDA = 1200).
    spec2 = {"numerator_add": ["operating_income", "depreciation_amortization"]}
    r2 = c.ratio_from_spec("X", spec2, "FY2022", "ebitda")
    assert r2["ok"] and r2["value"] == 1200.0 and "/" not in r2["formula"]

    # A concept the filing doesn't report → clean miss, not a crash.
    spec3 = {"numerator_add": ["goodwill"], "denominator_add": ["revenue"]}
    r3 = c.ratio_from_spec("X", spec3, "FY2022", "goodwill_intensity")
    assert not r3["ok"] and "missing" in r3["error"]
