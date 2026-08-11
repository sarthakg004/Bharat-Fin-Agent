"""Smoke tests — fast, no network, no API keys.

They verify the package wiring stays intact after refactors: the public
imports resolve, the FastAPI app builds, and the deployed agent's LangGraph
compiles with the expected nodes (graph construction is lazy w.r.t. LLMs and
retrievers, so this needs neither a key nor the vector store).
"""

from __future__ import annotations

import inspect


def test_package_imports():
    import finagent
    from finagent.graph import AgenticRAGv4, FinAgent

    assert finagent.__version__
    assert FinAgent is AgenticRAGv4


def test_api_app_builds():
    from finagent.api.main import app

    assert app.title == "FinAgent API"


def test_health_is_liveness_only():
    """The SPA polls /api/health every 10-30s to tell a cold start from a dead
    backend. It must stay free: no LLM probe (burns quota), no index probe
    (slows every poll). A 200 is the whole contract."""
    from starlette.testclient import TestClient

    from finagent.api.main import app

    with TestClient(app) as client:
        resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_deployed_image_ships_no_vector_data():
    """The corpus lives in Qdrant, not in the image.

    Baking it in was what forced a 2.2 GB image and a read-only index. Now that
    the store is a remote server, dynamic fetch can write to the shared index
    (PERSIST_DYNAMIC_FETCH=true) because the server handles concurrent
    read/write and deterministic point ids make a duplicate write harmless.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    dockerfile = root / "Dockerfile"
    text = dockerfile.read_text()
    assert "COPY data/" not in text, "the image must not bake in corpus data"
    assert "PERSIST_DYNAMIC_FETCH=true" in text
    assert "data/" in (root / ".dockerignore").read_text()


def test_image_bakes_every_model_loaded_offline():
    """The runtime sets HF_HUB_OFFLINE=1, so any model it loads must be baked
    into the image at build time.

    The BM25 sparse encoder was missed on the Qdrant cutover: the container
    started and /api/health passed, but the first real query died with
    "Could not load model Qdrant/bm25". Nothing catches that except asking the
    Dockerfile whether it bakes what the code names.
    """
    from pathlib import Path

    from finagent.config import settings
    from finagent.retrieval.hybrid import HybridRetriever
    from finagent.vectorstore import DEFAULT_EMBED_MODEL, SPARSE_MODEL

    text = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text()
    assert "HF_HUB_OFFLINE=1" in text
    # Only LOCAL models need baking. An API-served model (Gemini embeddings,
    # `cohere:` rerank) is an HTTPS call with no weights to cache, so requiring
    # it in the Dockerfile would be nonsense — but the check still has to run
    # for whatever local model remains, because that is the one that dies
    # offline on the first query. `settings` is the value the SERVED path
    # passes (rag_service), so it is the one that can actually reach the
    # runtime unbaked; `HybridRetriever.DEFAULT_RERANKER` is the local
    # fallback the Cohere path drops to and must always be present.
    def is_local(model: str) -> bool:
        return not (model.startswith("gemini-embedding")
                    or model.startswith("cohere:"))

    for model in (DEFAULT_EMBED_MODEL, SPARSE_MODEL,
                  HybridRetriever.DEFAULT_RERANKER, settings.reranker_model):
        if is_local(model):
            assert model in text, f"{model} is loaded at runtime but never baked"
    # The fallback cross-encoder is never API-served, so it is unconditional.
    assert HybridRetriever.DEFAULT_RERANKER in text


def _build_agent():
    from finagent.graph import AgenticRAGv4

    return AgenticRAGv4(
        collection_name="us_filings_v3",
        reranker_model="BAAI/bge-reranker-base",
        pool_top_k=16, final_top_k=5,
        max_critic_retries=1,
        web_top_k=10,
    )


def test_agent_graph_has_expected_nodes():
    agent = _build_agent()
    nodes = {n for n in agent.graph.get_graph().nodes if not n.startswith("__")}
    expected = {
        "planner", "router", "retrieve",
        "market_data", "web_search", "evidence_builder", "synthesize", "critic",
        "refuse",
    }
    assert expected <= nodes, f"missing nodes: {expected - nodes}"
    # English-only: the bilingual translation nodes must be gone.
    assert not ({"detect_lang", "translate_in", "translate_out"} & nodes)


def test_critic_is_the_only_gate():
    """The confidence gate and the numeric verifier are both gone — the critic
    is the only thing that may suppress an answer now. It reaches `refuse`
    directly when it could not support the draft's claims."""
    agent = _build_agent()
    g = agent.graph.get_graph()
    nodes = {n for n in g.nodes if not n.startswith("__")}
    assert not ({"confidence", "answer_with_warning", "low_confidence",
                 "verify_numbers", "grader", "rewrite"} & nodes)
    targets = {(e.source, e.target) for e in g.edges}
    assert ("critic", "refuse") in targets
    assert any(s == "critic" and t.startswith("__end__") for s, t in targets)
    for gone in ("confidence_node", "answer_with_warning_node",
                 "withhold_low_confidence_node", "_confidence_gate",
                 "verify_numbers_node", "grader_node", "rewrite_node"):
        assert not hasattr(agent, gone), gone


def test_evidence_builder_normalises_all_lanes():
    """#3: every lane projects into one {kind,fact,value,unit,source,citation,
    confidence,sub_query} shape, with per-source confidence ordering preserved
    (XBRL > web)."""
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
    assert by_kind["filing"]["confidence"] == agent._EVIDENCE_BASE_CONF["filing"]
    assert by_kind["xbrl"]["confidence"] > by_kind["web"]["confidence"]


def test_audit_trail_has_all_required_sections():
    """#13: every answer carries an audit trail — sources used, calculations
    performed, and verification status."""
    from finagent.api.rag_service import _build_audit
    state = {
        "status": "answered", "query_routes": ["numeric"],
        "evidence": [{"kind": "xbrl", "source": "SEC XBRL",
                      "citation": "us-gaap:Revenues", "confidence": 0.98}],
        "calc_results": [{"metric": "growth", "ticker": "AMZN", "value_str": "+30.8%",
                          "formula": "(b-a)/a",
                          "inputs": [{"concept": "revenue", "value_str": "$136B", "fy": 2016}]}],
        "grading_score": 1.0, "critic_feedback": [],
    }
    a = _build_audit(state)
    assert a["sources_used"] and a["sources_used"][0]["kind"] == "xbrl"
    assert a["calculations"] and a["calculations"][0]["result"] == "+30.8%"
    assert a["verification"]["claims_supported"] == 1.0
    assert "confidence" not in a


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
        "web_results": [{"title": "a"}],
    })
    assert th["xbrl"] == {"calls": 2, "ok": 1, "failed": 1}
    assert th["market"]["ok"] == 1
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
    """#7: status reports critic attempts vs cap and flags exhaustion, but
    defers the refuse decision to `_critic_router`."""
    from finagent.api.rag_service import _retrieval_status
    ex = _retrieval_status({"critic_iterations": 2, "needs_retry": True}, 2)
    assert ex["exhausted"] is True and ex["refusal_handled_by"] == "critic"
    ok = _retrieval_status({"critic_iterations": 0, "grading_score": 0.9}, 2)
    assert ok["exhausted"] is False
    # Pure tool path — the critic never ran.
    none = _retrieval_status({}, 2)
    assert none["exhausted"] is False


def test_new_nodes_degrade_gracefully_on_bad_state(monkeypatch=None):
    """#14: the evidence_builder / verification-report additions must never
    crash the run on a malformed state — they degrade to a safe dict."""
    agent = _build_agent()
    # Malformed calc_results (inputs is not a list of dicts) must not raise.
    bad = {"calc_results": [{"value": 0.3, "inputs": "oops"}],
           "xbrl_facts": [{"value": None}], "draft_answer": "x"}
    out = agent.evidence_builder_node(bad)
    assert "evidence" in out                      # returned, didn't raise


def test_audit_degraded_summary():
    """#14: the audit flags lanes that ran but returned nothing usable."""
    from finagent.api.rag_service import _build_audit
    a = _build_audit({"market_data": [{"ok": False}, {"ok": False}],
                      "xbrl_facts": [{"ok": True}]})
    assert a["degraded"]["is_degraded"] is True
    assert "market" in a["degraded"]["lanes"] and "xbrl" not in a["degraded"]["lanes"]


def test_critic_router_follows_the_remedy_and_spends_one_budget():
    """The critic picks the recovery, and the agent gets exactly one.

    Which recovery is the CRITIC's call now, not a guess from how much evidence
    happens to be lying around. The old rule keyed on evidence richness, and
    since `has_evidence` is true whenever any lane returned anything, the
    `retrieve` branch was unreachable in production: every recovery was a
    re-draft over byte-identical evidence.
    """
    agent = _build_agent()          # max_critic_retries=1
    agent._log = lambda s, m: None

    # The critic says the evidence is missing → go and get more.
    assert agent._critic_router(
        {"needs_retry": True, "critic_remedy": "gather"}) == "retrieve"
    # The critic says the draft over-claimed → re-write, same evidence.
    assert agent._critic_router(
        {"needs_retry": True, "critic_remedy": "redraft"}) == "resynthesize"
    # Rich evidence does NOT override a `gather` verdict any more.
    assert agent._critic_router(
        {"needs_retry": True, "critic_remedy": "gather",
         "xbrl_facts": [{"ok": True}], "evidence": [1, 2]}) == "retrieve"
    # Missing remedy degrades to the old behaviour rather than erroring.
    assert agent._critic_router({"needs_retry": True}) == "resynthesize"
    # Nothing unsupported → done.
    assert agent._critic_router({"needs_retry": False, "xbrl_facts": [1]}) == "end"

    # ONE recovery, enforced by the GATES rather than by this router: by the
    # time the router runs, the counter already includes the recovery it is
    # dispatching, so re-checking it here would veto the very pass that just
    # incremented it. `critic_node` and `_web_fallback_signal` each refuse to
    # raise their flag once the budget is spent.
    spent = {"critic_iterations": 1}
    assert agent._web_fallback_signal(
        {"draft_answer": "not explicitly stated in the provided evidence", **spent}
    ) == {"web_fallback_pending": False}
    assert agent._critic_router(
        {"needs_retry": False, "grading_score": 0.2, **spent}) == "refuse"

    targets = {(e.source, e.target) for e in agent.graph.get_graph().edges}
    for t in ("synthesize", "retrieve", "web_search", "refuse"):
        assert ("critic", t) in targets, t


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


def test_exactly_one_recovery_then_refuse():
    """Drive the real critic node twice and prove the budget holds.

    The router alone cannot prove this: the bound lives in the gates
    (`critic_node` and `_web_fallback_signal`), and the failure mode being
    guarded is two recoveries slipping through because each mechanism counted
    itself separately. So run the actual sequence.
    """
    from finagent.graph.state import ClaimVerdict, CriticReport

    agent = _build_agent()          # max_critic_retries=1
    agent._log = lambda s, m: None

    def stub_critic(remedy):
        report = CriticReport(
            verdicts=[ClaimVerdict(claim="Revenue was $9bn", supported=False,
                                   reason="not in evidence")],
            remedy=remedy)
        agent._get_llm = lambda role: type(
            "L", (), {"with_structured_output":
                      lambda self, schema: type(
                          "B", (), {"invoke": lambda self, msgs: report})()})()

    state = {"question": "q", "draft_answer": "Revenue was $9bn [1].",
             "retrieved_chunks": [{"text": "t", "source": "s"}], "errors": []}

    # Pass 1: the critic wants more evidence, so one recovery is granted.
    stub_critic("gather")
    first = agent.critic_node(state)
    assert first["needs_retry"] is True
    assert first["critic_iterations"] == 1
    assert agent._critic_router({**state, **first}) == "retrieve"
    # It must search something NEW, not the query that already failed.
    assert first["sub_queries"] != [state["question"]]
    assert first["retrieval_query"] == ""

    # Pass 2: same complaint, budget spent. No second recovery, and the draft
    # is refused rather than shipped mostly-unsupported.
    stub_critic("gather")
    second = agent.critic_node({**state, **first})
    assert second["needs_retry"] is False
    assert agent._critic_router({**state, **first, **second}) == "refuse"

    # A redraft recovery is bounded by the same single budget.
    stub_critic("redraft")
    only = agent.critic_node(state)
    assert agent._critic_router({**state, **only}) == "resynthesize"
    assert only["critic_iterations"] == 1
    # …and it does NOT clobber the planner's decomposition, since a re-draft
    # retrieves nothing.
    assert "sub_queries" not in only


def test_fetched_filing_is_not_diluted_with_web_results():
    """A filing fetched on demand from EDGAR must NOT trigger a web escalation.

    `fetch_filing` runs before retrieval and pulls a missing US company's 10-K,
    but on the EPHEMERAL (cloud) path it leaves `company_in_corpus` False —
    corpus retrieval was skipped in favour of the filing just fetched. Keying
    the escalation on that flag therefore sent the authoritative filing to the
    web anyway, which is how a stock-forecast page ($26,161M) beat the filed
    figure ($34,229M) for 3M's FY2022 revenue.

    Only genuinely empty retrieval may escalate; "chunks that don't answer" is
    the critic's insufficient-draft path, not a guess made up front.
    """
    agent = _build_agent()
    agent._log = lambda s, m: None
    searched: list = []
    agent._web = type("W", (), {"search": lambda self, q: searched.append(q) or []})()

    base = {"question": "What was 3M's revenue in FY2022?",
            "sub_queries": ["3M revenue FY2022"], "query_routes": ["narrative"],
            "errors": []}

    # Ephemeral fetch succeeded: chunks present, company_in_corpus False.
    agent.web_search_node({**base, "company_in_corpus": False,
                           "retrieved_chunks": [{"text": "Net sales $34,229M",
                                                 "source": "[MMM 10-K 2022]"}],
                           "fetch_status": {"decision": "fetch", "ephemeral": True}})
    assert searched == [], f"fetched filing was diluted with web: {searched}"

    # Nothing retrieved and nothing fetched (e.g. a non-US filer) → web is the
    # only source left, so it must still escalate.
    searched.clear()
    agent.web_search_node({**base, "company_in_corpus": False,
                           "retrieved_chunks": [], "fetch_status": {}})
    assert searched, "an empty corpus with no fetch must still reach the web"


def test_qwen_structured_output_uses_json_schema():
    """Qwen 3.x on Groq returns tool-call scalars as STRINGS, so Groq rejects
    the call ("`/answerable`: expected boolean, but got string") under
    LangChain's default `method="function_calling"`.

    That broke every structured extraction on the pinned router model: the XBRL
    and calculator lanes both failed, the agent fell back to web search, and it
    answered 3M's FY2022 revenue as $26,161M off a stock-tip page instead of
    the filed $34,229M — a wrong number shown to the user with no error. The
    request must therefore go out with `method="json_schema"`.
    """
    from finagent.graph.state import XBRLQuery
    from finagent.llm import RotatingChatModel

    def bound_kwargs(model: str) -> dict:
        rot = RotatingChatModel(provider="groq", chat_model=model, keys=["k"])
        return rot.with_structured_output(XBRLQuery).kw

    assert bound_kwargs("qwen/qwen3.6-27b").get("method") == "json_schema"

    # Non-Qwen models keep LangChain's default — this is a Qwen quirk, not a
    # blanket preference, and gpt-oss's function calling works.
    assert "method" not in bound_kwargs("openai/gpt-oss-120b")
    # …and an explicit choice from the caller still wins.
    rot = RotatingChatModel(provider="groq", chat_model="qwen/qwen3.6-27b", keys=["k"])
    assert rot.with_structured_output(
        XBRLQuery, method="function_calling").kw["method"] == "function_calling"


def test_router_is_pinned_to_the_free_groq_pool():
    """The `router` role (XBRL/calc/formula/EDGAR/corpus-gate extraction) is
    one-shot structured output — good enough on Groq's free Qwen, and pinning
    it there keeps the small Gemini per-day budget for the reasoning roles.

    The pin must carry the PROVIDER and the KEY with it: sending a user's
    Gemini key to Groq authenticates nothing."""
    from finagent.runtime import ROUTER_PIN, RuntimeContext

    ctx = RuntimeContext(provider="gemini", api_key="user-gemini-key")
    assert ctx.provider_for("router") == ROUTER_PIN[0] == "groq"
    assert ctx.model_for("router") == ROUTER_PIN[1]
    assert ctx.key_for("router") is None          # never the Gemini key
    assert ctx.model_for("router") != ctx.model_for("synth")

    # A Groq request keeps its own key on the pinned role.
    groq = RuntimeContext(provider="groq", api_key="user-groq-key")
    assert groq.key_for("router") == "user-groq-key"

    # With no Groq key anywhere the pin must fall back rather than 401.
    import finagent.llm as _llm
    real = _llm.collect_provider_keys
    _llm.collect_provider_keys = lambda p: []
    try:
        assert ctx.provider_for("router") == "gemini"
        assert ctx.model_for("router") == ctx.model_for("synth")
    finally:
        _llm.collect_provider_keys = real


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


def test_mostly_supported_answer_is_not_hard_refused():
    """A mostly-supported draft ships as written; only one the critic could
    barely support (< REFUSE_BELOW_SUPPORT) hard-refuses, and only once the
    retries are spent."""
    agent = _build_agent()
    spent = {"critic_iterations": 99, "needs_retry": False}
    assert agent._critic_router({"grading_score": 0.9, **spent}) == "end"
    assert agent._critic_router({"grading_score": 0.2, **spent}) == "refuse"
    # Same bad score with retries left never refuses — it retries first.
    assert agent._critic_router(
        {"grading_score": 0.2, "critic_iterations": 0, "needs_retry": False}) == "end"


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
        "web_results": [], "market_data": [],
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
    assert sig == {"web_fallback_pending": True, "web_fallback_used": True,
                   # The escalation spends the single recovery budget like any
                   # other retry; it used to be free and invisible.
                   "critic_iterations": 1}
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
    exc = m._exhausted(Exception("429 rate limit"), [])
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


def test_rate_limit_reset_is_read_off_the_429():
    """A 429 says exactly when it clears; we surface that instead of guessing.

    Groq puts the wait in the body ("try again in 2m59.56s", resolved against
    whichever bucket tripped) and in `retry-after` / `x-ratelimit-reset-*`
    headers (one per bucket, so they don't say which one failed). The body must
    win, the value must survive the wrapping LangGraph/executor does to it, and
    the pool's own reset is the EARLIEST of its keys'.
    """
    import time

    import pytest

    import finagent.llm as L
    from finagent.api.main import _classify_provider_error

    assert L.parse_duration("7.66s") == 7.66
    assert L.parse_duration("500ms") == 0.5
    assert L.parse_duration("1h2m3s") == 3723.0
    assert L.parse_duration("30") == 30.0          # bare HTTP Retry-After
    assert L.parse_duration("soon") is None
    assert (L.format_wait(7.66), L.format_wait(179.6), L.format_wait(7500)) \
        == ("8s", "3m 0s", "2h 5m")

    class Err(Exception):
        response = type("R", (), {"headers": {"retry-after": "60",
                                              "x-ratelimit-reset-tokens": "7.66s"}})()

    body = Err("Rate limit reached ... Please try again in 2m59.56s")
    assert abs(L.retry_after_seconds(body) - 179.56) < 1e-9   # body beats headers
    assert L.retry_after_seconds(Err("429 too many requests")) == 60.0  # fallback
    assert L.retry_after_seconds(ValueError("unrelated")) is None
    try:                                          # wrapped, as it arrives in prod
        try:
            raise body
        except Exception as inner:
            raise RuntimeError("node failed") from inner
    except Exception as outer:
        assert abs(L.retry_after_seconds(outer) - 179.56) < 1e-9

    m = L.RotatingChatModel(provider="groq", chat_model="llama-3.1-8b-instant",
                            keys=["k1", "k2"], chat_kwargs={"temperature": 0})
    try:
        L._EXHAUSTED_UNTIL.pop("groq", None)
        L._RESET_AT.pop("groq", None)
        # Two keys, two resets → the pool is usable again when the first one is.
        assert m._exhausted(Exception("429"), [3600.0, 12.0]).retry_after == 12.0
        # A daily bucket resets hours out. The user is told the truth; the
        # process-wide fail-fast latch is clamped so keys in another org (whose
        # day rolls over separately) aren't locked out behind it.
        m._exhausted(Exception("429"), [7200.0])
        assert L._EXHAUSTED_UNTIL["groq"] - time.time() <= L._EXHAUST_COOLDOWN_MAX_S
        assert L._RESET_AT["groq"] - time.time() > 7000
        with pytest.raises(L.AllKeysExhaustedError) as caught:
            m._check_cooldown()
        assert caught.value.retry_after > 7000 and "2h" in str(caught.value)
    finally:
        L._EXHAUSTED_UNTIL.pop("groq", None)
        L._RESET_AT.pop("groq", None)

    # …and it reaches the browser on the SSE error event.
    ev = _classify_provider_error(body, None)
    assert ev["code"] == "rate_limit" and ev["retry_after"] == 179.6
    assert "Retry in 3m 0s." in ev["message"]
    # Provider said nothing → no timer, and the copy stays vague as before.
    vague = _classify_provider_error(Exception("429 rate limit"), None)
    assert "retry_after" not in vague and "wait a minute" in vague["message"]


def test_oversized_prompt_is_not_a_rate_limit():
    """Groq rejects an over-sized prompt with HTTP 413 whose body says
    `"code": "rate_limit_exceeded"`. Matching that as a rate limit burned all 12
    keys in 0.6s and then told the user to retry in 14s — a retry that could
    never work, because every key has the same per-request ceiling. It is the
    prompt that must shrink, so this must NOT rotate and must NOT promise a wait.
    """
    import pytest

    import finagent.llm as L
    from finagent.api.main import _classify_provider_error

    body = ('{"error":{"message":"Request too large for model `openai/gpt-oss-120b` '
            'on tokens per minute (TPM): Limit 8000, Requested 8482, please reduce '
            'your message size and try again","type":"tokens",'
            '"code":"rate_limit_exceeded"}}')
    too_big = Exception(body)
    assert L.is_request_too_large(too_big)
    # Still matches the rate-limit hints — that is exactly the trap, so the
    # specific check has to be consulted first everywhere it matters.
    assert L.is_rate_limit_error(too_big)

    m = L.RotatingChatModel(provider="groq", chat_model="x",
                            keys=["a", "b", "c"], chat_kwargs={"temperature": 0})
    assert m._recover(too_big) is False, "must not rotate the pool"
    calls = []
    with pytest.raises(Exception) as caught:
        m._retry(lambda llm: calls.append(1) or (_ for _ in ()).throw(too_big))
    assert len(calls) == 1, f"tried {len(calls)} keys; must stop at the first"
    assert not isinstance(caught.value, L.AllKeysExhaustedError)

    ev = _classify_provider_error(too_big, None)
    assert ev["code"] == "too_large" and "retry_after" not in ev
    assert "waiting will not help" in ev["message"]

    # A genuine 429 still rotates.
    assert m._recover(Exception("429 rate limit reached")) is True


def test_per_minute_limit_is_waited_out_not_failed():
    """A TPM bucket refills in seconds. Rotating the whole pool in 0.6s and then
    giving up threw away a recoverable failure — sleep the reported reset and
    try the pool again. A DAILY quota is not waited on: it is hours away."""
    import finagent.llm as L

    m = L.RotatingChatModel(provider="groq", chat_model="x", keys=["a", "b"],
                            chat_kwargs={"temperature": 0})
    slept: list[float] = []

    # Swap the MODULE REFERENCE in llm's namespace, not `time.sleep` itself —
    # mutating the real module would leak a broken sleep into every other test.
    class _NoSleep:
        def __init__(self, real):
            self._real = real

        def sleep(self, s):
            slept.append(s)

        def __getattr__(self, name):
            return getattr(self._real, name)

    real_time = L.time
    L.time = _NoSleep(real_time)
    try:
        m._last_daily = False
        assert m._wait_out([3.0, 9.0], Exception("429")) is True
        assert slept and 3.0 <= slept[0] <= 4.0, slept   # earliest key, small pad

        # Longer than a user will wait → fail instead of holding the socket.
        slept.clear()
        assert m._wait_out([L.MAX_INLINE_WAIT_S + 1], Exception("429")) is False
        assert not slept

        # Daily quota → no wait at any duration.
        m._last_daily = True
        assert m._wait_out([5.0], Exception("429")) is False
        assert not slept
    finally:
        L.time = real_time


def test_free_tier_evidence_is_bounded():
    """The shared Groq pool rejects >8000-token requests outright, and traces
    ran 7581-7815 against that ceiling — a follow-up question's history was
    enough to break it. Evidence is the dominant term, so it gets a budget.
    A user on their own key keeps the full context."""
    from finagent.runtime import RuntimeContext

    shared = RuntimeContext(provider="groq")
    assert shared.evidence_budget() == RuntimeContext.FREE_TIER_EVIDENCE_CHARS
    assert RuntimeContext(provider="groq", api_key="sk-own").evidence_budget() is None
    # The wall is Groq's, not everyone's — the default (Gemini) is unbounded.
    assert RuntimeContext().evidence_budget() is None
    assert RuntimeContext(provider="gemini", api_key="k").evidence_budget() is None

    agent = _build_agent()
    agent._log = lambda s, m: None
    budget = shared.evidence_budget()
    chunks = [{"text": "x" * 4000, "sub_query": f"q{i}"} for i in range(8)]
    with __import__("finagent.runtime", fromlist=["use_context"]).use_context(shared):
        kept = agent._fit_budget({"errors": []}, chunks)
    assert sum(len(c["text"]) for c in kept) <= budget
    assert len(kept) == budget // 4000, f"dropped from the bottom, got {len(kept)}"

    # One passage larger than the whole budget is truncated, never dropped —
    # answering with no evidence at all is worse.
    with __import__("finagent.runtime", fromlist=["use_context"]).use_context(shared):
        solo = agent._fit_budget({"errors": []},
                                 [{"text": "y" * 50_000, "sub_query": "q"}])
    assert len(solo) == 1 and len(solo[0]["text"]) == budget


def test_planner_can_run_on_its_own_provider():
    """A free planner beside a paid answer model. The answer picker used to own
    the provider outright, so choosing Gemini silently moved the planner onto
    the paid key and hid the free models. Provider AND key must move together
    per role — sending Gemini's key to Groq authenticates nothing."""
    from finagent.runtime import RuntimeContext

    ctx = RuntimeContext(provider="gemini", api_key="gem-key",
                         planner_provider="groq", planner_api_key=None)
    assert ctx.provider_for("planner") == "groq"
    assert ctx.key_for("planner") is None          # → the server's Groq pool
    assert ctx.model_for("planner") == "openai/gpt-oss-120b"
    # Everything else follows the answer provider — except `router`, which is
    # pinned to the free Groq pool (see test_router_is_pinned_to_the_free_groq_pool).
    for role in ("synth", "critic"):
        assert ctx.provider_for(role) == "gemini", role
        assert ctx.key_for(role) == "gem-key", role

    # Unset → everything follows the one provider, exactly as before.
    plain = RuntimeContext(provider="gemini", api_key="k")
    assert plain.provider_for("planner") == "gemini"
    assert plain.key_for("planner") == "k"


def test_client_chat_history_is_the_only_memory():
    """The server is stateless: memory is exactly the turns the client replayed,
    capped at the last 6 and truncated. A request with no history yields none —
    never a crash, and never turns invented from a server-side store."""
    from finagent.api import main as M
    from finagent.api.models import ChatTurn, QueryRequest

    turns = [ChatTurn(role="user", content="News on Fluence Energy?"),
             ChatTurn(role="assistant", content="FLNC surged on the Siemens/Nvidia tie-up.")]
    req = QueryRequest(question="Any confirmation of the tie-up?", chat_history=turns)
    assert M._agent_history(req) == [
        {"role": "user", "content": "News on Fluence Energy?"},
        {"role": "assistant", "content": "FLNC surged on the Siemens/Nvidia tie-up."},
    ]

    # No client history → empty memory.
    assert M._agent_history(QueryRequest(question="q")) == []

    # Only the last 6 turns survive, and each is truncated to 1200 chars.
    long_turns = [ChatTurn(role="user", content="x" * 2000)] * 8
    got = M._agent_history(QueryRequest(question="q", chat_history=long_turns))
    assert len(got) == 6
    assert all(len(t["content"]) == 1200 for t in got)


def test_session_id_reaches_the_tracer(monkeypatch):
    """Regression: `session_id` is only useful if it survives the whole way to
    run_agentic, which tags the Langfuse trace with it. The previous field
    (chat_id) was wired here but never sent by the SPA, so session grouping was
    silently dead — assert the value actually arrives."""
    import json

    from starlette.testclient import TestClient

    from finagent.api import main as M
    from finagent.api import rag_service

    seen: dict = {}

    def fake_run_agentic(question, company_filter=None, chat_history=None,
                         provider="groq", synth_model=None, api_key=None, **kwargs):
        seen.update(kwargs)
        return {"answer": "ok", "chunks": [], "charts": [], "metadata": {}}

    monkeypatch.setattr(rag_service, "run_agentic", fake_run_agentic)

    with TestClient(M.app) as client:
        resp = client.post("/api/query",
                           json={"question": "Revenue?", "session_id": "thread-abc"})
        assert resp.status_code == 200
        resp.text          # drain the stream so the run completes

    assert seen.get("session_id") == "thread-abc"


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
    """The eval's metrics layer counts an explicit abstention as a refusal."""
    from finagent.evaluation.financebench.parallel import _is_refusal

    assert _is_refusal("**Insufficient evidence to answer.** The filings ...")
    assert _is_refusal("I don't have enough information to answer this from ...")
    assert not _is_refusal("Revenue was $394.3 billion [1].")


def test_count_rows_tolerates_partial_shards(tmp_path):
    """The orchestrator's progress poll counts rows across shard files and
    skips one that is mid-rewrite (unparseable) or missing."""
    from finagent.evaluation.financebench.parallel import _count_rows

    good = tmp_path / "shard0_outputs.json"
    good.write_text('[{"a": 1}, {"a": 2}]')
    torn = tmp_path / "shard1_outputs.json"
    torn.write_text('[{"a": 1}, {"a"')          # mid-write
    missing = tmp_path / "shard2_outputs.json"  # never created
    assert _count_rows([good, torn, missing]) == 2


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

    def fake_input(ticker, concept, period, quarterly=False, fp=None):
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


def test_xbrl_dual_scale_money():
    """XBRL money strings restate the filed value in millions/billions so a
    judge (or reader) can match '$32,780 million' against the evidence."""
    from finagent.tools.xbrl import dual_scale_money
    assert dual_scale_money(32_780_000_000) == \
        "$32,780,000,000 ($32,780 million; $32.78 billion)"
    assert dual_scale_money(4_625_000_000).startswith("$4,625,000,000 ($4,625 million")
    assert dual_scale_money(1_577_000_000).endswith("($1,577 million; $1.58 billion)")
    assert dual_scale_money(950_000) == "$950,000"          # small: raw only
    assert "million" in dual_scale_money(-2_000_000_000)    # sign-safe


def test_calc_result_formats_derivation():
    """A derived-metric chunk states the result AND its formula + inputs, so
    the figure is verifiable from the chunk alone."""
    from finagent.graph.agent import AgenticRAGv4
    text = AgenticRAGv4._format_calc_result({
        "ticker": "ATVI", "metric": "fixed_asset_turnover", "value_str": "24.26",
        "formula": "revenue / avg(ppe_net)",
        "inputs": [
            {"concept": "revenue", "period": "FY2019", "value_str": "$6,489M"},
            {"concept": "ppe_net", "period": "FY2018", "value_str": "$282M"},
        ],
        "source": "computed from SEC XBRL",
    })
    assert "ATVI fixed asset turnover = 24.26" in text
    assert "Derivation: revenue / avg(ppe_net)" in text
    assert "revenue (FY2019) = $6,489M" in text and "ppe_net (FY2018)" in text


def test_cap_pool_keeps_every_sub_query():
    """The global pool cap must not let one sub-query's chunks crowd out the
    others entirely — each sub-query keeps its best chunk."""
    agent = _build_agent()
    agent.retrieve_cap = 4
    agent._log = lambda s, m: None

    class FakeReranker:
        def predict(self, pairs):
            # Chunks 0-4 (sub A) score highest; 5-6 (sub B) would be crowded out.
            return [9, 8, 7, 6, 5, 3, 2][: len(pairs)]

    import finagent.retrieval.reranker as R
    chunks = ([{"text": f"a{i}", "sub_query": "A"} for i in range(5)]
              + [{"text": f"b{i}", "sub_query": "B"} for i in range(2)])
    orig = R._get_shared_reranker
    R._get_shared_reranker = lambda model: FakeReranker()
    try:
        kept = agent._cap_pool({"question": "q"}, chunks)
    finally:
        R._get_shared_reranker = orig
    assert len(kept) == 4
    subs = {c["sub_query"] for c in kept}
    assert subs == {"A", "B"}, f"sub-query B was crowded out: {kept}"
    assert kept[0]["text"] == "a0"          # global best still first


def test_cap_pool_rescores_the_merged_pool_against_the_question():
    """The cap ranks by re-scoring every chunk against the ORIGINAL question.

    §11 replaced this with the score each chunk carried from its own sub-query;
    §13 re-ran the A/B after chunks gained context headers and it reversed
    (58 vs 61 of 99 — RETRIEVAL_EXPERIMENTS.md §13). Skipping the re-score is
    therefore a regression, not an optimisation.
    """
    agent = _build_agent()
    agent.retrieve_cap = 2
    agent._log = lambda s, m: None

    import finagent.retrieval.reranker as R
    seen = {}
    chunks = [{"text": "low", "sub_query": "A"},
              {"text": "high", "sub_query": "A"},
              {"text": "mid", "sub_query": "A"}]
    scores = {"low": 0.1, "high": 9.0, "mid": 5.0}

    def fake(model):
        return type("F", (), {"predict": staticmethod(
            lambda pairs: [seen.setdefault("q", pairs[0][0]) and 0 or
                           scores[t] for _, t in pairs])})()

    orig = R._get_shared_reranker
    R._get_shared_reranker = fake
    try:
        kept = agent._cap_pool({"question": "the question"}, chunks)
    finally:
        R._get_shared_reranker = orig
    assert seen["q"] == "the question", "must score against the question, not a sub-query"
    assert [c["text"] for c in kept] == ["high", "mid"]


def test_derived_metric_query_gains_the_line_items_that_carry_the_numbers():
    """A filing never says "quick ratio" — it says "Total current assets".
    Expansion is what closes that gap (§13); without it those questions sit at
    ~0.00 coverage no matter how deep the pool goes."""
    from finagent.retrieval.expansion import expand_query

    out = expand_query("Verizon quick ratio FY2022")
    assert out.startswith("Verizon quick ratio FY2022"), "original query must survive"
    assert "total current liabilities" in out
    assert "consolidated balance sheets" in out
    # A query naming no derived metric must be left exactly alone, or every
    # ordinary lookup pays for padding it does not need.
    assert expand_query("3M total revenue 2022") == "3M total revenue 2022"


def test_question_query_is_gated_on_planner_routing():
    """Retrieving on the original question must not override the planner: when
    every sub-query was routed to yfinance/web/EDGAR, the filings are not the
    source and the question must not drag them back in (§13)."""
    agent = _build_agent()
    agent._log = lambda s, m: None
    searched = []
    agent._get_hybrids = lambda: [type("H", (), {
        "search": staticmethod(lambda q, top_k=None: searched.append(q) or []),
        "infer_filter": staticmethod(lambda q: None)})()]

    agent.hybrid_retrieve_node({
        "question": "What is AAPL trading at today?",
        "sub_queries": ["AAPL current share price"],
        "query_routes": ["market"], "errors": []})
    assert searched == [], f"routed away from filings but still searched: {searched}"

    searched.clear()
    agent.hybrid_retrieve_node({
        "question": "What drove AMD revenue change in FY22?",
        "sub_queries": ["AMD segment revenue FY22"],
        "query_routes": ["numeric"], "errors": []})
    assert "What drove AMD revenue change in FY22?" in searched, \
        "the question itself must be retrieved on when filings are in play"


def test_retrieval_runs_on_the_rewrite_not_the_subqueries():
    """§14: the filings are searched with ONE query in the filing's vocabulary
    plus the raw question — 72/99 hit@8 against 67 for retrieving on the
    decomposition. The sub-queries still route the tool lanes and still gate
    this node; they just stopped being search keys."""
    agent = _build_agent()
    agent._log = lambda s, m: None
    searched, budgets = [], []
    agent._get_hybrids = lambda: [type("H", (), {
        "search": staticmethod(lambda q, top_k=None:
                               (searched.append(q), budgets.append(top_k), [])[2]),
        "infer_filter": staticmethod(lambda q: None)})()]
    base = {"question": "What drove AMD revenue change in FY22?",
            "sub_queries": ["AMD segment revenue FY22", "AMD FY22 revenue drivers"],
            "query_routes": ["numeric", "narrative"], "errors": []}
    rq = "AMD 2022 2021 segment information net revenue Data Center Client Gaming"

    agent.hybrid_retrieve_node({**base, "retrieval_query": rq})
    assert searched == [rq, base["question"]], \
        f"must search the rewrite + the question only, got {searched}"
    # The lone §14 query gets the FULL cap, not `final_top_k`. At 5 it would
    # supply 5 + QUESTION_SLOTS = 7 into an 8-passage cap, leaving it unfilled
    # and discarding its own 6th-8th hits — `final_top_k` is the budget for one
    # of SEVERAL competing sub-queries, which this is not.
    # This request has a `narrative` route, so the question gets the wider
    # budget (§14c): keyword queries are a poor dense match for MD&A prose, and
    # at 2 slots the question's evidence never survives into the cap.
    assert budgets == [agent.retrieve_cap, agent.NARRATIVE_QUESTION_SLOTS], budgets

    # A purely numeric request keeps the narrow §13 budget — the sweep showed
    # widening it buys nothing there, and it costs hit@5.
    searched.clear(); budgets.clear()
    agent.hybrid_retrieve_node({**base, "query_routes": ["numeric", "numeric"],
                                "retrieval_query": rq})
    assert budgets == [agent.retrieve_cap, agent.QUESTION_SLOTS], budgets

    # No rewrite (planner failed, or a rung below v3) → the old sub-query path,
    # so a failed rewrite costs nothing rather than losing the turn.
    searched.clear()
    agent.hybrid_retrieve_node({**base, "retrieval_query": ""})
    assert searched == base["sub_queries"] + [base["question"]], searched

    # Routed entirely away from the filings: the rewrite must NOT drag them
    # back in. `_retrieval_query` also declines to spend a call in this case.
    searched.clear()
    agent.hybrid_retrieve_node({
        "question": "What is AAPL trading at today?",
        "sub_queries": ["AAPL current share price"], "query_routes": ["market"],
        "retrieval_query": "Apple 2024 consolidated statements of operations",
        "errors": []})
    assert searched == [], f"routed to market but still searched: {searched}"

    # A retry re-searches the critic's own hint queries, so the stale §14 query
    # must be cleared — otherwise the retry repeats the search that just failed.
    from finagent.graph.corrective import AgenticRAGv2
    assert '"retrieval_query"' in inspect.getsource(AgenticRAGv2.critic_node), \
        "critic_node must clear retrieval_query"


def test_retrieval_query_rejects_an_answer_masquerading_as_a_query():
    """The model sometimes ANSWERS instead of writing a query (a FinanceBench
    run produced "Approximately 1.33.") or returns nothing. Either would send
    garbage to the retriever, so both fall back to the sub-queries."""
    from finagent.graph import full as F

    agent = _build_agent()
    agent._log = lambda s, m: None
    replies = iter(["Approximately 1.33.", "", "  \n ",
                    "AMD 2022 consolidated balance sheets total current assets"])
    agent._get_llm = lambda role: type("L", (), {
        "invoke": staticmethod(lambda msgs: type("R", (), {"content": next(replies)})())})()

    for _ in range(3):
        assert agent._retrieval_query({"errors": []}, "q", ["numeric"]) == ""
    assert agent._retrieval_query({"errors": []}, "q", ["numeric"]).startswith("AMD 2022")

    # Every route away from the filings → no call at all (the iterator would
    # raise StopIteration if one were made).
    assert agent._retrieval_query({"errors": []}, "q", ["market", "external"]) == ""

    assert F._first_line('  "Nike 2023 revenue"  \nignored') == "Nike 2023 revenue"
    assert F._first_line("Query: Nike 2023 revenue") == "Nike 2023 revenue"
    assert F._first_line("") == ""


def test_element_captions_name_a_table_that_lost_its_heading():
    """`chunk_by_title` cuts a statement's heading away from its numbers, so the
    chunk holding "Total assets" carries no company/year/statement name and every
    company's balance sheet embeds to nearly the same vector. The caption walker
    is what puts that identity back — see RETRIEVAL_EXPERIMENTS.md §12.
    """
    from finagent.ingestion.ingest import _element_captions, CorpusIngester

    class E:
        def __init__(self, text, kind="Text"):
            self.text = text
            self.__class__ = type(kind, (E,), {}) if kind != "Text" else E

    def el(text, kind="Text"):
        e = E.__new__(type(kind, (), {"text": text}))
        return e

    els = [el("49"), el("Table of Contents"), el("3M Company and Subsidiaries"),
           el("Consolidated Balance Sheet"), el("At December 31"),
           el("Total assets | 46,455", "Table")]
    caps = _element_captions(els)
    # Page number and TOC boilerplate must not become a caption.
    assert "49" not in caps[-1] and "Table of Contents" not in caps[-1]
    assert "Consolidated Balance Sheet" in caps[-1]
    assert "3M Company and Subsidiaries" in caps[-1]

    # Prose resets the caption: a heading must not leak onto an unrelated table.
    els2 = els[:5] + [el("The Company operates in four business segments. " * 4),
                      el("Some other table | 1", "Table")]
    assert "Consolidated Balance Sheet" not in _element_captions(els2)[-1]

    head = CorpusIngester._context_header(
        {"company": "3M", "year": "2022", "doc_type": "10-K"},
        "Consolidated Balance Sheet")
    assert head == "3M 2022 10-K · Consolidated Balance Sheet"


def test_behaviour_metrics_aggregate_latency_and_tokens(tmp_path):
    """summarize_outputs reports latency percentiles + token totals when the
    runner recorded them, and omits them cleanly for older outputs."""
    import json
    from finagent.evaluation.financebench.parallel import summarize_outputs

    rows = [{"question": f"q{i}", "answer": f"${i} million [1].", "qtype": "numeric",
             "gold": f"${i}.00", "confidence": 0.9, "answer_status": "answered",
             "latency_s": float(i + 1), "input_tokens": 1000, "output_tokens": 100,
             "error": None} for i in range(10)]
    p = tmp_path / "out.json"; p.write_text(json.dumps(rows))
    b = summarize_outputs(p)
    assert b["latency"]["mean_s"] == 5.5
    assert b["latency"]["p50_s"] == 6.0 and b["latency"]["p95_s"] == 10.0
    assert b["tokens"] == {"mean_per_question": 1100, "total": 11000}

    for r in rows:                          # old outputs: fields absent
        for k in ("latency_s", "input_tokens", "output_tokens"): r.pop(k)
    p.write_text(json.dumps(rows))
    b2 = summarize_outputs(p)
    assert b2["latency"] is None and b2["tokens"] is None


def test_averaged_ratio_targets_latest_named_year():
    """Regression (caught via Langfuse trace): 'FY2021 inventory turnover using
    average inventory between FY2020 and FY2021' must compute FY2021, even when
    the LLM extraction returns only the earlier year as the period."""
    agent = _build_agent()
    calls = {}

    class FakeCalc:
        def ratio(self, ticker, metric, period):
            calls.update(ticker=ticker, metric=metric, period=period)
            return {"ok": True, "metric": metric, "period": period, "value": 3.46}
        def run(self, **kw):
            raise AssertionError("should have taken the averaged-ratio shortcut")

    agent._calc = FakeCalc()

    class Q:
        metric, ticker, concept = "inventory turnover", "NKE", ""
        periods = ["FY2020"]                       # the bad extraction

    sub_q = ("When primarily referencing the income statement, what is the FY2021 "
             "inventory turnover ratio for Nike? Defined as FY2021 COGS / "
             "average inventory between FY2020 and FY2021.")
    res = agent._run_calc({}, sub_q, Q())
    assert res["ok"] and calls["period"] == "FY2021" and calls["metric"] == "inventory_turnover"


def test_short_company_tokens_resolve():
    """A distinctive SHORT name must resolve to its company.

    The single-token alias rule used to require 5+ characters, which excluded
    every short name in the corpus: "What was Ulta's revenue?" matched nothing,
    the company filter was dropped, and the question competed against all 72
    filings' near-identical accounting language. (MGM had been hand-written into
    COMPANY_ALIASES to paper over exactly this.) The guards that keep it safe
    are single-ownership and the generic-word list, not the length.
    """
    from finagent.retrieval.filters import build_company_vocab, infer_filter

    metas = [{"company": c, "ticker": t, "year": "2022"} for c, t in (
        ("Ulta Beauty", "ULTA"), ("AES Corporation", "AES"),
        ("MGM Resorts", "MGM"), ("CVS Health", "CVS"),
        ("American Express", "AXP"), ("American Water Works", "AWK"),
        ("General Mills", "GIS"), ("3M", "MMM"))]
    vocab, years = build_company_vocab(metas)

    def co(q):
        return (infer_filter(q, vocab, years) or {}).get("companies")

    assert co("What was Ulta's revenue in FY2023?") == ["Ulta Beauty"]
    assert co("AES total debt 2022") == ["AES Corporation"]
    assert co("MGM occupancy 2022") == ["MGM Resorts"]
    assert co("CVS pharmacy revenue") == ["CVS Health"]
    # Longest-first, so the specific name beats the generic word it contains.
    assert co("American Water Works capex") == ["American Water Works"]
    # …and generic words still must NOT name a company on their own.
    assert co("a general discussion of water works") is None
    assert co("the company issued new shares") is None


def test_refusal_is_not_a_retrieval_query():
    """A LONG refusal must be rejected as a retrieval query.

    `MIN_RETRIEVAL_QUERY_WORDS` only catches a rewrite that is too SHORT (the
    model answering "Approximately 1.33."). A refusal is the other failure
    shape and clears the floor easily — one FinanceBench generation returned
    "I'm not able to retrieve the specific figures from Walmart's filings…",
    which was then searched verbatim. Both must degrade to the sub-queries.
    """
    from finagent.graph.full import MIN_RETRIEVAL_QUERY_WORDS, _NOT_A_QUERY

    refusal = ("I'm not able to retrieve the specific figures from Walmart's "
               "filings needed to compute the EBITDA for FY2021")
    assert len(refusal.split()) >= MIN_RETRIEVAL_QUERY_WORDS, "the floor lets it through"
    assert _NOT_A_QUERY.search(refusal), "so this must catch it"

    for bad in ("I cannot determine the value from the provided filings",
                "Sorry, the document does not provide that figure",
                "As an AI, I do not have access to that filing"):
        assert _NOT_A_QUERY.search(bad), bad

    # A real keyword query must NOT trip it — including captions containing
    # words like "income" that appear in no refusal pattern.
    for good in ("AMD 2022 consolidated balance sheets cash and cash equivalents "
                 "inventories total current liabilities",
                 "Boeing 2022 2021 note income taxes provision for income taxes",
                 "Verizon 2022 Item 7 Liquidity and Capital Resources"):
        assert not _NOT_A_QUERY.search(good), good


def test_daily_quota_benches_a_key_but_not_the_run():
    """A key whose DAILY budget is gone is retired, not treated as fatal.

    The six Gemini keys are six separate projects and their daily counters
    reset at different wall-clock times, so a build launched near the boundary
    meets a mix of fresh and stale keys. Aborting on the first stale one killed
    a run at 17 of 72 filings while five live keys could have finished it.

    Only an all-keys-dead pool is fatal, and a dead key is never retried — the
    request would be a guaranteed-wasted round trip against a budget already
    spent.
    """
    from finagent.vectorstore import EmbeddingQuotaExhausted, GeminiEmbeddings

    from finagent.vectorstore import _HttpError

    emb = GeminiEmbeddings.__new__(GeminiEmbeddings)   # no API key needed
    emb.model, emb.dim = "gemini-embedding-2", 8
    emb.keys, emb._dead = ["dead1", "dead2", "live"], {}

    calls: list[str] = []

    def fake_post(key, texts, task):
        calls.append(key)
        if key != "live":
            # The 429 body carries the quotaId; only that says PerDay vs
            # PerMinute, which is why `_HttpError` keeps the body.
            raise _HttpError(429, '{"error": {"details": [{"quotaId": '
                             '"EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier"}]}}')
        return [[1.0] + [0.0] * 7 for _ in texts]

    emb._post = fake_post
    try:
        got = emb._embed_batch(["x"], "RETRIEVAL_DOCUMENT", 0)
        assert len(got) == 1, "the live key must still serve the batch"
        assert "dead1" in emb._dead and "live" not in emb._dead, emb._dead
        assert calls.count("dead1") == 1, "a retired key is never retried"
        assert calls[-1] == "live"
        # Rotation skips the benched key, so a second batch costs no wasted
        # round trip at all.
        calls.clear()
        emb._embed_batch(["y"], "RETRIEVAL_DOCUMENT", 0)
        assert "dead1" not in calls, calls

        # Every key benched -> fatal, and it names the cache as the way back.
        import time as _t
        emb._dead = {k: _t.time() + 900 for k in emb.keys}
        try:
            emb._embed_batch(["x"], "RETRIEVAL_DOCUMENT", 0)
            raise AssertionError("an all-benched pool must raise")
        except EmbeddingQuotaExhausted as e:
            assert "embed_cache" in str(e)

        # ...but the bench EXPIRES. A key that rolled over mid-build has to come
        # back, otherwise a build started inside the reset stagger dies with a
        # full quota available (measured: twice, at 17 of 72 filings).
        emb._dead = {k: _t.time() - 1 for k in emb.keys}
        calls.clear()
        assert len(emb._embed_batch(["z"], "RETRIEVAL_DOCUMENT", 0)) == 1
        assert "live" in calls
    finally:
        pass
