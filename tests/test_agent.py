"""The agent's control flow: routing, the critic's single recovery, refusal,
and the numbered evidence. No network: model calls are stubbed."""

from __future__ import annotations

from finagent.agent import FinAgent
from finagent.agent import answer as A
from finagent.agent import external as E
from finagent.agent.state import ClaimVerdict, CriticReport


def _agent_with_critic(verdicts, remedy="redraft", missing=False):
    """An agent whose critic model returns a fixed report."""
    agent = FinAgent(collection="stub")
    report = CriticReport(remedy=remedy, draft_says_evidence_missing=missing, verdicts=[
        ClaimVerdict(claim=c, supported=ok, reason="") for c, ok in verdicts])
    model = type("M", (), {"with_structured_output": lambda self, schema: self,
                           "invoke": lambda self, msgs: report})()
    agent.llm = lambda role: model
    return agent


def test_graph_has_the_expected_steps():
    nodes = {n for n in FinAgent(collection="stub").graph.get_graph().nodes if not n.startswith("__")}
    assert nodes == {"planner", "fetch_filing", "retrieve", "xbrl", "calculator", "market_data",
                     "web_search", "edgar_search", "gather", "synthesize", "critic", "refuse"}


def test_only_a_narrative_part_sends_a_question_through_filing_search():
    assert FinAgent._after_plan({"query_routes": ["numeric", "narrative"]}) == "filings"
    assert FinAgent._after_plan({"query_routes": ["numeric", "market"]}) == "tools"
    assert FinAgent._after_plan({}) == "filings"


def test_supported_answer_ends_without_a_recovery():
    agent = _agent_with_critic([("a", True), ("b", True)])
    out = A.critic(agent, {"answer": "fine", "evidence": []})
    assert out["support_score"] == 1.0 and not out["needs_retry"]
    assert agent._after_critic(out) == "end"


def test_overstated_draft_is_rewritten_and_missing_evidence_is_searched_for():
    redraft = A.critic(_agent_with_critic([("a", True), ("b", False)], "redraft"),
                       {"answer": "x", "evidence": []})
    assert redraft["needs_retry"] and redraft["recoveries"] == 1 and redraft["retry_queries"] == []
    assert FinAgent(collection="s")._after_critic(redraft) == "synthesize"

    gather = A.critic(_agent_with_critic([("a", False)], "gather"), {"answer": "x", "evidence": []})
    assert gather["retry_queries"] == ["supporting evidence for: a"]
    assert FinAgent(collection="s")._after_critic(gather) == "retrieve"


def test_exactly_one_recovery_then_refuse():
    """After the one recovery, a draft under 50% supported is refused; a mostly
    supported one is shown."""
    agent = _agent_with_critic([("a", False), ("b", False), ("c", True)])
    out = A.critic(agent, {"answer": "x", "evidence": [], "recoveries": 1})
    assert not out["needs_retry"] and "recoveries" not in out          # budget spent
    assert agent._after_critic({**out, "recoveries": 1}) == "refuse"
    refusal = A.refuse(agent, {"unsupported_claims": ["a", "b"]})
    assert refusal["refused"] and refusal["answer"].startswith("I don't have enough information")

    mostly = _agent_with_critic([("a", True), ("b", True), ("c", False)])
    out = A.critic(mostly, {"answer": "x", "evidence": [], "recoveries": 1})
    assert mostly._after_critic({**out, "recoveries": 1}) == "end"


def test_a_draft_that_admits_it_cannot_answer_goes_to_the_web_once():
    agent = _agent_with_critic([("a", True)])
    draft = "The provided evidence does not contain the store count."
    out = A.critic(agent, {"answer": draft, "evidence": []})
    assert out["web_fallback_pending"] and out["recoveries"] == 1
    assert agent._after_critic(out) == "web_search"
    # Not again once the web lane has run.
    again = A.critic(agent, {"answer": draft, "evidence": [], "web_fallback_used": True})
    assert not again["web_fallback_pending"]


def test_a_numeric_draft_that_admits_a_missing_figure_searches_the_filings_once():
    # Wording the old pattern missed; the critic's flag catches it.
    draft = "The ratio cannot be settled: cash from operations is missing."
    agent = _agent_with_critic([("cash from operations is missing", True)], missing=True)
    state = {"answer": draft, "evidence": [], "query_routes": ["numeric"]}
    out = A.critic(agent, state)
    assert out["corpus_fallback_pending"] and out["recoveries"] == 1
    assert agent._after_critic(out) == "filings"
    assert FinAgent._after_retrieve({**state, **out}) == "synthesize"     # then rewrite
    # The one recovery is spent: the next draft is not sent anywhere.
    again = A.critic(agent, {**state, **out})
    assert not again["corpus_fallback_pending"] and agent._after_critic(again) == "end"


def test_a_failed_fact_check_ships_the_draft_and_tells_the_user():
    agent = FinAgent(collection="stub")

    def boom(role):
        raise RuntimeError("HTTP 503 service unavailable")

    agent.llm = boom
    state = {"answer": "x", "evidence": [], "log": [], "notices": []}
    out = A.critic(agent, state)
    assert out["support_score"] is None and agent._after_critic(out) == "end"
    assert state["notices"] == ["Fact-check was skipped: the model provider is busy."]


def test_a_retry_skips_the_tool_lanes():
    """On a recovery pass retrieve goes straight to the writer; re-running XBRL
    on the retry queries used to wipe the figures already found."""
    assert FinAgent._after_retrieve({}) == "xbrl"
    assert FinAgent._after_retrieve({"recoveries": 1}) == "synthesize"
    assert FinAgent._after_retrieve({"corpus_fallback_used": True}) == "synthesize"


def test_evidence_order_is_the_citation_order():
    """`[N]` in the answer must point at card N. One list feeds the writer, the
    critic and the API, most authoritative source first."""
    state = {
        "xbrl_facts": [{"entity": "Apple", "concept": "revenue", "fy": 2024,
                        "value_str": "$391,035 million", "tag": "Revenues", "source": "SEC"}],
        "calc_results": [{"ticker": "AAPL", "metric": "gross_margin", "value_str": "46.2%"}],
        "retrieved_chunks": [{"text": "MD&A text", "source": "[AAPL 10-K 2024]"}],
        "market_data": [{"ok": True, "tool": "get_quote", "data": {"symbol": "AAPL", "lastPrice": 1}},
                        {"ok": False, "tool": "get_news"}],
        "web_results": [{"title": "old", "url": "u1", "published_date": "2024-01-01"},
                        {"title": "new", "url": "u2", "published_date": "2026-01-01"}],
        "edgar_results": [{"query": "going concern", "companies": [{"company": "X", "url": "u"}]}],
    }
    items = A.build_evidence(state)
    assert [e["kind"] for e in items] == ["xbrl", "calc", "text", "market", "web", "web", "edgar"]
    assert [e["source_url"] for e in items if e["kind"] == "web"] == ["u2", "u1"]   # newest first
    assert items[0]["prompt"].startswith("XBRL FACT")


def test_tool_lanes_with_nothing_fall_back_to_filing_search_once():
    agent = FinAgent(collection="stub")
    state = {"query_routes": ["numeric"], "log": []}
    assert A.gather(agent, state)["corpus_fallback_pending"]
    assert agent._after_gather({"corpus_fallback_pending": True}) == "filings"
    assert not A.gather(agent, {**state, "corpus_fallback_used": True})["corpus_fallback_pending"]
    # A narrative question already searched the filings.
    assert not A.gather(agent, {"query_routes": ["narrative"]})["corpus_fallback_pending"]


def test_web_search_is_not_added_on_top_of_filing_evidence():
    """Web pages once outranked a fetched 10-K and gave a wrong revenue figure."""
    agent = FinAgent(collection="stub")
    searched = []
    agent._web = type("W", (), {"search": lambda self, q: searched.append(q) or []})()
    base = {"question": "What are 3M's segments?", "sub_queries": ["3M segments"],
            "query_routes": ["narrative"], "log": []}
    E.web_search(agent, {**base, "retrieved_chunks": [{"text": "segment table"}]})
    assert searched == []
    E.web_search(agent, {**base, "retrieved_chunks": []})       # nothing found: the web is the fallback
    assert searched == ["What are 3M's segments?"]


def test_a_company_named_in_the_sub_query_overrides_a_guessed_ticker():
    """The extractor once turned "Amcor" into AMR, another company's ticker."""
    from types import SimpleNamespace

    from finagent.agent import numeric
    from finagent.agent.prompts import XBRL_EXTRACT_PROMPT, XBRL_EXTRACT_SYSTEM
    from finagent.agent.state import XBRLQuery, XBRLQueryBatch

    known = {"amcor": {"cik": "1", "ticker": "AMCR", "match": "name_exact"},
             "amr": {"cik": "2", "ticker": "AMR", "match": "ticker"},
             "googl": {"cik": "3", "ticker": "GOOGL", "match": "ticker"}}
    resolver = SimpleNamespace(resolve=lambda q: known.get(q.lower(), {"cik": None, "match": "none"}))
    guesses = iter(["AMR", "GOOGL"])
    llm = SimpleNamespace(with_structured_output=lambda schema: SimpleNamespace(
        invoke=lambda msgs: XBRLQuery(answerable=True, ticker=next(guesses), concept="revenue")))
    agent = SimpleNamespace(xbrl=SimpleNamespace(resolver=resolver), llm=lambda role: llm,
                            log=lambda state, msg: state.setdefault("log", []).append(msg))

    def ticker(sub_query):
        return numeric._extract(agent, {"question": "q"}, [sub_query], XBRLQueryBatch, XBRLQuery,
                                XBRL_EXTRACT_SYSTEM, XBRL_EXTRACT_PROMPT)[0][1].ticker

    assert ticker("Amcor's net accounts receivable, FY2020") == "AMCR"
    # No registrant is called "Google": the model's ticker is all there is.
    assert ticker("Google revenue FY2023") == "GOOGL"


def test_a_packed_numeric_sub_query_looks_up_every_line_item_and_year():
    from finagent.agent import numeric
    from finagent.agent.state import XBRLQuery

    calls = []
    agent = FinAgent(collection="stub")
    query = XBRLQuery(answerable=True, ticker="MMM", concept="capital expenditures",
                      other_concepts=["depreciation and amortization"],
                      period="FY2021", other_periods=["FY2022"], fiscal_quarter="")
    agent.llm = lambda role: type("M", (), {"with_structured_output": lambda self, s: self,
                                            "invoke": lambda self, msgs: query})()
    agent._xbrl = type("X", (), {
        "resolver": type("R", (), {"resolve": lambda self, name: {}})(),
        "run": lambda self, **kw: calls.append((kw["concept"], kw["period"])) or {"ok": True}})()
    state = {"question": "q", "sub_queries": ["3M capex and D&A for FY2021 and FY2022"],
             "query_routes": ["numeric"], "log": []}
    assert len(numeric.xbrl(agent, state)["xbrl_facts"]) == 4
    assert calls == [("capital expenditures", "FY2021"), ("capital expenditures", "FY2022"),
                     ("depreciation and amortization", "FY2021"),
                     ("depreciation and amortization", "FY2022")]
