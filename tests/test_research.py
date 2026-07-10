"""Deep Research Mode tests — fast, no network, no API keys.

The orchestrator's runner and LLM calls are injected/stubbed, so these
exercise the real orchestration mechanics: planning, sequencing, retry,
failure isolation, caching, citation renumbering, and report assembly.
"""

from __future__ import annotations

import pytest

from finagent.research import DeepResearch, SPECIALISTS
from finagent.research.orchestrator import _CACHE, shift_citations
from finagent.research.specialists import (
    DEFAULT_SPECIALISTS, THESIS_TASK_ID, ResearchScope,
)


@pytest.fixture(autouse=True)
def _clear_task_cache():
    _CACHE.clear()
    yield
    _CACHE.clear()


def _fake_run_fn(question: str) -> dict:
    """Canned production-agent result: one chunk, one citation, 0.9 confidence."""
    return {
        "answer": f"Finding for: {question[:40]} [1].",
        "chunks": [{"id": 0, "text": "evidence", "company": "AAPL",
                    "citation": "[AAPL 10-K 2023, p. 5]", "kind": "text"}],
        "metadata": {"confidence": 0.9, "answer_status": "answered",
                     "input_tokens": 100, "output_tokens": 20},
    }


def _stubbed(run_fn=_fake_run_fn, specialists=("company", "financials"),
             **kwargs) -> DeepResearch:
    """Orchestrator with the scope/cross-check/report LLM calls stubbed out."""
    dr = DeepResearch(run_fn=run_fn, provider="groq", **kwargs)
    dr.scope = lambda q, h=None: ResearchScope(          # type: ignore[method-assign]
        company="Apple", ticker="AAPL", objective="assess AAPL",
        specialists=list(specialists))
    dr._cross_check = lambda digest: []                  # type: ignore[method-assign]
    dr._write_report = lambda *a, **k: ("# Report\n\nDone [1].", 50, 10)  # type: ignore[method-assign]
    return dr


# --------------------------------------------------------------------------- #
# Registry + primitives
# --------------------------------------------------------------------------- #

def test_registry_shape():
    """Every specialist is well-formed data and its template formats cleanly."""
    assert len(SPECIALISTS) == 14
    fields = {"company": "Apple", "question": "Should I invest?",
              "competitors": " (Microsoft)"}
    for s in SPECIALISTS.values():
        assert s.id and s.label and s.section and s.template
        rendered = s.template.format_map(fields)
        assert "{" not in rendered, f"{s.id}: unformatted placeholder"
    assert set(DEFAULT_SPECIALISTS) <= set(SPECIALISTS)


def test_shift_citations():
    assert shift_citations("Revenue rose [1] then [2,3].", 10) == \
        "Revenue rose [11] then [12,13]."
    # Full-width markers normalise to ASCII while shifting.
    assert shift_citations("Grew 30% 【4】", 2) == "Grew 30% [6]"
    assert shift_citations("no cites", 5) == "no cites"
    assert shift_citations("[1] stays", 0) == "[1] stays"


def test_plan_caps_agents_and_orders_custom_last():
    dr = DeepResearch(run_fn=_fake_run_fn, max_agents=4)
    scope = ResearchScope(company="Apple",
                          specialists=["custom", "company", "financials",
                                       "valuation", "risk", "macro"])
    tasks = dr.plan(scope, "How will tariffs affect Apple?")
    assert len(tasks) == 4
    assert tasks[-1].id == "custom"
    assert tasks[-1].question == "How will tariffs affect Apple?"
    assert "Apple" in tasks[0].question


def test_plan_falls_back_to_defaults_on_unknown_ids():
    dr = DeepResearch(run_fn=_fake_run_fn)
    tasks = dr.plan(ResearchScope(company="Apple", specialists=["bogus"]), "q")
    assert [t.id for t in tasks] == DEFAULT_SPECIALISTS[: dr.max_agents]


def test_scope_falls_back_when_llm_fails(monkeypatch):
    dr = DeepResearch(run_fn=_fake_run_fn)
    monkeypatch.setattr(DeepResearch, "_llm",
                        lambda self: (_ for _ in ()).throw(ValueError("boom")))
    scope = dr.scope("Should I invest in Apple?")
    assert scope.specialists == list(DEFAULT_SPECIALISTS)


# --------------------------------------------------------------------------- #
# Orchestration mechanics
# --------------------------------------------------------------------------- #

def test_run_merges_evidence_onto_global_numbering():
    """Two specialists × one chunk each → global chunk ids 0,1 and the second
    finding's [1] shifted to [2]; agent provenance stamped on each chunk."""
    dr = _stubbed()
    events: list[dict] = []
    out = dr.run("Should I invest in Apple? (merge)", on_event=events.append)

    chunks = out["chunks"]
    assert [c["id"] for c in chunks] == [0, 1]
    assert chunks[0]["agent"] == "Company Research"
    assert chunks[1]["agent"] == "Financial Statements"

    plan = next(e for e in events if e["type"] == "research_plan")
    assert [t["id"] for t in plan["tasks"]] == ["company", "financials",
                                                THESIS_TASK_ID]
    done = [e for e in events if e["type"] == "agent_done"]
    assert [e["status"] for e in done] == ["done", "done", "done"]

    meta = out["metadata"]
    assert meta["mode"] == "research"
    assert meta["agents_run"] == 2 and meta["agents_failed"] == 0
    assert meta["confidence"] == 0.9
    # Specialist tokens summed + report writer's own usage.
    assert meta["input_tokens"] == 250 and meta["output_tokens"] == 50
    assert out["report"].startswith("# Report")


def test_merge_shifts_finding_citations():
    dr = _stubbed()
    findings = [
        {"label": "A", "answer": "a [1].",
         "chunks": [{"id": 0}, {"id": 1}]},
        {"label": "B", "answer": "b [1,2].",
         "chunks": [{"id": 0}, {"id": 1}]},
    ]
    merged = dr._merge_chunks(findings)
    assert [c["id"] for c in merged] == [0, 1, 2, 3]
    assert findings[0]["global_answer"] == "a [1]."
    assert findings[1]["global_answer"] == "b [3,4]."


def test_failed_specialist_is_isolated():
    """One specialist failing (after its retry) must not sink the run — it is
    reported as failed and the report still gets written."""
    def flaky(question: str) -> dict:
        if "financial statements" in question:  # the financials template
            raise TimeoutError("lane down")
        return _fake_run_fn(question)

    dr = _stubbed(run_fn=flaky)
    events: list[dict] = []
    out = dr.run("Should I invest in Apple? (flaky)", on_event=events.append)
    meta = out["metadata"]
    assert meta["agents_run"] == 1 and meta["agents_failed"] == 1
    statuses = {t["id"]: t["status"] for t in meta["tasks"]}
    assert statuses["company"] == "done" and statuses["financials"] == "failed"
    assert any(e["type"] == "agent_done" and e["status"] == "failed"
               for e in events)


def test_all_specialists_failing_raises():
    def dead(question: str) -> dict:
        raise TimeoutError("everything down")

    dr = _stubbed(run_fn=dead)
    with pytest.raises(RuntimeError, match="Every research specialist failed"):
        dr.run("Should I invest in Apple? (dead)")


def test_specialist_retry_then_success():
    calls = {"n": 0}

    def once_flaky(question: str) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("transient")
        return _fake_run_fn(question)

    dr = _stubbed(run_fn=once_flaky, specialists=("company",))
    out = dr.run("Should I invest in Apple? (retry)")
    assert out["metadata"]["agents_run"] == 1
    assert calls["n"] == 2                       # failed once, retried once


def test_task_cache_reuses_specialist_runs():
    calls = {"n": 0}

    def counting(question: str) -> dict:
        calls["n"] += 1
        return _fake_run_fn(question)

    dr = _stubbed(run_fn=counting)
    dr.run("Should I invest in Apple? (cache)")
    assert calls["n"] == 2
    out2 = dr.run("Should I invest in Apple? (cache)")
    assert calls["n"] == 2                       # both served from cache
    # Cached runs still produce correctly renumbered fresh chunk copies.
    assert [c["id"] for c in out2["chunks"]] == [0, 1]
    assert "cached" in out2["metadata"]["tasks"][0]["detail"]


def test_exhausted_keys_abort_the_run():
    from finagent.llm import AllKeysExhaustedError

    def exhausted(question: str) -> dict:
        raise AllKeysExhaustedError("groq keys hit their rate limit")

    dr = _stubbed(run_fn=exhausted)
    with pytest.raises(AllKeysExhaustedError):
        dr.run("Should I invest in Apple? (exhausted)")


def test_report_writer_failure_falls_back_to_sections(monkeypatch):
    dr = _stubbed()
    monkeypatch.setattr(DeepResearch, "_llm",
                        lambda self: (_ for _ in ()).throw(ValueError("down")))
    # _write_report was stubbed by _stubbed(); restore the real one.
    dr._write_report = DeepResearch._write_report.__get__(dr)  # type: ignore[method-assign]
    out = dr.run("Should I invest in Apple? (fallback)")
    assert "Company Overview" in out["report"]           # sections assembled
    assert "report writer was unavailable" in out["report"]


def test_web_pool_is_capped_across_sub_queries():
    """Regression (found by the live Deep Research smoke run): a many-sub-query
    question escalating every sub-query to web piled 40+ hits into the synth
    prompt and 413'd Groq's TPM cap. web_search_node must bound the total to
    web_top_k, round-robin so every sub-query keeps its best (trusted) hits."""
    from finagent.graph import AgenticRAGv4

    agent = AgenticRAGv4(collection_name="us_filings", provider="groq",
                         reranker_model="BAAI/bge-reranker-base", web_top_k=6)
    agent._log = lambda s, m: None

    class FakeWeb:
        def search(self, q):
            return [{"title": f"{q}-{i}", "content": "x",
                     "tier": "trusted" if i < 2 else "web"} for i in range(10)]

    agent._web = FakeWeb()
    subs = [f"query {i}" for i in range(4)]
    out = agent.web_search_node({
        "question": "q", "sub_queries": subs, "query_routes": ["external"] * 4,
    })
    hits = out["web_results"]
    assert len(hits) == 6                                 # bounded to web_top_k
    assert {h["sub_query"] for h in hits} == set(subs)    # no sub-query starved
    assert all(h["tier"] == "trusted" for h in hits)      # best hits kept

    # The common case — one search within budget (WebSearcher returns at most
    # web_top_k per search) — passes through completely unchanged, order intact.
    class SmallWeb:
        def search(self, q):
            return [{"title": f"{q}-{i}", "content": "x", "tier": "web"}
                    for i in range(4)]

    agent._web = SmallWeb()
    out1 = agent.web_search_node({
        "question": "q", "sub_queries": ["only"], "query_routes": ["external"],
    })
    assert [h["title"] for h in out1["web_results"]] == [f"only-{i}" for i in range(4)]


# --------------------------------------------------------------------------- #
# Evaluation harness
# --------------------------------------------------------------------------- #

def test_eval_scores_report_structure():
    from finagent.evaluation.research_eval import score_output, summarize

    filler = "This paragraph carries enough substance to be scored properly " * 4
    report = (
        "## Executive Summary\n\n" + filler + " Revenue grew 8% [1].\n\n"
        "## Investment Thesis\n\n### Bull Case\n\n" + filler + " [2]\n\n"
        "### Bear Case\n\n" + filler + "\n\n### Base Case\n\ny\n\n## Risks\n\nz [9]"
    )
    chunks = [{"kind": "xbrl"}, {"kind": "text"}]
    s = score_output(report, chunks)
    assert s["completed"] and s["sections"] == 3
    assert s["required_coverage"] == 1.0              # all six markers present
    # 3 substantive paragraphs, 2 cited.
    assert s["citation_coverage"] == round(2 / 3, 3)
    assert s["citation_validity"] == round(2 / 3, 3)  # [9] dangles
    assert s["source_diversity"] == 2

    rows = [{"question": "q", "research": {**s, "latency_s": 10.0,
                                           "agent_success_rate": 1.0,
                                           "contradictions": 0}}]
    summary = summarize(rows)
    assert summary["n"] == 1
    assert summary["research"]["required_coverage"] == 1.0

    # Empty output scores as incomplete, not a crash.
    empty = score_output("", [])
    assert empty["completed"] is False and empty["citation_coverage"] == 0.0


# --------------------------------------------------------------------------- #
# API surface
# --------------------------------------------------------------------------- #

def test_research_route_registered():
    from finagent.api.main import app

    routes = {r.path: getattr(r, "methods", set()) for r in app.routes}
    assert "POST" in routes.get("/api/research", set())
    assert "POST" in routes.get("/api/query", set())     # chat path untouched


def test_research_endpoint_streams_events(monkeypatch):
    """POST /api/research end-to-end over the ASGI app with the orchestrator
    stubbed: the SSE stream must carry plan → agent events → sources → report
    chunks → metrics → done, in order."""
    import json

    from starlette.testclient import TestClient

    import finagent.research as research_pkg
    from finagent.api.main import app

    class FakeResearch:
        def __init__(self, **kwargs):
            assert callable(kwargs["run_fn"])

        def run(self, question, chat_history=None, on_event=None):
            on_event({"type": "research_plan", "company": "Apple",
                      "objective": "assess", "tasks": [{"id": "company",
                                                        "label": "Company Research"}]})
            on_event({"type": "agent_start", "id": "company"})
            on_event({"type": "agent_done", "id": "company", "status": "done",
                      "detail": "1 evidence items"})
            return {"report": "# Report\n\nApple is fine [1].",
                    "chunks": [{"id": 0, "text": "e", "kind": "text"}],
                    "metadata": {"mode": "research", "confidence": 0.9}}

    monkeypatch.setattr(research_pkg, "DeepResearch", FakeResearch)

    with TestClient(app) as client:
        resp = client.post("/api/research",
                           json={"question": "Should I invest in Apple?"})
        assert resp.status_code == 200
        events = [json.loads(line[len("data: "):])
                  for line in resp.text.splitlines() if line.startswith("data: ")]

    types = [e["type"] for e in events]
    for expected in ("chat", "research_plan", "agent_start", "agent_done",
                     "sources", "chunk", "metrics", "done"):
        assert expected in types, f"missing {expected} in {types}"
    # Ordering: plan before agent events, sources before report chunks.
    assert types.index("research_plan") < types.index("agent_start") < \
        types.index("agent_done") < types.index("sources") < types.index("chunk")
    report = "".join(e["content"] for e in events if e["type"] == "chunk")
    assert report == "# Report\n\nApple is fine [1]."
    metrics = next(e for e in events if e["type"] == "metrics")
    assert metrics["agentic"]["mode"] == "research"


def test_research_request_validation():
    from finagent.api.models import ResearchRequest

    req = ResearchRequest(question="Should I invest in Apple?", max_agents=5)
    assert req.max_agents == 5
    with pytest.raises(Exception):
        ResearchRequest(question="q", max_agents=99)     # above the cap
    with pytest.raises(Exception):
        ResearchRequest(question="")                     # empty question
