"""
Thread-safety of per-request LLM configuration.

The agent is shared and cached; provider/model/api_key used to be *fields on it*
that each request overwrote before invoking the graph. That is safe only while
exactly one request runs at a time (RAG_MAX_WORKERS=1). These tests pin the
invariant that replaced it: request configuration travels in the LangGraph
config, so concurrent runs on one agent never observe each other's values.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from finagent.runtime import DEFAULTS, RuntimeContext, current_context


def test_model_for_resolves_roles_and_overrides():
    ctx = RuntimeContext(provider="groq")
    assert ctx.model_for("synth") == DEFAULTS["groq"]["synth"]
    # Derived roles follow the role they were defined against.
    assert ctx.model_for("router") == ctx.model_for("synth")
    assert ctx.model_for("grader") == ctx.model_for("planner")
    assert ctx.model_for("verifier") == ctx.model_for("critic")

    # The UI's "model" overrides the synth family only — not the planner.
    over = RuntimeContext(provider="groq", synth_model="my-model")
    assert over.model_for("synth") == "my-model"
    assert over.model_for("router") == "my-model"
    assert over.model_for("planner") == DEFAULTS["groq"]["planner"]


def test_context_is_frozen():
    """A node must not be able to write request config back into shared state."""
    import dataclasses

    import pytest

    ctx = RuntimeContext()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.api_key = "leaked"          # type: ignore[misc]


def test_current_context_falls_back_outside_a_graph_run():
    """The eval harness and __main__ demos build LLMs with no graph running."""
    assert current_context() == RuntimeContext()


def test_concurrent_runs_do_not_share_request_config():
    """64 threads, ONE compiled graph, a distinct api_key each.

    Every run must read back its own key. Against the previous design — where
    the key was assigned to the shared agent before invoking — this leaks:
    whichever thread wrote last wins for everyone still in flight.
    """
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class S(TypedDict, total=False):
        seen: str

    # Stands in for a node: reads config through the same ambient accessor the
    # real `_get_llm` uses, from a helper that was never handed the context.
    def node(state):
        return {"seen": current_context().api_key}

    g = StateGraph(S)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    app = g.compile()

    def run(i: int) -> str:
        ctx = RuntimeContext(provider="groq", api_key=f"key-{i}")
        out = app.invoke({}, config={"configurable": {"runtime_context": ctx}})
        return out["seen"]

    with ThreadPoolExecutor(max_workers=16) as ex:
        got = list(ex.map(run, range(64)))

    assert got == [f"key-{i}" for i in range(64)]


def test_fetch_cache_is_per_conversation():
    """A fetched filing is reused across the turns of ONE chat and is
    unreachable from any other — the key is the client's session id.

    Without a session id (eval harness, CLIs) nothing is cached, which is the
    behaviour that existed before the cache.
    """
    import finagent.graph.nodes.fetch as F
    from finagent.runtime import RuntimeContext, use_context

    F._FETCH_CACHE.clear()
    chunks = [{"text": "risk factors", "page": 12}]

    with use_context(RuntimeContext(session_id="chat-A")):
        key_a = F._fetch_cache_key("PLTR", "10-K:1")
        assert F._fetch_cache_get(key_a) is None        # cold
        F._fetch_cache_put(key_a, chunks)
        assert F._fetch_cache_get(key_a) == chunks      # same chat → reused

    with use_context(RuntimeContext(session_id="chat-B")):
        key_b = F._fetch_cache_key("PLTR", "10-K:1")
        assert key_b != key_a
        assert F._fetch_cache_get(key_b) is None        # other user → isolated

    with use_context(RuntimeContext()):                 # no session id
        assert F._fetch_cache_key("PLTR", "10-K:1") is None
        assert F._fetch_cache_get(None) is None

    # The cache hands back copies, so a caller mutating its list cannot corrupt
    # what the next turn of the conversation sees.
    with use_context(RuntimeContext(session_id="chat-A")):
        got = F._fetch_cache_get(key_a)
        got.append({"text": "injected"})
        assert len(F._fetch_cache_get(key_a)) == 1
    F._FETCH_CACHE.clear()


def test_fetch_cache_is_bounded():
    """It must not grow without limit — one entry per (session, filing)."""
    import finagent.graph.nodes.fetch as F

    F._FETCH_CACHE.clear()
    for i in range(F._FETCH_MAX + 10):
        F._fetch_cache_put((f"chat-{i}", "PLTR", "10-K:1"), [{"text": "x"}])
    assert len(F._FETCH_CACHE) == F._FETCH_MAX
    F._FETCH_CACHE.clear()


def test_agent_holds_no_request_state():
    """The shared agent must expose no provider/model/key attribute at all.

    These were fields that `get_agentic` overwrote per request. Asserting their
    absence is what stops them being reintroduced.
    """
    from finagent.graph import AgenticRAGv4

    agent = AgenticRAGv4(collection_name="us_filings",
                         reranker_model="BAAI/bge-reranker-base")
    for attr in ("provider", "api_key", "_llms", "synth_model", "planner_model",
                 "critic_model", "grader_model", "router_model", "code_model",
                 "verifier_model"):
        assert not hasattr(agent, attr), f"{attr} is back on the shared agent"


def test_real_agent_llm_follows_the_request_context(monkeypatch):
    """`_get_llm` on the production agent must resolve provider/model/key from
    the context in scope, not from anything stored on the agent."""
    import finagent.runtime as R
    from finagent.graph import AgenticRAGv4

    seen: list[tuple] = []
    monkeypatch.setattr(R, "build_llm",
                        lambda p, m, k, **kw: seen.append((p, m, k)) or "llm",
                        raising=False)
    # create_llm imports build_llm from finagent.llm at call time.
    import finagent.llm as L
    monkeypatch.setattr(L, "build_llm",
                        lambda p, m, k, **kw: seen.append((p, m, k)) or "llm")

    agent = AgenticRAGv4(collection_name="us_filings",
                         reranker_model="BAAI/bge-reranker-base")

    with R.use_context(R.RuntimeContext(provider="openai", api_key="key-a")):
        agent._get_llm("synth")
    with R.use_context(R.RuntimeContext(provider="groq", api_key="key-b")):
        agent._get_llm("synth")

    assert seen == [("openai", R.DEFAULTS["openai"]["synth"], "key-a"),
                    ("groq", R.DEFAULTS["groq"]["synth"], "key-b")]
