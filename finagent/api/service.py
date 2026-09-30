"""Run one question through the agent and shape the result for the API.

The evaluation calls `run_agent` too, so it measures the answers a user gets.
"""

from __future__ import annotations

import contextlib
import os
import time
from threading import Lock
from typing import Callable, Optional

from finagent.agent import FinAgent
from finagent.config import settings
from finagent.runtime import RuntimeContext

# One agent per collection. It holds no request state, so it is shared.
_lock = Lock()
_agents: dict[str, FinAgent] = {}


def get_agent(collection: Optional[str] = None) -> FinAgent:
    coll = collection or settings.us_collection
    with _lock:
        if coll not in _agents:
            _agents[coll] = FinAgent(collection=coll)
        return _agents[coll]


def _step_detail(node: str, delta: dict) -> Optional[str]:
    """A short outcome line for a finished step, e.g. "8 passages"."""
    def plural(n, word):
        return f"{n} {word}{'' if n == 1 else 's'}"

    try:
        if node == "planner":
            routes = delta.get("query_routes") or []
            counts = {r: routes.count(r) for r in dict.fromkeys(routes)}
            return " · ".join(f"{k}×{v}" if v > 1 else k for k, v in counts.items()) or None
        if node == "fetch_filing":
            fs = delta.get("fetch_status") or {}
            if fs.get("status") == "fetched":
                return f"fetched from EDGAR ({fs.get('chunks_added')} chunks)"
            if fs.get("status") == "error":
                return "fetch failed"
            return "already in the index" if fs.get("decision") == "already_indexed" else None
        if node == "retrieve":
            n = len(delta.get("retrieved_chunks") or [])
            return plural(n, "passage") if n else "no relevant passages"
        if node == "xbrl":
            n = len(delta.get("xbrl_facts") or [])
            return f"{plural(n, 'exact figure')} from SEC XBRL" if n else None
        if node == "calculator":
            n = len(delta.get("calc_results") or [])
            return plural(n, "derived metric") if n else None
        if node == "market_data":
            n = sum(1 for m in delta.get("market_data") or [] if m.get("ok"))
            return (plural(n, "market call") + (" + chart" if delta.get("charts") else "")) if n else None
        if node == "web_search":
            n = len(delta.get("web_results") or [])
            return plural(n, "web result") if n else None
        if node == "edgar_search":
            n = sum(len(r.get("companies") or []) for r in delta.get("edgar_results") or [])
            return f"{n} compan{'y' if n == 1 else 'ies'} matched" if n else None
        if node == "synthesize":
            return f"draft written ({len((delta.get('answer') or '').split())} words)"
        if node == "critic":
            score = delta.get("support_score")
            return f"{score:.0%} of claims supported" if isinstance(score, (int, float)) else "not checked"
    except Exception:
        pass
    return None


def _langfuse_handler():
    """A Langfuse tracing callback when LANGFUSE_* keys are set, else None."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except Exception as e:
        print(f"[langfuse] tracing disabled ({type(e).__name__}: {e})")
        return None


def run_agent(question: str,
              chat_history: Optional[list[dict]] = None,
              provider: Optional[str] = None,
              model: Optional[str] = None,
              api_key: Optional[str] = None,
              collection: Optional[str] = None,
              session_id: Optional[str] = None,
              on_step: Optional[Callable[[str], None]] = None,
              on_step_done: Optional[Callable[[str, Optional[str]], None]] = None) -> dict:
    """Answer one question. Returns `{answer, chunks, charts, metadata}`.

    `provider` / `model` / `api_key` override the writer for this request only.
    `on_step(name)` fires when a step starts and `on_step_done(name, detail)`
    when it finishes, so the API can stream progress.
    """
    agent = get_agent(collection)
    ctx = RuntimeContext(provider=provider, model=model, api_key=api_key)
    state: dict = {"question": question, "log": [], "notices": []}
    if chat_history:
        state["chat_history"] = chat_history

    from langchain_core.callbacks import UsageMetadataCallbackHandler

    usage = UsageMetadataCallbackHandler()
    langfuse = _langfuse_handler()
    config: dict = {"recursion_limit": 50,
                    "configurable": {"runtime_context": ctx},
                    "callbacks": [cb for cb in (usage, langfuse) if cb]}
    trace = contextlib.nullcontext()
    if langfuse is not None:
        from langfuse import propagate_attributes
        attrs: dict = {"trace_name": "finagent-query",
                       "tags": [ctx.resolve("writer")[0], agent.collection]}
        if session_id:
            attrs["session_id"] = str(session_id)     # groups a chat's turns
        trace = propagate_attributes(**attrs)

    node_seconds: dict[str, float] = {}
    last = time.time()
    with trace:
        # "tasks" fires when a step starts, "updates" when it finishes,
        # "values" carries the full state after each round.
        for mode, data in agent.graph.stream(
                state, stream_mode=["updates", "values", "tasks"], config=config):
            if mode == "values":
                state = data
            elif mode == "tasks" and isinstance(data, dict):
                if "result" not in data and "error" not in data and on_step:
                    on_step(data.get("name", ""))
            elif mode == "updates" and isinstance(data, dict):
                now = time.time()
                for name, delta in data.items():
                    node_seconds[name] = round(node_seconds.get(name, 0.0) + now - last, 3)
                    if on_step_done:
                        on_step_done(name, _step_detail(name, delta or {}))
                last = now

    if langfuse is not None:
        # Cloud Run stops giving CPU once the response is sent, so flush now.
        with contextlib.suppress(Exception):
            from langfuse import get_client
            get_client().flush()

    per_model = getattr(usage, "usage_metadata", {}) or {}
    cards = [{"id": i, "page": "—", **{k: v for k, v in e.items() if k != "prompt"}}
             for i, e in enumerate(state.get("evidence") or [])]
    return {
        "answer": state.get("answer") or "",
        "chunks": cards,                       # card N-1 is what `[N]` cites
        "charts": list(state.get("charts") or []),
        "metadata": {
            "model": ctx.resolve("writer")[1],
            "input_tokens": sum(m.get("input_tokens") or 0 for m in per_model.values()),
            "output_tokens": sum(m.get("output_tokens") or 0 for m in per_model.values()),
            "node_seconds": node_seconds,
            "sub_queries": state.get("sub_queries", []),
            "query_routes": state.get("query_routes", []),
            "retrieval_query": state.get("retrieval_query", ""),
            "support_score": state.get("support_score"),
            "unsupported_claims": state.get("unsupported_claims") or [],
            "recoveries": state.get("recoveries", 0),
            "refused": bool(state.get("refused")),
            "fetch_status": state.get("fetch_status") or {},
            "notices": state.get("notices") or [],
            "log": state.get("log") or [],
        },
    }
