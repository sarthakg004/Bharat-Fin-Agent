"""
runner.py  ·  finagent/agents/runner.py

Execute the agent graph. Canonical home for the run/trace utilities.

`run_with_trace` / `run_with_route_trace` / `run_with_full_trace` / `format_trace`
are re-exported from `finagent.graph.runner` (kept with the class chain).
`run_agent` is the synchronous one-call entry point, delegating to the deployed
service so notebook/CLI callers get the same answer a user would.

Note: the SSE streaming generator (`stream_agent` equivalent) lives in the API
layer (`finagent.api.main._stream_answer`) because it is tied to the HTTP
response and chat-history plumbing; it is intentionally not duplicated here.
"""

from __future__ import annotations

from typing import Optional

from finagent.graph.runner import (  # noqa: F401
    run_with_trace,
    run_with_route_trace,
    run_with_full_trace,
    format_trace,
)


def run_agent(
    question: str,
    market: str = "us",
    top_k: int = 5,
    provider: str = "groq",
    synth_model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """Synchronously answer one question with the production agent.

    Returns the normalized result dict (`answer`, `chunks`, `charts`,
    `metadata`) from `rag_service.run_agentic`.
    """
    from finagent.api.rag_service import run_agentic

    return run_agentic(
        market=market, question=question, top_k=top_k,
        provider=provider, synth_model=synth_model, api_key=api_key,
    )


__all__ = [
    "run_agent",
    "run_with_trace",
    "run_with_route_trace",
    "run_with_full_trace",
    "format_trace",
]
