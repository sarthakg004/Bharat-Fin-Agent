"""LangGraph agent layer (canonical public namespace).

The agentic RAG implementation is a class chain — `AgenticRAG` → `AgenticRAGv2`
→ `AgenticRAGv3` → `AgenticRAGv4` (the deployed agent). Per the agreed
"structural move, no logic rewrite" scope, that chain stays intact in
`finagent.graph.*` (its nodes are bound methods, so decomposing them into
standalone function modules is a rewrite, deferred); this package is the
**canonical front door** that re-exports the public surface so callers can use
clean `finagent.agents` imports.

`AgentState` has been physically moved here (`finagent.agents.state`), with a
re-export shim left at `finagent.graph.state`.

Public exports:
    AgentState            — the shared graph state (TypedDict).
    build_graph/build_agent — compile / fetch the deployed agent + its graph.
    run_agent             — synchronous one-call run.
    run_with_trace, run_with_route_trace, run_with_full_trace, format_trace.
    AgenticRAG, AgenticRAGv2, AgenticRAGv3, AgenticRAGv4 — the class chain
                            (resolved lazily; see note below).

Depends on: config, llm, retrieval, tools, prompts, corpus.

Import-cycle note: the class-chain classes are exposed lazily via module
``__getattr__`` (PEP 562). Eagerly importing them here would re-enter a
half-initialised ``finagent.graph.base`` through the ``finagent.graph.state``
→ ``finagent.agents.state`` shim, so they are resolved only on first access.
"""

from typing import TYPE_CHECKING

from finagent.agents.state import AgentState
from finagent.agents.graph import build_agent, build_graph, NODE_REFERENCE
from finagent.agents.runner import (
    run_agent,
    run_with_trace,
    run_with_route_trace,
    run_with_full_trace,
    format_trace,
)

# The class chain lives in finagent.graph.*; resolve lazily to avoid an import
# cycle (see module docstring).
_CLASS_CHAIN = {
    "AgenticRAG": ("finagent.graph.base", "AgenticRAG"),
    "AgenticRAGv2": ("finagent.graph.corrective", "AgenticRAGv2"),
    "AgenticRAGv3": ("finagent.graph.full", "AgenticRAGv3"),
    "AgenticRAGv4": ("finagent.graph.agent", "AgenticRAGv4"),
}


def __getattr__(name: str):
    """PEP 562 lazy attribute access for the agent class chain."""
    if name in _CLASS_CHAIN:
        import importlib

        module_path, attr = _CLASS_CHAIN[name]
        return getattr(importlib.import_module(module_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # for type checkers / IDEs only — not executed at runtime
    from finagent.graph.base import AgenticRAG
    from finagent.graph.corrective import AgenticRAGv2
    from finagent.graph.full import AgenticRAGv3
    from finagent.graph.agent import AgenticRAGv4


__all__ = [
    "AgentState",
    "build_agent", "build_graph", "NODE_REFERENCE",
    "run_agent", "run_with_trace", "run_with_route_trace",
    "run_with_full_trace", "format_trace",
    "AgenticRAG", "AgenticRAGv2", "AgenticRAGv3", "AgenticRAGv4",
]
