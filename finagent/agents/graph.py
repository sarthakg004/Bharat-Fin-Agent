"""
graph.py  ·  finagent/agents/graph.py

Graph construction entry points for the agent. Canonical home for `build_graph`.

The compiled LangGraph and its construction currently live with the agent class
chain in `finagent.graph` (kept intact per the "no logic rewrite" scope). This
module re-exports the builder so the canonical path is `finagent.agents.graph`.
"""

from __future__ import annotations

from finagent.graph.builder import build_agent, build_graph, NODE_REFERENCE  # noqa: F401

__all__ = ["build_agent", "build_graph", "NODE_REFERENCE"]
