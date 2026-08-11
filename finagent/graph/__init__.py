"""LangGraph agent.

The pipeline is built as an inheritance ladder, each layer adding a capability:

    base.py        AgenticRAG    — planner → retrieve → synthesize → critic
    corrective.py  AgenticRAGv2  — + hybrid retrieval, critic retry loop
    full.py        AgenticRAGv3  — + sub-query router
    agent.py       AgenticRAGv4  — + XBRL, calculator, SEC fetch, market data,
                                    web search, memory

`AgenticRAGv4` is what the API serves. `FinAgent` is a friendly alias.
"""

from finagent.graph.agent import AgenticRAGv4
from finagent.graph.agent import AgenticRAGv4 as FinAgent

__all__ = ["AgenticRAGv4", "FinAgent"]
