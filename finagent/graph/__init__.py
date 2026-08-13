"""The LangGraph agent.

`AgenticRAGv4` (`agent.py`) is the class the API serves; `FinAgent` is an alias
for it. It is assembled from the node mixins in `nodes/` plus three base layers
that exist only to be inherited: `base.py` (planner, critic), `corrective.py`
(hybrid retrieval, critic retry), `full.py` (query router, retrieval rewrite).
"""

from finagent.graph.agent import AgenticRAGv4
from finagent.graph.agent import AgenticRAGv4 as FinAgent

__all__ = ["AgenticRAGv4", "FinAgent"]
