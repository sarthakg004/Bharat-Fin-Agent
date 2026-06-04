"""
planner.py  ·  finagent/prompts/planner.py

Planning / routing prompts (planner decomposition, query router, market-intent
planner). Canonical import surface; the string bodies currently live with the
nodes that use them in `finagent.graph.*` and are re-exported here (physical
relocation deferred under the "no logic rewrite" scope).
"""

from finagent.graph.base import PLANNER_PROMPT  # noqa: F401
from finagent.graph.full import ROUTER_SYSTEM, ROUTER_PROMPT  # noqa: F401
from finagent.graph.agent import (  # noqa: F401
    MARKET_PLANNER_SYSTEM,
    MARKET_PLANNER_PROMPT,
)

__all__ = [
    "PLANNER_PROMPT", "ROUTER_SYSTEM", "ROUTER_PROMPT",
    "MARKET_PLANNER_SYSTEM", "MARKET_PLANNER_PROMPT",
]
