"""
critic.py  ·  finagent/prompts/critic.py

Critic + numeric-verification + refusal prompts. Canonical import surface; the
string bodies currently live in `finagent.graph.base` / `finagent.graph.agent`
and are re-exported here (physical relocation deferred under the "no logic
rewrite" scope).
"""

from finagent.graph.base import CRITIC_SYSTEM, CRITIC_PROMPT  # noqa: F401
from finagent.graph.nodes.verification import (  # noqa: F401
    NUM_VERIFY_SYSTEM,
    NUM_VERIFY_PROMPT,
    REFUSAL_TEMPLATE,
)

__all__ = [
    "CRITIC_SYSTEM", "CRITIC_PROMPT",
    "NUM_VERIFY_SYSTEM", "NUM_VERIFY_PROMPT", "REFUSAL_TEMPLATE",
]
