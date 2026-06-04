"""System prompts, organised by agent role.

Canonical import surface for prompt strings:
    planner.*      — PLANNER_PROMPT, ROUTER_SYSTEM/PROMPT, MARKET_PLANNER_*.
    synthesizer.*  — SYNTHESIZER_SYSTEM, SYNTHESIZER_PROMPT.
    critic.*       — CRITIC_SYSTEM/PROMPT, NUM_VERIFY_SYSTEM/PROMPT, REFUSAL_TEMPLATE.

Deviation note: the architectural ideal is for `prompts` to be a zero-import
leaf (pure strings) that the nodes depend on. Today these modules *re-export*
the constants from their `finagent.graph.*` definition sites to avoid editing
the class-chain files or duplicating strings (which could drift). Flipping the
dependency — moving the bodies here and importing them into the nodes — is a
follow-up once the node decomposition happens.
"""

from finagent.prompts import planner, synthesizer, critic  # noqa: F401

__all__ = ["planner", "synthesizer", "critic"]
