"""System prompts, organised by agent role.

Canonical home of prompt strings — zero-import leaf modules the graph nodes
depend on:
    planner.*      — PLANNER_PROMPT, ROUTER_SYSTEM/PROMPT, MARKET_PLANNER_*.
    synthesizer.*  — SYNTHESIZER_SYSTEM, SYNTHESIZER_PROMPT.
    critic.*       — CRITIC_SYSTEM/PROMPT, NUM_VERIFY_SYSTEM/PROMPT, REFUSAL_TEMPLATE.
"""

from finagent.prompts import critic, planner, synthesizer  # noqa: F401

__all__ = ["planner", "synthesizer", "critic"]
