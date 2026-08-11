"""
critic.py  ·  finagent/prompts/critic.py

Critic + refusal prompts. Zero-import leaf: the graph nodes import these
strings from here.
"""

CRITIC_SYSTEM = """\
You are a hallucination critic. Given an answer and the source excerpts it was
based on, extract each distinct factual claim from the answer and decide whether
the excerpts SUPPORT it. Judge only against the excerpts, not your own knowledge.
"""

CRITIC_PROMPT = """\
Source excerpts:
{context}

---
Answer to check:
{answer}

---
Extract the factual claims and mark each supported / not supported.

If any claim is NOT supported, also say which fix would work, because the agent
gets exactly one retry and has to spend it on the right thing:
  - "redraft" when the excerpts DO contain what is needed and the answer
    overstated, mis-stated, or mis-attributed it. Rewriting against these same
    excerpts is enough.
  - "gather" when the excerpts simply do not contain the fact. Rewriting cannot
    invent it, so the agent must go and retrieve more evidence.
"""

REFUSAL_TEMPLATE = (
    "I don't have enough information to answer this from the available "
    "filings{web_clause}. The claim-checking critic could not support the "
    "draft's claims from the gathered evidence{detail}."
)

__all__ = ["CRITIC_SYSTEM", "CRITIC_PROMPT", "REFUSAL_TEMPLATE"]
