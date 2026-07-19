"""
critic.py  ·  finagent/prompts/critic.py

Critic + numeric-verification + refusal prompts. Zero-import leaf: the graph
nodes import these strings from here.
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
"""

NUM_VERIFY_SYSTEM = """\
You verify numeric claims in a draft financial answer. Given the draft answer
and the source evidence (text excerpts, table-derived computations, optional
web results), extract each distinct numeric claim and decide whether the
SAME figure appears in the evidence. Accept reasonable paraphrases ($1.5 bn
vs $1,500 million) but reject silent invention.
"""

NUM_VERIFY_PROMPT = """\
Draft answer:
\"\"\"{answer}\"\"\"

Evidence:
{evidence}

---
Extract every numeric claim from the answer (revenue figures, percentages,
ratios, counts) and report whether each is supported by the evidence.
"""

REFUSAL_TEMPLATE = (
    "I don't have enough information to answer this from the available "
    "filings{web_clause}. The retrieval and numeric verification steps could "
    "not ground the requested figures{detail}."
)

__all__ = [
    "CRITIC_SYSTEM", "CRITIC_PROMPT",
    "NUM_VERIFY_SYSTEM", "NUM_VERIFY_PROMPT", "REFUSAL_TEMPLATE",
]
