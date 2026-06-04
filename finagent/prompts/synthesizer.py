"""
synthesizer.py  ·  finagent/prompts/synthesizer.py

Synthesizer prompts (analyst-voice system prompt + answer template). Canonical
import surface; the string bodies currently live in `finagent.graph.base` and
are re-exported here (physical relocation deferred under the "no logic rewrite"
scope).
"""

from finagent.graph.base import SYNTHESIZER_SYSTEM, SYNTHESIZER_PROMPT  # noqa: F401

__all__ = ["SYNTHESIZER_SYSTEM", "SYNTHESIZER_PROMPT"]
