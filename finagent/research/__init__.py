"""Deep Research Mode — multi-specialist investment research orchestration.

An additional execution path next to the chat pipeline: the orchestrator
scopes the request, runs specialist research tasks through the existing
production agent (injected as a runner), cross-checks the findings, and
writes a cited investment report. See `docs/deep-research.md`.
"""

from finagent.research.orchestrator import DeepResearch
from finagent.research.specialists import SPECIALISTS, ResearchScope, Specialist

__all__ = ["DeepResearch", "SPECIALISTS", "ResearchScope", "Specialist"]
