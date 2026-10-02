"""Which model does which job, and the one thing a user can change: the writer.

The agent is built once and shared. Anything that differs per request (the
writer model and the user's API key) travels in a `RuntimeContext` through
LangGraph's config, so concurrent requests never see each other's values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# role -> (provider, model). This table is the single source of truth: the API
# serves it to the frontend, so the UI never hardcodes a model name.
ROLES: dict[str, tuple[str, str]] = {
    # Splits the question, writes the search query, picks the market-data call.
    "planner":   ("groq", "qwen/qwen3.8-27b"),
    # One-shot structured extraction: company gate, XBRL, calculator, EDGAR.
    "extractor": ("groq", "qwen/qwen3.8-27b"),
    # Writes the answer. On Gemini because Groq's free tier caps a request at
    # 8,000 tokens, which truncates the evidence.
    "writer":    ("gemini", "gemini-3.5-flash"),
    # Fact-checks the draft. A different model from the writer on purpose.
    "critic":    ("gemini", "gemini-3.6-flash"),
}

# Models the picker offers for the writer. The first one is the provider's default.
WRITER_MODELS: dict[str, list[str]] = {
    "gemini": ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"],
    "groq": ["qwen/qwen3.8-27b"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
    "anthropic": ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-4-5-20251001"],
}

# Groq's free tier rejects any request over 8,000 tokens. Filing text runs about
# 3.6 characters per token, so 14,000 characters of evidence leaves room for the
# prompt, the history and the answer.
FREE_GROQ_EVIDENCE_CHARS = 14_000


# How hard Gemini "thinks" before it answers. Measured on one filing question:
# the writer took 45 s at the default and 8 s at "low", with every claim still
# supported. ponytail: set on speed, not on an accuracy run; raise to "high" and
# compare if answer quality matters more than latency.
GEMINI_THINKING = "low"

# Qwen on Groq does not reason unless asked. With "low" the extractors think
# before filling the form. Measured on 14 extraction cases x 4 runs: 56 of 56
# correct, against 50 of 56 without. The planned formula gained most (unadjusted
# EBITDA stopped picking up interest expense). Costs about 2 s and 200 output
# tokens per call.
EXTRACTOR_REASONING = "low"


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """One request's writer choice. Frozen so a node cannot change it."""

    provider: Optional[str] = None      # None = the default writer provider
    model: Optional[str] = None         # None = that provider's first model
    api_key: Optional[str] = None       # the user's own key for the writer
    temperature: float = 0.0

    def resolve(self, role: str) -> tuple[str, str, Optional[str]]:
        """(provider, model, api_key) for a role. Only the writer is overridable."""
        if role == "critic":
            # The critic must be a different model from the writer: when the user
            # picks the critic's model to write, check with the default writer's.
            same = self.resolve("writer")[:2] == ROLES["critic"]
            return (*(ROLES["writer"] if same else ROLES["critic"]), None)
        if role != "writer":
            return (*ROLES[role], None)
        provider = (self.provider or ROLES["writer"][0]).lower()
        default = (ROLES["writer"][1] if provider == ROLES["writer"][0]
                   else WRITER_MODELS[provider][0])
        return provider, self.model or default, self.api_key

    def evidence_budget(self) -> Optional[int]:
        """Characters of evidence the writer may receive. None = no limit."""
        provider, _, key = self.resolve("writer")
        return FREE_GROQ_EVIDENCE_CHARS if provider == "groq" and not key else None


def create_llm(ctx: RuntimeContext, role: str):
    """The chat model for a role under this request's context."""
    from finagent.llm import build_llm

    provider, model, key = ctx.resolve(role)
    extra = {}
    if provider == "gemini":
        extra = {"thinking_level": GEMINI_THINKING}
    elif role == "extractor":
        extra = {"reasoning_effort": EXTRACTOR_REASONING}
    return build_llm(provider, model, key, temperature=ctx.temperature, **extra)


def current_context() -> RuntimeContext:
    """The running request's context, or the defaults outside a graph run."""
    try:
        from langgraph.config import get_config

        ctx = (get_config().get("configurable") or {}).get("runtime_context")
    except Exception:
        ctx = None
    return ctx if isinstance(ctx, RuntimeContext) else RuntimeContext()
