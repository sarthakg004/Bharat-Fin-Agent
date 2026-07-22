"""
runtime.py  ·  finagent/runtime.py

Per-request execution context.

The agent is a long-lived, shared, immutable object: compiled graph, retriever,
reranker, embedder, vector store. Everything that varies per REQUEST — which
provider, which model, whose API key — lives here instead, in a frozen dataclass
created fresh for each question and thrown away after.

It reaches the nodes through LangGraph's `configurable`, which is backed by a
contextvar. That means a node's nested helpers read it without being handed it,
and concurrent runs on the same compiled graph never see each other's values.
Previously these fields lived on the agent and were overwritten per request —
safe only because the API served one request at a time.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

# Per-provider model defaults. These used to be `AgenticRAG.DEFAULTS`; they moved
# here because the agent must no longer know what a provider is.
#
# Groq tier picks:
#   planner / grader / router (structured output)  → qwen/qwen3.6-27b
#   synth + critic  (long-form writing, reasoning) → openai/gpt-oss-120b
# The planner's decomposition drives retrieval and the grader DROPS chunks, so
# both sit on the quality path — small (≤8B) models there measurably
# mis-decomposed multi-hop questions and mis-graded relevant chunks. Qwen
# (replacing the deprecated llama-3.3-70b) keeps these calls on a different
# quota bucket from the 120B synth/critic, so the roles don't hit rate limits
# in lockstep.
DEFAULTS: dict[str, dict[str, str]] = {
    "groq": {
        "planner": "qwen/qwen3.6-27b",
        "synth":   "openai/gpt-oss-120b",
        "critic":  "openai/gpt-oss-120b",
    },
    "gemini": {
        "planner": "gemini-2.5-flash",
        "synth":   "gemini-2.5-flash",
        "critic":  "gemini-2.5-flash",
    },
    "openai": {
        "planner": "gpt-4o-mini",
        "synth":   "gpt-4o",
        "critic":  "gpt-4o",
    },
    "anthropic": {
        "planner": "claude-haiku-4-5",
        "synth":   "claude-sonnet-4-6",
        "critic":  "claude-sonnet-4-6",
    },
}

# Roles that don't get their own default, mapped to the one they follow. This
# reproduces exactly what the mixins did with `grader_model or self.planner_model`,
# `router_model or self.synth_model`, and so on.
_DERIVED = {
    "grader":         "planner",
    "market_planner": "planner",
    "router":         "synth",
    "code":           "synth",
    "verifier":       "critic",
}


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """One request's LLM configuration. Frozen so a node cannot write to it."""

    provider: str = "groq"
    # The Settings UI's "model" — overrides the synthesizer family only, which
    # is what it has always meant in the API contract.
    synth_model: Optional[str] = None
    # None → `build_llm` reads the provider's env keys and rotates through the
    # pool on rate limits.
    api_key: Optional[str] = None
    temperature: float = 0.0
    top_k: int = 5
    # The client's conversation id. Scopes anything cached BETWEEN turns of one
    # chat (a fetched filing, a research finding) so it is reused within the
    # conversation and invisible to every other user. None → no cross-turn
    # caching at all, which is the safe default for the eval harness and CLIs.
    session_id: Optional[str] = None

    def model_for(self, role: str) -> str:
        """Resolve a role ('planner', 'synth', 'router', …) to a model name."""
        base = _DERIVED.get(role, role)
        if base == "synth" and self.synth_model:
            return self.synth_model
        try:
            return DEFAULTS[self.provider][base]
        except KeyError:
            raise ValueError(
                f"No default model for provider={self.provider!r} role={role!r}. "
                f"Providers: {list(DEFAULTS)}."
            ) from None


_DEFAULT = RuntimeContext()


def create_llm(ctx: RuntimeContext, role: str):
    """Build the chat client for a role.

    Deliberately uncached: constructing a LangChain wrapper costs microseconds
    against a multi-second inference call, and a cache keyed on the agent is
    precisely the shared mutable state this module exists to remove.
    """
    from finagent.llm import build_llm

    return build_llm(ctx.provider, ctx.model_for(role), ctx.api_key,
                     temperature=ctx.temperature)


# Scoped override for callers that build LLMs OUTSIDE a graph run — the CLIs,
# the eval harness, notebooks. A ContextVar (not a global) so it stays
# thread-local and is unwound on exit.
_override: ContextVar[Optional[RuntimeContext]] = ContextVar(
    "finagent_runtime_context", default=None)


@contextmanager
def use_context(ctx: RuntimeContext):
    """Make `ctx` the current context for the duration of the block."""
    token = _override.set(ctx)
    try:
        yield ctx
    finally:
        _override.reset(token)


def current_context() -> RuntimeContext:
    """The running request's context.

    Resolution order: the graph invocation's config (most specific — this is
    the API path), then any `use_context` block, then defaults. The fallback is
    what lets the eval harness, notebooks, and `__main__` demos build LLMs
    without going through the API.
    """
    try:
        from langgraph.config import get_config

        ctx = (get_config().get("configurable") or {}).get("runtime_context")
    except Exception:       # not inside a graph run
        ctx = None
    if not isinstance(ctx, RuntimeContext):
        ctx = _override.get()
    return ctx if isinstance(ctx, RuntimeContext) else _DEFAULT
