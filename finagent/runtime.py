"""Per-request execution context.

The agent is a long-lived, shared, immutable object: compiled graph, retriever,
reranker, embedder, vector store. Everything that varies per REQUEST — which
provider, which model, whose API key — lives here instead, in a frozen dataclass
created fresh for each question and thrown away after.

It reaches the nodes through LangGraph's `configurable`, which is backed by a
contextvar, so a node's nested helpers read it without being handed it and
concurrent runs on the same compiled graph never see each other's values. These
fields used to live on the agent and be overwritten per request — safe only while
the API served one request at a time.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

# Per-provider model defaults. These used to be `AgenticRAG.DEFAULTS`; they moved
# here because the agent must no longer know what a provider is.
#
# EVERY role a node asks for must appear here or in `_DERIVED`, and the value
# must be the model that role ACTUALLY runs on. That was not true before:
# `planner` said qwen while the deployed fused plan+route node called
# `_get_llm("router")`, `router` derived from `synth`, and so the planner ran on
# gpt-oss-120b. Setting `planner` changed nothing, the comment here described a
# configuration that was not in effect, and the UI's model picker silently
# re-pointed the planner (it overrides the synth family, and the planner was
# inside it). Roles that differ are now spelled out rather than derived.
#
# The planner's decomposition drives retrieval, so every reasoning role sits on
# the quality tier — small (≤8B) models there measurably mis-decomposed
# multi-hop questions.
DEFAULTS: dict[str, dict[str, str]] = {
    "groq": {
        "planner": "openai/gpt-oss-120b",
        "synth":   "openai/gpt-oss-120b",
        "critic":  "openai/gpt-oss-120b",
    },
    # Gemini free-tier picks. MEASURED 2026-08-12 off the 429 itself: the
    # binding limit is 20 requests per MINUTE, per key, per model
    # (`generate_content_free_tier_requests, limit: 20` with a ~10s
    # retry-after) — NOT per day, as this comment previously guessed. A long
    # run is therefore throughput-limited, not day-limited: it needs backoff,
    # never an overnight wait. Splitting planner+synth (3.6) from the critic
    # (3.5) doubles the pool by using two independent buckets.
    "gemini": {
        "planner": "gemini-3.6-flash",
        "synth":   "gemini-3.6-flash",
        "critic":  "gemini-3.5-flash",
    },
    "openai": {
        "planner": "gpt-4o",
        "synth":   "gpt-4o",
        "critic":  "gpt-4o",
    },
    "anthropic": {
        "planner": "claude-sonnet-4-6",
        "synth":   "claude-sonnet-4-6",
        "critic":  "claude-sonnet-4-6",
    },
}

# Roles that follow another role rather than carrying their own default. Keep
# this list SHORT: a derived role is one the operator cannot point at a
# different model, which is exactly the trap `planner` fell into.
_DERIVED = {
    # Cheap structured extraction that rides the synth tier.
    "market_planner": "synth",
    # The re-route classifier and the small tool extractors (corpus gate, XBRL
    # concept, EDGAR query, formula spec). Normally pinned to the free Groq
    # pool — see ROUTER_PIN — so the small Gemini RPD budget is spent on the
    # reasoning roles; this entry is only the fallback when no Groq key exists.
    "router":         "synth",
}

# The extractor calls behind `router` are one-shot structured outputs; Qwen on
# the free Groq pool is good enough for them and a Gemini call there would buy
# quality nothing downstream uses. Falls back to the context provider when no
# Groq key is configured.
ROUTER_PIN = ("groq", "qwen/qwen3.6-27b")


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """One request's LLM configuration. Frozen so a node cannot write to it."""

    provider: str = "gemini"
    # The Settings UI's two model pickers. `synth_model` overrides the
    # synthesizer family (synth + market planner; the extractors are pinned,
    # see ROUTER_PIN); `planner_model` overrides decomposition ONLY. They are separate because
    # the two jobs pull in different directions: the planner writes structured
    # retrieval keys, the synthesizer writes long-form prose, and the best model
    # for one is not always the best for the other. None → provider default.
    synth_model: Optional[str] = None
    planner_model: Optional[str] = None
    # None → `build_llm` reads the provider's env keys and rotates through the
    # pool on rate limits.
    api_key: Optional[str] = None
    # The planner may run on a DIFFERENT provider from the answer model. The
    # two jobs have opposite economics: the planner makes several small
    # structured calls where a free tier is fine, while the answer model does
    # the long-form writing worth paying for. Forcing both onto one provider
    # meant choosing a paid answer model silently moved the planner onto the
    # same paid key. None → follow `provider` / `api_key`.
    planner_provider: Optional[str] = None
    planner_api_key: Optional[str] = None
    temperature: float = 0.0
    # The client's conversation id. Scopes anything cached BETWEEN turns of one
    # chat (a fetched filing, a research finding) so it is reused within the
    # conversation and invisible to every other user. None → no cross-turn
    # caching at all, which is the safe default for the eval harness and CLIs.
    session_id: Optional[str] = None

    # Shared-tier prompt ceiling, in CHARACTERS of retrieved evidence.
    #
    # Groq's free tier rejects any single request over 8000 tokens outright —
    # HTTP 413, not a rate limit, so no wait and no other key helps. Measured
    # traces ran 7581-7815 tokens against that 8000 ceiling, i.e. 95-98%, and a
    # follow-up question's history tipped it over. Evidence is the dominant
    # term (~6.9k tokens at cap 8), so bounding it is what keeps us under.
    #
    # ~3.6 chars/token on filing text, so 14000 chars ~= 3900 tokens, leaving
    # ~4000 for the system prompt, the question, history and the answer.
    FREE_TIER_EVIDENCE_CHARS = 14_000

    def evidence_budget(self) -> Optional[int]:
        """Characters of evidence this request may carry, None = unbounded.

        Only the SHARED Groq pool is capped. A user who brought their own key
        is on their own quota with their own (usually far larger) window, so
        their requests are left exactly as they are.
        """
        if self.provider == "groq" and not self.api_key:
            return self.FREE_TIER_EVIDENCE_CHARS
        return None

    def _router_pinned(self) -> bool:
        """The extractor pin only holds if a Groq key is actually available."""
        if self.provider == ROUTER_PIN[0]:
            return True
        from finagent.llm import collect_provider_keys

        return bool(collect_provider_keys(ROUTER_PIN[0]))

    def provider_for(self, role: str) -> str:
        """Which provider serves a role. Only planner and router can differ."""
        if role == "router" and self._router_pinned():
            return ROUTER_PIN[0]
        if _DERIVED.get(role, role) == "planner" and self.planner_provider:
            return self.planner_provider
        return self.provider

    def key_for(self, role: str) -> Optional[str]:
        """The API key for a role, matched to `provider_for(role)`.

        Must move together with the provider: sending the answer model's Gemini
        key to Groq authenticates nothing.
        """
        if role == "router" and self._router_pinned():
            # A user key for another provider must not be sent to Groq; None
            # makes `build_llm` read the Groq env pool.
            return self.api_key if self.provider == ROUTER_PIN[0] else None
        if _DERIVED.get(role, role) == "planner" and self.planner_provider:
            return self.planner_api_key
        return self.api_key

    def model_for(self, role: str) -> str:
        """Resolve a role ('planner', 'synth', 'router', …) to a model name."""
        if role == "router" and self._router_pinned():
            return ROUTER_PIN[1]
        base = _DERIVED.get(role, role)
        provider = self.provider_for(role)
        if base == "synth" and self.synth_model:
            return self.synth_model
        if base == "planner" and self.planner_model:
            return self.planner_model
        try:
            return DEFAULTS[provider][base]
        except KeyError:
            raise ValueError(
                f"No default model for provider={provider!r} role={role!r}. "
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

    return build_llm(ctx.provider_for(role), ctx.model_for(role),
                     ctx.key_for(role), temperature=ctx.temperature)


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
