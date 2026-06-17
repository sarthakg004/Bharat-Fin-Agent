"""
llm.py  ·  finagent/llm.py

Single place that knows how to build a chat LLM for each supported provider.
Used by the naive RAG, the RAGAS judge, and the agentic graph so provider/key
handling stays consistent and adding a provider is a one-line change here.

Supported providers: "groq", "gemini", "openai", "anthropic".

Model choice is always the caller's: every pipeline exposes its own model
arguments (e.g. a fast planner vs. a strong synthesizer vs. a strong judge), so
you can mix providers and models freely. This module only resolves the API key
and instantiates the right client.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # so provider keys in .env are visible even if imported directly

# Provider → the .env variable holding its API key.
API_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

SUPPORTED_PROVIDERS = tuple(API_KEY_ENV)


def resolve_api_key(provider: str, api_key: Optional[str] = None) -> str:
    """Return the API key for a provider, from the argument or its env var.

    For rotation, use `collect_provider_keys` instead — this just returns the
    first key found and is used for one-shot validation.
    """
    provider = provider.lower()
    if provider not in API_KEY_ENV:
        raise ValueError(
            f"Unknown provider {provider!r}. Choose one of {list(API_KEY_ENV)}."
        )
    # .strip() guards against a stray newline/space on the key (e.g. a secret
    # created with `echo`), which would make an illegal HTTP Authorization header.
    key = (api_key or os.getenv(API_KEY_ENV[provider]) or "").strip()
    if not key:
        # The pool may be configured as numbered keys only (GROQ_API_KEY1, …)
        # with no bare GROQ_API_KEY — validate against the first pool key then.
        pool = collect_provider_keys(provider)
        if pool:
            return pool[0]
        raise ValueError(
            f"{API_KEY_ENV[provider]} not found. Set it in your .env file or "
            f"pass api_key= explicitly."
        )
    return key


# Highest numbered-key suffix scanned by `collect_provider_keys`. Scanning a
# fixed range (instead of stopping at the first gap) means GROQ_API_KEY1..N
# all load even when the bare GROQ_API_KEY is absent or the numbering skips.
_MAX_KEY_INDEX = 32


def collect_provider_keys(provider: str) -> list[str]:
    """All API keys available for a provider, in rotation order.

    Reads the bare var (`GROQ_API_KEY`) plus every numbered variant
    (`GROQ_API_KEY1` … `GROQ_API_KEY32`), tolerating gaps and either naming
    scheme, deduplicated in order. Set multiple keys in .env to let
    `RotatingChat` swap to the next one when one hits a rate limit.
    """
    provider = provider.lower()
    if provider not in API_KEY_ENV:
        raise ValueError(
            f"Unknown provider {provider!r}. Choose one of {list(API_KEY_ENV)}."
        )
    base = API_KEY_ENV[provider]
    keys: list[str] = []
    seen: set[str] = set()
    for name in (base, *(f"{base}{i}" for i in range(1, _MAX_KEY_INDEX + 1))):
        k = (os.getenv(name) or "").strip()      # strip stray newline/space
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


# --------------------------------------------------------------------------- #
# Rate-limit detection + key rotation
# --------------------------------------------------------------------------- #

_RATE_LIMIT_HINTS = (
    "rate limit", "ratelimit", "rate_limit",
    "429", "too many requests",
    "quota", "resourceexhausted", "resource exhausted",
)

_DAILY_QUOTA_HINTS = (
    "tokens per day", "per day (tpd)", "tpd",
    "requests per day", "per day (rpd)", "rpd",
    "daily quota", "daily limit",
)


def is_rate_limit_error(exc: BaseException) -> bool:
    """True if `exc` looks like a provider rate-limit / quota error.

    Covers groq.RateLimitError, openai.RateLimitError, anthropic.RateLimitError,
    google.api_core.exceptions.ResourceExhausted, and any wrapped variant that
    mentions a 429 or "rate limit" in the message.

    Walks the exception chain (``__cause__`` / ``__context__``) because by the
    time the error bubbles up through LangGraph + the thread executor the
    provider's RateLimitError is usually *wrapped* — the original check only saw
    the generic outer wrapper and so showed users a raw traceback instead of the
    friendly "limit reached" message.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    depth = 0
    while cur is not None and id(cur) not in seen and depth < 10:
        seen.add(id(cur))
        name = type(cur).__name__.lower()
        if "ratelimit" in name or "resourceexhausted" in name:
            return True
        if any(hint in (str(cur) or "").lower() for hint in _RATE_LIMIT_HINTS):
            return True
        cur = cur.__cause__ or cur.__context__
        depth += 1
    return False


_AUTH_HINTS = (
    "invalid api key", "invalid_api_key", "incorrect api key",
    "401", "unauthorized", "authentication",
)


def is_auth_error(exc: BaseException) -> bool:
    """True if `exc` looks like an invalid/revoked API key (HTTP 401).

    A multi-key pool can contain a key that has been revoked since it was
    configured; requests landing on it must not fail — `RotatingChatModel`
    drops such a key from the pool and continues on the remaining ones."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    depth = 0
    while cur is not None and id(cur) not in seen and depth < 10:
        seen.add(id(cur))
        if "authenticationerror" in type(cur).__name__.lower():
            return True
        if any(hint in (str(cur) or "").lower() for hint in _AUTH_HINTS):
            return True
        cur = cur.__cause__ or cur.__context__
        depth += 1
    return False


def is_daily_quota_error(exc: BaseException) -> bool:
    """True for *daily* quota errors (TPD/RPD) specifically.

    Most providers scope these per-organisation per-model, so rotating across
    keys *within the same org* cannot recover. If you have keys from multiple
    orgs, rotation still helps until every org's daily bucket is drained — at
    which point only a 24h wait helps.
    """
    if not is_rate_limit_error(exc):
        return False
    seen: set[int] = set()
    cur: BaseException | None = exc
    depth = 0
    while cur is not None and id(cur) not in seen and depth < 10:
        seen.add(id(cur))
        if any(hint in (str(cur) or "").lower() for hint in _DAILY_QUOTA_HINTS):
            return True
        cur = cur.__cause__ or cur.__context__
        depth += 1
    return False


# --------------------------------------------------------------------------- #
# RotatingChatModel — a real BaseChatModel that rotates keys on rate-limits
# --------------------------------------------------------------------------- #

# Imports kept module-level so the class body resolves; the langchain_core
# pieces are cheap and present whenever any chat client is installed.
from typing import Any, List   # noqa: E402  (deliberate placement)

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.outputs import ChatResult  # noqa: E402
from langchain_core.runnables import Runnable  # noqa: E402
from pydantic import Field, PrivateAttr  # noqa: E402


class AllKeysExhaustedError(Exception):
    """Every configured key for a provider is rate-limited right now.

    Raised after a full rotation cycle fails, and immediately (no network)
    while the provider is in cooldown — so the request fails FAST instead of
    every node grinding through the whole key pool again. The message contains
    'rate limit' so `is_rate_limit_error` classifies it, and the original
    provider error is chained (`from`) so `is_daily_quota_error` still works.
    """


# Provider → unix time until which every key is considered exhausted. Shared
# process-wide: once one call proves the whole pool is limited, every other
# call short-circuits for the cooldown window instead of re-proving it.
_EXHAUSTED_UNTIL: dict[str, float] = {}
_EXHAUST_COOLDOWN_S = 60.0

# Total exhaustion events per provider across this process lifetime.
# Incremented every time _exhausted() fires (i.e., every full rotation cycle
# that found every key rate-limited). Callers can snapshot this counter before
# a batch and compare after to detect how many full-cycle failures occurred.
_EXHAUST_COUNT: dict[str, int] = {}


# Round-robin start-index per provider: each RotatingChatModel instance (one
# per LLM role — planner, synth, critic, grader, …) starts on a DIFFERENT key,
# so the roles spread across the key pool instead of all hammering key 1 and
# rotating together on the same 429.
_START_COUNTERS: dict[str, int] = {}


def _next_start_index(provider: str, n_keys: int) -> int:
    if n_keys <= 1:
        return 0
    idx = _START_COUNTERS.get(provider, 0)
    _START_COUNTERS[provider] = idx + 1
    return idx % n_keys


class RotatingChatModel(BaseChatModel):
    """A `BaseChatModel` holding N keys for one provider/model that swaps to
    the next key on rate-limit errors.

    Because it IS a `BaseChatModel`, it works wherever a real chat model is
    expected — including ragas' `LangchainLLMWrapper`, which type-checks.

    Rotation strategy
    -----------------
    * Per-minute rate limit on key K → rotate to K+1 (often immediate recovery).
    * Daily-quota error (TPD/RPD) on key K → rotate to K+1; if every key is
      from the same organisation those will all share the bucket and we'll
      cycle through to the original error in a few seconds; if your keys are
      from different orgs each has its own bucket. Either way, after exhausting
      every key we re-raise the last exception. Use `is_daily_quota_error()`
      on the raised exception to decide whether to wait 24h vs minutes.
    """

    provider: str = Field(description="groq | gemini | openai | anthropic")
    chat_model: str = Field(description="Underlying model name")
    keys: List[str] = Field(description="API keys to rotate through, in order")
    chat_kwargs: dict = Field(
        default_factory=dict,
        description="Extra kwargs forwarded to the inner chat client",
    )

    _idx: int = PrivateAttr(default=0)
    _llm: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:  # pydantic v2
        if not self.keys:
            raise ValueError("RotatingChatModel needs at least one key")
        self._idx = _next_start_index(self.provider, len(self.keys))
        self._llm = _build_single(
            self.provider, self.chat_model, self.keys[self._idx], **self.chat_kwargs
        )

    @property
    def _llm_type(self) -> str:
        return f"rotating-{self.provider}"

    # ------------------------------------------------------------------ #
    # Rotation
    # ------------------------------------------------------------------ #

    def _rotate(self, exc: BaseException) -> None:
        self._idx = (self._idx + 1) % len(self.keys)
        print(
            f"[RotatingChat:{self.provider}] {type(exc).__name__}; switched to "
            f"key {self._idx + 1}/{len(self.keys)}"
        )
        self._llm = _build_single(
            self.provider, self.chat_model, self.keys[self._idx], **self.chat_kwargs
        )

    def _drop_current_key(self, exc: BaseException) -> None:
        """Remove a revoked/invalid key from the pool and continue on the rest.
        Unlike a rate limit (transient), a 401 never recovers — leaving the key
        in rotation would keep paying a failed round-trip on every cycle."""
        bad = self.keys.pop(self._idx)
        self._idx %= len(self.keys)
        print(
            f"[RotatingChat:{self.provider}] {type(exc).__name__}: dropped "
            f"invalid key …{bad[-4:]}; {len(self.keys)} key(s) remain"
        )
        self._llm = _build_single(
            self.provider, self.chat_model, self.keys[self._idx], **self.chat_kwargs
        )

    def _recover(self, exc: BaseException) -> bool:
        """Try to recover from `exc` by switching keys. False → not recoverable."""
        if is_rate_limit_error(exc):
            self._rotate(exc)
            return True
        if is_auth_error(exc) and len(self.keys) > 1:
            self._drop_current_key(exc)
            return True
        return False

    def _check_cooldown(self) -> None:
        """Fail fast while the whole pool is known-exhausted (no network)."""
        until = _EXHAUSTED_UNTIL.get(self.provider, 0.0)
        left = until - time.time()
        if left > 0:
            raise AllKeysExhaustedError(
                f"All {self.provider} API keys hit their rate limit; "
                f"cooling down for another {int(left) + 1}s."
            )

    def _exhausted(self, last: BaseException) -> AllKeysExhaustedError:
        """A full rotation cycle failed — latch the cooldown for the provider."""
        _EXHAUSTED_UNTIL[self.provider] = time.time() + _EXHAUST_COOLDOWN_S
        _EXHAUST_COUNT[self.provider] = _EXHAUST_COUNT.get(self.provider, 0) + 1
        print(f"[RotatingChat:{self.provider}] every key rate-limited; "
              f"failing fast for {_EXHAUST_COOLDOWN_S:.0f}s "
              f"(exhaustion #{_EXHAUST_COUNT[self.provider]} this session)")
        return AllKeysExhaustedError(
            f"All {len(self.keys)} {self.provider} API keys hit their rate "
            f"limit ({type(last).__name__})."
        )

    def _retry(self, op):
        """Run op(self._llm) across all keys, switching on recoverable errors."""
        self._check_cooldown()
        last: Any = None
        for _ in range(max(1, len(self.keys))):
            try:
                return op(self._llm)
            except Exception as e:
                last = e
                if not self._recover(e):
                    raise
        raise self._exhausted(last) from last

    # ------------------------------------------------------------------ #
    # BaseChatModel hooks
    # ------------------------------------------------------------------ #

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        return self._retry(
            lambda llm: llm._generate(messages, stop=stop, run_manager=run_manager, **kw)
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        # Async variant — same retry loop but awaiting the inner call.
        self._check_cooldown()
        last: Any = None
        for _ in range(max(1, len(self.keys))):
            try:
                return await self._llm._agenerate(
                    messages, stop=stop, run_manager=run_manager, **kw
                )
            except Exception as e:
                last = e
                if not self._recover(e):
                    raise
        raise self._exhausted(last) from last

    # ------------------------------------------------------------------ #
    # Structured-output / tool-binding — must rebuild the chain inside the
    # retry loop so a rotated `_llm` is actually used after a swap.
    # ------------------------------------------------------------------ #

    def with_structured_output(self, schema, **so_kw) -> Runnable:
        return _RotatingBound(self, "with_structured_output", (schema,), so_kw)

    def bind_tools(self, tools, **kw) -> Runnable:
        return _RotatingBound(self, "bind_tools", (tools,), kw)


class _RotatingBound(Runnable):
    """Re-derives `with_structured_output(...)` / `bind_tools(...)` from the
    (possibly rotated) inner LLM on every invoke, so rotation propagates."""

    def __init__(self, rot: RotatingChatModel, method: str, args: tuple, kw: dict):
        self.rot, self.method, self.args, self.kw = rot, method, args, kw

    def invoke(self, input, config=None, **ikw):
        return self.rot._retry(
            lambda llm: getattr(llm, self.method)(*self.args, **self.kw)
            .invoke(input, config=config, **ikw)
        )

    async def ainvoke(self, input, config=None, **ikw):
        self.rot._check_cooldown()
        last: Any = None
        for _ in range(max(1, len(self.rot.keys))):
            try:
                chain = getattr(self.rot._llm, self.method)(*self.args, **self.kw)
                return await chain.ainvoke(input, config=config, **ikw)
            except Exception as e:
                last = e
                if not self.rot._recover(e):
                    raise
        raise self.rot._exhausted(last) from last


def build_llm(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 3,
    rotate: bool = True,
):
    """Instantiate a LangChain chat model for the given provider + model.

    Key handling:
      * If `api_key` is given → use exactly that key, no rotation.
      * Else → collect every `{ENV}`, `{ENV}2`, … from .env for the provider.
        If multiple keys are found AND `rotate=True`, return a `RotatingChat`
        that swaps to the next key on rate-limit errors; otherwise return a
        plain chat model built with the first key.

    Pass `rotate=False` when the caller needs a real `BaseChatModel` (e.g.
    ragas `LangchainLLMWrapper`, which type-checks).
    """
    provider = provider.lower()
    if provider not in API_KEY_ENV:
        raise ValueError(
            f"Unknown provider {provider!r}. Choose one of {list(API_KEY_ENV)}."
        )

    if api_key:
        keys = [api_key.strip()]
    else:
        keys = collect_provider_keys(provider)
        if not keys:
            raise ValueError(
                f"{API_KEY_ENV[provider]} not found. Set it in your .env file "
                f"or pass api_key= explicitly."
            )

    if rotate and len(keys) > 1:
        # Internal retries on the same key just delay rotation; keep them low.
        return RotatingChatModel(
            provider=provider,
            chat_model=model,
            keys=keys,
            chat_kwargs={
                "temperature": temperature,
                "max_retries": min(max_retries, 2),
            },
        )
    return _build_single(provider, model, keys[0],
                         temperature=temperature, max_retries=max_retries)


def _build_single(provider: str, model: str, api_key: str,
                  *, temperature: float = 0.0, max_retries: int = 3):
    """One chat instance for one key. Imports are local per provider."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key,
            temperature=temperature, max_retries=max_retries,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model, api_key=api_key,
            temperature=temperature, max_retries=max_retries,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, api_key=api_key,
            temperature=temperature, max_retries=max_retries,
        )
    # default: groq
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model, api_key=api_key,
        temperature=temperature, max_retries=max_retries,
    )
