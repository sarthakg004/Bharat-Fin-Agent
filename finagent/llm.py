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
        raise ValueError(
            f"{API_KEY_ENV[provider]} not found. Set it in your .env file or "
            f"pass api_key= explicitly."
        )
    return key


def collect_provider_keys(provider: str) -> list[str]:
    """All API keys available for a provider, in rotation order.

    Reads `GROQ_API_KEY`, `GROQ_API_KEY2`, `GROQ_API_KEY3`, ... until one is
    missing (same pattern for the other providers). Set multiple keys in .env
    to let `RotatingChat` swap to the next one when one hits a rate limit.
    """
    provider = provider.lower()
    if provider not in API_KEY_ENV:
        raise ValueError(
            f"Unknown provider {provider!r}. Choose one of {list(API_KEY_ENV)}."
        )
    base = API_KEY_ENV[provider]
    keys: list[str] = []
    first = (os.getenv(base) or "").strip()      # strip stray newline/space
    if first:
        keys.append(first)
    i = 2
    while True:
        k = (os.getenv(f"{base}{i}") or "").strip()
        if not k:
            break
        keys.append(k)
        i += 1
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
    """
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "resourceexhausted" in name:
        return True
    msg = (str(exc) or "").lower()
    return any(hint in msg for hint in _RATE_LIMIT_HINTS)


def is_daily_quota_error(exc: BaseException) -> bool:
    """True for *daily* quota errors (TPD/RPD) specifically.

    Most providers scope these per-organisation per-model, so rotating across
    keys *within the same org* cannot recover. If you have keys from multiple
    orgs, rotation still helps until every org's daily bucket is drained — at
    which point only a 24h wait helps.
    """
    if not is_rate_limit_error(exc):
        return False
    return any(hint in (str(exc) or "").lower() for hint in _DAILY_QUOTA_HINTS)


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
        self._llm = _build_single(
            self.provider, self.chat_model, self.keys[0], **self.chat_kwargs
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
            f"[RotatingChat:{self.provider}] {type(exc).__name__} on key "
            f"{self._idx}/{len(self.keys)}; switched to key "
            f"{self._idx + 1}/{len(self.keys)}"
        )
        self._llm = _build_single(
            self.provider, self.chat_model, self.keys[self._idx], **self.chat_kwargs
        )

    def _retry(self, op):
        """Run op(self._llm) across all keys, rotating on rate-limit errors."""
        last: Any = None
        for _ in range(len(self.keys)):
            try:
                return op(self._llm)
            except Exception as e:
                last = e
                if not is_rate_limit_error(e):
                    raise
                self._rotate(e)
        raise last

    # ------------------------------------------------------------------ #
    # BaseChatModel hooks
    # ------------------------------------------------------------------ #

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        return self._retry(
            lambda llm: llm._generate(messages, stop=stop, run_manager=run_manager, **kw)
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        # Async variant — same retry loop but awaiting the inner call.
        last: Any = None
        for _ in range(len(self.keys)):
            try:
                return await self._llm._agenerate(
                    messages, stop=stop, run_manager=run_manager, **kw
                )
            except Exception as e:
                last = e
                if not is_rate_limit_error(e):
                    raise
                self._rotate(e)
        raise last

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
        last: Any = None
        for _ in range(len(self.rot.keys)):
            try:
                chain = getattr(self.rot._llm, self.method)(*self.args, **self.kw)
                return await chain.ainvoke(input, config=config, **ikw)
            except Exception as e:
                last = e
                if not is_rate_limit_error(e):
                    raise
                self.rot._rotate(e)
        raise last


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
