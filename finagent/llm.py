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
import re
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

    Reads three forms, deduplicated in order:
      * the consolidated var (`GROQ_API_KEYS`) — the whole pool as a
        comma/whitespace-separated list;
      * the bare var (`GROQ_API_KEY`);
      * every numbered variant (`GROQ_API_KEY1` … `GROQ_API_KEY32`).
    It tolerates gaps and either naming scheme. Prefer the consolidated form in
    the cloud: Secret Manager bills per active secret *version*, so keeping the
    whole pool in ONE secret (instead of one secret per key) keeps it inside the
    free tier. Set multiple keys to let `RotatingChat` swap to the next one when
    one hits a rate limit.
    """
    provider = provider.lower()
    if provider not in API_KEY_ENV:
        raise ValueError(
            f"Unknown provider {provider!r}. Choose one of {list(API_KEY_ENV)}."
        )
    base = API_KEY_ENV[provider]
    keys: list[str] = []
    seen: set[str] = set()
    # Consolidated pool (e.g. GROQ_API_KEYS="k1,k2,...") first, then the bare +
    # numbered forms for back-compat.
    consolidated = re.split(r"[,\s]+", os.getenv(f"{base}S") or "")
    numbered = (os.getenv(name) or ""
                for name in (base, *(f"{base}{i}"
                                     for i in range(1, _MAX_KEY_INDEX + 1))))
    for raw in (*consolidated, *numbered):
        k = raw.strip()                          # strip stray newline/space
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


def text_of(response) -> str:
    """Plain text of a chat response's `.content`.

    Most models return a str, but Gemini 3.x returns a LIST of typed content
    blocks (text blocks carry thought signatures in `extras`); any consumer
    that regexes or strips the answer must flatten through this instead of
    using `.content` raw.
    """
    c = getattr(response, "content", response)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(
            (b.get("text") or "") if isinstance(b, dict) else str(b)
            for b in c
            if not (isinstance(b, dict) and b.get("type") in ("thinking", "reasoning"))
        )
    return str(c or "")


def _chain(exc: BaseException):
    """Walk `exc` and its ``__cause__`` / ``__context__`` ancestry.

    Every classifier below needs this: by the time a provider error bubbles up
    through LangGraph and the thread executor it is usually *wrapped*, and a
    check that only inspects the outer wrapper shows users a raw traceback
    instead of the friendly "limit reached" message. Bounded and cycle-safe.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen and len(seen) < 10:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def is_rate_limit_error(exc: BaseException) -> bool:
    """True if `exc` looks like a provider rate-limit / quota error.

    Covers groq.RateLimitError, openai.RateLimitError, anthropic.RateLimitError,
    google.api_core.exceptions.ResourceExhausted, and any wrapped variant that
    mentions a 429 or "rate limit" in the message.
    """
    return any(
        "ratelimit" in type(c).__name__.lower()
        or "resourceexhausted" in type(c).__name__.lower()
        or any(hint in (str(c) or "").lower() for hint in _RATE_LIMIT_HINTS)
        for c in _chain(exc)
    )


_AUTH_HINTS = (
    "invalid api key", "invalid_api_key", "incorrect api key",
    "401", "unauthorized", "authentication",
)


def is_auth_error(exc: BaseException) -> bool:
    """True if `exc` looks like an invalid/revoked API key (HTTP 401).

    A multi-key pool can contain a key that has been revoked since it was
    configured; requests landing on it must not fail — `RotatingChatModel`
    drops such a key from the pool and continues on the remaining ones."""
    return any(
        "authenticationerror" in type(c).__name__.lower()
        or any(hint in (str(c) or "").lower() for hint in _AUTH_HINTS)
        for c in _chain(exc)
    )


def is_daily_quota_error(exc: BaseException) -> bool:
    """True for *daily* quota errors (TPD/RPD) specifically.

    Most providers scope these per-organisation per-model, so rotating across
    keys *within the same org* cannot recover. If you have keys from multiple
    orgs, rotation still helps until every org's daily bucket is drained — at
    which point only a 24h wait helps.
    """
    return is_rate_limit_error(exc) and any(
        hint in (str(c) or "").lower()
        for c in _chain(exc) for hint in _DAILY_QUOTA_HINTS
    )


# --------------------------------------------------------------------------- #
# "How long until it clears?" — reading the reset off the 429 itself
# --------------------------------------------------------------------------- #

# Go-style durations, which is what Groq emits: "7.66s", "2m59.56s", "500ms".
_DUR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)")
_DUR_MULT = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
_TRY_AGAIN_RE = re.compile(r"try again in\s+([\d.hms\s]+)", re.I)

# Per-bucket reset headers, longest-lived bucket first. Used only as a fallback:
# they say when EACH bucket refills, not which one tripped, so we take the first
# that is present rather than guessing — see `retry_after_seconds`.
_RESET_HEADERS = ("retry-after", "x-ratelimit-reset-requests",
                  "x-ratelimit-reset-tokens")


def parse_duration(text: str) -> Optional[float]:
    """Seconds from a Go-style duration ("7.66s", "2m59.56s", "1h2m3s") or a
    bare number ("30", the plain HTTP `Retry-After` form). None if unparseable.
    """
    t = (text or "").strip()
    if not t:
        return None
    parts = _DUR_RE.findall(t)
    if parts:
        return sum(float(n) * _DUR_MULT[u] for n, u in parts)
    try:
        return float(t)
    except ValueError:
        return None


def _headers_of(exc: BaseException) -> dict[str, str]:
    """Lower-cased response headers off an SDK error, or {} if it has none."""
    resp = getattr(exc, "response", None)
    raw = getattr(resp, "headers", None) or getattr(exc, "headers", None)
    try:
        return {str(k).lower(): str(v) for k, v in dict(raw).items()}
    except Exception:
        return {}


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """How long until this provider error clears, in seconds — None if unknown.

    Providers already answer this on the 429 and we were throwing it away, so
    the UI could only say "wait a minute" whether the true wait was 8 seconds
    or 3 hours. Groq (and every OpenAI-shaped SDK) puts it in two places:

      * the body — "Please try again in 2m59.56s", computed against whichever
        limit ACTUALLY tripped;
      * the headers — `retry-after` plus per-bucket
        `x-ratelimit-reset-{requests,tokens}`.

    The body wins because it names the bucket that failed; the headers report
    every bucket's refill regardless of which one was hit, so reading the wrong
    one would tell the user to retry too early. `AllKeysExhaustedError` carries
    a pre-resolved value, so a latched cooldown answers without re-parsing.
    """
    for cur in _chain(exc):
        cached = getattr(cur, "retry_after", None)
        if isinstance(cached, (int, float)) and not isinstance(cached, bool):
            return float(cached)
        m = _TRY_AGAIN_RE.search(str(cur) or "")
        if m and (secs := parse_duration(m.group(1))) is not None:
            return secs
        headers = _headers_of(cur)
        for name in _RESET_HEADERS:
            if name in headers and (secs := parse_duration(headers[name])) is not None:
                return secs
    return None


def format_wait(seconds: float) -> str:
    """A wait a human reads at a glance: "8s", "3m 0s", "2h 5m"."""
    s = max(0, int(seconds + 0.5))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {s % 3600 // 60}m"


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

    `retry_after` is the pool's real reset in seconds when the provider told us
    (see `retry_after_seconds`), else None.
    """

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


# Provider → unix time until which every key is considered exhausted. Shared
# process-wide: once one call proves the whole pool is limited, every other
# call short-circuits for the cooldown window instead of re-proving it.
_EXHAUSTED_UNTIL: dict[str, float] = {}
_EXHAUST_COOLDOWN_S = 60.0
# Ceiling on a provider-reported cooldown. A drained DAILY bucket resets hours
# out, and latching that process-wide would keep failing fast long after keys
# from another org (whose day rolls over separately) have recovered. The user
# still sees the true reset — only the internal fail-fast window is clamped.
_EXHAUST_COOLDOWN_MAX_S = 300.0

# Provider → unix time the provider itself said its limit clears. Separate from
# the clamped fail-fast window above because this is the number shown to the
# user, and it must stay honest even when it is three hours away.
_RESET_AT: dict[str, float] = {}

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

    def _get_ls_params(self, stop: Optional[List[str]] = None, **kwargs):
        """Standard LangChain tracing params. Delegate to the inner client so
        tracers (Langfuse/LangSmith) attribute each generation to the REAL
        model name — enabling per-model cost/latency analytics — instead of
        seeing only this wrapper class."""
        try:
            return self._llm._get_ls_params(stop=stop, **kwargs)
        except Exception:
            from langchain_core.language_models.chat_models import LangSmithParams
            return LangSmithParams(ls_provider=self.provider,
                                   ls_model_name=self.chat_model,
                                   ls_model_type="chat")

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

    def _note_failure(self, exc: BaseException, waits: list) -> bool:
        """Record what the provider said about the reset, then try to recover.

        Called from every rotation loop so `_exhausted` can report a real
        countdown instead of a guess. False → the error is not recoverable.
        """
        wait = retry_after_seconds(exc)
        if wait is not None:
            waits.append(wait)
        return self._recover(exc)

    def _check_cooldown(self) -> None:
        """Fail fast while the whole pool is known-exhausted (no network)."""
        left = _EXHAUSTED_UNTIL.get(self.provider, 0.0) - time.time()
        if left > 0:
            # Report the provider's own reset when we have it; the latch may be
            # shorter (clamped) but the user wants the truth, not our schedule.
            true_left = max(_RESET_AT.get(self.provider, 0.0) - time.time(), left)
            raise AllKeysExhaustedError(
                f"All {self.provider} API keys hit their rate limit; "
                f"resets in {format_wait(true_left)}.",
                retry_after=true_left,
            )

    def _exhausted(self, last: BaseException,
                   waits: "list | tuple" = ()) -> AllKeysExhaustedError:
        """A full rotation cycle failed — latch the cooldown for the provider.

        The pool recovers when its EARLIEST key does, so take the min of what
        each key's 429 reported.
        """
        wait = min(waits) if waits else None
        cooldown = (_EXHAUST_COOLDOWN_S if wait is None
                    else min(max(wait, 1.0), _EXHAUST_COOLDOWN_MAX_S))
        now = time.time()
        _EXHAUSTED_UNTIL[self.provider] = now + cooldown
        if wait is not None:
            _RESET_AT[self.provider] = now + wait
        _EXHAUST_COUNT[self.provider] = _EXHAUST_COUNT.get(self.provider, 0) + 1
        print(f"[RotatingChat:{self.provider}] every key rate-limited; "
              f"failing fast for {cooldown:.0f}s "
              f"(provider reset: {format_wait(wait) if wait is not None else 'unknown'}; "
              f"exhaustion #{_EXHAUST_COUNT[self.provider]} this session)")
        return AllKeysExhaustedError(
            f"All {len(self.keys)} {self.provider} API keys hit their rate "
            f"limit ({type(last).__name__}).",
            retry_after=wait,
        )

    def _retry(self, op):
        """Run op(self._llm) across all keys, switching on recoverable errors."""
        self._check_cooldown()
        last: Any = None
        waits: list[float] = []
        for _ in range(max(1, len(self.keys))):
            try:
                return op(self._llm)
            except Exception as e:
                last = e
                if not self._note_failure(e, waits):
                    raise
        raise self._exhausted(last, waits) from last

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
        waits: list[float] = []
        for _ in range(max(1, len(self.keys))):
            try:
                return await self._llm._agenerate(
                    messages, stop=stop, run_manager=run_manager, **kw
                )
            except Exception as e:
                last = e
                if not self._note_failure(e, waits):
                    raise
        raise self._exhausted(last, waits) from last

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
        waits: list[float] = []
        for _ in range(max(1, len(self.rot.keys))):
            try:
                chain = getattr(self.rot._llm, self.method)(*self.args, **self.kw)
                return await chain.ainvoke(input, config=config, **ikw)
            except Exception as e:
                last = e
                if not self.rot._note_failure(e, waits):
                    raise
        raise self.rot._exhausted(last, waits) from last


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

    kwargs = {}
    if model.startswith("qwen/"):
        # Qwen 3.x are reasoning models: without this they prepend a <think>…
        # block to every plain-text completion, which would leak into rewritten
        # queries and judged answers. "hidden" strips it server-side.
        kwargs["reasoning_format"] = "hidden"
    return ChatGroq(
        model=model, api_key=api_key,
        temperature=temperature, max_retries=max_retries, **kwargs,
    )
