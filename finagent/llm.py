"""Chat models for every provider, with key rotation and error classification.

One key per provider is enough to run. With several keys, a key that hits a
limit is swapped for the next one, and the request only fails once every key
has failed.
"""

from __future__ import annotations

import asyncio
import copy
import os
import re
import time
from dataclasses import dataclass
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, PrivateAttr

load_dotenv()

# Provider -> the env var holding its key. Cohere is the reranker, not a chat
# provider; it is listed so the reranker can reuse `collect_provider_keys`.
API_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "cohere": "COHERE_API_KEY",
}
CHAT_PROVIDERS = ("groq", "gemini", "openai", "anthropic")
PROVIDER_LABELS = {"groq": "Groq", "gemini": "Gemini", "openai": "OpenAI",
                   "anthropic": "Anthropic"}


def collect_provider_keys(provider: str) -> list[str]:
    """Every key configured for a provider, in order, without duplicates.

    Reads `GROQ_API_KEYS` (comma or space separated), `GROQ_API_KEY`, and
    `GROQ_API_KEY1` .. `GROQ_API_KEY32`. In the cloud use the first form: one
    secret holds the whole pool.
    """
    base = API_KEY_ENV[provider.lower()]
    raw = re.split(r"[,\s]+", os.getenv(f"{base}S") or "")
    raw += [os.getenv(name) or "" for name in (base, *(f"{base}{i}" for i in range(1, 33)))]
    return list(dict.fromkeys(k.strip() for k in raw if k.strip()))


def text_of(response) -> str:
    """The plain text of a chat response. Gemini returns a list of blocks."""
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


# --------------------------------------------------------------------------- #
# Error classification
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ErrorInfo:
    """What went wrong with a provider call.

    kind: rate_limit (per-minute, clears in seconds) | quota (daily, clears
    tomorrow) | busy (provider overloaded or unreachable) | too_large (prompt
    over the model's limit) | auth (bad key) | not_found (model gone) | other.
    """
    kind: str
    retry_after: Optional[float] = None
    status: Optional[int] = None

    @property
    def retryable(self) -> bool:
        return self.kind in ("rate_limit", "busy")


class ProviderError(Exception):
    """Every key for one provider failed. Carries the classified reason."""

    def __init__(self, message: str, kind: str, provider: str,
                 retry_after: Optional[float] = None):
        super().__init__(message)
        self.kind, self.provider, self.retry_after = kind, provider, retry_after


# A 429 means two different things. These words mark the daily one; Gemini only
# says so in a camelCase quota id ("...PerDayPerProjectPerModel-FreeTier").
_DAILY_HINTS = ("perday", "per day", "tokens per day", "requests per day",
                "(tpd)", "(rpd)", "daily quota", "daily limit")
_RATE_HINTS = ("rate limit", "ratelimit", "rate_limit", "too many requests",
               "quota", "resource exhausted", "resource_exhausted")
# Groq reports an over-sized prompt as HTTP 413 with code "rate_limit_exceeded",
# so this is checked before the rate-limit hints.
_TOO_LARGE_HINTS = ("request too large", "reduce your message size",
                    "please reduce the length", "context length", "maximum context",
                    "string too long", "prompt is too long")
_AUTH_HINTS = ("invalid api key", "invalid_api_key", "incorrect api key",
               "api key not valid", "api_key_invalid", "unauthorized")
_NOT_FOUND_HINTS = ("model_not_found", "does not exist", "is not found for api version")
_BUSY_HINTS = ("overloaded", "service unavailable", "internal server error",
               "bad gateway", "timed out", "timeout", "connection error",
               "temporarily unavailable")


def _chain(exc: BaseException):
    """`exc` and the exceptions it wraps. Provider errors arrive wrapped."""
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen and len(seen) < 10:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def _status_of(exc: BaseException) -> Optional[int]:
    """The HTTP status an SDK error carries, if any."""
    for c in _chain(exc):
        for holder in (c, getattr(c, "response", None)):
            for attr in ("status_code", "code", "status"):
                v = getattr(holder, attr, None)
                if isinstance(v, int) and not isinstance(v, bool) and 400 <= v < 600:
                    return v
    return None


# "try again in 2m59.56s" (Groq), "retry in 10.7s" / retryDelay: "10s" (Gemini).
_TRY_AGAIN_RE = re.compile(
    r"(?:try again in|retry in|retrydelay[\"']?\s*:\s*[\"']?)\s*([\d.hms]+)", re.I)
_DUR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)")
_DUR_MULT = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_duration(text: str) -> Optional[float]:
    """Seconds from "7.66s", "2m59.56s", "1h2m" or a bare number."""
    t = (text or "").strip()
    parts = _DUR_RE.findall(t)
    if parts:
        return sum(float(n) * _DUR_MULT[u] for n, u in parts)
    try:
        return float(t)
    except ValueError:
        return None


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """How long the provider says to wait, from the error body or headers."""
    for c in _chain(exc):
        if isinstance(getattr(c, "retry_after", None), (int, float)):
            return float(c.retry_after)
        m = _TRY_AGAIN_RE.search(str(c) or "")
        if m and (secs := parse_duration(m.group(1))) is not None:
            return secs
        headers = getattr(getattr(c, "response", None), "headers", None) or {}
        try:
            secs = parse_duration(dict(headers).get("retry-after", ""))
        except Exception:
            secs = None
        if secs is not None:
            return secs
    return None


def classify_error(exc: BaseException) -> ErrorInfo:
    """Turn any provider exception into an `ErrorInfo`.

    The HTTP status is checked first and the message second, because providers
    reuse one status for different problems (429 is both "slow down" and "come
    back tomorrow").
    """
    for c in _chain(exc):
        if isinstance(c, ProviderError):
            return ErrorInfo(c.kind, c.retry_after)

    status = _status_of(exc)
    text = " ".join(str(c) for c in _chain(exc)).lower()
    names = " ".join(type(c).__name__.lower() for c in _chain(exc))

    def has(hints) -> bool:
        return any(h in text for h in hints)

    if status == 413 or has(_TOO_LARGE_HINTS):
        return ErrorInfo("too_large", status=status)
    if (status == 429 or "ratelimit" in names or "resourceexhausted" in names
            or has(_RATE_HINTS) or re.search(r"\b429\b", text)):
        if has(_DAILY_HINTS):
            return ErrorInfo("quota", status=status)
        return ErrorInfo("rate_limit", retry_after_seconds(exc), status)
    if status in (401, 403) or "authentication" in names or has(_AUTH_HINTS):
        return ErrorInfo("auth", status=status)
    if status == 404 or "notfound" in names or has(_NOT_FOUND_HINTS):
        return ErrorInfo("not_found", status=status)
    if ((status or 0) >= 500 or "timeout" in names or "connection" in names
            or has(_BUSY_HINTS)):
        return ErrorInfo("busy", retry_after_seconds(exc), status)
    return ErrorInfo("other", status=status)


def format_wait(seconds: float) -> str:
    """"8s", "3m 0s", "2h 5m"."""
    s = max(0, int(seconds + 0.5))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {s % 3600 // 60}m"


# --------------------------------------------------------------------------- #
# Key rotation
# --------------------------------------------------------------------------- #

# How long one request may sleep waiting for a per-minute limit to clear, and
# how many times. The eval raises both through the environment.
MAX_INLINE_WAIT_S = float(os.getenv("LLM_MAX_INLINE_WAIT_S", "45"))
MAX_WAIT_RETRIES = int(os.getenv("LLM_MAX_WAIT_RETRIES", "1"))

# Each new model starts on the next key, so the roles spread across the pool.
_next_start: dict[str, int] = {}


class RotatingChatModel(BaseChatModel):
    """A chat model that holds a pool of keys and moves on when one fails."""

    provider: str = Field(description="groq | gemini | openai | anthropic")
    chat_model: str
    keys: List[str]
    chat_kwargs: dict = Field(default_factory=dict)

    _idx: int = PrivateAttr(default=0)
    _llm: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        if not self.keys:
            raise ValueError("RotatingChatModel needs at least one key")
        n = _next_start.get(self.provider, 0)
        _next_start[self.provider] = n + 1
        self._idx = n % len(self.keys)
        self._build()

    def _build(self) -> None:
        self._llm = _build_single(self.provider, self.chat_model,
                                  self.keys[self._idx], **self.chat_kwargs)

    @property
    def _llm_type(self) -> str:
        return f"rotating-{self.provider}"

    def _get_ls_params(self, stop: Optional[List[str]] = None, **kwargs):
        # Let tracers see the real model name, not this wrapper.
        return self._llm._get_ls_params(stop=stop, **kwargs)

    def _handle(self, exc: BaseException, waits: list) -> str:
        """Move to another key after a failure. Re-raises when no key can help."""
        info = classify_error(exc)
        if info.kind == "auth" and len(self.keys) > 1:
            self.keys.pop(self._idx)            # a dead key never recovers
            self._idx %= len(self.keys)
        elif info.kind in ("rate_limit", "quota", "busy"):
            self._idx = (self._idx + 1) % len(self.keys)
            if info.kind != "quota":
                waits.append(info.retry_after)
        elif info.kind == "other":
            raise exc                           # not a provider problem (e.g. unparseable output)
        else:                                   # too_large, not_found, or a bad single key
            label = PROVIDER_LABELS.get(self.provider, self.provider)
            raise ProviderError(f"{label} rejected the request ({info.kind}) on "
                                f"{self.chat_model}.", info.kind, self.provider) from exc
        self._build()
        return info.kind

    @staticmethod
    def _pool_wait(kinds: list, waits: list) -> Optional[float]:
        """Seconds to wait before trying the pool again, or None to give up."""
        if all(k == "quota" for k in kinds):
            return None                          # daily limit: waiting will not help
        known = [w for w in waits if w is not None]
        wait = min(known) if known else (2.0 if "busy" in kinds else None)
        return wait if wait is not None and wait <= MAX_INLINE_WAIT_S else None

    def _exhausted(self, kinds: list, waits: list) -> ProviderError:
        kind = ("quota" if all(k == "quota" for k in kinds)
                else "busy" if all(k == "busy" for k in kinds) else "rate_limit")
        known = [w for w in waits if w is not None]
        label = PROVIDER_LABELS.get(self.provider, self.provider)
        return ProviderError(
            f"Every {label} key failed ({kind}) on {self.chat_model}.", kind,
            self.provider, min(known) if known and kind != "quota" else None)

    def _run(self, op):
        """Call `op(llm)`, trying each key, then waiting once for a per-minute limit."""
        last: Any = None
        kinds: list = []
        waits: list = []
        for attempt in range(MAX_WAIT_RETRIES + 1):
            kinds, waits = [], []
            for _ in range(len(self.keys)):
                try:
                    return op(self._llm)
                except Exception as e:
                    last = e
                    kinds.append(self._handle(e, waits))
            wait = self._pool_wait(kinds, waits)
            if wait is None or attempt == MAX_WAIT_RETRIES:
                break
            print(f"[llm] {self.chat_model}: every key is limited, waiting {wait:.0f}s", flush=True)
            time.sleep(wait + 0.5)
        raise self._exhausted(kinds, waits) from last

    async def _arun(self, op):
        """Async twin of `_run` (RAGAS calls the judge asynchronously)."""
        last: Any = None
        kinds: list = []
        waits: list = []
        for attempt in range(MAX_WAIT_RETRIES + 1):
            kinds, waits = [], []
            for _ in range(len(self.keys)):
                try:
                    return await op(self._llm)
                except Exception as e:
                    last = e
                    kinds.append(self._handle(e, waits))
            wait = self._pool_wait(kinds, waits)
            if wait is None or attempt == MAX_WAIT_RETRIES:
                break
            print(f"[llm] {self.chat_model}: every key is limited, waiting {wait:.0f}s", flush=True)
            await asyncio.sleep(wait + 0.5)
        raise self._exhausted(kinds, waits) from last

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        return self._run(lambda llm: llm._generate(
            messages, stop=stop, run_manager=run_manager, **kw))

    async def _agenerate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        return await self._arun(lambda llm: llm._agenerate(
            messages, stop=stop, run_manager=run_manager, **kw))

    def with_structured_output(self, schema, **kw) -> Runnable:
        return _Structured(self, schema, kw)


def _strict_schema(model) -> dict:
    """A pydantic model's JSON schema with every field required.

    Qwen on Groq skips any field that has a default and returns a nearly empty
    object. Marking all fields required (Groq's strict mode) makes it fill them.
    Measured on 11 extraction cases x 4 runs: 40 of 44 correct, against 28 of 44
    with tool calling.
    """
    schema = copy.deepcopy(model.model_json_schema())

    def fix(node):
        if isinstance(node, dict):
            if "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            node.pop("default", None)
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)

    fix(schema)
    return schema


class _Structured(Runnable):
    """`with_structured_output` rebuilt on every call, so it follows key rotation."""

    def __init__(self, rot: RotatingChatModel, schema, kw: dict):
        self.rot, self.schema, self.kw = rot, schema, kw

    def _strict(self, llm):
        return llm.with_structured_output(
            _strict_schema(self.schema), method="json_schema", strict=True
        ) | self.schema.model_validate

    def _is_qwen(self) -> bool:
        return (self.rot.provider == "groq" and self.rot.chat_model.startswith("qwen/")
                and isinstance(self.schema, type) and issubclass(self.schema, BaseModel))

    def invoke(self, input, config=None, **ikw):
        def op(llm):
            if self._is_qwen():
                try:
                    return self._strict(llm).invoke(input, config=config, **ikw)
                except Exception as e:
                    if classify_error(e).kind != "other":
                        raise               # a provider problem: let rotation handle it
                    # Qwen wrote invalid JSON: fall through to tool calling.
            return llm.with_structured_output(self.schema, **self.kw).invoke(
                input, config=config, **ikw)
        return self.rot._run(op)

    async def ainvoke(self, input, config=None, **ikw):
        async def op(llm):
            if self._is_qwen():
                try:
                    return await self._strict(llm).ainvoke(input, config=config, **ikw)
                except Exception as e:
                    if classify_error(e).kind != "other":
                        raise
            return await llm.with_structured_output(self.schema, **self.kw).ainvoke(
                input, config=config, **ikw)
        return await self.rot._arun(op)


def build_llm(provider: str, model: str, api_key: Optional[str] = None,
              temperature: float = 0.0, **model_kwargs) -> RotatingChatModel:
    """A chat model for `provider`/`model`.

    With `api_key`, only that key is used. Without it, every key configured for
    the provider is loaded and rotated.
    """
    provider = provider.lower()
    if provider not in CHAT_PROVIDERS:
        raise ValueError(f"Unknown chat provider {provider!r}. Choose one of {CHAT_PROVIDERS}.")
    keys = [api_key.strip()] if api_key else collect_provider_keys(provider)
    if not keys:
        raise ValueError(f"No {PROVIDER_LABELS[provider]} API key. Set "
                         f"{API_KEY_ENV[provider]} or pass one with the request.")
    return RotatingChatModel(
        provider=provider, chat_model=model, keys=keys,
        chat_kwargs={"temperature": temperature, "max_retries": 1, **model_kwargs})


def _build_single(provider: str, model: str, api_key: str, **kw):
    """One LangChain chat client for one key."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, **kw)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key, **kw)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=api_key, **kw)
    from langchain_groq import ChatGroq
    if model.startswith("qwen/"):
        # When Qwen is asked to reason (`reasoning_effort`, set per role in
        # runtime.py), "hidden" keeps its <think> block out of the reply.
        kw.setdefault("reasoning_format", "hidden")
    return ChatGroq(model=model, api_key=api_key, **kw)
