"""Error classification, key rotation and per-request model choice.

No network: provider errors are fakes shaped like the real ones.
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor

import pytest

from finagent import llm as L
from finagent.runtime import ROLES, RuntimeContext, current_context


class FakeHTTPError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status_code = status


@pytest.mark.parametrize("status,message,kind", [
    # The same 429 means "wait a few seconds" or "come back tomorrow".
    (429, "Rate limit reached. Please try again in 7.66s.", "rate_limit"),
    (429, "Quota exceeded. quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier, "
          "retryDelay: '36s'", "quota"),
    (429, "Rate limit reached on tokens per day (TPD). Please try again in 7m12s.", "quota"),
    # Groq labels an over-sized prompt as a rate limit. It is not one.
    (413, "Request too large. code: rate_limit_exceeded", "too_large"),
    (503, "The model is overloaded. Please try again later.", "busy"),
    (500, "Internal server error", "busy"),
    (401, "Invalid API Key", "auth"),
    (404, "The model `qwen/qwen3.6-27b` does not exist", "not_found"),
    (400, "Failed to generate JSON", "other"),
])
def test_errors_are_classified_by_status_then_message(status, message, kind):
    assert L.classify_error(FakeHTTPError(status, message)).kind == kind


def test_classification_sees_through_wrapped_errors_and_reads_the_wait():
    try:
        try:
            raise FakeHTTPError(429, "Please try again in 2m59.56s")
        except FakeHTTPError as inner:
            raise RuntimeError("graph node failed") from inner
    except RuntimeError as wrapped:
        info = L.classify_error(wrapped)
    assert info.kind == "rate_limit" and info.retryable
    assert abs(info.retry_after - 179.56) < 0.01


class FakeChat:
    """Stands in for one provider client bound to one key."""

    def __init__(self, key, script):
        self.key, self.script = key, script

    def _generate(self, messages, **kw):
        outcome = self.script[self.key].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _rotating(monkeypatch, script, keys=("k1", "k2")):
    monkeypatch.setattr(L, "_build_single", lambda provider, model, key, **kw: FakeChat(key, script))
    monkeypatch.setattr(L, "_next_start", {})
    sleeps: list[float] = []
    monkeypatch.setattr(L.time, "sleep", sleeps.append)
    return L.RotatingChatModel(provider="groq", chat_model="m", keys=list(keys)), sleeps


def test_a_rate_limited_key_is_swapped_for_the_next_one(monkeypatch):
    model, sleeps = _rotating(monkeypatch, {
        "k1": [FakeHTTPError(429, "rate limit, try again in 5s")], "k2": ["answer"]})
    assert model._generate([]) == "answer"
    assert sleeps == []                       # another key worked: no waiting


def test_a_per_minute_limit_on_every_key_is_waited_out_once(monkeypatch):
    model, sleeps = _rotating(monkeypatch, {
        "k1": [FakeHTTPError(429, "try again in 3s"), "answer"],
        "k2": [FakeHTTPError(429, "try again in 9s")]})
    assert model._generate([]) == "answer"
    assert sleeps == [3.5]                    # the shortest reported wait, plus a margin


def test_a_daily_quota_fails_fast_and_says_so(monkeypatch):
    daily = "Quota exceeded for quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    model, sleeps = _rotating(monkeypatch, {
        "k1": [FakeHTTPError(429, daily)], "k2": [FakeHTTPError(429, daily)]})
    with pytest.raises(L.ProviderError) as err:
        model._generate([])
    assert err.value.kind == "quota" and err.value.provider == "groq"
    assert sleeps == []                       # waiting cannot fix a daily limit


def test_a_missing_model_is_not_retried(monkeypatch):
    model, _ = _rotating(monkeypatch, {
        "k1": [FakeHTTPError(404, "model does not exist")], "k2": ["never reached"]})
    with pytest.raises(L.ProviderError) as err:
        model._generate([])
    assert err.value.kind == "not_found"


def test_strict_schema_requires_every_field():
    """Qwen leaves out any field that has a default unless the schema requires it."""
    from finagent.agent.state import XBRLQueryBatch

    schema = L._strict_schema(XBRLQueryBatch)
    item = schema["$defs"]["XBRLQuery"]
    assert set(item["required"]) == set(item["properties"])
    assert item["additionalProperties"] is False
    assert all("default" not in p for p in item["properties"].values())


def test_only_the_writer_can_be_overridden():
    ctx = RuntimeContext(provider="openai", model="gpt-4o", api_key="user-key")
    assert ctx.resolve("writer") == ("openai", "gpt-4o", "user-key")
    for role in ("planner", "extractor", "critic"):
        assert ctx.resolve(role) == (*ROLES[role], None)     # the user's key is never sent elsewhere
    assert RuntimeContext().resolve("writer") == (*ROLES["writer"], None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.api_key = "leaked"                               # type: ignore[misc]


def test_evidence_is_trimmed_only_for_the_free_groq_writer():
    assert RuntimeContext().evidence_budget() is None
    assert RuntimeContext(provider="groq").evidence_budget() == 14_000
    assert RuntimeContext(provider="groq", api_key="own").evidence_budget() is None


def test_concurrent_requests_do_not_share_a_context():
    """64 runs on one compiled graph, each with its own key, must read back their own."""
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class S(TypedDict, total=False):
        seen: str

    g = StateGraph(S)
    g.add_node("n", lambda state: {"seen": current_context().api_key})
    g.add_edge(START, "n")
    g.add_edge("n", END)
    app = g.compile()

    def run(i: int) -> str:
        ctx = RuntimeContext(api_key=f"key-{i}")
        return app.invoke({}, config={"configurable": {"runtime_context": ctx}})["seen"]

    with ThreadPoolExecutor(max_workers=16) as ex:
        assert list(ex.map(run, range(64))) == [f"key-{i}" for i in range(64)]


def test_critic_is_never_the_writer_model():
    from finagent.runtime import ROLES, RuntimeContext
    assert RuntimeContext().resolve("critic")[:2] == ROLES["critic"]
    same = RuntimeContext(provider=ROLES["critic"][0], model=ROLES["critic"][1])
    assert same.resolve("critic")[:2] != same.resolve("writer")[:2]
