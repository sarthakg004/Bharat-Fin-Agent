"""The HTTP layer: config, the errors the user sees, and stopping abandoned runs."""

from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

from finagent.api import main
from finagent.api.models import QueryRequest
from finagent.llm import ProviderError
from finagent.runtime import RuntimeContext


def test_config_serves_the_model_table_the_frontend_renders():
    body = TestClient(main.app).get("/api/config").json()
    assert set(body["roles"]) == {"planner", "extractor", "writer", "critic"}
    writer = body["roles"]["writer"]
    assert writer["model"] == body["writer_models"][writer["provider"]][0]
    assert "gpt-oss" not in str(body)


def test_a_writer_the_server_has_no_key_for_is_rejected_up_front(monkeypatch):
    monkeypatch.setattr(main, "collect_provider_keys", lambda provider: [])
    client = TestClient(main.app)
    r = client.post("/api/query", json={"question": "q", "writer": {"provider": "openai"}})
    assert r.status_code == 400 and "OpenAI API key" in r.json()["detail"]


@pytest.mark.parametrize("kind,retryable", [
    ("rate_limit", True), ("busy", True), ("quota", False),
    ("too_large", False), ("auth", False), ("not_found", False)])
def test_every_provider_failure_gets_its_own_message(kind, retryable):
    event = main.error_event(ProviderError("x", kind, "gemini", retry_after=8.0))
    assert event["code"] == kind and event["retryable"] is retryable
    assert event["message"] and "ProviderError" not in event["message"]
    assert ("retry_after" in event) is retryable          # only transient errors are auto-retried


def test_a_daily_quota_is_not_reported_as_a_short_wait():
    """Gemini sends a 36-second retry hint on a quota that resets tomorrow."""
    err = RuntimeError("429 quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier retryDelay: '36s'")
    event = main.error_event(err)
    assert event["code"] == "quota" and "retry_after" not in event


def test_a_closed_browser_tab_stops_the_run(monkeypatch):
    """Cancelling the future is not enough once the worker thread has started,
    so the run checks for the disconnect as each step begins."""
    steps_run: list[int] = []
    aborted = threading.Event()

    def fake_run_agent(question, *, on_step=None, **kw):
        try:
            for i in range(200):
                on_step("retrieve")
                steps_run.append(i)
                threading.Event().wait(0.01)
            return {"answer": "never", "chunks": [], "charts": [], "metadata": {}}
        except main.ClientGone:
            aborted.set()
            raise

    monkeypatch.setattr(main.service, "run_agent", fake_run_agent)

    async def drive():
        gen = main._stream_answer(QueryRequest(question="q"), RuntimeContext())
        await gen.__anext__()           # the up-front planner frame
        await gen.__anext__()           # a frame from the running graph
        await gen.aclose()              # the browser goes away
        await asyncio.get_event_loop().run_in_executor(None, aborted.wait, 5)

    asyncio.new_event_loop().run_until_complete(drive())
    assert aborted.is_set() and len(steps_run) < 200
