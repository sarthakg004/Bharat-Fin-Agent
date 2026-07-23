"""Closing the tab must actually stop the graph run.

Before this, an abandoned run held the single executor worker to completion and
kept calling the provider: the next question queued behind it and the shared
key pool absorbed the 429s. Cancelling the Future alone does not do it —
`run_in_executor` cannot interrupt a thread that has already started — so the
run has to notice the disconnect itself at the next node boundary.
"""

import asyncio
import threading

from finagent.api import main
from finagent.api.models import QueryRequest


def test_disconnect_aborts_a_running_graph(monkeypatch):
    nodes_run: list[str] = []
    aborted = threading.Event()

    def fake_run_agentic(question, *, on_step=None, **kw):
        """Stand-in for the graph: calls on_step per node, like graph.stream."""
        try:
            for i in range(200):
                on_step("retrieve")           # the cancellation checkpoint
                nodes_run.append(f"node{i}")
                threading.Event().wait(0.01)  # pretend to rerank
            return {"answer": "should never finish", "chunks": [], "metadata": {}}
        except main.ClientGone:
            aborted.set()
            raise

    monkeypatch.setattr(main.rag_service, "run_agentic", fake_run_agentic)

    async def drive():
        gen = main._stream_answer(QueryRequest(question="q"))
        await gen.__anext__()   # the up-front planner frame
        # Second frame comes from the running graph, so by now the task exists
        # and we are suspended inside the wait loop — where a real disconnect
        # lands. (Stopping after the first frame would close the generator
        # before it ever started the run, and prove nothing.)
        await gen.__anext__()
        await gen.aclose()      # <- the browser goes away
        # The worker only checks at a node boundary, so give it one.
        await asyncio.get_event_loop().run_in_executor(None, aborted.wait, 5)

    asyncio.new_event_loop().run_until_complete(drive())

    assert aborted.is_set(), "run kept going after the client disconnected"
    assert len(nodes_run) < 200, "run reached the end despite the disconnect"


def test_abandon_is_a_noop_once_the_run_finished():
    """The happy path must not cancel anything — `finally` runs there too."""
    async def check():
        done = asyncio.get_event_loop().create_future()
        done.set_result("answer")
        flag = threading.Event()
        main._abandon(done, flag)
        assert not flag.is_set() and not done.cancelled()
        assert done.result() == "answer"

    asyncio.new_event_loop().run_until_complete(check())


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
