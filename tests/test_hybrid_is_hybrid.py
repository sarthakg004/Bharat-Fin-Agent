"""The retriever must actually issue the fused (dense + sparse) query.

Regression test. There used to be a `use_mmr` flag, and it defaulted to True
while the comment beside it said "off by default". LangChain's
`max_marginal_relevance_search` is dense-only — it sends a NearestQuery with
`using="dense"` and no prefetch — so the sparse/BM25 half of every collection
was written at ingest and never read at query time. The bug was invisible:
retrieval still returned plausible chunks, and the eval harness called
`similarity_search` directly, so it measured a path production never ran.

The flag is gone now, which makes the bug unreachable rather than merely
untrue, so these tests assert the call that reaches the store instead of the
value of a setting.

No cluster needed: a stub store records which retrieval call is made.
"""

from __future__ import annotations

import pytest

from finagent.retrieval.hybrid import HybridRetriever


class StubStore:
    """Records which retrieval method the retriever chose."""

    collection_name = "stub"

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def similarity_search(self, query, k=None, filter=None, **kw):
        self.calls.append(("similarity_search", {"k": k}))
        return []

    def max_marginal_relevance_search(self, query, k=None, fetch_k=None,
                                      lambda_mult=None, filter=None, **kw):
        self.calls.append(("max_marginal_relevance_search", {"k": k}))
        return []


def _retriever(**kw):
    # auto_filter off so the stub never has to serve a metadata scroll.
    return HybridRetriever(StubStore(), auto_filter=False, **kw)


def test_default_issues_the_fused_query_not_mmr():
    r = _retriever()
    r._pool("what were total operating expenses in 2023?")
    assert [c[0] for c in r.store.calls] == ["similarity_search"]
    assert not hasattr(r, "use_mmr"), (
        "the MMR flag is back; it is dense-only and silently disables the "
        "sparse half of every collection")


def test_pool_depth_is_passed_through():
    r = _retriever(pool_top_k=96)
    r._pool("revenue")
    assert r.store.calls[0][1]["k"] == 96


def test_mmr_is_unreachable():
    """MMR is not an option any more, in either direction.

    It was kept for a while as an opt-in "diversity vs lexical recall" trade,
    but nothing ever opted in and leaving the branch there left a one-keyword
    path back to a silently dense-only retriever.
    """
    with pytest.raises(TypeError):
        _retriever(use_mmr=True)
    r = _retriever()
    r._pool("revenue")
    assert all(c[0] != "max_marginal_relevance_search" for c in r.store.calls)


def test_agent_builds_retrievers_that_fuse():
    """The agent must not re-enable MMR when it constructs its retrievers."""
    import inspect

    from finagent.graph.corrective import AgenticRAGv2

    src = inspect.getsource(AgenticRAGv2._get_hybrids)
    assert "use_mmr" not in src, (
        "the agent now sets use_mmr explicitly — update this test and re-measure "
        "recall before trusting the change")


@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_store_is_configured_for_both_vectors(mode):
    """Both named vectors must exist in the store config, or fusion silently
    degrades to whichever half is present."""
    from finagent import vectorstore as vs

    assert vs.DENSE_VECTOR == "dense" and vs.SPARSE_VECTOR == "sparse"
    src = inspect_source(vs.build_store)
    assert "RetrievalMode.HYBRID" in src, "build_store must request hybrid retrieval"
    assert f"{mode}_vector_name" in src or f"{mode}_vector" in src.lower()


def inspect_source(fn) -> str:
    import inspect

    return inspect.getsource(fn)


def test_collection_is_sized_for_the_embedder_in_use():
    """A collection must be created at the dimensionality of the embedder that
    will write to it.

    `build_store(..., create=True)` used to call `ensure_collection` with the
    hardcoded 384 default, so pointing the ingester at bge-large created a
    384-dim collection and then failed EVERY upsert with "collection is
    configured for dense vectors with 384 dimensions". Nothing caught it until
    a reindex was already running.
    """
    from finagent.vectorstore import (
        DEFAULT_EMBED_MODEL, GEMINI_EMBED_DIM, dim_for)

    # Every embedder the project can be pointed at resolves to its own width,
    # WITHOUT an API call. `DENSE_DIM` is deliberately not the reference here:
    # it is bge-large's 1024, and asserting against it only held while bge was
    # the default. What must stay true is that the default embedder, whatever
    # it is, reports a width.
    assert dim_for(DEFAULT_EMBED_MODEL) > 0
    assert dim_for("gemini-embedding-2") == GEMINI_EMBED_DIM == 1536
    assert dim_for("BAAI/bge-large-en-v1.5") == 1024
    assert dim_for("BAAI/bge-base-en-v1.5") == 768

    import inspect

    from finagent.vectorstore import build_store, ensure_collection

    # ensure_collection must derive the size, not default it.
    assert "dim_for(embedding_model)" in inspect.getsource(ensure_collection)
    # and build_store must pass the model through when it creates.
    assert "embedding_model=embedding_model" in inspect.getsource(build_store)
