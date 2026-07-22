"""The retriever must actually issue the fused (dense + sparse) query.

Regression test. `use_mmr` defaulted to True while the comment beside it said
"off by default", and nothing in the codebase passed the flag. LangChain's
`max_marginal_relevance_search` is dense-only — it sends a NearestQuery with
`using="dense"` and no prefetch — so the sparse/BM25 half of every collection
was written at ingest and never read at query time. The bug was invisible:
retrieval still returned plausible chunks, and the eval harness called
`similarity_search` directly, so it measured a path production never ran.

No cluster needed: a stub store records which retrieval call the branch makes.
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
    assert r.use_mmr is False, "MMR is dense-only; it disables sparse retrieval"
    r._pool("what were total operating expenses in 2023?")
    assert [c[0] for c in r.store.calls] == ["similarity_search"]


def test_pool_depth_is_passed_through():
    r = _retriever(pool_top_k=96)
    r._pool("revenue")
    assert r.store.calls[0][1]["k"] == 96


def test_mmr_remains_available_but_opt_in():
    """Not dead code — it's a deliberate trade (diversity for lexical recall)
    that a caller can still make explicitly."""
    r = _retriever(use_mmr=True)
    r._pool("revenue")
    assert [c[0] for c in r.store.calls] == ["max_marginal_relevance_search"]


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
