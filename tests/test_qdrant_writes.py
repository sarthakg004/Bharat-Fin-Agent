"""Concurrent-writer safety for the shared corpus.

Skipped unless QDRANT_URL is set — the rest of the suite is offline, and this
one needs a live cluster. Run it after any change to `chunk_point_id` or the
ingest write path:

    QDRANT_URL=... QDRANT_API_KEY=... pytest tests/test_qdrant_writes.py

The property under test is why dynamic fetch can write to the served index at
all. We do not prevent two requests fetching the same filing simultaneously;
we make the second write land on the same point ids as the first, so the race
converges instead of duplicating.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from dotenv import load_dotenv

# Load .env explicitly: without it the guard would depend on whether some other
# module happened to be imported first, so this file ran or skipped by accident
# of collection order.
load_dotenv()

pytestmark = pytest.mark.skipif(
    not (os.getenv("QDRANT_URL") or os.getenv("QDRANT_CLUSTER_ENDPOINT")),
    reason="needs a live Qdrant cluster (QDRANT_URL)",
)

COLLECTION = "_test_write_convergence"


@pytest.fixture
def store():
    from finagent.vectorstore import build_store, delete_collection

    delete_collection(COLLECTION)
    yield build_store(COLLECTION, create=True)
    delete_collection(COLLECTION)


def _docs(n: int = 30):
    from langchain_core.documents import Document

    # parent_id repeats every 3 docs, mirroring parent-document chunking — the
    # exact shape that made (source_url, parent_id) alone collide.
    return [Document(page_content=f"Chunk number {i} about revenue growth.",
                     metadata={"source_url": "http://example/10k",
                               "parent_id": i // 3, "company": "TESTCO",
                               "year": "2024"})
            for i in range(n)]


def test_repeated_and_concurrent_writes_converge(store):
    from finagent.vectorstore import chunk_point_id, count

    docs = _docs()
    ids = [chunk_point_id(d.metadata, d.page_content) for d in docs]
    assert len(set(ids)) == len(docs), "sibling chunks must get distinct ids"

    store.add_documents(docs, ids=ids)
    assert count(COLLECTION) == len(docs)

    store.add_documents(docs, ids=ids)          # plain re-ingest
    assert count(COLLECTION) == len(docs), "re-ingest must overwrite, not append"

    with ThreadPoolExecutor(max_workers=8) as ex:   # 8 writers racing
        list(ex.map(lambda _: store.add_documents(docs, ids=ids), range(8)))
    assert count(COLLECTION) == len(docs), "concurrent writers must converge"

    assert store.similarity_search("revenue growth", k=3), "written chunks must be searchable"


def test_identical_text_in_one_parent_collapses():
    """Byte-identical chunks of the same parent share an id on purpose — they
    are duplicates. Distinct text must not."""
    from finagent.vectorstore import chunk_point_id

    meta = {"source_url": "http://example/10k", "parent_id": 1}
    assert chunk_point_id(meta, "Services") == chunk_point_id(meta, "Services")
    assert chunk_point_id(meta, "Services") != chunk_point_id(meta, "Products")
    # Same text under a different parent is a different point.
    assert chunk_point_id(meta, "Services") != chunk_point_id(
        {**meta, "parent_id": 2}, "Services")


def test_facet_index_matches_a_full_scroll(store):
    """The metadata-filter vocabulary is built from `company_facet_index` (facet
    API) instead of scrolling every point's full payload — the old cold-start
    cost. This asserts the facet index produces the exact same company→years map
    (and therefore the same vocab) as a full-payload scroll, so the speedup is
    free of behaviour change."""
    from langchain_core.documents import Document

    from finagent.vectorstore import (company_facet_index, scroll_payloads)
    from finagent.retrieval.filters import build_company_vocab

    docs, ids = [], []
    # "APPLIED MATERIALS INC /DE" is the dynamic-fetch shape: `company` holds the
    # SEC registrant title while `ticker` holds AMAT, so the two fields disagree.
    plan = {"AAPL": (["2022", "2023"], "AAPL"), "MSFT": (["2023"], "MSFT"),
            "3M": (["2021", "2022"], "MMM"),
            "APPLIED MATERIALS INC /DE": (["2025"], "AMAT")}
    i = 0
    for company, (years, ticker) in plan.items():
        for y in years:
            m = {"source_url": f"http://x/{company}/{y}", "parent_id": i,
                 "company": company, "ticker": ticker, "year": y}
            docs.append(Document(page_content=f"{company} {y} revenue", metadata=m))
            i += 1
    from finagent.vectorstore import chunk_point_id
    ids = [chunk_point_id(d.metadata, d.page_content) for d in docs]
    store.add_documents(docs, ids=ids)

    idx = company_facet_index(COLLECTION)
    assert {c: set(f["years"]) for c, f in idx.items()} == {
        c: set(ys) for c, (ys, _) in plan.items()}
    assert {c: set(f["tickers"]) for c, f in idx.items()} == {
        c: {t} for c, (_, t) in plan.items()}

    # Same vocabulary as the full-scroll path it replaces. This is what keeps
    # the cheap facet path honest: drop `ticker` from either side and the two
    # vocabularies diverge here rather than silently in production.
    synth = [{"company": c, "ticker": t, "year": y}
             for c, f in idx.items() for y in f["years"] for t in f["tickers"]]
    v_facet, y_facet = build_company_vocab(synth)
    v_scroll, y_scroll = build_company_vocab(scroll_payloads(COLLECTION))
    assert v_facet == v_scroll
    assert {k: set(v) for k, v in y_facet.items()} == {k: set(v) for k, v in y_scroll.items()}

    # The point of all of it: a question that names the company by TICKER
    # resolves to the same canonical entity as one that spells the name out.
    from finagent.retrieval.filters import infer_filter
    amat = "APPLIED MATERIALS INC /DE"
    for phrasing in ("AMAT net sales by segment FY2025",
                     "Applied Materials net sales by segment FY2025"):
        assert infer_filter(phrasing, v_facet, y_facet)["companies"] == [amat]
