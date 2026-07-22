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
