"""
vectorstore.py · finagent/vectorstore.py

One place that builds the vector store, so the agent code never constructs a
backend directly. The store is a remote **Qdrant** collection.

Why remote: the corpus used to be an on-disk Chroma directory baked into the
container image, because BM25 needed the whole corpus in memory to build a
lexical index at startup. That made the image huge, made cold starts slow, and
made the index effectively read-only — a write concurrent with a read segfaults
chroma/hnswlib's shared native segment, and any write also left the in-memory
BM25 index stale.

Qdrant removes all three. Lexical retrieval is a *sparse vector* index inside
the database (BM25 with server-side IDF, so it stays correct as documents are
added), concurrency is the server's problem, and the image ships no data.

Points use DETERMINISTIC ids (`point_id`), so re-ingesting a filing overwrites
its chunks instead of duplicating them. That is what makes concurrent writers
safe: the double write is harmless rather than prevented.

Environment
-----------
    QDRANT_URL / QDRANT_CLUSTER_ENDPOINT   cluster endpoint
    QDRANT_API_KEY                         API key
"""

from __future__ import annotations

import os
import uuid
from functools import lru_cache
from typing import Any, Iterator, Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_DIM = 384                     # bge-small-en-v1.5
# BM25 as sparse vectors, computed client-side; Qdrant supplies the IDF half
# server-side (see `ensure_collection`), which is what keeps lexical scoring
# correct after a write instead of frozen at index-build time.
SPARSE_MODEL = "Qdrant/bm25"

# LangChain's Qdrant integration nests document metadata under this payload key,
# so every filter path is "metadata.<field>".
META_KEY = "metadata"
CONTENT_KEY = "page_content"
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# Metadata fields we filter on; indexed so filtering doesn't scan the payload.
_INDEXED_FIELDS = ("company", "ticker", "year", "item", "table_id", "source_url")

# Stable namespace for deterministic point ids.
_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def qdrant_url() -> str:
    url = (os.getenv("QDRANT_URL") or os.getenv("QDRANT_CLUSTER_ENDPOINT") or "").strip()
    if not url:
        raise RuntimeError(
            "QDRANT_URL (or QDRANT_CLUSTER_ENDPOINT) is not set. The corpus lives "
            "in a Qdrant cluster; set it in .env or the deploy environment."
        )
    return url


def point_id(*parts: Any) -> str:
    """Deterministic id from arbitrary stable parts."""
    return str(uuid.uuid5(_ID_NAMESPACE, "|".join(str(p) for p in parts)))


def chunk_point_id(meta: dict, text: str) -> str:
    """Deterministic id for one indexed chunk.

    Identity is (document, parent position, chunk content). Content has to be in
    the key: parent-document chunking stores several children per parent and
    `element_index` equals `parent_id`, so document+position alone collides —
    migrating with that key silently collapsed 71k chunks into 34k.

    Consequences, both wanted:
      * re-ingesting the same filing with the same chunker produces the same
        ids, so the write is an overwrite and concurrent writers converge;
      * two byte-identical chunks of one parent collapse to one point, which is
        correct — they are duplicates.

    Note: changing the CHUNKER changes every id, so a chunk-size/strategy change
    needs `reset_collection()` rather than a plain re-ingest, or the old points
    linger alongside the new ones.
    """
    import hashlib

    digest = hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]
    return point_id(meta.get("source_url") or meta.get("local_path", ""),
                    meta.get("parent_id", ""), digest)


@lru_cache(maxsize=4)
def get_embeddings(model_name: str = DEFAULT_EMBED_MODEL):
    """Shared dense embedding function (GPU when available, else CPU)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    from finagent.device import get_device

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": get_device()},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_sparse_embeddings():
    """Shared BM25 sparse encoder (tokenise + term weights; no neural net)."""
    from langchain_qdrant import FastEmbedSparse

    return FastEmbedSparse(model_name=SPARSE_MODEL)


@lru_cache(maxsize=1)
def get_client():
    """Shared QdrantClient. One per process; the client is thread-safe."""
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=qdrant_url(),
        api_key=(os.getenv("QDRANT_API_KEY") or "").strip() or None,
        timeout=int(os.getenv("QDRANT_TIMEOUT", "60")),
        prefer_grpc=False,
    )


def collection_exists(collection_name: str) -> bool:
    return get_client().collection_exists(collection_name)


def ensure_collection(collection_name: str, dim: int = DENSE_DIM) -> None:
    """Create the collection with named dense + sparse vectors if it's missing.

    `modifier=IDF` is the important flag: Qdrant then computes the inverse
    document frequency term of BM25 from the live collection, so lexical scores
    stay correct as filings are added. An in-process index froze IDF at build
    time and silently went stale on every write.
    """
    from qdrant_client import models

    client = get_client()
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=dim, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    modifier=models.Modifier.IDF),
            },
        )
    for field in _INDEXED_FIELDS:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=f"{META_KEY}.{field}",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass            # already indexed


def build_store(collection_name: str, embedding_model: str = DEFAULT_EMBED_MODEL,
                create: bool = False):
    """Return a hybrid (dense + sparse) LangChain `QdrantVectorStore`.

    Exposes `.similarity_search(query, k=..., filter=...)` and
    `.max_marginal_relevance_search(...)` like any LangChain VectorStore.
    Retrieval fuses the dense and sparse hits with Reciprocal Rank Fusion
    server-side, so the candidate mix adapts per query instead of being a fixed
    lexical/semantic split.
    """
    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    if create:
        ensure_collection(collection_name)

    return QdrantVectorStore(
        client=get_client(),
        collection_name=collection_name,
        embedding=get_embeddings(embedding_model),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR,
        sparse_vector_name=SPARSE_VECTOR,
        content_payload_key=CONTENT_KEY,
        metadata_payload_key=META_KEY,
    )


# --------------------------------------------------------------------------- #
# Backend-neutral admin helpers
#
# These exist so no caller reaches into store internals (the old code used
# `store._collection`, a chroma-only attribute, in six places).
# --------------------------------------------------------------------------- #

def scroll_payloads(collection_name: str, qfilter=None,
                    limit: Optional[int] = None,
                    batch: int = 1000) -> Iterator[dict]:
    """Yield each point's metadata dict, optionally filtered. Pages internally."""
    client = get_client()
    if not client.collection_exists(collection_name):
        return
    offset, seen = None, 0
    while True:
        points, offset = client.scroll(
            collection_name=collection_name, scroll_filter=qfilter,
            limit=batch if limit is None else min(batch, limit - seen),
            with_payload=True, with_vectors=False, offset=offset,
        )
        for p in points:
            yield (p.payload or {}).get(META_KEY, {}) or {}
            seen += 1
            if limit is not None and seen >= limit:
                return
        if offset is None or not points:
            return


def exists_where(collection_name: str, field: str, value: str) -> bool:
    """True when at least one point has metadata[field] == value."""
    from qdrant_client import models

    client = get_client()
    if not client.collection_exists(collection_name):
        return False
    got, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(must=[models.FieldCondition(
            key=f"{META_KEY}.{field}", match=models.MatchValue(value=value))]),
        limit=1, with_payload=False, with_vectors=False,
    )
    return bool(got)


def distinct_values(collection_name: str, field: str,
                    where_field: Optional[str] = None,
                    where_value: Optional[str] = None,
                    limit: int = 5000) -> set[str]:
    """Distinct metadata values for `field`, optionally scoped by another field."""
    from qdrant_client import models

    qfilter = None
    if where_field and where_value:
        qfilter = models.Filter(must=[models.FieldCondition(
            key=f"{META_KEY}.{where_field}",
            match=models.MatchValue(value=where_value))])
    return {str(m.get(field, "")) for m in
            scroll_payloads(collection_name, qfilter, limit=limit)}


def count(collection_name: str) -> int:
    client = get_client()
    if not client.collection_exists(collection_name):
        return 0
    return client.count(collection_name, exact=True).count


def delete_collection(collection_name: str) -> None:
    client = get_client()
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
