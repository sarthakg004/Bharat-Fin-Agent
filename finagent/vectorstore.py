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

DEFAULT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DENSE_DIM = 1024                    # bge-large-en-v1.5

# Output dimensionality per embedding model. `ensure_collection` must size the
# dense vector from the model actually in use — creating a collection at the
# default 384 and then writing 1024-d vectors into it fails every upsert with
# "collection is configured for dense vectors with 384 dimensions".
_MODEL_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}


def dim_for(model_name: str) -> int:
    """Dense dimensionality for an embedding model.

    Unknown models are measured once by embedding a probe string rather than
    guessed — a wrong dimension only surfaces at write time, thousands of chunks
    into an ingest run.
    """
    known = _MODEL_DIMS.get(model_name)
    if known:
        return known
    return len(get_embeddings(model_name).embed_query("dimension probe"))


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
_INDEXED_FIELDS = ("company", "ticker", "year", "item", "table_id",
                   "source_url", "local_path")

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


# Models trained with an ASYMMETRIC query/document prefix, as (query, document).
# These are not decoration: e5 and nomic were contrastively trained with the
# prefix in every pair, and scoring a bare query against a bare passage puts
# them near random — a benchmark that omits them measures the prefix, not the
# model. BGE-family models take no document prefix, so the incumbent is
# unaffected (empty pair = behave exactly as before).
_EMBED_PROMPTS = {
    "intfloat/e5-large-v2": ("query: ", "passage: "),
    "intfloat/e5-base-v2": ("query: ", "passage: "),
    "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: "),
    "Snowflake/snowflake-arctic-embed-l-v2.0": ("query: ", ""),
}


@lru_cache(maxsize=4)
def get_embeddings(model_name: str = DEFAULT_EMBED_MODEL):
    """Shared dense embedding function (GPU when available, else CPU)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    from finagent.device import get_device

    q_prompt, d_prompt = _EMBED_PROMPTS.get(model_name, ("", ""))
    doc_kwargs = {"normalize_embeddings": True}
    if d_prompt:
        doc_kwargs["prompt"] = d_prompt
    # `query_encode_kwargs` REPLACES `encode_kwargs` for queries rather than
    # merging with it, so it has to carry normalize_embeddings itself or
    # queries come back un-normalised against normalised documents.
    query_kwargs = dict(doc_kwargs)
    query_kwargs.pop("prompt", None)
    if q_prompt:
        query_kwargs["prompt"] = q_prompt

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": get_device(), "trust_remote_code": True},
        encode_kwargs=doc_kwargs,
        query_encode_kwargs=query_kwargs,
    )


@lru_cache(maxsize=1)
def get_sparse_embeddings():
    """Shared BM25 sparse encoder (tokenise + term weights; no neural net)."""
    from langchain_qdrant import FastEmbedSparse

    return FastEmbedSparse(model_name=SPARSE_MODEL)


@lru_cache(maxsize=1)
def get_client():
    """Shared QdrantClient. One per process; the client is thread-safe.

    `QDRANT_EVAL_URL` points evaluation runs at a different cluster — normally a
    local `qdrant/qdrant` container. The eval corpus is ~60% of all points and is
    never queried when answering a user, so keeping it out of the managed cluster
    is what leaves room there for the served index. Set it in the eval's
    environment only; production never defines it.

    Use a real local SERVER, not embedded `path=` mode: embedded takes an
    exclusive file lock (the parallel eval runner spawns worker processes) and
    silently ignores payload indexes, so every company/year filter would degrade
    to a linear scan.
    """
    from qdrant_client import QdrantClient

    eval_url = (os.getenv("QDRANT_EVAL_URL") or "").strip()
    return QdrantClient(
        url=eval_url or qdrant_url(),
        api_key=((os.getenv("QDRANT_EVAL_API_KEY") if eval_url
                  else os.getenv("QDRANT_API_KEY")) or "").strip() or None,
        timeout=int(os.getenv("QDRANT_TIMEOUT", "60")),
        prefer_grpc=False,
    )


def ensure_collection(collection_name: str, dim: Optional[int] = None,
                      embedding_model: str = DEFAULT_EMBED_MODEL) -> None:
    """Create the collection with named dense + sparse vectors if it's missing.

    `modifier=IDF` is the important flag: Qdrant then computes the inverse
    document frequency term of BM25 from the live collection, so lexical scores
    stay correct as filings are added. An in-process index froze IDF at build
    time and silently went stale on every write.
    """
    from qdrant_client import models

    if dim is None:
        dim = dim_for(embedding_model)
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
        ensure_collection(collection_name, embedding_model=embedding_model)

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
# `store._collection`, a backend-specific attribute, in six places).
# --------------------------------------------------------------------------- #

def scroll_payloads(collection_name: str, qfilter=None,
                    limit: Optional[int] = None,
                    batch: int = 1000,
                    select: Optional[tuple] = None) -> Iterator[dict]:
    """Yield each point's metadata dict, optionally filtered. Pages internally.

    `select` limits the transfer to those metadata fields (e.g. ("company",
    "year")) instead of the whole payload — the payload also carries
    `page_content` + `parent_text` (~4k chars) + `text_as_html` (~6k chars) per
    point, so selecting a couple of scalar fields moves orders of magnitude less
    data when all you need is metadata.
    """
    client = get_client()
    if not client.collection_exists(collection_name):
        return
    with_payload: Any = True
    if select:
        from qdrant_client import models
        with_payload = models.PayloadSelectorInclude(
            include=[f"{META_KEY}.{f}" for f in select])
    offset, seen = None, 0
    while True:
        points, offset = client.scroll(
            collection_name=collection_name, scroll_filter=qfilter,
            limit=batch if limit is None else min(batch, limit - seen),
            with_payload=with_payload, with_vectors=False, offset=offset,
        )
        for p in points:
            yield (p.payload or {}).get(META_KEY, {}) or {}
            seen += 1
            if limit is not None and seen >= limit:
                return
        if offset is None or not points:
            return


def facet_values(collection_name: str, field: str,
                 where_field: Optional[str] = None,
                 where_value: Optional[str] = None,
                 limit: int = 1000) -> set:
    """Distinct values of a metadata `field`, via Qdrant's server-side facet
    (counts distinct values without scrolling every point). Optionally scoped by
    another field == value. Raises if the server predates the facet API."""
    from qdrant_client import models

    ffilter = None
    if where_field and where_value is not None:
        ffilter = models.Filter(must=[models.FieldCondition(
            key=f"{META_KEY}.{where_field}",
            match=models.MatchValue(value=where_value))])
    res = get_client().facet(collection_name=collection_name,
                             key=f"{META_KEY}.{field}",
                             facet_filter=ffilter, limit=limit)
    return {h.value for h in res.hits}


def company_facet_index(collection_name: str, max_workers: int = 12) -> dict:
    """`{company -> {"years": set, "tickers": set}}` for a collection, cheaply.

    The retriever's metadata-filter vocabulary needs the distinct company values
    and, per company, its filing years and its ticker(s). Reading that by
    scrolling every point's full payload was the dominant cold-start cost (~54s
    on 80k points — hundreds of MB moved to learn ~40 names). The facet API
    returns distinct values server-side; the per-company facets run in parallel
    (the client is thread-safe). ~16x faster, and exact.

    `tickers` matters because `company` and `ticker` do NOT always agree: the
    curated corpus stores the ticker in BOTH ("NVDA"/"NVDA"), while a
    dynamically fetched filing stores the SEC registrant title in `company`
    ("APPLIED MATERIALS INC /DE") and the ticker in `ticker`. A vocabulary built
    from `company` alone therefore cannot resolve "AMAT", and the question falls
    through to unfiltered retrieval. Read both; let either name the entity.

    Falls back to a selective-payload scroll when the server predates facets —
    still far cheaper than the old full-payload scroll.

    ponytail: parallel per-company facets are fine for a bounded corpus (tens of
    companies); a corpus with thousands of issuers would want the scroll path.
    """
    client = get_client()
    if not client.collection_exists(collection_name):
        return {}
    try:
        companies = {c for c in facet_values(collection_name, "company",
                                             limit=5000) if c}

        def _facets(co):
            return co, {
                "years": {y for y in facet_values(collection_name, "year",
                                                  "company", co, limit=200) if y},
                "tickers": {t for t in facet_values(collection_name, "ticker",
                                                    "company", co, limit=50) if t},
            }

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return dict(ex.map(_facets, companies))
    except Exception:
        idx: dict = {}
        for m in scroll_payloads(collection_name,
                                 select=("company", "year", "ticker")):
            c = m.get("company") or ""
            if not c:
                continue
            entry = idx.setdefault(c, {"years": set(), "tickers": set()})
            for key, field in (("years", "year"), ("tickers", "ticker")):
                v = str(m.get(field) or "")
                if v:
                    entry[key].add(v)
        return idx


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
