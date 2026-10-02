"""The vector store: a remote Qdrant collection with two vectors per chunk.

    dense   Gemini embedding, 1536 dimensions (semantic match)
    sparse  BM25 term weights (exact-word match); Qdrant computes the IDF half

Qdrant fuses both rankings server-side. Point ids are derived from the chunk's
content, so ingesting the same filing twice overwrites instead of duplicating.

    QDRANT_URL / QDRANT_CLUSTER_ENDPOINT, QDRANT_API_KEY   the served cluster
    QDRANT_EVAL_URL                                        the local eval Qdrant
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import threading
import time
import uuid
from array import array
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()

# The embedder. A collection and its embedder are one choice: querying a
# collection with a different model returns nothing, without an error.
DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")

# Gemini's native width is 3072. The vectors are Matryoshka-trained, so the
# first 1536 values are still a usable vector at half the storage. A truncated
# vector is not unit length, which is why `_l2` re-normalises it.
GEMINI_EMBED_DIM = 1536
GEMINI_EMBED_BATCH = 100        # texts per request (the API maximum)
GEMINI_EMBED_CONCURRENCY = 3    # requests in flight at once

SPARSE_MODEL = "Qdrant/bm25"
META_KEY = "metadata"           # LangChain nests chunk metadata under this key
CONTENT_KEY = "page_content"
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# Metadata fields that get a Qdrant index so filtering does not scan payloads.
_INDEXED_FIELDS = ("company", "ticker", "year", "item", "table_id",
                   "source_url", "local_path")
_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


class EmbeddingQuotaExhausted(RuntimeError):
    """Every Gemini key is out of embedding quota for today."""


class _HttpError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:400]}")
        self.status, self.body = status, body


def _l2(vec: list[float]) -> list[float]:
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm else vec


class GeminiEmbeddings(Embeddings):
    """Gemini embeddings over plain REST, with a disk cache and key rotation.

    REST instead of the SDK: the SDK merged a batch of texts into one vector on
    this model. `_embed_batch` still checks it got one vector per text.

    Questions are embedded as RETRIEVAL_QUERY and chunks as RETRIEVAL_DOCUMENT;
    the model is trained on that pairing.

    The cache (sqlite, one row per text) makes a re-run free. A key that hits
    its per-minute limit (100 texts a minute on the free tier) rests until the
    reset the API names, and the batch moves to the next key; it sleeps only when
    every key is resting. A 10-K is about 1,000 texts, so one key alone needs
    ten minutes and the whole pool about two.
    """

    MAX_WAITS = 12              # sleeps with every key resting before giving up
    DEAD_COOLDOWN_S = 900       # how long a key sits out after a daily-quota error

    def __init__(self, model: str = DEFAULT_EMBED_MODEL, dim: int = GEMINI_EMBED_DIM,
                 batch_size: int = GEMINI_EMBED_BATCH):
        from finagent.llm import collect_provider_keys

        self.model, self.dim, self.batch_size = model, dim, batch_size
        self.keys = collect_provider_keys("gemini")
        if not self.keys:
            raise RuntimeError("No Gemini API key found. Set GEMINI_API_KEYS or "
                               "GEMINI_API_KEY to embed.")
        self._dead: dict[str, float] = {}       # key -> time it may be tried again (daily)
        self._resting: dict[str, float] = {}    # key -> end of its per-minute limit
        self._local = threading.local()         # one sqlite connection per thread

    def _post(self, key: str, texts: list[str], task: str) -> list[list[float]]:
        """One `:batchEmbedContents` call."""
        import json
        import urllib.error
        import urllib.request

        body = json.dumps({"requests": [
            {"model": f"models/{self.model}",
             "content": {"parts": [{"text": t}]},
             "taskType": task,
             "outputDimensionality": self.dim}
            for t in texts
        ]}).encode()
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:batchEmbedContents",
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise _HttpError(e.code, e.read().decode("utf-8", "replace")) from e
        return [e["values"] for e in payload.get("embeddings", [])]

    def _embed_batch(self, texts: list[str], task: str, key_idx: int) -> list[list[float]]:
        """Embed one batch. Raises instead of dropping texts."""
        last: Optional[Exception] = None
        waits = attempt = 0
        while True:
            now = time.time()
            live = [k for k in self.keys if self._dead.get(k, 0) <= now]
            if not live:
                raise EmbeddingQuotaExhausted(
                    f"All {len(self.keys)} Gemini keys are out of embedding quota "
                    f"for today. Vectors already embedded are cached, so the next "
                    f"run resumes.") from last
            ready = [k for k in live if self._resting.get(k, 0) <= now]
            if not ready:                                # every key is resting
                if waits >= self.MAX_WAITS:
                    break
                time.sleep(min(self._resting[k] for k in live) - now + 0.5)
                waits += 1
                continue
            key = ready[(key_idx + attempt) % len(ready)]
            attempt += 1
            try:
                got = self._post(key, texts, task)
                if len(got) != len(texts):
                    raise RuntimeError(f"{self.model} returned {len(got)} vectors "
                                       f"for {len(texts)} texts.")
                return [_l2(v) for v in got]
            except _HttpError as e:
                last = e
                if e.status in (401, 403):               # bad key: drop it for good
                    self._dead[key] = time.time() + 10 * 365 * 24 * 3600
                elif e.status != 429:
                    raise
                elif "PerDay" in e.body:                 # daily quota: bench the key
                    self._dead[key] = time.time() + self.DEAD_COOLDOWN_S
                else:                                    # per-minute: rest this key
                    m = re.search(r"retryDelay['\"]?:\s*['\"](\d+(?:\.\d+)?)s", e.body)
                    self._resting[key] = time.time() + (float(m.group(1)) + 1 if m else 30)
        raise RuntimeError(f"Gemini embeddings still rate-limited after "
                           f"{self.MAX_WAITS} waits ({last})") from last

    def _conn(self):
        db = getattr(self._local, "db", None)
        if db is None:
            path = Path(os.getenv("FINAGENT_EMBED_CACHE_DIR", "data/embed_cache"))
            path.mkdir(parents=True, exist_ok=True)
            db = self._local.db = sqlite3.connect(
                path / f"{self.model}_{self.dim}.sqlite", timeout=30)
            db.execute("CREATE TABLE IF NOT EXISTS vec (k TEXT PRIMARY KEY, v BLOB)")
        return db

    @staticmethod
    def _key(task: str, text: str) -> str:
        return hashlib.sha1(f"{task}\x00{text}".encode()).hexdigest()

    def _embed_all(self, texts: list[str], task: str) -> list[list[float]]:
        """Embed `texts` in order: read the cache, call the API for the rest."""
        keys = [self._key(task, t) for t in texts]
        db = self._conn()
        cached: dict[str, list[float]] = {}
        uniq = list(dict.fromkeys(keys))
        for i in range(0, len(uniq), 900):               # sqlite allows 999 variables
            window = uniq[i:i + 900]
            q = f"SELECT k, v FROM vec WHERE k IN ({','.join('?' * len(window))})"
            for k, blob in db.execute(q, window):
                cached[k] = array("f", blob).tolist()

        todo = list(dict.fromkeys(k for k in keys if k not in cached))
        if todo:
            text_of_key = dict(zip(keys, texts))
            pending = [text_of_key[k] for k in todo]
            work = list(enumerate(pending[i:i + self.batch_size]
                                  for i in range(0, len(pending), self.batch_size)))
            if len(work) == 1:
                fresh = self._embed_and_cache(work[0], task)
            else:
                with ThreadPoolExecutor(max_workers=GEMINI_EMBED_CONCURRENCY) as ex:
                    fresh = [v for batch in ex.map(
                        lambda w: self._embed_and_cache(w, task), work) for v in batch]
            cached.update(zip(todo, fresh))
        return [cached[k] for k in keys]

    def _embed_and_cache(self, work: tuple[int, list[str]], task: str) -> list[list[float]]:
        """Embed one batch and save it before returning, so paid quota is never lost."""
        key_idx, texts = work
        # Round-trip through float32 so a fresh vector equals its cached copy.
        vecs = [array("f", v).tolist() for v in self._embed_batch(texts, task, key_idx)]
        db = self._conn()
        db.executemany("INSERT OR REPLACE INTO vec VALUES (?, ?)",
                       [(self._key(task, t), array("f", v).tobytes())
                        for t, v in zip(texts, vecs)])
        db.commit()
        return vecs

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_all(list(texts), "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed_all([text], "RETRIEVAL_QUERY")[0]


@lru_cache(maxsize=2)
def get_embeddings(model_name: str = DEFAULT_EMBED_MODEL) -> GeminiEmbeddings:
    """The shared embedder."""
    return GeminiEmbeddings(model_name)


@lru_cache(maxsize=1)
def get_sparse_embeddings():
    """The shared BM25 encoder (tokenise and weight terms; no neural net)."""
    from langchain_qdrant import FastEmbedSparse

    # The image bakes the model and runs offline (HF_HUB_OFFLINE=1). fastembed
    # 0.8.1's offline check wants two files the Hugging Face repo does not ship
    # (mock.file, tamil.txt), so it refuses its own cache. Point it at the baked
    # snapshot directly; without one (local dev, online), let it download.
    cache = Path(os.getenv("FASTEMBED_CACHE_PATH", ""))
    snapshots = sorted(cache.glob("models--Qdrant--bm25/snapshots/*")) if cache.name else []
    if snapshots:
        return FastEmbedSparse(model_name=SPARSE_MODEL, specific_model_path=str(snapshots[-1]))
    return FastEmbedSparse(model_name=SPARSE_MODEL)


def qdrant_url() -> str:
    url = (os.getenv("QDRANT_URL") or os.getenv("QDRANT_CLUSTER_ENDPOINT") or "").strip()
    if not url:
        raise RuntimeError("QDRANT_URL (or QDRANT_CLUSTER_ENDPOINT) is not set.")
    return url


@lru_cache(maxsize=1)
def get_client():
    """The shared Qdrant client. `QDRANT_EVAL_URL` points it at the local eval
    Qdrant instead of the served cluster; production never sets it."""
    from qdrant_client import QdrantClient

    eval_url = (os.getenv("QDRANT_EVAL_URL") or "").strip()
    return QdrantClient(
        url=eval_url or qdrant_url(),
        api_key=((os.getenv("QDRANT_EVAL_API_KEY") if eval_url
                  else os.getenv("QDRANT_API_KEY")) or "").strip() or None,
        timeout=int(os.getenv("QDRANT_TIMEOUT", "60")),
        prefer_grpc=False,
    )


def chunk_point_id(meta: dict, text: str) -> str:
    """Deterministic point id from (filing, parent position, chunk text).

    Ingesting the same chunk twice gives the same id, so a re-run overwrites
    instead of duplicating. The stored points were written with this formula.
    """
    digest = hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]
    parts = (meta.get("source_url") or meta.get("local_path", ""),
             meta.get("parent_id", ""), digest)
    return str(uuid.uuid5(_ID_NAMESPACE, "|".join(str(p) for p in parts)))


def ensure_collection(collection_name: str) -> None:
    """Create the collection (dense + sparse vectors, payload indexes) if missing."""
    from qdrant_client import models

    client = get_client()
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={DENSE_VECTOR: models.VectorParams(
                size=GEMINI_EMBED_DIM, distance=models.Distance.COSINE)},
            # modifier=IDF: Qdrant keeps BM25's document-frequency term current
            # as filings are added.
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(
                modifier=models.Modifier.IDF)},
        )
    for field in _INDEXED_FIELDS:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=f"{META_KEY}.{field}",
                field_schema=models.PayloadSchemaType.KEYWORD)
        except Exception:
            pass                                         # already indexed


def build_store(collection_name: str, embedding_model: str = DEFAULT_EMBED_MODEL,
                create: bool = False):
    """A LangChain `QdrantVectorStore` in hybrid mode (dense + sparse, fused by
    Reciprocal Rank Fusion on the server)."""
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
# Read helpers
# --------------------------------------------------------------------------- #

def _match(field: str, value: str):
    from qdrant_client import models

    return models.Filter(must=[models.FieldCondition(
        key=f"{META_KEY}.{field}", match=models.MatchValue(value=value))])


def scroll_payloads(collection_name: str, qfilter=None, limit: Optional[int] = None,
                    select: Optional[tuple] = None) -> Iterator[dict]:
    """Yield each point's metadata. `select` limits which fields are transferred."""
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
            limit=1000 if limit is None else min(1000, limit - seen),
            with_payload=with_payload, with_vectors=False, offset=offset)
        for p in points:
            yield (p.payload or {}).get(META_KEY, {}) or {}
            seen += 1
            if limit is not None and seen >= limit:
                return
        if offset is None or not points:
            return


def facet_values(collection_name: str, field: str, where_field: Optional[str] = None,
                 where_value: Optional[str] = None, limit: int = 1000) -> set:
    """Distinct values of a metadata field, counted by Qdrant (no scroll)."""
    ffilter = _match(where_field, where_value) if where_field and where_value is not None else None
    res = get_client().facet(collection_name=collection_name,
                             key=f"{META_KEY}.{field}", facet_filter=ffilter, limit=limit)
    return {h.value for h in res.hits}


def company_facet_index(collection_name: str, max_workers: int = 12) -> dict:
    """`{company: {"years": set, "tickers": set}}` for a collection.

    Both `company` and `ticker` are read because they differ: a live-fetched
    filing stores the SEC legal name as company and the symbol as ticker.
    """
    if not get_client().collection_exists(collection_name):
        return {}
    try:
        companies = {c for c in facet_values(collection_name, "company", limit=5000) if c}

        def facets(co):
            return co, {
                "years": {y for y in facet_values(collection_name, "year", "company", co, 200) if y},
                "tickers": {t for t in facet_values(collection_name, "ticker", "company", co, 50) if t},
            }

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return dict(ex.map(facets, companies))
    except Exception:                                    # server without facets
        idx: dict = {}
        for m in scroll_payloads(collection_name, select=("company", "year", "ticker")):
            if not m.get("company"):
                continue
            entry = idx.setdefault(m["company"], {"years": set(), "tickers": set()})
            for key, field in (("years", "year"), ("tickers", "ticker")):
                if m.get(field):
                    entry[key].add(str(m[field]))
        return idx


def exists_where(collection_name: str, field: str, value: str) -> bool:
    """True when at least one point has metadata[field] == value."""
    client = get_client()
    if not client.collection_exists(collection_name):
        return False
    got, _ = client.scroll(collection_name=collection_name,
                           scroll_filter=_match(field, value), limit=1,
                           with_payload=False, with_vectors=False)
    return bool(got)


def distinct_values(collection_name: str, field: str, where_field: Optional[str] = None,
                    where_value: Optional[str] = None, limit: int = 5000) -> set[str]:
    """Distinct metadata values for `field`, optionally scoped by another field."""
    qfilter = _match(where_field, where_value) if where_field and where_value else None
    return {str(m.get(field, "")) for m in
            scroll_payloads(collection_name, qfilter, limit=limit, select=(field,))}


def count(collection_name: str) -> int:
    client = get_client()
    if not client.collection_exists(collection_name):
        return 0
    return client.count(collection_name, exact=True).count


def delete_collection(collection_name: str) -> None:
    client = get_client()
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
