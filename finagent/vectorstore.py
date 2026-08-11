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
import re
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Optional

from dotenv import load_dotenv

load_dotenv()

# The embedder every caller uses unless it names its own: the agent, the hybrid
# retriever, the dynamic SEC fetch, and the ingester all read this constant.
#
# Env-backed because the served index and the local retrieval sweeps disagree.
# Production embeds with Gemini (the served collection is 1536-d, and querying
# it with a 1024-d local model retrieves nothing), while the sweep collections
# in `evaluation/` are bge-large and are re-scored with `--embed`. A query
# embedded by the wrong model is not an error, it is silently empty results,
# so this has to be one switch rather than a literal repeated per call site.
DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
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

# --------------------------------------------------------------------------- #
# Google Gemini embeddings
# --------------------------------------------------------------------------- #

# Models served by the Gemini embedding API rather than by a local
# sentence-transformer. Matched by prefix so a dated variant still routes here.
GEMINI_EMBED_PREFIX = "gemini-embedding"

# Native width is 3072. Gemini's embeddings are Matryoshka-trained, so a
# truncated prefix is still a usable vector — 1536 halves both the Qdrant
# storage and the wire cost at a published quality loss near zero, and 3072
# would put this corpus at ~550 MB of vectors.
#
# A truncated vector is NOT unit-norm (only the full 3072 is), so `_l2` below is
# mandatory rather than tidiness: Qdrant's COSINE distance normalises at compare
# time, but the retriever also does raw dot products against these vectors on
# the ephemeral-fetch path, and those would be silently wrong.
GEMINI_EMBED_DIM = 1536

# Texts per request. 100 is the documented per-request maximum.
#
# What this number does NOT do is buy quota, and that misreading cost a full
# day's budget. The quotaId is *named* `EmbedContentRequests…PerMinute/PerDay`,
# but the free tier meters the CONTENTS inside the request, not the request:
# a batch of 100 texts spends 100 units, so batching buys THROUGHPUT and
# nothing else. Measured, not read from docs — one 100-text batch immediately
# tripped the per-minute limit, and 5,125 contents (854/key) exhausted the day.
#
# The practical consequence: budget a build in CHUNKS, never in requests.
# ~1000 contents/day/key, dimensioned PER MODEL (each embedding model has its
# own bucket), so a 6-key pool is ~6000 chunks/day against a ~37k-chunk corpus
# — a multi-day build that resumes off the sqlite cache below.
#
# `_embed_batch` verifies the response count, so if a model silently ignores
# the batch we fail loudly rather than write merged vectors.
GEMINI_EMBED_BATCH = 100

# Concurrent in-flight requests. Deliberately small: the 100/min ceiling is
# shared across the whole project, so more workers only convert quota into 429s
# and backoff. This is not a throughput knob.
GEMINI_EMBED_CONCURRENCY = 3


# `QdrantVectorStore` type-CHECKS its embedder against this ABC and rejects
# anything else with a bare "Invalid `embeddings` type" — duck-typing
# embed_documents/embed_query is not enough. langchain_core is already a hard
# dependency, so importing it here costs nothing.
from langchain_core.embeddings import Embeddings  # noqa: E402


class _HttpError(RuntimeError):
    """A non-2xx from the embedding endpoint, with the status and body kept.

    Both matter: the status separates a 429 from a real error, and only the
    body says whether the 429 was per-MINUTE (wait) or per-DAY (bench the key).
    """

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:400]}")
        self.status = status
        self.body = body


class EmbeddingQuotaExhausted(RuntimeError):
    """The provider's DAILY embedding budget is gone.

    A distinct type because it must abort the whole ingest rather than be
    retried or counted as one more failed file: every further attempt spends a
    request from a budget that is already empty, and the per-file `except` in
    `CorpusIngester.ingest_all` would otherwise swallow it 65 times over.
    """


def _l2(vec: list[float]) -> list[float]:
    import math

    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class GeminiEmbeddings(Embeddings):
    """LangChain-compatible embedder backed by the Gemini embedding API.

    Three things here are not boilerplate.

    **Raw REST, not the SDK.** Requests go to `:batchEmbedContents` over plain
    urllib. The google-genai SDK's `embed_content(contents=[…])` path merges a
    batch into ONE vector on `gemini-embedding-2` instead of erroring (measured:
    the merged vector sits at cosine 0.87 / 0.85 to the two true vectors, so
    nothing downstream looks obviously broken) — that would embed every chunk
    against its neighbours' text and the only symptom would be quietly bad
    retrieval. The REST endpoint batches correctly (verified 100 texts → 100
    vectors). The SDK also reports a 429 as "Cannot send a request, as the
    client has been closed", which made quota debugging guesswork. The response
    count is still checked against the batch: a cheap invariant on the path
    where being wrong is invisible.

    **Asymmetric task types.** The model is trained with RETRIEVAL_QUERY for the
    question and RETRIEVAL_DOCUMENT for the passage; scoring a query embedded as
    a document costs real accuracy, the same trap `_EMBED_PROMPTS` documents for
    e5/nomic.

    **A budget that is easy to destroy.** Each key carries ~1000 CONTENTS/day
    (see `GEMINI_EMBED_BATCH` — the unit is texts, not requests), and it is
    spent by ATTEMPTS, not by successes: see `_embed_batch` for how an
    over-eager retry loop spent ~15,000 requests on 429 retries and drained
    every key after 7 of 72 filings. That is why the backoff waits instead of
    rotating.

    The two 429s are therefore handled in opposite ways. A PER-MINUTE limit
    clears on its own, so it sleeps for exactly the delay the server names and
    retries the same key. A PER-DAY limit will not clear soon, so that key is
    BENCHED for `DEAD_COOLDOWN_S` and the batch moves to a live one; only a
    pool with every key benched raises `EmbeddingQuotaExhausted`.

    The bench is timed rather than permanent, and that is the part that was
    learned the hard way. The keys are separate projects whose daily counters
    roll over at different wall-clock times, so for an hour or so around the
    boundary the pool is a MIX of fresh and stale keys. Two builds died there:
    the first treated the first stale key as fatal to the whole run (17 of 72
    filings), and the second benched each key permanently as it was met and
    reached "all keys dead" while the survivors were minutes from resetting —
    both while ~10 requests of a 6,000/day budget had actually been spent.
    """

    # Waits before giving up on a per-minute limit. Small on purpose: each
    # attempt is a real request against the daily budget.
    MAX_ATTEMPTS = 4

    # How long a key sits out after a PER-DAY 429 before it is worth one more
    # request. Long enough that a genuinely spent key costs ~4 wasted requests
    # an hour, short enough that a key which rolls over mid-build comes back.
    DEAD_COOLDOWN_S = 900

    def __init__(self, model: str = "gemini-embedding-001",
                 dim: int = GEMINI_EMBED_DIM,
                 batch_size: int = GEMINI_EMBED_BATCH):
        from finagent.llm import collect_provider_keys

        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self._local = None          # thread-local sqlite handles, see `_conn`
        self.keys = collect_provider_keys("gemini")
        if not self.keys:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEYS (several keys "
                "separated by commas or whitespace), GEMINI_API_KEY, or "
                "GEMINI_API_KEY1..N to use a gemini-embedding model.")
        # key -> the time it may be tried again after a PER-DAY 429. Not a
        # permanent set: the keys are separate projects whose daily counters
        # roll over at different wall-clock times, so "exhausted" is a
        # statement about NOW, not about the rest of the run.
        self._dead: dict[str, float] = {}

    # -- plumbing ---------------------------------------------------------- #

    def _post(self, key: str, texts: list[str], task: str) -> list[list[float]]:
        """One `:batchEmbedContents` call. Raises `_HttpError` on a non-2xx so
        the caller can read the real status and body (the SDK hid both)."""
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
        """One request, with a WAIT-FIRST backoff. Raises rather than silently
        dropping chunks — an index missing rows is worse than a failed ingest.

        The retry policy is the load-bearing part, and the first version of it
        was actively harmful. It answered every 429 by immediately re-issuing
        against the next key, up to 2 full cycles of the pool. Each of those is
        a real HTTP request that COUNTS AGAINST THE DAILY QUOTA, so a per-minute
        throttle — which clears in seconds — was converted into ~12 wasted
        requests per batch, and a corpus build spent roughly 15,000 requests on
        retries and drained a 6,000/day budget after 7 of 72 filings.

        A rate limit means "slow down", so the fix is to slow down: sleep for
        exactly the delay the server names, then retry the SAME key. Rotation
        still happens, but only once per wait, as a cheap hedge in case the keys
        really are on separate projects. This is the same lesson `llm.py` records
        for Groq, where rotating on an unretryable error burned all 12 keys in
        0.6s.
        """
        import time

        last: Exception | None = None
        waits = 0
        attempt = 0
        while waits < self.MAX_ATTEMPTS:
            now = time.time()
            live = [k for k in self.keys if self._dead.get(k, 0) <= now]
            if not live:
                raise EmbeddingQuotaExhausted(
                    f"All {len(set(self.keys))} Gemini keys returned a DAILY "
                    f"embedding quota error within the last "
                    f"{self.DEAD_COOLDOWN_S // 60} minutes (~1000 contents/"
                    f"day/key for {self.model}). Vectors already in "
                    f"data/embed_cache are reusable, so the next build resumes "
                    f"rather than restarts.") from last
            key = live[(key_idx + attempt) % len(live)]
            attempt += 1
            try:
                got = self._post(key, texts, task)
                if len(got) != len(texts):
                    # The merge bug. Fail loudly: the alternative is a corpus
                    # embedded against the wrong text that looks fine.
                    raise RuntimeError(
                        f"{self.model} returned {len(got)} embedding(s) for a "
                        f"batch of {len(texts)} — this model does not honour "
                        f"batching and would silently merge the inputs. Set "
                        f"batch_size=1 for it.")
                return [_l2(v) for v in got]
            except _HttpError as e:
                last = e
                # A revoked or mistyped key is that KEY's problem, not the
                # pool's. Retire it for the rest of the run and carry on:
                # re-raising meant one dead key among eight killed every batch
                # that happened to land on it, which is how a single bad entry
                # in a consolidated `GEMINI_API_KEYS` secret would take down
                # embedding entirely. Measured: key 8 of 8 returned 401 while
                # the other seven were healthy.
                if e.status in (401, 403):
                    self._dead[key] = time.time() + 10 * 365 * 24 * 3600
                    print(f"  [embed] key …{key[-4:]} rejected ({e.status}); "
                          f"dropped from the pool for this run", flush=True)
                    continue
                if e.status != 429:
                    raise
                msg = e.body
                # A DAILY quota does not refill for hours, so this key is done
                # for the run — retire it and move to the next one WITHOUT
                # sleeping. Retiring rather than aborting matters because the
                # six keys are six separate projects whose daily counters reset
                # at different wall-clock times: a build launched inside that
                # stagger meets a mix of fresh and stale keys, and treating the
                # first stale one as fatal threw away a run that five live keys
                # could have finished (measured: died at 17 of 72 filings while
                # every key was minutes from resetting).
                if "PerDay" in msg:
                    self._dead[key] = time.time() + self.DEAD_COOLDOWN_S
                    # Printed, because this used to happen silently inside the
                    # worker threads: a run died reporting "every key is
                    # exhausted" with exactly ONE 429 visible in the log, and
                    # the five other retirements left no trace at all.
                    print(f"  [embed] key …{key[-4:]} out of DAILY quota — "
                          f"benched {self.DEAD_COOLDOWN_S // 60}m, "
                          f"{len([k for k in set(self.keys) if self._dead.get(k, 0) <= time.time()])}"
                          f"/{len(set(self.keys))} keys live", flush=True)
                    continue
                # Per-minute: wait exactly as long as the server asks, then try
                # again. Never burn a request to "check" whether it cleared.
                m = re.search(r"retryDelay['\"]?:\s*['\"](\d+(?:\.\d+)?)s", msg)
                time.sleep(float(m.group(1)) + 1 if m else 30)
                waits += 1
        raise RuntimeError(
            f"Gemini embeddings still rate-limited after "
            f"{self.MAX_ATTEMPTS} waits ({last})") from last

    # -- disk cache -------------------------------------------------------- #
    #
    # An embedding is a pure function of (model, dim, task, text), and the free
    # tier is metered, so paying for the same vector twice is pure waste. This
    # makes re-running a build FREE and makes a build after a chunker tweak cost
    # only the chunks whose text actually changed — which is what lets the same
    # corpus be re-indexed while iterating without draining the day's quota.
    #
    # sqlite because it is stdlib and one file; vectors are stored as raw
    # float32 (~6 KB each at 1536 dims, so ~270 MB for a 44.5k-chunk corpus).
    #
    # The connection is THREAD-LOCAL, and that is not optional. sqlite refuses a
    # connection used from a thread other than the one that opened it, and this
    # object is shared: `get_embeddings` is `lru_cache`d, so every caller in the
    # process gets the same instance. Within one `_embed_all` all DB access is
    # on the calling thread (the fan-out below only wraps network calls), which
    # is why this looked safe, but two concurrent callers — the API serving two
    # requests, or the concurrent-upsert path — land on one connection and
    # raise "SQLite objects created in a thread can only be used in that same
    # thread". Per-thread connections to one file are fine here: sqlite
    # serialises writers itself and the rows are immutable once written.

    def _conn(self):
        import sqlite3
        import threading

        local = getattr(self, "_local", None)
        if local is None:
            local = self._local = threading.local()
        db = getattr(local, "db", None)
        if db is None:
            path = Path(os.getenv("FINAGENT_EMBED_CACHE_DIR", "data/embed_cache"))
            path.mkdir(parents=True, exist_ok=True)
            db = local.db = sqlite3.connect(path / f"{self.model}_{self.dim}.sqlite")
            db.execute(
                "CREATE TABLE IF NOT EXISTS vec (k TEXT PRIMARY KEY, v BLOB)")
        return db

    def _key(self, task: str, text: str) -> str:
        import hashlib

        return hashlib.sha1(f"{task}\x00{text}".encode()).hexdigest()

    def _embed_all(self, texts: list[str], task: str) -> list[list[float]]:
        """Embed, preserving input order: cache first, then the API in parallel
        across the key pool for whatever is left."""
        from array import array
        from concurrent.futures import ThreadPoolExecutor

        keys = [self._key(task, t) for t in texts]
        db = self._conn()
        cached: dict[str, list[float]] = {}
        # Chunked IN(): sqlite caps a statement at 999 variables by default.
        uniq = list(dict.fromkeys(keys))
        for i in range(0, len(uniq), 900):
            window = uniq[i:i + 900]
            q = f"SELECT k, v FROM vec WHERE k IN ({','.join('?' * len(window))})"
            for k, blob in db.execute(q, window):
                cached[k] = array("f", blob).tolist()

        # One entry per DISTINCT miss — a corpus repeats boilerplate verbatim,
        # and the same text twice in one call is one embedding.
        todo = list(dict.fromkeys(k for k in keys if k not in cached))
        if todo:
            text_of_key = dict(zip(keys, texts))
            pending = [text_of_key[k] for k in todo]
            batches = [pending[i:i + self.batch_size]
                       for i in range(0, len(pending), self.batch_size)]
            if len(batches) == 1:
                fresh = self._embed_batch(batches[0], task, 0)
            else:
                with ThreadPoolExecutor(max_workers=GEMINI_EMBED_CONCURRENCY) as ex:
                    fresh = [v for batch in ex.map(
                        lambda ib: self._embed_batch(ib[1], task, ib[0]),
                        list(enumerate(batches))) for v in batch]
            # Round-trip through float32 before returning, not just before
            # storing. The API hands back float64 and the cache stores float32,
            # so without this a cold call and a warm call for the same text
            # return subtly different vectors — which would make "rebuild the
            # index" a silent source of drift and any A/B unreproducible.
            fresh = [array("f", v).tolist() for v in fresh]
            db.executemany("INSERT OR REPLACE INTO vec VALUES (?, ?)",
                           [(k, array("f", v).tobytes())
                            for k, v in zip(todo, fresh)])
            db.commit()
            cached.update(zip(todo, fresh))

        return [cached[k] for k in keys]

    # -- LangChain Embeddings interface ------------------------------------ #

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_all(list(texts), "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed_all([text], "RETRIEVAL_QUERY")[0]


def dim_for(model_name: str) -> int:
    """Dense dimensionality for an embedding model.

    Unknown models are measured once by embedding a probe string rather than
    guessed — a wrong dimension only surfaces at write time, thousands of chunks
    into an ingest run.
    """
    known = _MODEL_DIMS.get(model_name)
    if known:
        return known
    if model_name.startswith(GEMINI_EMBED_PREFIX):
        # Known from the request we make, so this costs no API call (and no
        # quota) on a path that runs at every collection create.
        return GEMINI_EMBED_DIM
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
    """Shared dense embedding function (GPU when available, else CPU).

    A `gemini-embedding*` name routes to the hosted API instead — same
    interface, so nothing downstream knows the difference.
    """
    if model_name.startswith(GEMINI_EMBED_PREFIX):
        return GeminiEmbeddings(model_name)

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
