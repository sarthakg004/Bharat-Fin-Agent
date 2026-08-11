"""
reranker.py  ·  finagent/retrieval/reranker.py

Cross-encoder reranking. A reranker reads the query and a candidate passage
*together* and scores relevance directly — far more precise than embedding
distance — so it reorders a candidate pool to put the best passages on top.

The cross-encoder is loaded once per model name and shared process-wide
(`_get_shared_reranker`): the agent builds one retriever per collection, and
without sharing each loaded its own ~1 GB copy and blew past the memory budget.
"""

from __future__ import annotations

import threading

# Process-wide cache so every retriever reuses a single CrossEncoder per model
# name instead of loading a ~1.1 GB copy each.
_SHARED_RERANKERS: dict = {}

# The load takes ~139 s for v2-m3 on 2 vCPU, which is a wide enough window for
# two threads to miss the cache together — the startup warm-up thread and a
# first query racing it do exactly that. Two concurrent loads peak at twice the
# resident weights (the loser is then garbage) and that peak is what OOM-kills
# the container, so serialize them.
_LOAD_LOCK = threading.Lock()


# Scoring window, in tokens. The reranker scores the PARENT (collapse runs
# before rerank), so this must cover a whole parent or the tail is invisible to
# ranking. bge-reranker-base caps at 512 (~2000 chars) — fine for 1500-char
# parents, which is why base was never truncating today. v2-m3 allows 8192; we
# cap it at 1024 because that already covers a 2500-char parent and a larger
# window costs memory and time for nothing.
_MAX_LENGTH = {"BAAI/bge-reranker-v2-m3": 1024,
               # Same 1024 as v2-m3 so the two are compared on identical text
               # rather than on how much of the parent each one can see.
               "mixedbread-ai/mxbai-rerank-base-v2": 1024}
_DEFAULT_MAX_LENGTH = 512


# Prefix that routes a model name to Cohere's hosted rerank API instead of a
# local cross-encoder: "cohere:rerank-v4.0-pro".
COHERE_PREFIX = "cohere:"

# What a `cohere:` reranker degrades to when the key pool is spent. Must be a
# model the Dockerfile bakes, since the runtime sets HF_HUB_OFFLINE=1.
LOCAL_FALLBACK_RERANKER = "BAAI/bge-reranker-v2-m3"


class CohereReranker:
    """Cohere's hosted rerank API behind the `CrossEncoder.predict` interface.

    Exposes exactly `predict(pairs) -> list[float]` so it drops into
    `HybridRetriever._rerank` and `_cap_pool` with no other change.

    The shape mismatch is the whole job: a CrossEncoder scores N independent
    (query, passage) pairs, while Cohere scores ONE query against a LIST of
    documents. Our callers always pass a single repeated query, so the pairs are
    grouped by query and each group becomes one request — which is also what
    makes this affordable, since Cohere bills a "search unit" per query-with-up-
    to-100-documents rather than per passage.

    Cohere returns only ranked (index, relevance_score) entries, so scores are
    scattered back into the caller's original order; anything the API omits
    keeps 0.0 and simply sorts last.

    Keys rotate on a rate limit. MEASURED on this workload (48 parents of ~2.5k
    chars, ~30K tokens per request): one key sustained 25 calls/minute with no
    429 at a median 2.0s, and the keys are independent buckets.
    """

    # Cohere recommends against more than 1000 documents per request; our pools
    # are 48, so this only guards a caller that raises `pool_top_k` a lot.
    MAX_DOCS = 500

    def __init__(self, model: str):
        from finagent.llm import collect_provider_keys

        self.model = model
        self.keys = collect_provider_keys("cohere")
        if not self.keys:
            raise RuntimeError(
                "No Cohere API key found. Set COHERE_API_KEYS (several keys "
                "separated by commas or whitespace), COHERE_API_KEY, or "
                "COHERE_API_KEY1..N to use a cohere: reranker.")
        self._idx = 0

    def _post(self, query: str, docs: list[str]) -> list[float]:
        import json
        import time
        import urllib.error
        import urllib.request

        body = json.dumps({"model": self.model, "query": query,
                           "documents": docs, "top_n": len(docs)}).encode()
        last = None
        for attempt in range(len(self.keys) * 2):
            key = self.keys[(self._idx + attempt) % len(self.keys)]
            req = urllib.request.Request(
                "https://api.cohere.com/v2/rerank", data=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    out = json.loads(resp.read())
                # Stick with the key that worked, so we do not walk the pool
                # from key 1 on every call.
                self._idx = (self._idx + attempt) % len(self.keys)
                scores = [0.0] * len(docs)
                for r in out.get("results", []):
                    scores[r["index"]] = float(r.get("relevance_score", 0.0))
                return scores
            except urllib.error.HTTPError as e:
                last = e
                if e.code not in (429, 500, 502, 503, 504):
                    raise
                if attempt and attempt % len(self.keys) == 0:
                    time.sleep(10)      # a full cycle failed; let the window roll
        raise RuntimeError(f"every Cohere key is rate-limited ({last})")

    def predict(self, pairs):
        """`pairs` is [(query, passage), …]; returns one score per pair."""
        pairs = list(pairs)
        scores = [0.0] * len(pairs)
        by_query: dict = {}
        for i, (q, doc) in enumerate(pairs):
            by_query.setdefault(q, []).append((i, doc))
        for query, group in by_query.items():
            for start in range(0, len(group), self.MAX_DOCS):
                window = group[start:start + self.MAX_DOCS]
                got = self._post(query, [d for _, d in window])
                for (i, _), s in zip(window, got):
                    scores[i] = s
        return scores


class FallbackReranker:
    """Cohere first, the local cross-encoder when Cohere is unavailable.

    Cohere is the better reranker and is the default, but it is a free tier on
    a metered API: ten keys at roughly 1000 calls/month each. When that runs
    out, the alternative to a fallback is an exception out of
    `HybridRetriever._rerank`, which has no handler and takes the whole request
    down. Degrading to `bge-reranker-v2-m3` costs some ranking quality and
    keeps the product working, which is the right trade for a free deployment.

    The cooldown is the part worth explaining. Discovering that the pool is
    spent costs two full passes over ten keys plus a 10s sleep, and that
    verdict does not change between one request and the next. Latching it for
    `COOLDOWN_S` means the first request after exhaustion pays that price and
    the rest go straight to the local model. It expires rather than sticking,
    because the monthly quota does eventually roll over and a permanent latch
    would need a redeploy to clear.
    """

    COOLDOWN_S = 900

    def __init__(self, remote, local_name: str):
        self.remote = remote
        self.local_name = local_name
        self._local = None
        self._blocked_until = 0.0

    def _local_reranker(self):
        if self._local is None:
            self._local = _get_shared_reranker(self.local_name)
        return self._local

    def predict(self, pairs):
        import time

        if time.time() >= self._blocked_until:
            try:
                return self.remote.predict(pairs)
            except Exception as e:
                # Every failure mode lands here on purpose: a spent key pool
                # raises RuntimeError, a bad key raises HTTPError, and a
                # network blip raises URLError. None of them is worth failing
                # a user's question over when a local model is sitting in the
                # image.
                self._blocked_until = time.time() + self.COOLDOWN_S
                print(f"  [rerank] Cohere unavailable ({type(e).__name__}: "
                      f"{str(e)[:120]}); falling back to {self.local_name} "
                      f"for {self.COOLDOWN_S // 60}m", flush=True)
        return self._local_reranker().predict(pairs)


def _get_shared_reranker(model_name: str):
    """Return the shared reranker for `model_name`, loading it on first use.

    A `cohere:` prefix returns the hosted API adapter wrapped in
    `FallbackReranker`, so an exhausted key pool degrades to the local
    cross-encoder instead of failing the request. It is cached the same way so
    the key rotation and the fallback cooldown are shared process-wide.
    """
    reranker = _SHARED_RERANKERS.get(model_name)
    if reranker is None:
        with _LOAD_LOCK:
            # Re-check under the lock: whoever waited here while another thread
            # loaded must reuse that model, not load a second copy.
            reranker = _SHARED_RERANKERS.get(model_name)
            if reranker is None:
                if model_name.startswith(COHERE_PREFIX):
                    reranker = FallbackReranker(
                        CohereReranker(model_name[len(COHERE_PREFIX):]),
                        LOCAL_FALLBACK_RERANKER)
                    _SHARED_RERANKERS[model_name] = reranker
                    return reranker

                from sentence_transformers import CrossEncoder

                from finagent.device import get_device

                # trust_remote_code: several current rerankers (mxbai-v2)
                # ship a custom head. It executes code from the model repo, so
                # keep the set of models we actually load explicit.
                reranker = CrossEncoder(
                    model_name, device=get_device(), trust_remote_code=True,
                    max_length=_MAX_LENGTH.get(model_name, _DEFAULT_MAX_LENGTH))
                _SHARED_RERANKERS[model_name] = reranker
    return reranker
