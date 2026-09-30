"""Reranking: score (question, passage) pairs directly and reorder the candidates.

A reranker reads the question and a passage together, which is more accurate
than comparing two embeddings. Cohere's API is the default; the local
cross-encoder takes over when Cohere is unavailable.
"""

from __future__ import annotations

import os
import threading
import time
from functools import lru_cache

COHERE_PREFIX = "cohere:"                       # "cohere:rerank-v4.0-pro"
LOCAL_FALLBACK_RERANKER = "BAAI/bge-reranker-v2-m3"
# The reranker scores the parent passage (up to 2,500 characters), so its window
# must cover one. 1,024 tokens does; the model allows 8,192.
_MAX_LENGTH = 1024

_shared: dict = {}
_load_lock = threading.Lock()       # the local model is about 1 GB: load it once


@lru_cache(maxsize=1)
def get_device() -> str:
    """cuda, mps or cpu. Override with FINAGENT_DEVICE."""
    forced = os.getenv("FINAGENT_DEVICE")
    if forced:
        return forced.strip().lower()
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class CohereReranker:
    """Cohere's rerank API behind `predict(pairs) -> scores`.

    Cohere scores one query against a list of documents, so pairs are grouped
    by query and each group is one request.
    """

    MAX_DOCS = 500

    def __init__(self, model: str):
        from finagent.llm import collect_provider_keys

        self.model = model
        self.keys = collect_provider_keys("cohere")
        if not self.keys:
            raise RuntimeError("No Cohere API key found. Set COHERE_API_KEYS or "
                               "COHERE_API_KEY to use a cohere: reranker.")
        self._idx = 0

    def _post(self, query: str, docs: list[str]) -> list[float]:
        import json
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
                self._idx = (self._idx + attempt) % len(self.keys)   # stay on the key that worked
                scores = [0.0] * len(docs)
                for r in out.get("results", []):
                    scores[r["index"]] = float(r.get("relevance_score", 0.0))
                return scores
            except urllib.error.HTTPError as e:
                last = e
                if e.code not in (429, 500, 502, 503, 504):
                    raise
                if attempt and attempt % len(self.keys) == 0:
                    time.sleep(10)                  # every key failed once: let the minute roll
        raise RuntimeError(f"every Cohere key is rate-limited ({last})")

    def predict(self, pairs) -> list[float]:
        """`pairs` is [(query, passage), ...]; returns one score per pair."""
        pairs = list(pairs)
        scores = [0.0] * len(pairs)
        by_query: dict = {}
        for i, (q, doc) in enumerate(pairs):
            by_query.setdefault(q, []).append((i, doc))
        for query, group in by_query.items():
            for start in range(0, len(group), self.MAX_DOCS):
                window = group[start:start + self.MAX_DOCS]
                for (i, _), s in zip(window, self._post(query, [d for _, d in window])):
                    scores[i] = s
        return scores


class FallbackReranker:
    """Cohere first; the local cross-encoder when Cohere fails.

    After a failure Cohere is skipped for 15 minutes, so only the first request
    pays for discovering that it is down.
    """

    COOLDOWN_S = 900

    def __init__(self, remote, local_name: str):
        self.remote, self.local_name = remote, local_name
        self._blocked_until = 0.0

    def predict(self, pairs):
        if time.time() >= self._blocked_until:
            try:
                return self.remote.predict(pairs)
            except Exception as e:
                self._blocked_until = time.time() + self.COOLDOWN_S
                print(f"[rerank] Cohere unavailable ({type(e).__name__}: {str(e)[:120]}); "
                      f"using {self.local_name} for {self.COOLDOWN_S // 60}m", flush=True)
        return get_reranker(self.local_name).predict(pairs)


def get_reranker(model_name: str):
    """The shared reranker for `model_name`, loaded on first use."""
    reranker = _shared.get(model_name)
    if reranker is None:
        with _load_lock:
            reranker = _shared.get(model_name)
            if reranker is None:
                if model_name.startswith(COHERE_PREFIX):
                    reranker = FallbackReranker(
                        CohereReranker(model_name[len(COHERE_PREFIX):]),
                        LOCAL_FALLBACK_RERANKER)
                else:
                    from sentence_transformers import CrossEncoder

                    reranker = CrossEncoder(model_name, device=get_device(),
                                            max_length=_MAX_LENGTH)
                _shared[model_name] = reranker
    return reranker
