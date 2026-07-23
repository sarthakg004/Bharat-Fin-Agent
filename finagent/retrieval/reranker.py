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
_MAX_LENGTH = {"BAAI/bge-reranker-v2-m3": 1024}
_DEFAULT_MAX_LENGTH = 512


def _get_shared_reranker(model_name: str):
    """Return the shared `CrossEncoder` for `model_name`, loading it on first use."""
    reranker = _SHARED_RERANKERS.get(model_name)
    if reranker is None:
        with _LOAD_LOCK:
            # Re-check under the lock: whoever waited here while another thread
            # loaded must reuse that model, not load a second copy.
            reranker = _SHARED_RERANKERS.get(model_name)
            if reranker is None:
                from sentence_transformers import CrossEncoder

                from finagent.device import get_device

                reranker = CrossEncoder(
                    model_name, device=get_device(),
                    max_length=_MAX_LENGTH.get(model_name, _DEFAULT_MAX_LENGTH))
                _SHARED_RERANKERS[model_name] = reranker
    return reranker
