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

# Process-wide cache so every retriever reuses a single CrossEncoder per model
# name instead of loading a ~1.1 GB copy each.
_SHARED_RERANKERS: dict = {}


def _get_shared_reranker(model_name: str):
    """Return the shared `CrossEncoder` for `model_name`, loading it on first use."""
    reranker = _SHARED_RERANKERS.get(model_name)
    if reranker is None:
        from sentence_transformers import CrossEncoder

        from finagent.device import get_device

        reranker = CrossEncoder(model_name, device=get_device())
        _SHARED_RERANKERS[model_name] = reranker
    return reranker
