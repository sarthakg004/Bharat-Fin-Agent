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

from typing import Optional

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


class CrossEncoderReranker:
    """OOP wrapper over the shared cross-encoder.

    Args:
        model_name: HuggingFace cross-encoder name. Defaults to the configured
            `settings.reranker_model`.
    """

    def __init__(self, model_name: Optional[str] = None):
        if model_name is None:
            from finagent.config import settings

            model_name = settings.reranker_model
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """The lazily-loaded, process-shared CrossEncoder."""
        if self._model is None:
            self._model = _get_shared_reranker(self.model_name)
        return self._model

    def rerank(
        self, query: str, candidates: list[tuple[str, dict]], top_k: Optional[int] = None
    ) -> list[tuple[str, dict]]:
        """Reorder `(text, metadata)` candidates by relevance to `query`."""
        if not candidates:
            return []
        scores = self.model.predict([(query, text) for text, _ in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        out = [c for c, _ in ranked]
        return out[:top_k] if top_k else out

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model_name={self.model_name!r})"
