"""
embedder.py  ·  finagent/ingestion/embedder.py

The embedding step, exposed standalone so the notebook can show text becoming a
vector and demonstrate semantic similarity. Uses the SAME model as everything
else (`BAAI/bge-small-en-v1.5`, L2-normalized) via the shared `get_embeddings`
cache, so a query embedded here matches what's stored in Chroma.
"""

from __future__ import annotations

import numpy as np

from finagent.vectorstore import DEFAULT_EMBED_MODEL, get_embeddings


def embed_text(text: str, model_name: str = DEFAULT_EMBED_MODEL) -> np.ndarray:
    """Embed a single string into a normalized float32 vector."""
    vec = get_embeddings(model_name).embed_query(text)
    return np.asarray(vec, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (≈ dot product when normalized)."""
    a, b = np.asarray(a), np.asarray(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)
