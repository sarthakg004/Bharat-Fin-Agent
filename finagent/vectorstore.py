"""
vectorstore.py · finagent/vectorstore.py

One place that builds the vector store, so the agent code never constructs a
backend directly. We use a local, on-disk **Chroma** store: in production the
prebuilt `data/chroma` is baked into the container image, which keeps the full
hybrid retrieval (BM25 + dense + cross-encoder rerank) working — BM25 needs the
whole corpus in memory, so shipping the DB with the image is the simplest way to
keep it.

Environment
-----------
    CHROMA_DIR   directory of the Chroma store (default "data/chroma";
                 the container image sets it to the baked-in path).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=4)
def get_embeddings(model_name: str = DEFAULT_EMBED_MODEL):
    """Shared embedding function (GPU when available, else CPU)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    from finagent.device import get_device

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": get_device()},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_store(
    collection_name: str,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    chroma_dir: Optional[str] = None,
):
    """Return a LangChain ``Chroma`` store for ``collection_name``.

    Exposes ``.similarity_search(query, k=...)`` like any LangChain VectorStore.
    """
    from langchain_chroma import Chroma

    from finagent.chroma_client import chroma_kwargs_for_langchain

    if chroma_dir is None:
        chroma_dir = os.getenv("CHROMA_DIR", "data/chroma")
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(embedding_model),
        **chroma_kwargs_for_langchain(chroma_dir),
    )
