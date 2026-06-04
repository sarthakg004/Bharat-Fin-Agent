"""
client.py  ·  finagent/corpus/client.py

Object-oriented wrapper around the on-disk Chroma store. Gives the rest of the
codebase one typed handle for collection access — counts, the LangChain store
for a collection, raw record fetches — instead of reaching for the low-level
helpers directly.

This wraps (does not replace) the foundational `finagent.chroma_client` and
`finagent.vectorstore` modules, which remain the single place that knows where
Chroma lives. Configuration defaults come from `finagent.config.settings`.
"""

from __future__ import annotations

from typing import Optional

from finagent.config import settings


class ChromaClient:
    """A thin handle over the persistent Chroma store.

    Args:
        chroma_dir: Persistent Chroma directory. Defaults to `settings.chroma_dir`.
        embedding_model: Embedding model for LangChain stores. Defaults to
            `settings.embedding_model`.
    """

    def __init__(
        self,
        chroma_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.chroma_dir = chroma_dir or settings.chroma_dir
        self.embedding_model = embedding_model or settings.embedding_model

    def get_store(self, collection: str):
        """Return a LangChain `Chroma` store for `collection`.

        The store exposes `.similarity_search(query, k=...)` and the underlying
        `._collection` for raw access.
        """
        from finagent.vectorstore import build_store

        return build_store(collection, self.embedding_model, self.chroma_dir)

    def count(self, collection: str) -> int:
        """Number of chunks indexed in `collection`."""
        return self.get_store(collection)._collection.count()

    def list_collections(self) -> list[str]:
        """Names of all collections in the store."""
        from finagent.chroma_client import get_chroma_client

        return [c.name for c in get_chroma_client().list_collections()]

    def fetch_all(self, collection: str, include=("documents", "metadatas"),
                  batch_size: int = 5000) -> dict:
        """Page through a collection and return all requested fields.

        Returns a dict with the requested keys, each a flat list across batches.
        """
        col = self.get_store(collection)._collection
        out: dict = {k: [] for k in include}
        offset = 0
        while True:
            batch = col.get(include=list(include), limit=batch_size, offset=offset)
            docs = batch.get("documents") if "documents" in include else batch.get(list(include)[0])
            got = docs or []
            if not got:
                break
            for k in include:
                out[k].extend(batch.get(k) or [])
            offset += len(got)
            if len(got) < batch_size:
                break
        return out

    def __repr__(self) -> str:
        return f"ChromaClient(chroma_dir={self.chroma_dir!r})"
