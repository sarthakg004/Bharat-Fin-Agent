"""
base.py  ·  finagent/retrieval/base.py

Abstract interface for the retrieval layer. Implementations (BM25, dense,
hybrid) expose a common `retrieve(query, k)` so the agent can be pointed at any
retriever without code changes.

Return type note: concrete retrievers in this codebase return ``(text, metadata)``
tuples (the shape the agent's nodes consume), so `retrieve` is typed as
``list`` rather than forcing ``list[Document]``. New implementations may return
either, as long as they're consistent for their consumer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """Common interface for all retrievers."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list:
        """Return up to `k` results most relevant to `query`."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
