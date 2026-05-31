"""
web_search.py  ·  src/graph/web_search.py

Web search for out-of-corpus questions, with a graceful local fallback.

Strategy:
    1. If `TAVILY_API_KEY` is set → `TavilyClient.search` (best quality).
    2. Otherwise (or on Tavily error) → similarity-search the local `news`
       Chroma collection, which we built from the Indian_Financial_News dataset.

Both paths return a uniform list[dict] shape:

    [{"title", "url", "content", "score", "source"}, ...]

so the synthesizer can cite them identically. `source` is either `"tavily"` or
`"news_local"` so the answer text can mark the provenance with a tag like
`<News: ...>` while staying inside the inline-citation regex used elsewhere.

Usage as a library
------------------
    from src.graph.web_search import WebSearcher

    ws = WebSearcher(collection_name="news")
    hits = ws.search("Reliance Q3 FY24 results", k=3)
    for h in hits:
        print(h["source"], h["title"])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv

load_dotenv()


class WebSearcher:
    """Unified web/news search with Tavily → local-news fallback."""

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        chroma_dir: Union[str, Path] = "data/chroma",
        collection_name: str = "news",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        top_k: int = 3,
    ):
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.chroma_dir = str(chroma_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.top_k = top_k

        self._tavily = None
        self._news_store = None

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def search(self, query: str, k: Optional[int] = None) -> list[dict]:
        """Return top-k results. Tavily first; falls back to local news on any error."""
        k = k or self.top_k
        if self.tavily_api_key:
            try:
                return self._tavily_search(query, k)
            except Exception as e:
                print(f"[WebSearcher] Tavily failed ({type(e).__name__}: {e}); "
                      f"falling back to local news collection.")
        return self._local_news_search(query, k)

    # ------------------------------------------------------------------ #
    # Tavily
    # ------------------------------------------------------------------ #

    def _tavily_client(self):
        if self._tavily is None:
            from tavily import TavilyClient
            self._tavily = TavilyClient(api_key=self.tavily_api_key)
        return self._tavily

    def _tavily_search(self, query: str, k: int) -> list[dict]:
        client = self._tavily_client()
        # search_depth="basic" is the free tier; "advanced" costs more credits.
        resp = client.search(query=query, max_results=k, search_depth="basic")
        results = resp.get("results", []) if isinstance(resp, dict) else []
        return [
            {
                "title": (r.get("title") or "")[:240],
                "url": r.get("url") or "",
                "content": (r.get("content") or "")[:1200],
                "score": float(r.get("score", 0.0) or 0.0),
                "source": "tavily",
            }
            for r in results
        ]

    # ------------------------------------------------------------------ #
    # Local news fallback
    # ------------------------------------------------------------------ #

    def _news_store_handle(self):
        if self._news_store is None:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings

            emb = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                encode_kwargs={"normalize_embeddings": True},
            )
            self._news_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=emb,
                persist_directory=self.chroma_dir,
            )
        return self._news_store

    def _local_news_search(self, query: str, k: int) -> list[dict]:
        try:
            store = self._news_store_handle()
            docs = store.similarity_search(query, k=k)
        except Exception as e:
            print(f"[WebSearcher] news collection unavailable ({e}); returning [].")
            return []
        return [
            {
                "title": (d.metadata.get("title") or "")[:240],
                "url": d.metadata.get("url") or "",
                "content": (d.page_content or "")[:1200],
                "score": 1.0,           # similarity scores aren't exposed by the wrapper
                "source": "news_local",
                "date": d.metadata.get("date", ""),
            }
            for d in docs
        ]
