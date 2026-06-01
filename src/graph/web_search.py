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


TRUSTED_FINANCIAL_DOMAINS: tuple[str, ...] = (
    # India
    "moneycontrol.com",
    "economictimes.indiatimes.com",
    "livemint.com",
    "business-standard.com",
    "thehindubusinessline.com",
    "tradingview.com",
    "screener.in",
    "nseindia.com",
    "bseindia.com",
    "sebi.gov.in",
    "rbi.org.in",
    # Global
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "cnbc.com",
    "barrons.com",
    "marketwatch.com",
    "finance.yahoo.com",
    "seekingalpha.com",
    "investopedia.com",
    "sec.gov",
)


class WebSearcher:
    """Unified web/news search with Tavily → local-news fallback.

    The Tavily path runs **two passes** per call: one restricted to
    `TRUSTED_FINANCIAL_DOMAINS` (Moneycontrol, ET, TradingView, Reuters,
    Bloomberg, …) and one unrestricted general search. Trusted hits are
    returned first so the synthesizer can prefer them; each hit carries a
    `tier` of `"trusted"` or `"web"` for that decision.
    """

    # Tavily's `max_results` accepts up to 20. We default to 10 so the
    # synthesizer has enough material to answer multi-faceted questions
    # ("performance over the last year" rarely fits in 3 snippets).
    MAX_RESULTS_CAP = 20
    TRUSTED_DOMAINS = TRUSTED_FINANCIAL_DOMAINS

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        chroma_dir: Union[str, Path] = "data/chroma",
        collection_name: str = "news",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        top_k: int = 10,
        trusted_ratio: float = 0.7,
    ):
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.chroma_dir = str(chroma_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.top_k = min(top_k, self.MAX_RESULTS_CAP)
        # Share of slots reserved for trusted-domain results (the rest go to
        # general web). 0.7 → with k=10, ~7 trusted + ~3 general.
        self.trusted_ratio = max(0.0, min(1.0, trusted_ratio))

        self._tavily = None
        self._news_store = None

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def search(self, query: str, k: Optional[int] = None) -> list[dict]:
        """Return top-k results. Tavily first; falls back to local news on any error."""
        k = min(k or self.top_k, self.MAX_RESULTS_CAP)
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
        """Two-pass search: trusted-domain first, then unrestricted general.

        We do separate calls because Tavily's `include_domains` is a hard
        filter — combining it with general results in a single call isn't
        possible. Returns trusted hits first so the synthesizer can prefer
        them; duplicates by URL are dropped.
        """
        client = self._tavily_client()
        trusted_k = max(1, int(round(k * self.trusted_ratio)))
        general_k = max(0, k - trusted_k)
        # search_depth="basic" is the free tier; "advanced" costs more credits.
        common = dict(search_depth="basic")

        hits: list[dict] = []
        seen_urls: set[str] = set()

        # --- 1. Trusted financial domains -----------------------------------
        try:
            resp = client.search(
                query=query, max_results=trusted_k,
                include_domains=list(self.TRUSTED_DOMAINS),
                **common,
            )
            for r in (resp.get("results") or []):
                norm = self._normalize_tavily(r, tier="trusted")
                if norm["url"] and norm["url"] not in seen_urls:
                    seen_urls.add(norm["url"])
                    hits.append(norm)
        except Exception as e:
            print(f"[Tavily] trusted-domain search failed ({type(e).__name__}: {e})")

        # --- 2. General web (only if we still have budget) ------------------
        if general_k > 0:
            try:
                resp = client.search(query=query, max_results=general_k, **common)
                for r in (resp.get("results") or []):
                    norm = self._normalize_tavily(r, tier="web")
                    if norm["url"] and norm["url"] not in seen_urls:
                        seen_urls.add(norm["url"])
                        hits.append(norm)
            except Exception as e:
                print(f"[Tavily] general search failed ({type(e).__name__}: {e})")

        return hits

    @staticmethod
    def _normalize_tavily(r: dict, tier: str) -> dict:
        return {
            "title": (r.get("title") or "")[:240],
            "url": r.get("url") or "",
            "content": (r.get("content") or "")[:1500],
            "score": float(r.get("score") or 0.0),
            "source": "tavily",
            "tier": tier,
        }

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
