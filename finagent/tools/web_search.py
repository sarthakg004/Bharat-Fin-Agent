"""
web_search.py  ·  finagent/graph/web_search.py

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
    from finagent.graph.web_search import WebSearcher

    ws = WebSearcher(collection_name="news")
    hits = ws.search("Reliance Q3 FY24 results", k=3)
    for h in hits:
        print(h["source"], h["title"])
"""

from __future__ import annotations

import os
import re
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
    # Market-data / screeners / analysis (Tavily extracts these server-side, so
    # sites that block direct scraping are still searchable through it).
    "investing.com",
    "stockanalysis.com",
    "finviz.com",
    "benzinga.com",
    "fool.com",
    "simplywall.st",
    "morningstar.com",
    "zacks.com",
    "moneyview.in",
    "businesstoday.in",
    "ndtvprofit.com",
)


# Social / UGC / video domains that are noise for a financial answer. The
# general (untrusted) pass excludes these so we never surface an Instagram or
# Reddit post as "market news".
JUNK_DOMAINS: tuple[str, ...] = (
    "instagram.com", "facebook.com", "tiktok.com", "twitter.com", "x.com",
    "reddit.com", "pinterest.com", "youtube.com", "quora.com", "linkedin.com",
    "threads.net", "medium.com",
)


# Recency cues in the query → narrower Tavily `time_range`. Without these,
# Tavily returns articles by relevance with no time bias, so a "premarket"
# question can come back with last quarter's premarket move.
_RECENCY_DAY_MARKERS = (
    "premarket", "pre-market", "today", "this morning",
    "right now", "current price", "live price", "as of now",
)
_RECENCY_WEEK_MARKERS = (
    "yesterday", "this week", "latest", "this quarter results",
)
# Stock / company news is generally only useful within ~30 days.
_DEFAULT_NEWS_RANGE = "month"


def infer_time_range(query: str) -> str:
    """day | week | month — used by Tavily to drop stale articles."""
    q = (query or "").lower()
    if any(m in q for m in _RECENCY_DAY_MARKERS):
        return "day"
    if any(m in q for m in _RECENCY_WEEK_MARKERS):
        return "week"
    return _DEFAULT_NEWS_RANGE


# Past-year / historical-fact markers. A question about FY2021-2023 acquisitions
# is a HISTORICAL lookup, not "latest news" — applying a recency window to it
# returns recent articles that merely mention the topic for unrelated companies.
_PAST_YEAR_RE = re.compile(r"\b(?:fy\s*)?(19|20)\d{2}\b", re.I)
_HISTORICAL_MARKERS = (
    "acquisition", "acquisitions", "acquire", "acquired", "merger", "merged",
    "divestiture", "divest", "spin-off", "spinoff", "history", "historical",
    "over the years", "in the past", "previously", "founded", "since",
)


def is_historical(query: str) -> bool:
    """True when the query is a historical/factual lookup rather than fresh news,
    so the web search should run ALL-TIME by relevance (no recency window)."""
    import datetime
    q = (query or "").lower()
    if any(m in q for m in _RECENCY_DAY_MARKERS + _RECENCY_WEEK_MARKERS):
        return False
    cur_year = datetime.date.today().year
    for m in _PAST_YEAR_RE.finditer(q):
        if int(m.group(0)[-4:]) < cur_year:        # a real past-year reference
            return True
    return any(t in q for t in _HISTORICAL_MARKERS)


class WebSearcher:
    """Unified web/news search with Tavily → local-news fallback.

    The Tavily path runs **two passes** per call: one restricted to
    `TRUSTED_FINANCIAL_DOMAINS` (Moneycontrol, ET, TradingView, Reuters,
    Bloomberg, …) and one unrestricted general search. Trusted hits are
    returned first so the synthesizer can prefer them; each hit carries a
    `tier` of `"trusted"` or `"web"` for that decision.

    All Tavily calls request `topic="news"` and a `time_range` inferred from
    the query (day / week / month), so dated content gets filtered server-side
    rather than being lumped into the synthesizer's context.
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

    # Fallback widths when the inferred time_range returns too few hits.
    # Order matters — we expand outwards: day → week → month.
    _TIME_RANGE_FALLBACK = {
        "day":   ["day", "week", "month"],
        "week":  ["week", "month"],
        "month": ["month"],
    }
    # Stop widening once we've collected this many useful hits.
    _ENOUGH_HITS = 4

    def _tavily_search(self, query: str, k: int) -> list[dict]:
        """Trusted-first search with progressive time-range fallback.

        Pass 1 queries ONLY the trusted financial domains (Moneycontrol, ET,
        TradingView, Reuters, Bloomberg, Yahoo Finance, …), widening the time
        window day→week→month until it has enough hits. Pass 2 runs a general
        web search **only to fill the remaining slots**, and EXCLUDES social /
        UGC domains so an Instagram or Reddit post never shows up as market news.
        Recency comes from `time_range`, inferred from the query, so the latest
        coverage is preferred.

        `topic` is left at Tavily's default — combining `topic="news"` with
        `include_domains` and a tight `time_range` over-filters into generic
        headlines instead of stock-specific results.
        """
        client = self._tavily_client()
        hits: list[dict] = []
        seen_urls: set[str] = set()
        # Historical/factual lookups (FY2021-2023 acquisitions, company history)
        # search ALL-TIME by relevance — a recency window would return recent
        # articles that merely mention the topic for unrelated companies. Fresh
        # questions keep the day→week→month recency fallback. `None` time_range
        # means "no time filter" (omitted from the Tavily call below).
        if is_historical(query):
            ranges: list = [None]
            general_topic = "general"      # not "news" — we want reference pages
        else:
            ranges = self._TIME_RANGE_FALLBACK[infer_time_range(query)]
            general_topic = None           # Tavily default

        def _collect(resp, tier):
            for r in (resp.get("results") or []):
                norm = self._normalize_tavily(r, tier=tier)
                if norm["url"] and norm["url"] not in seen_urls:
                    seen_urls.add(norm["url"])
                    hits.append(norm)

        def _search(max_results, time_range, **extra):
            kw = dict(query=query, max_results=max_results, search_depth="basic", **extra)
            if time_range is not None:
                kw["time_range"] = time_range
            return client.search(**kw)

        # --- Pass 1: trusted financial domains first ----------------------
        for time_range in ranges:
            if len(hits) >= self._ENOUGH_HITS:
                break
            try:
                _collect(_search(k, time_range,
                                 include_domains=list(self.TRUSTED_DOMAINS)), tier="trusted")
            except Exception as e:
                print(f"[Tavily] trusted ({time_range}) failed ({type(e).__name__}: {e})")

        # --- Pass 2: general web ONLY to fill the gap, excluding junk -------
        if len(hits) < k:
            extra = {"exclude_domains": list(JUNK_DOMAINS)}
            if general_topic:
                extra["topic"] = general_topic
            for time_range in ranges:
                if len(hits) >= k:
                    break
                try:
                    _collect(_search(k - len(hits), time_range, **extra), tier="web")
                except Exception as e:
                    print(f"[Tavily] general ({time_range}) failed ({type(e).__name__}: {e})")

        return hits[:k]

    @staticmethod
    def _normalize_tavily(r: dict, tier: str) -> dict:
        # Tavily's `published_date` for news topic is ISO-8601; first 10 chars
        # are the YYYY-MM-DD we want to surface to the synthesizer.
        pub = (r.get("published_date") or "")[:10]
        return {
            "title": (r.get("title") or "")[:240],
            "url": r.get("url") or "",
            "content": (r.get("content") or "")[:1500],
            "score": float(r.get("score") or 0.0),
            "source": "tavily",
            "tier": tier,
            "published_date": pub,
        }

    # ------------------------------------------------------------------ #
    # Local news fallback
    # ------------------------------------------------------------------ #

    def _news_store_handle(self):
        if self._news_store is None:
            from finagent.vectorstore import build_store

            self._news_store = build_store(
                self.collection_name, self.embedding_model, self.chroma_dir
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
                "tier": "web",
                "published_date": (d.metadata.get("date") or "")[:10],
            }
            for d in docs
        ]
