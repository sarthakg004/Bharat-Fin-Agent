"""Agent tools.

Live tools (migrated here, re-exported from old `finagent.graph` paths):
    market.*      — yfinance market data (get_quote, get_history, …, call_tool).
    web_search.*  — Tavily web search with local-news fallback (WebSearcher).

Shared infrastructure:
    TickerCIKResolver (Phase 2) — ticker/name → SEC CIK; backs XBRL, fetch, FTS.

Structured-data tools:
    XBRLClient (Phase 3)        — exact reported figures from SEC XBRL company-facts.
    FinancialCalculator (Phase 4) — margins/ratios/growth/CAGR/trends over XBRL inputs.

Corpus-expansion tools:
    SecFilingFetcher (Phase 5)  — fetch + ingest a missing US 10-K on demand.

Cross-document tools:
    EdgarFullTextSearch (Phase 6) — full-text search across many companies' filings.

Infrastructure:
    BaseTool — abstract tool interface.

Depends on: config, llm, retrieval (some tools).
"""

from finagent.tools.base import BaseTool

# Live tools
from finagent.tools.market import call_tool, get_quote, get_company_info
from finagent.tools.web_search import WebSearcher

# Roadmap stubs
from finagent.tools.resolver import TickerCIKResolver
from finagent.tools.xbrl import XBRLClient
from finagent.tools.calculator import FinancialCalculator
from finagent.tools.sec_fetch import SecFilingFetcher
from finagent.tools.edgar_search import EdgarFullTextSearch

__all__ = [
    "BaseTool",
    "call_tool", "get_quote", "get_company_info", "WebSearcher",
    "TickerCIKResolver", "XBRLClient", "FinancialCalculator",
    "SecFilingFetcher", "EdgarFullTextSearch",
]
