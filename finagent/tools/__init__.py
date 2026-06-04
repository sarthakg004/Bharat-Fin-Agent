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

Roadmap tools (stubs — `BaseTool` subclasses, implemented in later phases):
    EdgarFullTextSearch (Phase 6).

Infrastructure:
    BaseTool, ToolRegistry — interface + name→tool registry.

Depends on: config, llm, retrieval (some tools).
"""

from finagent.tools.base import BaseTool, ToolRegistry

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
    "BaseTool", "ToolRegistry",
    "call_tool", "get_quote", "get_company_info", "WebSearcher",
    "TickerCIKResolver", "XBRLClient", "FinancialCalculator",
    "SecFilingFetcher", "EdgarFullTextSearch",
]
