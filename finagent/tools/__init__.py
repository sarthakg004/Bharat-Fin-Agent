"""Agent tools.

    market            yfinance market data (get_quote, get_history, call_tool…)
    web_search        Tavily search with a local-news fallback (WebSearcher)
    resolver          TickerCIKResolver — ticker/name → SEC CIK; backs the three
                      SEC tools below
    xbrl              XBRLClient — exact reported figures from company-facts
    calculator        FinancialCalculator — margins/ratios/growth/CAGR
    sec_fetch         SecFilingFetcher — fetch + ingest a missing 10-K on demand
    edgar_search      EdgarFullTextSearch — search across many companies
    base              BaseTool, the abstract interface
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
