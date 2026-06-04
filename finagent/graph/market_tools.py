"""Back-compat shim — market tools MOVED to `finagent.tools.market`.

Re-exported here so existing imports keep working, e.g.:
    from finagent.graph.market_tools import call_tool
Remove this shim once every consumer imports from `finagent.tools.market`.
"""

from finagent.tools.market import (  # noqa: F401
    normalise_ticker,
    get_quote,
    get_history,
    get_company_info,
    get_news,
    compare,
    call_tool,
)

__all__ = [
    "normalise_ticker", "get_quote", "get_history", "get_company_info",
    "get_news", "compare", "call_tool",
]
