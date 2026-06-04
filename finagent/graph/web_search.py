"""Back-compat shim — web search MOVED to `finagent.tools.web_search`.

Re-exported here so existing imports keep working, e.g.:
    from finagent.graph.web_search import WebSearcher
Remove this shim once every consumer imports from `finagent.tools.web_search`.
"""

from finagent.tools.web_search import (  # noqa: F401
    WebSearcher,
    infer_time_range,
    TRUSTED_FINANCIAL_DOMAINS,
)

__all__ = ["WebSearcher", "infer_time_range", "TRUSTED_FINANCIAL_DOMAINS"]
