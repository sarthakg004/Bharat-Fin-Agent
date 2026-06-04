"""
edgar_search.py  ·  finagent/tools/edgar_search.py

EDGAR full-text search (🚧 Phase 6 — stub).

Answers cross-document questions ("which companies disclosed X") by querying
EDGAR's full-text search across many filings at once.
"""

from __future__ import annotations

from finagent.tools.base import BaseTool


class EdgarFullTextSearch(BaseTool):
    """Search EDGAR full-text across filings. Implemented in Phase 6."""

    name = "edgar_full_text_search"
    description = "Search the full text of SEC filings across many companies."

    def run(self, q: str, form: str = "10-K", n: int = 5) -> dict:
        raise NotImplementedError("EdgarFullTextSearch lands in Phase 6.")
