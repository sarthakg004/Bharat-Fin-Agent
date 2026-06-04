"""
sec_fetch.py  ·  finagent/tools/sec_fetch.py

Dynamic SEC filing fetch (🚧 Phase 5 — stub).

For companies not yet in the corpus: resolve the ticker, pull the latest filing
via the SEC submissions API, ingest it through the production pipeline, then
retrieve — a self-expanding corpus.
"""

from __future__ import annotations

from finagent.tools.base import BaseTool


class SecFilingFetcher(BaseTool):
    """Fetch + ingest an SEC filing on demand. Implemented in Phase 5."""

    name = "sec_fetch"
    description = "Fetch and ingest a company's latest SEC filing if not already indexed."

    def run(self, ticker: str, filing_type: str = "10-K", n: int = 1) -> dict:
        raise NotImplementedError("SecFilingFetcher lands in Phase 5.")
