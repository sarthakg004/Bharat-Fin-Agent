"""
resolver.py  ·  finagent/tools/resolver.py

Ticker → CIK resolver (🚧 Phase 2 — stub).

Maps a ticker symbol or company name to the SEC's zero-padded 10-digit CIK,
the identifier required by the XBRL and EDGAR APIs. Shared infrastructure for
the XBRL, dynamic-fetch, and full-text-search tools.
"""

from __future__ import annotations

from finagent.tools.base import BaseTool


class TickerCIKResolver(BaseTool):
    """Resolve a ticker or company name to its SEC CIK. Implemented in Phase 2."""

    name = "ticker_cik_resolver"
    description = "Resolve a ticker symbol or company name to its SEC CIK."

    def run(self, query: str) -> dict:
        raise NotImplementedError("TickerCIKResolver lands in Phase 2.")
