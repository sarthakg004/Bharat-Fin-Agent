"""
xbrl.py  ·  finagent/tools/xbrl.py

XBRL facts tool (🚧 Phase 3 — stub).

Returns exact reported figures from SEC XBRL company-facts data, handling the
tag-heterogeneity problem (e.g. `Revenues` vs
`RevenueFromContractWithCustomerExcludingAssessedTax`).
"""

from __future__ import annotations

from finagent.tools.base import BaseTool


class XBRLClient(BaseTool):
    """Fetch structured financial facts from SEC XBRL. Implemented in Phase 3."""

    name = "xbrl_facts"
    description = "Look up an exact reported financial figure for a company/period."

    def run(self, ticker: str, concept: str, period: str) -> dict:
        raise NotImplementedError("XBRLClient lands in Phase 3.")
