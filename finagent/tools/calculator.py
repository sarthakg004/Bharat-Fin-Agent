"""
calculator.py  ·  finagent/tools/calculator.py

Financial calculator (🚧 Phase 4 — stub).

Computes derived metrics (margins, YoY growth, CAGR, ratios) from exact XBRL
inputs rather than retrieved prose, with verified formulas.
"""

from __future__ import annotations

from finagent.tools.base import BaseTool


class FinancialCalculator(BaseTool):
    """Compute derived financial metrics from XBRL inputs. Implemented in Phase 4."""

    name = "financial_calculator"
    description = "Compute margins, growth rates, ratios, and CAGR from exact inputs."

    def run(self, metric: str, **kwargs) -> dict:
        raise NotImplementedError("FinancialCalculator lands in Phase 4.")
