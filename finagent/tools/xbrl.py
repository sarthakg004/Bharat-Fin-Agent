"""
xbrl.py  ·  finagent/tools/xbrl.py

XBRL facts tool (Phase 3 — the big accuracy unlock).

Answers numeric questions from **structured** SEC data instead of retrieved
prose. Given a ticker (or company name), a financial concept, and a period, it
returns the *exact reported figure* a company filed — no LLM in the number path,
so there's nothing to hallucinate.

Pipeline:
  ticker/name --(Phase 2 resolver)--> CIK
  CIK --> data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json  (downloaded + cached)
  concept --> the right US-GAAP *tag*, then period --> the exact fact value.

The hard part is **tag heterogeneity**: the same economic concept is filed under
different XBRL tags by different companies (revenue is `Revenues` for some,
`RevenueFromContractWithCustomerExcludingAssessedTax` for others). We handle it
with two layers:
  1. a curated **concept → candidate-tags** map (ordered by preference), and
  2. a cheap **LLM fallback** (`tag_resolver`) that picks the best tag from the
     tags the company *actually* reports, when the map misses.

This tool is decoupled from the agent: it takes an optional `resolver`
(`TickerCIKResolver`) and an optional `tag_resolver` callable so it stays unit
-testable without the graph. The graph wires a router-tier LLM into
`tag_resolver` in Phase 3's `xbrl_node`.
"""

from __future__ import annotations

import json
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import requests

from finagent.tools.base import BaseTool
from finagent.tools.resolver import TickerCIKResolver

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
DEFAULT_CACHE_DIR = "data/cache/xbrl"
DEFAULT_TTL_DAYS = 7

# Concept → candidate US-GAAP tags, most-preferred first. The picker walks this
# list and takes the first tag the company actually reports. Covers the line
# items FinanceBench numeric questions ask about; the LLM fallback handles the
# long tail.
CONCEPT_TAGS: dict[str, list[str]] = {
    "revenue": [
        # `Revenues` is the consolidated top line when present (e.g. Walmart's
        # $572.7B total vs $567.8B net product sales, Pfizer's $100.3B total vs
        # $91.8B product). Companies that don't file it (Apple, Microsoft) fall
        # through to RevenueFromContract, which is then their total.
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "cost_of_revenue": [
        "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "operating_expenses": ["OperatingExpenses", "CostsAndExpenses"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "income_tax": ["IncomeTaxExpenseBenefit"],
    "rd_expense": ["ResearchAndDevelopmentExpense",
                   "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "sga_expense": ["SellingGeneralAndAdministrativeExpense"],
    "interest_expense": ["InterestExpense", "InterestExpenseDebt"],
    "depreciation_amortization": ["DepreciationDepletionAndAmortization",
                                  "DepreciationAmortizationAndAccretionNet",
                                  "DepreciationAndAmortization"],
    "eps_basic": ["EarningsPerShareBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "total_liabilities": ["Liabilities"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "stockholders_equity": ["StockholdersEquity",
                            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "inventory": ["InventoryNet"],
    "accounts_receivable": ["AccountsReceivableNetCurrent"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "total_debt": ["DebtLongtermAndShorttermCombinedAmount", "LongTermDebt"],
    "goodwill": ["Goodwill"],
    "ppe_net": ["PropertyPlantAndEquipmentNet"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "dividends_paid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "shares_outstanding": ["CommonStockSharesOutstanding",
                           "WeightedAverageNumberOfDilutedSharesOutstanding",
                           "WeightedAverageNumberOfSharesOutstandingBasic"],
}

# Free-text concept → canonical key. Matched by longest keyword found in the
# (lowercased) concept string, so "total revenue" and "net sales" both → revenue.
_CONCEPT_KEYWORDS: list[tuple[str, str]] = [
    ("cost of goods", "cost_of_revenue"), ("cost of revenue", "cost_of_revenue"),
    ("cost of sales", "cost_of_revenue"),
    ("gross profit", "gross_profit"), ("gross margin", "gross_profit"),
    ("operating income", "operating_income"), ("operating profit", "operating_income"),
    ("operating expense", "operating_expenses"),
    ("net income", "net_income"), ("net profit", "net_income"),
    ("net earnings", "net_income"), ("profit attributable", "net_income"),
    ("pretax", "pretax_income"), ("pre-tax", "pretax_income"),
    ("income tax", "income_tax"), ("tax expense", "income_tax"),
    ("research and development", "rd_expense"), ("r&d", "rd_expense"),
    ("selling, general", "sga_expense"), ("sg&a", "sga_expense"),
    ("interest expense", "interest_expense"),
    ("depreciation", "depreciation_amortization"), ("amortization", "depreciation_amortization"),
    ("diluted eps", "eps_diluted"), ("diluted earnings per share", "eps_diluted"),
    ("basic eps", "eps_basic"), ("basic earnings per share", "eps_basic"),
    ("earnings per share", "eps_diluted"), ("eps", "eps_diluted"),
    ("total assets", "total_assets"),
    ("current assets", "current_assets"),
    ("total liabilities", "total_liabilities"),
    ("current liabilities", "current_liabilities"),
    ("shareholders equity", "stockholders_equity"),
    ("shareholders' equity", "stockholders_equity"),
    ("stockholders equity", "stockholders_equity"),
    ("total equity", "stockholders_equity"), ("book value", "stockholders_equity"),
    ("cash and cash equivalents", "cash"), ("cash equivalents", "cash"),
    ("inventory", "inventory"), ("inventories", "inventory"),
    ("accounts receivable", "accounts_receivable"), ("receivable", "accounts_receivable"),
    ("long-term debt", "long_term_debt"), ("long term debt", "long_term_debt"),
    ("total debt", "total_debt"),
    ("goodwill", "goodwill"),
    ("property, plant", "ppe_net"), ("property plant", "ppe_net"),
    ("capital expenditure", "capex"), ("capex", "capex"),
    ("operating cash flow", "operating_cash_flow"),
    ("cash from operations", "operating_cash_flow"),
    ("cash flow from operating", "operating_cash_flow"),
    ("dividend", "dividends_paid"),
    ("shares outstanding", "shares_outstanding"),
    ("revenue", "revenue"), ("net sales", "revenue"), ("total sales", "revenue"),
    ("sales", "revenue"), ("turnover", "revenue"),
]


def _year_of(period) -> Optional[int]:
    """Pull a 4-digit fiscal year out of 'FY2022' / '2022' / 'fy22' / 2022."""
    if period is None:
        return None
    if isinstance(period, int):
        return period
    m = re.search(r"(19|20)\d{2}", str(period))
    if m:
        return int(m.group(0))
    m2 = re.search(r"\b(\d{2})\b", str(period))   # bare 'fy22'
    if m2:
        return 2000 + int(m2.group(1))
    return None


class XBRLClient(BaseTool):
    """Fetch an exact reported financial figure from SEC XBRL company-facts.

    Usage:
        x = XBRLClient()
        x.run("AAPL", "revenue", "FY2022")
        # -> {"ok": True, "value": 394328000000.0, "tag": "RevenueFromContract...",
        #     "fy": 2022, "form": "10-K", "value_str": "$394,328,000,000", ...}
    """

    name = "xbrl_facts"
    description = "Look up an exact reported financial figure for a company/period."

    def __init__(
        self,
        resolver: Optional[TickerCIKResolver] = None,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        ttl_days: int = DEFAULT_TTL_DAYS,
        user_agent: Optional[str] = None,
        tag_resolver: Optional[Callable[[str, list[str]], Optional[str]]] = None,
    ) -> None:
        self.resolver = resolver or TickerCIKResolver(user_agent=user_agent)
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_days * 86400
        self.user_agent = user_agent or self.resolver.user_agent
        # LLM (or any callable) fallback for tag heterogeneity the map misses.
        self.tag_resolver = tag_resolver
        # In-process LRU memo by CIK. companyfacts payloads are multi-MB each, so
        # cap the count — without a bound a long-running server answering about
        # many companies grows unboundedly. The on-disk cache still serves the
        # rest cheaply.
        self._facts_cache: "OrderedDict[str, dict]" = OrderedDict()
        self._facts_cache_max = 24

    # --- companyfacts load / cache ------------------------------------------

    def _load_companyfacts(self, cik: str) -> dict:
        if cik in self._facts_cache:
            self._facts_cache.move_to_end(cik)      # mark most-recently-used
            return self._facts_cache[cik]

        path = self.cache_dir / f"CIK{cik}.json"
        fresh = path.exists() and (time.time() - path.stat().st_mtime) < self.ttl_seconds
        if fresh:
            data = json.loads(path.read_text())
        else:
            try:
                resp = requests.get(
                    COMPANYFACTS_URL.format(cik=cik),
                    headers={"User-Agent": self.user_agent}, timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            except Exception:
                if path.exists():            # network down → serve stale cache
                    data = json.loads(path.read_text())
                else:
                    raise
        self._facts_cache[cik] = data
        while len(self._facts_cache) > self._facts_cache_max:
            self._facts_cache.popitem(last=False)   # evict least-recently-used
        return data

    # --- concept → tag -------------------------------------------------------

    @staticmethod
    def canonical_concept(concept: str) -> Optional[str]:
        """Map free-text concept ('total revenue') to a canonical key ('revenue').

        Accepts both the canonical keys themselves ('operating_income', passed by
        the calculator) and free text ('operating income', 'net sales').
        """
        q = (concept or "").strip().lower()
        # Fast path: the caller passed a canonical key verbatim.
        if q in CONCEPT_TAGS:
            return q
        # Normalise separators so 'operating_income' matches the 'operating
        # income' keyword just like the free-text form does.
        q = q.replace("_", " ").replace("-", " ")
        # Prefer the longest matching keyword so "operating income" beats "income".
        best: Optional[tuple[int, str]] = None
        for kw, key in _CONCEPT_KEYWORDS:
            if kw in q and (best is None or len(kw) > best[0]):
                best = (len(kw), key)
        return best[1] if best else None

    def _candidate_tags(self, concept: str, usgaap: dict) -> tuple[list[str], str]:
        """Return (ordered candidate tags present in `usgaap`, how).

        `how` records the resolution path for transparency: "exact_tag" (caller
        passed a literal tag), "concept_map", "llm_fallback", or "none".
        """
        if concept in usgaap:                       # caller passed a literal tag
            return [concept], "exact_tag"

        key = self.canonical_concept(concept)
        if key:
            present = [t for t in CONCEPT_TAGS.get(key, []) if t in usgaap]
            if present:
                return present, "concept_map"

        if self.tag_resolver is not None:           # cheap LLM picks from present tags
            try:
                chosen = self.tag_resolver(concept, sorted(usgaap.keys()))
            except Exception:
                chosen = None
            if chosen and chosen in usgaap:
                return [chosen], "llm_fallback"
        return [], "none"

    @staticmethod
    def _primary_unit(units: dict) -> Optional[str]:
        """The unit a value is reported in (USD for money, USD/shares for EPS)."""
        return ("USD" if "USD" in units
                else "USD/shares" if "USD/shares" in units
                else next(iter(units), None))

    # --- period → fact -------------------------------------------------------

    @staticmethod
    def _fiscal_year(f: dict) -> Optional[int]:
        """Infer the fiscal year a fact covers from its *period-end* date.

        The SEC `fy` field is unreliable for comparatives — a prior-year figure
        restated in a later 10-K carries that 10-K's `fy` (e.g. J&J's FY2021 R&D
        is tagged `fy=2023`). The end date is authoritative: a fiscal year is
        labelled by the calendar year it ends in, except 52/53-week filers whose
        year ends in the first days of January belong to the *prior* year (J&J
        ends 2022-01-02 → FY2021; Walmart ends 2022-01-31 → FY2022).
        """
        e = f.get("end")
        if not e:
            return None
        try:
            from datetime import date
            d = date.fromisoformat(e)
        except Exception:
            return None
        return d.year - 1 if (d.month == 1 and d.day <= 14) else d.year

    @classmethod
    def _select_fact(cls, unit_facts: list[dict], year: Optional[int]) -> Optional[dict]:
        """Choose the annual (10-K, full-year) fact for `year` from a unit's list.

        Flow concepts (revenue, income) are durations — we take the ~full-year
        span, not a quarter. Stock concepts (assets, equity) are instants — we
        take the period-end value. When `year` is None we return the most recent
        annual fact.
        """
        def is_full_year(f: dict) -> bool:
            s, e = f.get("start"), f.get("end")
            if not s or not e:
                return True   # instant fact (balance-sheet item)
            try:
                from datetime import date
                d = (date.fromisoformat(e) - date.fromisoformat(s)).days
                return d >= 300   # full year, not a quarter
            except Exception:
                return True

        annual = [f for f in unit_facts if f.get("fp") == "FY" and is_full_year(f)]
        pool = annual or [f for f in unit_facts if is_full_year(f)] or unit_facts

        if year is not None:
            # Match on the end-date-derived fiscal year (robust to restated
            # comparatives); fall back to the raw `fy` tag if that finds nothing.
            cands = [f for f in pool if cls._fiscal_year(f) == year]
            if not cands:
                cands = [f for f in pool if f.get("fy") == year]
            if not cands:
                return None
        else:
            cands = pool

        # Prefer the figure as reported on a 10-K, then the latest period-end.
        tenk = [f for f in cands if f.get("form", "").startswith("10-K")]
        cands = tenk or cands
        return max(cands, key=lambda f: f.get("end", ""))

    # --- public API ----------------------------------------------------------

    def run(self, ticker: str, concept: str, period: Optional[str] = None) -> dict:
        """Return the exact reported value for (ticker/name, concept, period)."""
        r = self.resolver.resolve(ticker)
        cik = r.get("cik")
        if not cik:
            return {"ok": False, "error": f"could not resolve '{ticker}' to a CIK",
                    "ticker": ticker, "concept": concept}

        try:
            data = self._load_companyfacts(cik)
        except Exception as e:
            return {"ok": False, "error": f"companyfacts fetch failed: {e}",
                    "ticker": ticker, "cik": cik, "concept": concept}

        usgaap = (data.get("facts", {}) or {}).get("us-gaap", {}) or {}
        candidates, how = self._candidate_tags(concept, usgaap)
        if not candidates:
            return {"ok": False,
                    "error": f"no XBRL tag found for concept '{concept}'",
                    "ticker": r.get("ticker", ticker), "cik": cik,
                    "entity": data.get("entityName"), "concept": concept,
                    "available_tags": sorted(usgaap.keys())[:60]}

        year = _year_of(period)
        # Walk candidate tags and pick the first that yields a REAL value for the
        # period. A tag can exist yet hold a placeholder 0 for some years (e.g.
        # J&J files R&D under ResearchAndDevelopmentExpense as 0 and the real
        # figure under ...ExcludingAcquiredInProcessCost) — so a non-zero hit on
        # a later candidate beats a zero on an earlier one. We remember the first
        # zero/placeholder hit only as a last resort.
        tag = unit_key = fact = None
        fallback: Optional[tuple[str, str, dict]] = None
        for cand in candidates:
            units = usgaap[cand].get("units", {}) or {}
            ukey = self._primary_unit(units)
            if not ukey:
                continue
            f = self._select_fact(units[ukey], year)
            if f is None:
                continue
            if f.get("val"):                        # truthy → real, non-zero value
                tag, unit_key, fact = cand, ukey, f
                break
            if fallback is None:
                fallback = (cand, ukey, f)
        if fact is None and fallback is not None:
            tag, unit_key, fact = fallback

        if fact is None:
            first = candidates[0]
            avail_years = sorted({
                self._fiscal_year(f) for f in
                usgaap[first].get("units", {}).get(self._primary_unit(usgaap[first].get("units", {})) or "", [])
                if self._fiscal_year(f)
            })
            return {"ok": False,
                    "error": f"no annual fact for {candidates} in period {period!r}",
                    "ticker": r.get("ticker", ticker), "cik": cik, "concept": concept,
                    "tag": first, "available_years": avail_years}

        value = fact.get("val")
        is_money = unit_key == "USD"
        value_str = (f"${value:,.0f}" if is_money and isinstance(value, (int, float))
                     else f"{value:,}" if isinstance(value, (int, float)) else str(value))
        fy = self._fiscal_year(fact) or fact.get("fy")
        return {
            "ok": True,
            "ticker": r.get("ticker", ticker),
            "cik": cik,
            "entity": data.get("entityName"),
            "concept": concept,
            "taxonomy": "us-gaap",
            "tag": tag,
            "tag_match": how,
            "unit": unit_key,
            "period": period,
            "fy": fy,
            "fp": fact.get("fp"),
            "form": fact.get("form"),
            "start": fact.get("start"),
            "end": fact.get("end"),
            "value": value,
            "value_str": value_str,
            "source": (f"SEC XBRL companyfacts (us-gaap:{tag}, "
                       f"FY{fy} {fact.get('form','')}, filed {fact.get('filed','?')})"),
        }
