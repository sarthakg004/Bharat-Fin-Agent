"""
calculator.py  ·  finagent/tools/calculator.py

Financial calculator (Phase 4) — derived metrics from EXACT XBRL inputs.

Margins, ratios, YoY growth, and CAGR are computed deterministically from the
figures the company actually filed, pulled via the Phase-3 `XBRLClient` — never
from retrieved prose. Every result carries the exact inputs it used (value, tag,
fiscal year, SEC source) so the number is fully auditable: you can see that
"operating margin 30.3%" came from operating income / revenue = $119,437M /
$394,328M, both straight from Apple's FY2022 10-K XBRL.

Supported metrics:
  margins   — gross_margin, operating_margin, net_margin
  ratios    — current_ratio, debt_to_equity, return_on_equity, return_on_assets,
              asset_turnover, interest_coverage
  wc days   — dio, dso, dpo, ccc (working-capital days; two-period averages)
  growth    — period-over-period % change of any concept
  cagr      — compound annual growth rate of any concept across N years
  trend     — any of the above metrics computed across a list of periods

This tool is decoupled from the agent: it takes an optional `xbrl` client so it
shares the same on-disk cache and stays unit-testable without the graph.
"""

from __future__ import annotations

from typing import Optional

from finagent.tools.base import BaseTool
from finagent.tools.xbrl import XBRLClient, _year_of

# metric -> (numerator concept, denominator concept, render_as_percent)
RATIOS: dict[str, tuple[str, str, bool]] = {
    "gross_margin": ("gross_profit", "revenue", True),
    "operating_margin": ("operating_income", "revenue", True),
    "net_margin": ("net_income", "revenue", True),
    "current_ratio": ("current_assets", "current_liabilities", False),
    "debt_to_equity": ("total_liabilities", "stockholders_equity", False),
    "return_on_equity": ("net_income", "stockholders_equity", True),
    "return_on_assets": ("net_income", "total_assets", True),
    "asset_turnover": ("revenue", "total_assets", False),
    "interest_coverage": ("operating_income", "interest_expense", False),
    # cash ratio is a simple two-concept ratio (most conservative liquidity ratio).
    "cash_ratio": ("cash", "current_liabilities", False),
    # "% of revenue" intensity ratios analysts compare across companies/years.
    "rd_to_revenue": ("rd_expense", "revenue", True),
    "sga_to_revenue": ("sga_expense", "revenue", True),
    "capex_to_revenue": ("capex", "revenue", True),
}

# Ratios whose numerator is a COMBINATION of concepts (so they can't be a single
# (num, den) pair). Each: the concepts to add, the concepts to subtract, the
# denominator concept, and whether to render as a percent. `optional` concepts
# default to 0 when a company doesn't report them (e.g. a services firm with no
# inventory → quick ratio collapses to the current ratio, which is correct).
COMPOSITE_RATIOS: dict[str, dict] = {
    # Quick (acid-test) ratio = (current assets − inventory) / current liabilities.
    "quick_ratio": {
        "add": ["current_assets"], "sub": ["inventory"],
        "den": "current_liabilities", "pct": False, "optional": {"inventory"},
    },
    # Unadjusted EBITDA margin = (operating income + D&A) / revenue.
    "ebitda_margin": {
        "add": ["operating_income", "depreciation_amortization"], "sub": [],
        "den": "revenue", "pct": True,
    },
}

# Working-capital "days" metrics (the FinanceBench convention): each uses the
# two-period AVERAGE of its balance-sheet input against a flow denominator.
#   dio = 365 × avg(inventory)            / COGS
#   dso = 365 × avg(accounts receivable)  / revenue
#   dpo = 365 × avg(accounts payable)     / (COGS + Δinventory)
#   ccc = dio + dso − dpo
WC_DAYS_METRICS = {"dio", "dso", "dpo", "ccc"}

# Friendly synonyms → canonical metric name.
ALIASES: dict[str, str] = {
    "gross_profit_margin": "gross_margin", "gross margin": "gross_margin",
    "operating_profit_margin": "operating_margin", "operating margin": "operating_margin",
    "ebit_margin": "operating_margin",
    "net_profit_margin": "net_margin", "profit_margin": "net_margin",
    "net margin": "net_margin",
    "roe": "return_on_equity", "return on equity": "return_on_equity",
    "roa": "return_on_assets", "return on assets": "return_on_assets",
    "de_ratio": "debt_to_equity", "debt_equity": "debt_to_equity",
    "debt/equity": "debt_to_equity", "debt to equity": "debt_to_equity",
    "liquidity_ratio": "current_ratio", "current ratio": "current_ratio",
    "quick ratio": "quick_ratio", "acid_test": "quick_ratio",
    "acid test": "quick_ratio", "acid_test_ratio": "quick_ratio",
    "acid test ratio": "quick_ratio", "quick_assets_ratio": "quick_ratio",
    "cash ratio": "cash_ratio",
    "asset turnover": "asset_turnover", "times_interest_earned": "interest_coverage",
    "interest coverage": "interest_coverage",
    "r&d as % of revenue": "rd_to_revenue", "r&d as a percentage of revenue": "rd_to_revenue",
    "r&d intensity": "rd_to_revenue", "rd as % of revenue": "rd_to_revenue",
    "research and development as % of revenue": "rd_to_revenue",
    "research and development as a percentage of revenue": "rd_to_revenue",
    "r&d to revenue": "rd_to_revenue", "rd revenue ratio": "rd_to_revenue",
    "sg&a as % of revenue": "sga_to_revenue", "sga as % of revenue": "sga_to_revenue",
    "capex as % of revenue": "capex_to_revenue",
    "capital expenditure as % of revenue": "capex_to_revenue",
    "ebitda margin": "ebitda_margin", "ebitda % margin": "ebitda_margin",
    "unadjusted ebitda margin": "ebitda_margin",
    "unadjusted_ebitda_margin": "ebitda_margin", "ebitda": "ebitda_margin",
    "cash conversion cycle": "ccc", "cash_conversion_cycle": "ccc",
    "days inventory outstanding": "dio", "days_inventory_outstanding": "dio",
    "inventory days": "dio",
    "days sales outstanding": "dso", "days_sales_outstanding": "dso",
    "receivable days": "dso", "days of sales outstanding": "dso",
    "days payable outstanding": "dpo", "days_payable_outstanding": "dpo",
    "days payables outstanding": "dpo", "payable days": "dpo",
}


def _canonical_metric(metric: str) -> str:
    m = (metric or "").strip().lower().replace("-", "_")
    return ALIASES.get(m, ALIASES.get(m.replace("_", " "), m))


def _fmt(value: float, as_percent: bool) -> str:
    if value is None:
        return "—"
    if as_percent:
        return f"{value * 100:.1f}%"
    return f"{value:,.2f}×"


class FinancialCalculator(BaseTool):
    """Compute derived financial metrics from exact XBRL inputs.

    Usage:
        c = FinancialCalculator()
        c.ratio("AAPL", "operating_margin", "FY2022")   # -> {"value": 0.3029, ...}
        c.growth("AAPL", "revenue", "FY2021", "FY2022")
        c.cagr("AAPL", "revenue", "FY2019", "FY2022")
        c.trend("AAPL", "operating_margin", ["FY2020", "FY2021", "FY2022"])
    """

    name = "financial_calculator"
    description = "Compute margins, growth rates, ratios, and CAGR from exact XBRL inputs."

    def __init__(self, xbrl: Optional[XBRLClient] = None) -> None:
        self.xbrl = xbrl or XBRLClient()

    # --- input helper --------------------------------------------------------

    def _input(self, ticker: str, concept: str, period: Optional[str]) -> dict:
        """Fetch one exact XBRL fact and shape it as an audit-friendly input."""
        f = self.xbrl.run(ticker=ticker, concept=concept, period=period)
        if not f.get("ok"):
            return {"ok": False, "concept": concept, "period": period,
                    "error": f.get("error", "lookup failed")}
        return {
            "ok": True, "concept": concept, "value": f["value"],
            "value_str": f["value_str"], "tag": f["tag"], "fy": f["fy"],
            "form": f["form"], "source": f["source"], "ticker": f.get("ticker", ""),
        }

    @staticmethod
    def _fail(metric: str, ticker: str, inputs: list[dict], why: str) -> dict:
        return {"ok": False, "metric": metric, "ticker": ticker,
                "error": why, "inputs": inputs}

    # --- ratios / margins ----------------------------------------------------

    def _composite_ratio(self, ticker: str, name: str, period: Optional[str]) -> dict:
        """Compute a ratio whose numerator combines several concepts, e.g. the
        quick ratio = (current assets − inventory) / current liabilities."""
        spec = COMPOSITE_RATIOS[name]
        optional = spec.get("optional", set())
        concepts = spec["add"] + spec["sub"] + [spec["den"]]
        got = {c: self._input(ticker, c, period) for c in dict.fromkeys(concepts)}
        inputs = list(got.values())

        def val(c: str) -> Optional[float]:
            inp = got[c]
            if inp["ok"]:
                return inp["value"]
            return 0.0 if c in optional else None      # optional missing → 0

        # Every REQUIRED concept must resolve.
        if any(val(c) is None for c in concepts):
            missing = [c for c in concepts if val(c) is None]
            return self._fail(name, ticker, inputs, f"missing XBRL input(s): {missing}")
        den = val(spec["den"])
        if not den:
            return self._fail(name, ticker, inputs, "denominator is zero")
        numerator = sum(val(c) for c in spec["add"]) - sum(val(c) for c in spec["sub"])
        value = numerator / den
        as_pct = spec.get("pct", False)
        num_desc = " − ".join(spec["add"] + [f"({s})" for s in spec["sub"]]) \
            if spec["sub"] else " + ".join(spec["add"])
        return {
            "ok": True, "metric": name, "ticker": self._tkr(got[spec["add"][0]]),
            "period": period, "fy": got[spec["add"][0]].get("fy"),
            "value": value, "value_str": _fmt(value, as_pct), "is_percent": as_pct,
            "formula": f"({num_desc}) / {spec['den']}",
            "inputs": inputs,
            "source": (f"computed from XBRL: ({num_desc}) / {spec['den']} "
                       f"(FY{got[spec['add'][0]].get('fy')})"),
        }

    def ratio(self, ticker: str, metric: str, period: Optional[str]) -> dict:
        """Compute a margin or balance-sheet ratio for a single period."""
        name = _canonical_metric(metric)
        if name in COMPOSITE_RATIOS:
            return self._composite_ratio(ticker, name, period)
        if name not in RATIOS:
            return self._fail(metric, ticker, [], f"unknown ratio metric '{metric}'")
        num_c, den_c, as_pct = RATIOS[name]
        num, den = self._input(ticker, num_c, period), self._input(ticker, den_c, period)
        inputs = [num, den]
        if not (num["ok"] and den["ok"]):
            return self._fail(name, ticker, inputs, "missing XBRL input(s)")
        if not den["value"]:
            return self._fail(name, ticker, inputs, "denominator is zero")
        value = num["value"] / den["value"]
        return {
            "ok": True, "metric": name, "ticker": self._tkr(num),
            "period": period, "fy": num.get("fy"),
            "value": value, "value_str": _fmt(value, as_pct), "is_percent": as_pct,
            "formula": f"{num_c} / {den_c}",
            "inputs": inputs,
            "source": (f"computed from XBRL: {num['value_str']} / {den['value_str']} "
                       f"({num_c}/{den_c}, FY{num.get('fy')})"),
        }

    # --- working-capital days (DIO / DSO / DPO / CCC) -------------------------

    def working_capital_days(self, ticker: str, metric: str,
                             period: Optional[str]) -> dict:
        """Compute a working-capital days metric for fiscal year t, averaging
        each balance-sheet input over (t−1, t) — the FinanceBench convention.

        `period` names year t; the prior year is derived from it (or from the
        latest filed year when no period is given).
        """
        name = _canonical_metric(metric)
        year = _year_of(period)
        if year is None:
            probe = self._input(ticker, "inventory", None)   # latest filed year
            if not probe["ok"] or not probe.get("fy"):
                return self._fail(name, ticker, [probe],
                                  "could not determine the fiscal year")
            year = probe["fy"]
        cur, prior = f"FY{year}", f"FY{year - 1}"

        # Pull every input the metric family needs once; reuse across components.
        need = {
            "dio": [("inventory", prior), ("inventory", cur), ("cost_of_revenue", cur)],
            "dso": [("accounts_receivable", prior), ("accounts_receivable", cur),
                    ("revenue", cur)],
            "dpo": [("accounts_payable", prior), ("accounts_payable", cur),
                    ("inventory", prior), ("inventory", cur), ("cost_of_revenue", cur)],
        }
        wanted = need[name] if name in need else need["dio"] + need["dso"] + need["dpo"]
        got: dict[tuple, dict] = {}
        for c, p in dict.fromkeys(wanted):
            got[(c, p)] = self._input(ticker, c, p)
        inputs = list(got.values())
        missing = [f"{c} {p}" for (c, p), inp in got.items() if not inp["ok"]]
        if missing:
            return self._fail(name, ticker, inputs,
                              f"missing XBRL input(s): {missing}")

        v = {k: float(inp["value"]) for k, inp in got.items()}

        def days(avg_a: float, avg_b: float, den: float, label: str):
            if not den:
                raise ZeroDivisionError(f"{label} denominator is zero")
            return 365.0 * ((avg_a + avg_b) / 2.0) / den

        components: dict[str, float] = {}
        try:
            if name in ("dio", "ccc"):
                components["dio"] = days(v[("inventory", prior)], v[("inventory", cur)],
                                         v[("cost_of_revenue", cur)], "dio")
            if name in ("dso", "ccc"):
                components["dso"] = days(v[("accounts_receivable", prior)],
                                         v[("accounts_receivable", cur)],
                                         v[("revenue", cur)], "dso")
            if name in ("dpo", "ccc"):
                d_inv = v[("inventory", cur)] - v[("inventory", prior)]
                components["dpo"] = days(v[("accounts_payable", prior)],
                                         v[("accounts_payable", cur)],
                                         v[("cost_of_revenue", cur)] + d_inv, "dpo")
        except ZeroDivisionError as e:
            return self._fail(name, ticker, inputs, str(e))

        value = (components["dio"] + components["dso"] - components["dpo"]
                 if name == "ccc" else components[name])
        formulas = {
            "dio": "365 × avg(inventory) / COGS",
            "dso": "365 × avg(accounts receivable) / revenue",
            "dpo": "365 × avg(accounts payable) / (COGS + Δinventory)",
            "ccc": "DIO + DSO − DPO",
        }
        comp_str = ", ".join(f"{k.upper()} = {x:.2f} days"
                             for k, x in components.items())
        # Components ride along as audit inputs so the verifier can ground the
        # intermediate DIO/DSO/DPO figures a worked answer states.
        for k, x in components.items():
            inputs.append({"ok": True, "concept": k, "value": x,
                           "value_str": f"{x:.2f} days", "fy": year,
                           "source": f"computed: {formulas[k]}"})
        any_inp = next(iter(got.values()))
        return {
            "ok": True, "metric": name, "ticker": self._tkr(any_inp),
            "period": cur, "fy": year,
            "value": value, "value_str": f"{value:.2f} days", "is_percent": False,
            "components": components,
            "formula": formulas[name],
            "inputs": inputs,
            "source": (f"computed from XBRL: {formulas[name]} "
                       f"(FY{year}, averaging FY{year - 1}–FY{year}; {comp_str})"),
        }

    # --- growth / CAGR -------------------------------------------------------

    def growth(self, ticker: str, concept: str,
               period_from: str, period_to: str) -> dict:
        """Period-over-period % change of `concept` (e.g. revenue YoY)."""
        a = self._input(ticker, concept, period_from)
        b = self._input(ticker, concept, period_to)
        inputs = [a, b]
        if not (a["ok"] and b["ok"]):
            return self._fail("growth", ticker, inputs, "missing XBRL input(s)")
        if not a["value"]:
            return self._fail("growth", ticker, inputs, "base-period value is zero")
        value = (b["value"] - a["value"]) / abs(a["value"])
        return {
            "ok": True, "metric": "growth", "concept": concept,
            "ticker": self._tkr(a),
            "period_from": period_from, "period_to": period_to,
            "value": value, "value_str": f"{value * 100:+.1f}%", "is_percent": True,
            "formula": f"({concept}[{period_to}] − {concept}[{period_from}]) / {concept}[{period_from}]",
            "inputs": inputs,
            "source": (f"computed from XBRL: ({b['value_str']} − {a['value_str']}) / "
                       f"{a['value_str']} ({concept}, FY{a.get('fy')}→FY{b.get('fy')})"),
        }

    def cagr(self, ticker: str, concept: str,
             start_period: str, end_period: str) -> dict:
        """Compound annual growth rate of `concept` between two fiscal years."""
        a = self._input(ticker, concept, start_period)
        b = self._input(ticker, concept, end_period)
        inputs = [a, b]
        if not (a["ok"] and b["ok"]):
            return self._fail("cagr", ticker, inputs, "missing XBRL input(s)")
        y0, y1 = _year_of(start_period), _year_of(end_period)
        n = (y1 - y0) if (y0 and y1) else (a.get("fy") and b.get("fy") and b["fy"] - a["fy"])
        if not n or n <= 0:
            return self._fail("cagr", ticker, inputs, "need ≥1 year between periods")
        if not a["value"] or a["value"] <= 0 or b["value"] <= 0:
            return self._fail("cagr", ticker, inputs, "CAGR needs positive endpoints")
        value = (b["value"] / a["value"]) ** (1 / n) - 1
        return {
            "ok": True, "metric": "cagr", "concept": concept,
            "ticker": self._tkr(a), "years": n,
            "start_period": start_period, "end_period": end_period,
            "value": value, "value_str": f"{value * 100:.1f}%", "is_percent": True,
            "formula": f"({concept}[{end_period}] / {concept}[{start_period}])^(1/{n}) − 1",
            "inputs": inputs,
            "source": (f"computed from XBRL: ({b['value_str']}/{a['value_str']})^(1/{n})−1 "
                       f"({concept}, FY{a.get('fy')}→FY{b.get('fy')})"),
        }

    # --- multi-period trend --------------------------------------------------

    def trend(self, ticker: str, metric: str, periods: list[str]) -> dict:
        """Compute `metric` (a ratio/margin, or a raw concept) across `periods`.

        Returns a per-period series plus the first→last change and, for a
        consistent multi-year span, the CAGR of the series. This backs questions
        like "operating margin trend over the last 3 years".
        """
        name = _canonical_metric(metric)
        is_ratio = name in RATIOS or name in COMPOSITE_RATIOS
        as_pct = (RATIOS[name][2] if name in RATIOS
                  else COMPOSITE_RATIOS[name].get("pct", False) if name in COMPOSITE_RATIOS
                  else False)

        series: list[dict] = []
        for p in periods:
            if is_ratio:
                r = self.ratio(ticker, name, p)
            else:                                  # raw concept (revenue, net income…)
                f = self._input(ticker, name, p)
                r = ({"ok": True, "period": p, "value": f["value"],
                      "value_str": f["value_str"], "fy": f.get("fy"), "inputs": [f]}
                     if f["ok"] else {"ok": False, "period": p, "error": f["error"]})
            series.append({"period": p, **({"value": r["value"],
                            "value_str": r["value_str"], "fy": r.get("fy")}
                           if r.get("ok") else {"value": None, "error": r.get("error")}),
                           "ok": r.get("ok", False)})

        good = [s for s in series if s["ok"]]
        if not good:
            return {"ok": False, "metric": name, "ticker": ticker,
                    "periods": periods, "error": "no period resolved", "series": series}

        first, last = good[0], good[-1]
        out = {
            "ok": True, "metric": name, "ticker": ticker, "is_percent": as_pct,
            "periods": periods, "series": series,
            "first": first, "last": last,
        }
        # Direction summary.
        if last["value"] is not None and first["value"] is not None:
            delta = last["value"] - first["value"]
            direction = "rose" if delta > 0 else "fell" if delta < 0 else "was flat"
            if as_pct:
                out["change_str"] = f"{direction} {abs(delta) * 100:.1f} pts"
            else:
                out["change_str"] = f"{direction} {abs(delta):,.2f}"
            out["summary"] = (
                f"{name.replace('_', ' ')} {direction} from {first['value_str']} "
                f"(FY{first.get('fy')}) to {last['value_str']} (FY{last.get('fy')})"
            )
        return out

    # --- dispatch ------------------------------------------------------------

    def run(self, metric: str, ticker: str = "", period: Optional[str] = None,
            concept: str = "", periods: Optional[list[str]] = None,
            period_from: Optional[str] = None, period_to: Optional[str] = None,
            start_period: Optional[str] = None, end_period: Optional[str] = None,
            **_: object) -> dict:
        """Single entry point. Routes on `metric` and the periods provided.

        - metric == "growth"  -> growth(concept, period_from, period_to)
        - metric == "cagr"    -> cagr(concept, start_period, end_period)
        - multiple `periods`  -> trend(metric, periods)
        - else                -> ratio(metric, period)
        """
        m = _canonical_metric(metric)
        if m in WC_DAYS_METRICS:
            # The LAST period named is year t (the averaging year t−1 is derived
            # internally), so ["FY2018", "FY2019"] computes the FY2019 metric.
            # `periods[-1]` outranks `period`: callers pass period=periods[0],
            # which for a two-period averaging question is the PRIOR year.
            return self.working_capital_days(
                ticker, m, (periods[-1] if periods else None) or period
                or period_to or end_period)
        if m == "growth":
            return self.growth(ticker, concept or "revenue",
                               period_from or start_period, period_to or end_period)
        if m == "cagr":
            return self.cagr(ticker, concept or "revenue",
                            start_period or period_from, end_period or period_to)
        if periods and len(periods) > 1:
            return self.trend(ticker, m, periods)
        return self.ratio(ticker, m, period or (periods[0] if periods else None))

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _tkr(inp: dict) -> str:
        return inp.get("ticker", "") if isinstance(inp, dict) else ""
