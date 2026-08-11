"""Checks for the wrong-fiscal-year failure class: relative periods resolved to
the newest FILED data, same-quarter YoY selection, and recency-aware retrieval
filtering. No network, no LLM — fakes throughout.

The deterministic period guard that used to strip LLM-guessed years is gone —
the extractors now get today's date and the tools resolve "latest" themselves
(`_select_fact` with year=None, `infer_filter`'s recency branch), which is what
these tests cover.
"""

from finagent.retrieval.filters import infer_filter, parse_years
from finagent.tools.calculator import FinancialCalculator
from finagent.tools.xbrl import XBRLClient


def test_short_fiscal_year_is_a_named_year():
    """The planner writes "FY22" as often as "FY2022" — sometimes both in one
    plan. The short form used to parse as NO year, which cost the retrieval
    filter its year clause (44% of measured sub-queries)."""
    assert parse_years("AMD Enterprise segment revenue FY22") == [2022]
    assert parse_years("AMD revenue FY 21") == [2021]
    assert parse_years("Apple gross margin FY2021 vs FY22") == [2021, 2022]
    assert parse_years("FY98 annual report") == [1998]
    # A bare 2-digit number is a quantity, not a year.
    assert parse_years("revenue grew across 22 states in 2022") == [2022]


def test_infer_filter_reads_short_fiscal_years():
    vocab, years = {"amd": "AMD"}, {"AMD": {"2021", "2022"}}
    assert infer_filter("AMD Enterprise segment revenue FY22",
                        vocab, years)["years"] == ["2022"]


def test_select_fact_pins_same_quarter_yoy():
    facts = [
        {"start": "2025-01-01", "end": "2025-03-31", "fp": "Q1", "val": 10, "form": "10-Q"},
        {"start": "2025-04-01", "end": "2025-06-30", "fp": "Q2", "val": 11, "form": "10-Q"},
        {"start": "2026-01-01", "end": "2026-03-31", "fp": "Q1", "val": 12, "form": "10-Q"},
    ]
    # fp pins the quarter: year 2025 + Q1 must NOT return Q2 2025.
    f = XBRLClient._select_fact(facts, 2025, quarterly=True, fp="Q1")
    assert f["val"] == 10
    # No year → most recent Q1.
    f = XBRLClient._select_fact(facts, None, quarterly=True, fp="Q1")
    assert f["val"] == 12


class FakeXBRL:
    """Deterministic stand-in for XBRLClient.run keyed on (concept, year, fp)."""

    # (concept, fy, fp) -> value.  Q1 margins: 2025 = 100/1000, 2026 = 150/1200.
    DATA = {
        ("operating_income", 2025, "Q1"): 100.0,
        ("revenue", 2025, "Q1"): 1000.0,
        ("operating_income", 2026, "Q1"): 150.0,
        ("revenue", 2026, "Q1"): 1200.0,
    }
    LATEST = (2026, "Q1")

    def run(self, ticker, concept, period=None, quarterly=False, fp=None):
        import re
        year = None
        m = re.search(r"(19|20)\d{2}", str(period or ""))
        if m:
            year = int(m.group(0))
        fy, fq = (year, fp) if year else self.LATEST
        val = self.DATA.get((concept, fy, fq or "Q1"))
        if val is None:
            return {"ok": False, "error": "no fact"}
        return {"ok": True, "value": val, "value_str": f"${val:,.0f}",
                "tag": concept, "fy": fy, "fp": fq or "Q1",
                "period_label": f"{fq or 'Q1'} {fy}", "form": "10-Q",
                "source": "fake", "ticker": ticker}


def test_quarterly_yoy_margin_trend():
    calc = FinancialCalculator(xbrl=FakeXBRL())
    out = calc.trend("NOW", "operating_margin", ["FY2025", "FY2026"],
                     quarterly=True, fp="Q1")
    assert out["ok"]
    vals = [s["value"] for s in out["series"]]
    assert abs(vals[0] - 0.10) < 1e-9 and abs(vals[1] - 0.125) < 1e-9
    assert "Q1 2025" in out["summary"] and "Q1 2026" in out["summary"]
    assert "rose 2.5 pts" in out["change_str"]


def test_growth_resolves_latest_period_when_none_given():
    calc = FinancialCalculator(xbrl=FakeXBRL())
    out = calc.growth("NOW", "revenue", None, None, quarterly=True)
    assert out["ok"]
    # Q1 2026 vs Q1 2025: (1200-1000)/1000 = +20% — not the silent 0% the
    # old same-fact-twice path produced.
    assert abs(out["value"] - 0.20) < 1e-9


def test_infer_filter_prefers_newest_years_for_yoy_questions():
    vocab = {"servicenow": "ServiceNow"}
    years = {"ServiceNow": {"2022", "2023", "2025", "2026"}}
    q = ("By how many percentage points did ServiceNow's operating margin "
         "improve year-over-year?")
    flt = infer_filter(q, vocab, years)
    assert flt["companies"] == ["ServiceNow"]
    assert flt["years"] == ["2025", "2026"]
    # A question naming no year and no recency cue keeps the full history.
    flt = infer_filter("ServiceNow revenue by segment in FY2022", vocab, years)
    assert flt.get("years") == ["2022", "2023"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
