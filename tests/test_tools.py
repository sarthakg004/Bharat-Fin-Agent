"""The number path and the live 10-K fetch. No network: SEC responses are fakes."""

from __future__ import annotations

import re
import sys

from finagent.tools.calculator import FinancialCalculator
from finagent.tools.sec_fetch import SecFilingFetcher
from finagent.tools.xbrl import XBRLClient


def test_fiscal_year_comes_from_the_period_end_date():
    """The SEC `fy` field is the filing's year, not the figure's. A year ending
    in early January belongs to the previous fiscal year."""
    assert XBRLClient._fiscal_year({"end": "2022-01-02"}) == 2021      # J&J
    assert XBRLClient._fiscal_year({"end": "2022-01-31"}) == 2022      # Walmart
    assert XBRLClient._fiscal_year({"end": "2022-09-24"}) == 2022      # Apple


def test_a_quarter_is_pinned_for_same_quarter_comparisons():
    facts = [
        {"start": "2025-01-01", "end": "2025-03-31", "fp": "Q1", "val": 10, "form": "10-Q"},
        {"start": "2025-04-01", "end": "2025-06-30", "fp": "Q2", "val": 11, "form": "10-Q"},
        {"start": "2026-01-01", "end": "2026-03-31", "fp": "Q1", "val": 12, "form": "10-Q"},
    ]
    assert XBRLClient._select_fact(facts, 2025, quarterly=True, fp="Q1")["val"] == 10
    assert XBRLClient._select_fact(facts, None, quarterly=True, fp="Q1")["val"] == 12   # newest


def test_a_named_quarter_uses_the_fiscal_year_and_never_returns_another_quarter():
    """Best Buy's Q2 FY2024 ended in July 2023. A balance is the latest instant
    in that 10-Q; a flow reported only year-to-date is a miss, not Q1."""
    facts = [
        {"start": "2023-04-30", "end": "2023-07-29", "fy": 2024, "fp": "Q2", "val": 2, "form": "10-Q"},
        {"start": "2023-07-30", "end": "2023-10-28", "fy": 2024, "fp": "Q3", "val": 3, "form": "10-Q"},
        {"end": "2023-07-29", "fy": 2024, "fp": "Q2", "val": 20, "form": "10-Q"},   # Q2 balance
        {"end": "2023-01-28", "fy": 2024, "fp": "Q2", "val": 19, "form": "10-Q"},   # prior year-end
        {"start": "2022-01-30", "end": "2022-07-30", "fy": 2023, "fp": "Q2", "val": 6, "form": "10-Q"},
    ]
    durations = [f for f in facts if "start" in f]
    assert XBRLClient._select_fact(durations, 2024, quarterly=True, fp="Q2")["val"] == 2
    assert XBRLClient._select_fact([f for f in facts if "start" not in f], 2024,
                                   quarterly=True, fp="Q2")["val"] == 20
    # FY2023 Q2 only has a 6-month figure: no answer rather than a wrong quarter.
    assert XBRLClient._select_fact(durations, 2023, quarterly=True, fp="Q2") is None


def test_an_llm_picked_tag_must_share_a_word_with_the_concept():
    """Otherwise a real figure is returned under the wrong name."""
    assert not XBRLClient._tag_relevant("restructuring costs", "AssetImpairmentCharges")
    assert XBRLClient._tag_relevant("asset impairment", "AssetImpairmentCharges")


class FakeXBRL:
    """Stands in for XBRLClient.run, keyed on (concept, year, quarter)."""

    DATA = {("operating_income", 2025, "Q1"): 100.0, ("revenue", 2025, "Q1"): 1000.0,
            ("operating_income", 2026, "Q1"): 150.0, ("revenue", 2026, "Q1"): 1200.0}

    def run(self, ticker, concept, period=None, quarterly=False, fp=None):
        m = re.search(r"(19|20)\d{2}", str(period or ""))
        fy, fq = (int(m.group(0)), fp) if m else (2026, "Q1")
        val = self.DATA.get((concept, fy, fq or "Q1"))
        if val is None:
            return {"ok": False, "error": "no fact"}
        return {"ok": True, "value": val, "value_str": f"${val:,.0f}", "tag": concept,
                "fy": fy, "fp": fq or "Q1", "period_label": f"{fq or 'Q1'} {fy}",
                "form": "10-Q", "source": "fake", "ticker": ticker}


def test_calculator_computes_margins_and_growth_from_the_facts():
    calc = FinancialCalculator(xbrl=FakeXBRL())
    trend = calc.trend("NOW", "operating_margin", ["FY2025", "FY2026"], quarterly=True, fp="Q1")
    assert [round(s["value"], 3) for s in trend["series"]] == [0.1, 0.125]
    assert "rose 2.5 pts" in trend["change_str"]
    # No period given: the latest period against the one before, not 0% growth.
    growth = calc.growth("NOW", "revenue", None, None, quarterly=True)
    assert abs(growth["value"] - 0.20) < 1e-9


def test_a_ratio_never_divides_figures_from_different_years():
    """Best Buy: gross profit runs to FY2026, the old revenue tag stops at
    FY2018. With no year named, both inputs must come from one year."""
    class Stale:
        LATEST = {"gross_profit": 2026, "revenue": 2018}

        def run(self, ticker, concept, period=None, quarterly=False, fp=None):
            fy = int(period[2:]) if period else self.LATEST[concept]
            if fy > self.LATEST[concept]:
                return {"ok": False, "error": "no fact"}
            return {"ok": True, "value": 10.0, "value_str": "$10", "tag": concept, "fy": fy,
                    "fp": "FY", "form": "10-K", "source": "fake", "ticker": ticker}

    r = FinancialCalculator(xbrl=Stale()).ratio("BBY", "gross_margin", None)
    assert not r["ok"] and [i.get("fy") for i in r["inputs"]] == [2026, None]


# Shape of data.sec.gov/submissions/CIK##########.json, trimmed to the keys read.
FAKE_SUBMISSIONS = {"filings": {"recent": {
    "form": ["8-K", "10-K", "10-K/A", "10-Q", "10-K"],
    "filingDate": ["2025-12-20", "2025-12-12", "2025-12-15", "2025-08-14", "2024-12-13"],
    "accessionNumber": ["0000000000-25-000001", "0001628280-25-056742", "0000000000-25-000002",
                        "0000000000-25-000003", "0000006951-24-000044"],
    "primaryDocument": ["ev.htm", "amat-20251026.htm", "amend.htm", "q3.htm", "amat-20241027.htm"],
}}}


class _Response:
    def __init__(self, payload=None, content=b""):
        self._payload, self.content = payload, content

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


def _fetcher(tmp_path, downloads):
    f = SecFilingFetcher(corpus_dir=tmp_path, collection_name="stub", resolver=type(
        "R", (), {"resolve": staticmethod(lambda q: {
            "cik": "0000006951", "ticker": "AMAT", "title": "APPLIED MATERIALS INC /DE"})})())

    def fake_get(url, timeout=None):
        if "data.sec.gov/submissions" in url:
            return _Response(payload=FAKE_SUBMISSIONS)
        downloads.append(url)
        return _Response(content=b"x" * 60_000)

    f._sec_get = fake_get
    return f


def test_fetch_downloads_the_latest_10ks_and_skips_amendments(tmp_path):
    downloads: list[str] = []
    records, form = _fetcher(tmp_path, downloads)._download("AMAT", "APPLIED MATERIALS INC /DE", "10-K", 2)
    assert form == "10-K" and [r["year"] for r in records] == ["2025", "2024"]    # 10-K/A is not a 10-K
    assert downloads[0] == ("https://www.sec.gov/Archives/edgar/data/6951/"
                            "000162828025056742/amat-20251026.htm")
    assert records[0]["status"] == "ok" and records[0]["company"] == "APPLIED MATERIALS INC /DE"


def test_a_dead_sec_endpoint_degrades_instead_of_raising(tmp_path):
    f = _fetcher(tmp_path, [])

    def boom(url, timeout=None):
        raise OSError("SEC unreachable")

    f._sec_get = boom
    assert f._download("AMAT", "", "10-K", 1) == ([], None)


def test_fetch_picks_the_10k_for_the_year_asked(tmp_path, monkeypatch):
    import copy
    from finagent.tools import sec_fetch
    subs = copy.deepcopy(FAKE_SUBMISSIONS)
    subs["filings"]["recent"]["reportDate"] = ["", "2025-10-26", "", "2025-07-27", "2024-10-27"]
    monkeypatch.setattr(sys.modules[__name__], "FAKE_SUBMISSIONS", subs)
    downloads: list[str] = []
    records, _ = _fetcher(tmp_path, downloads)._download("AMAT", "", "10-K", 1, fiscal_years=[2024])
    assert [r["source_url"].rsplit("/", 1)[-1] for r in records] == ["amat-20241027.htm"]
    # A 52/53-week year ending in early January is the year before (J&J FY2022).
    assert sec_fetch.fiscal_year("2023-01-01") == 2022 and sec_fetch.fiscal_year("2024-02-03") == 2024
