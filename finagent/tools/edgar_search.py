"""
edgar_search.py  ·  finagent/tools/edgar_search.py

EDGAR full-text search (Phase 6) — the cross-document question class.

Chunk retrieval answers "what did Apple say about X" by searching *one* company's
filings. It cannot answer "**which companies** disclosed X" — that needs a search
across the full text of *many* companies' filings at once. SEC's EDGAR full-text
search (`efts.sec.gov`, covering filings from 2001 on) does exactly that.

Given a phrase, this tool queries EDGAR FTS, then aggregates the raw filing hits
into a **distinct-company** list (most-recent filing per company) — the shape a
"which companies…" answer wants — while also returning the underlying hits with
direct links to each filing for citation.

HTTP-only (no embeddings / models), so it's cheap and memory-light.
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

import requests

from finagent.tools.base import BaseTool

EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
# "Apple Inc.  (AAPL)  (CIK 0000320193)" -> name, ticker, cik
_DISPLAY_RE = re.compile(r"^(?P<name>.*?)\s*(?:\((?P<ticker>[^)]+)\)\s*)?\(CIK\s*(?P<cik>\d+)\)\s*$")


def _user_agent() -> str:
    name = (os.getenv("SEC_UA_NAME") or "").strip() or "FinAgent Research"
    email = (os.getenv("SEC_UA_EMAIL") or "").strip() or "finagent@example.com"
    return f"{name} {email}"


def _parse_display(display: str) -> tuple[str, str, str]:
    m = _DISPLAY_RE.match((display or "").strip())
    if not m:
        return display, "", ""
    return m.group("name").strip(), (m.group("ticker") or "").strip(), m.group("cik")


class EdgarFullTextSearch(BaseTool):
    """Search the full text of SEC filings across many companies.

    Usage:
        e = EdgarFullTextSearch()
        e.run('"climate-related risks"', forms="10-K")
        # -> {"ok": True, "total": ..., "companies": [{name, ticker, cik, form,
        #     date, url}, ...], "hits": [...]}
    """

    name = "edgar_full_text_search"
    description = "Search the full text of SEC filings across many companies."

    def __init__(self, user_agent: Optional[str] = None, timeout: int = 25) -> None:
        self.user_agent = user_agent or _user_agent()
        self.timeout = timeout

    @staticmethod
    def _filing_url(cik: str, accession: str, doc_id: str) -> str:
        """Build the EDGAR archive URL for a hit.

        `doc_id` is the FTS `_id` ("{accession}:{filename}"); `accession` is the
        dashed form. The archive path uses the CIK (no zero-pad) and the
        no-dash accession folder.
        """
        filename = doc_id.split(":", 1)[1] if ":" in doc_id else ""
        acc_nodash = (accession or "").replace("-", "")
        cik_int = str(int(cik)) if cik.isdigit() else cik
        return (f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                f"{acc_nodash}/{filename}")

    def run(self, q: str, forms: str = "10-K", n: int = 10,
            startdt: Optional[str] = None, enddt: Optional[str] = None) -> dict:
        """Full-text search EDGAR for `q`, return distinct companies + raw hits.

        `q` is matched as written — wrap a multi-word concept in double quotes
        for an exact-phrase search ('"human capital"'). `n` caps the number of
        distinct companies returned.
        """
        params: dict[str, str] = {"q": q}
        if forms:
            params["forms"] = forms
        if startdt and enddt:
            params["dateRange"] = "custom"
            params["startdt"] = startdt
            params["enddt"] = enddt
        # EDGAR FTS is rate-sensitive and returns transient 500/429s under load;
        # retry a couple times with backoff before giving up.
        data = None
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = requests.get(EFTS_URL, params=params,
                                    headers={"User-Agent": self.user_agent},
                                    timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
        if data is None:
            return {"ok": False, "query": q, "error": f"EDGAR FTS failed: {last_err}",
                    "companies": [], "hits": []}

        hits_raw = (data.get("hits", {}) or {}).get("hits", []) or []
        total = (data.get("hits", {}) or {}).get("total", {}).get("value", 0)

        hits: list[dict] = []
        by_cik: dict[str, dict] = {}
        for h in hits_raw:
            src = h.get("_source", {}) or {}
            display = (src.get("display_names") or [""])[0]
            name, ticker, cik = _parse_display(display)
            cik = cik or (src.get("ciks") or [""])[0]
            accession = src.get("adsh", "")
            url = self._filing_url(cik, accession, h.get("_id", ""))
            entry = {
                "company": name, "ticker": ticker, "cik": cik,
                "form": src.get("form", ""), "date": src.get("file_date", ""),
                "accession": accession, "url": url,
            }
            hits.append(entry)
            # Keep the most recent filing per distinct company.
            prev = by_cik.get(cik)
            if prev is None or entry["date"] > prev["date"]:
                by_cik[cik] = entry

        companies = sorted(by_cik.values(), key=lambda e: e["date"], reverse=True)[:n]
        return {
            "ok": True,
            "query": q,
            "forms": forms,
            "total": total,                       # total matching filings (FTS-wide)
            "n_companies": len(companies),
            "companies": companies,
            "hits": hits[:max(n, len(companies))],
        }
