"""Fetch a company's 10-K from EDGAR when it is not in the index yet.

The gate sorts a company into one of three cases:

    already_indexed   retrieval will find it
    fetch             US-listed but not indexed: download, ingest, then retrieve
    not_us_listed     no SEC id: leave it to web search

A fetched filing is written into the served collection and stays there, so the
corpus grows with use.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from finagent.config import settings
from finagent.tools.resolver import TickerCIKResolver
from finagent.vectorstore import DEFAULT_EMBED_MODEL


def _form_key(form: str) -> str:
    """"10-K", "10 K" and "10k" all become "10K"."""
    return (form or "").upper().replace(" ", "").replace("-", "")


def _filing_rows(block: dict) -> list[dict]:
    """The SEC submissions index (one array per field) as one dict per filing."""
    cols = ("form", "filingDate", "accessionNumber", "primaryDocument", "reportDate")
    n = len(block.get("form") or [])
    return [dict(zip(("form", "date", "accession", "doc", "period"), row))
            for row in zip(*((block.get(c) or [""] * n) for c in cols))]


def fiscal_year(period_end: str) -> Optional[int]:
    """The fiscal year a report covers, from its period-end date. A 52/53-week
    year that ends in the first week of January belongs to the year before
    (J&J's fiscal 2022 ended on 1 January 2023)."""
    try:
        y, m, d = (int(x) for x in period_end[:10].split("-"))
    except ValueError:
        return None
    return y - 1 if m == 1 and d <= 7 else y


def is_filing_indexed(filing: dict, indexed_urls: set[str]) -> bool:
    """Is this SEC filing among the company's indexed `source_url`s? A fetched
    filing is stored under its SEC URL (which holds the accession number), a
    curated one under its local path "{TICKER}/{filing year}_{accession tail}.htm"."""
    acc = filing["accession"]
    tail = f"/{filing['date'][:4]}_{acc.split('-')[-1]}.htm"
    return any(acc.replace("-", "") in u or u.endswith(tail) for u in indexed_urls)


class SecFilingFetcher:
    # Fetched filings are saved under corpus_dir/dynamic-fetch/, apart from the
    # curated corpus, so rebuilding the corpus does not pick them up.
    SCRATCH_DIR = "dynamic-fetch"
    _SEC_TIMEOUT = 20

    def __init__(self, resolver: Optional[TickerCIKResolver] = None,
                 collection_name: str = settings.us_collection,
                 corpus_dir: str | Path = "data/us/pdfs",
                 embedding_model: str = DEFAULT_EMBED_MODEL) -> None:
        self.resolver = resolver or TickerCIKResolver()
        self.collection_name = collection_name
        self.corpus_dir = Path(corpus_dir)
        self.embedding_model = embedding_model

    # --- gate ---------------------------------------------------------------

    def is_indexed(self, ticker: str) -> bool:
        """Is any chunk tagged with this ticker?"""
        from finagent.vectorstore import exists_where

        t = (ticker or "").upper().strip()
        try:
            return bool(t) and exists_where(self.collection_name, "ticker", t)
        except Exception:
            return False

    def gate(self, query: str) -> dict:
        """`{"decision", "ticker", "cik", "company"}` for a company name or ticker."""
        r = self.resolver.resolve(query)
        cik, ticker = r.get("cik"), r.get("ticker")
        if not cik:
            return {"decision": "not_us_listed", "ticker": None, "cik": None,
                    "company": None, "query": query}
        decision = "already_indexed" if self.is_indexed(ticker) else "fetch"
        return {"decision": decision, "ticker": ticker, "cik": cik,
                "company": r.get("title"), "query": query}

    # --- download -----------------------------------------------------------

    def _sec_get(self, url: str, timeout: Optional[int] = None):
        """GET with the name + email User-Agent the SEC requires."""
        import requests

        name = (os.getenv("SEC_UA_NAME") or "").strip() or "FinAgent Research"
        email = (os.getenv("SEC_UA_EMAIL") or "").strip() or "finagent@example.com"
        return requests.get(url, headers={"User-Agent": f"{name} {email}"},
                            timeout=timeout or self._SEC_TIMEOUT)

    def _list_filings(self, cik: str, forms: tuple[str, ...], older: bool = False) -> list[dict]:
        """Filings for `cik` whose form is exactly one of `forms`, newest first.
        Amendments (10-K/A) do not match. The SEC lists only the latest ~1,000
        filings up front (for JPM that reaches back months, not years);
        `older=True` also reads the paged history behind it."""
        want = {_form_key(f) for f in forms}
        try:
            resp = self._sec_get(f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json")
            resp.raise_for_status()
            filings = resp.json().get("filings") or {}
            blocks = [filings.get("recent") or {}]
            for page in (filings.get("files") or []) if older else []:
                r = self._sec_get(f"https://data.sec.gov/submissions/{page['name']}")
                r.raise_for_status()
                blocks.append(r.json())
        except Exception:
            return []
        hits = [f for b in blocks for f in _filing_rows(b) if _form_key(f["form"]) in want]
        return sorted(hits, key=lambda f: f["date"], reverse=True)

    def _download_filing(self, cik: str, ticker: str, company: str,
                         filing: dict) -> Optional[dict]:
        """Save one filing's main document; return the record the ingester reads.
        `year` is the filing year, the convention the index uses."""
        acc = filing["accession"]
        url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
               f"{acc.replace('-', '')}/{filing['doc']}")
        out_path = (self.corpus_dir / self.SCRATCH_DIR / ticker.upper()
                    / f"{filing['date'][:4]}_{acc.split('-')[-1]}.htm")
        try:
            if not out_path.exists() or out_path.stat().st_size < 50_000:
                resp = self._sec_get(url, timeout=90)       # a 10-K is several MB
                resp.raise_for_status()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(resp.content)
        except Exception:
            return None
        return {
            "source_method": "sec",
            "ticker": ticker.upper(),
            "company": company or ticker.upper(),
            "year": filing["date"][:4],
            "filing_type": filing["form"],
            "accession_number": acc,
            "source_url": url,
            "local_path": str(out_path),
            "size_bytes": out_path.stat().st_size,
            "status": "ok",
            "market": "us",
        }

    def pick_filings(self, cik: str, filing_type: str = "10-K", n: int = 1,
                     fiscal_years: Optional[list[int]] = None,
                     accession: Optional[str] = None) -> list[dict]:
        """The filings a fetch would download: one exact filing (`accession`),
        the annual filings for `fiscal_years`, or the latest `n`. Annual tries
        10-K, then 20-F and 40-F for foreign and Canadian issuers. One form
        type per fetch."""
        forms = (filing_type,) if filing_type != "10-K" else ("10-K", "20-F", "40-F")

        def choose(filings):
            if not filings:
                return []
            if accession:
                return [x for x in filings if x["accession"] == accession]
            pick = [x for x in filings if x["form"] == filings[0]["form"]]
            return ([x for x in pick if fiscal_year(x.get("period") or "") in set(fiscal_years)]
                    if fiscal_years else pick)[:n]

        pick = choose(self._list_filings(str(cik), forms))
        if not pick and (accession or fiscal_years):        # older than the recent list
            pick = choose(self._list_filings(str(cik), forms, older=True))
        return pick

    def _download(self, ticker: str, company: str, filing_type: str, n: int,
                  fiscal_years: Optional[list[int]] = None,
                  accession: Optional[str] = None) -> tuple[list[dict], Optional[str]]:
        cik = self.resolver.resolve(ticker).get("cik")
        if not cik:
            return [], None
        pick = self.pick_filings(str(cik), filing_type, n, fiscal_years, accession)
        if not pick:
            return [], None
        used_form = pick[0]["form"]
        records = [rec for f in pick
                   if (rec := self._download_filing(str(cik), ticker, company, f))]
        return records, (used_form if records else None)

    # --- fetch + ingest -----------------------------------------------------

    def fetch_and_ingest(self, ticker: str, company: str = "", filing_type: str = "10-K",
                         n: int = 1, fiscal_years: Optional[list[int]] = None,
                         accession: Optional[str] = None) -> dict:
        """Download one filing (`accession`), the filings for `fiscal_years`, or
        the latest `n`, and ingest them."""
        from finagent.ingestion.ingest import CorpusIngester

        records, used_form = self._download(ticker, company, filing_type, n, fiscal_years,
                                            accession)
        if not records:
            return {"ok": False, "ticker": ticker, "chunks_added": 0,
                    "error": "no filing downloaded", "source_urls": []}

        manifest_path = self.corpus_dir / f"dynamic_fetch_{ticker}.json"
        manifest_path.write_text(json.dumps(records, indent=2))
        stats = CorpusIngester(collection_name=self.collection_name,
                               embedding_model=self.embedding_model).ingest_all(manifest_path)
        return {
            "ok": stats.total_chunks > 0,
            "ticker": ticker,
            "company": company or ticker,
            "form": used_form,
            "chunks_added": stats.total_chunks,
            "filings": len(records),
            "source_urls": [r.get("source_url", "") for r in records],
            "years": [r.get("year") for r in records],
        }
