"""
sec_fetch.py  ·  finagent/tools/sec_fetch.py

Dynamic SEC filing fetch (Phase 5) — a self-expanding corpus.

When a question is about a US-listed company we haven't indexed yet, we fetch its
latest 10-K from EDGAR on demand, run it through the *existing* ingestion
pipeline into the live collection, and then retrieve — so the corpus grows to
answer questions it couldn't a moment ago.

The corpus-membership **gate** distinguishes three cases cheaply:
  * already indexed         → nothing to do (retrieval will find it);
  * US-listed, not indexed  → fetch + ingest, then retrieve;
  * not US-listed           → leave it for the web-search branch.

The first two are decided by a cheap metadata query (`ticker == …`,
limit 1) and the Phase-2 resolver (a CIK means SEC-registered/US-listed).

Persistence note: this upserts straight into the Qdrant collection and
survives across runs. The Cloud Run scale-to-zero persistence problem
(ingested chunks vanish when the instance is recycled) is a *deploy-time*
concern handled in Phase 12 — it does not affect local behaviour here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from finagent.tools.base import BaseTool
from finagent.tools.resolver import TickerCIKResolver
from finagent.vectorstore import DEFAULT_EMBED_MODEL
from finagent.config import settings


def _sec_identity() -> tuple[str, str]:
    """SEC EDGAR requires a name + email in the User-Agent of every request."""
    name = (os.getenv("SEC_UA_NAME") or "").strip() or "FinAgent Research"
    email = (os.getenv("SEC_UA_EMAIL") or "").strip() or "finagent@example.com"
    return name, email


class SecFilingFetcher(BaseTool):
    """Fetch + ingest a company's latest SEC 10-K on demand (self-expanding corpus).

    Usage:
        f = SecFilingFetcher()  # collection defaults to settings.us_collection
        f.gate("AAPL")      # -> "already_indexed" | "fetch" | "not_us_listed"
        f.run("CRM")        # resolve -> fetch latest 10-K -> ingest -> report
    """

    name = "sec_fetch"
    description = "Fetch and ingest a company's latest SEC filing if not already indexed."

    def __init__(
        self,
        resolver: Optional[TickerCIKResolver] = None,
        collection_name: str = settings.us_collection,
        corpus_dir: str | Path = "data/us/pdfs",
        embedding_model: str = DEFAULT_EMBED_MODEL,
        market: str = "us",
    ) -> None:
        self.resolver = resolver or TickerCIKResolver()
        self.collection_name = collection_name
        self.corpus_dir = Path(corpus_dir)
        self.embedding_model = embedding_model
        self.market = market

    # --- membership gate -----------------------------------------------------

    def is_indexed(self, ticker: str) -> bool:
        """Cheap metadata check: is any chunk tagged with this ticker?"""
        from finagent.vectorstore import exists_where

        t = (ticker or "").upper().strip()
        if not t:
            return False
        try:
            return exists_where(self.collection_name, "ticker", t)
        except Exception:
            return False

    def gate(self, query: str) -> dict:
        """Classify a company query for the dynamic-fetch decision.

        Returns ``{"decision", "ticker", "cik", "company"}`` where decision ∈
        {"already_indexed", "fetch", "not_us_listed", "unresolved"}.
        """
        r = self.resolver.resolve(query)
        cik, ticker = r.get("cik"), r.get("ticker")
        if not cik:
            # Could not map to a US CIK → not US-listed (web branch handles it).
            return {"decision": "not_us_listed", "ticker": None, "cik": None,
                    "company": None, "query": query}
        if self.is_indexed(ticker):
            return {"decision": "already_indexed", "ticker": ticker, "cik": cik,
                    "company": r.get("title"), "query": query}
        return {"decision": "fetch", "ticker": ticker, "cik": cik,
                "company": r.get("title"), "query": query}

    # --- fetch + ingest ------------------------------------------------------

    def _download(self, ticker: str, company: str, filing_type: str, n: int) -> tuple[list[dict], Optional[str]]:
        """Download the latest `n` filings, sweeping the annual forms
        (10-K → 20-F → 40-F) since foreign private issuers (20-F) and Canadian
        issuers (40-F, e.g. Draganfly/DPRO) don't file 10-Ks. Returns the OK
        records (enriched with company/market) and the form that worked."""
        from finagent.ingestion.fetchPDFs import FetchPDFs

        name, email = _sec_identity()
        fetcher = FetchPDFs(output_dir=self.corpus_dir,
                            sec_user_agent_name=name, sec_user_agent_email=email)
        forms = [filing_type] if filing_type != "10-K" else ["10-K", "20-F", "40-F"]
        for form in forms:
            records = fetcher.from_sec(ticker, filing_type=form, num_filings=n)
            ok = [r for r in records if r.get("status") == "ok"]
            if ok:
                for r in ok:
                    r.setdefault("filing_type", form)
                    r.setdefault("company", company or ticker)
                    r.setdefault("market", self.market)
                return ok, form
        return [], None

    def fetch_chunks(self, ticker: str, company: str = "",
                     filing_type: str = "10-K", n: int = 1) -> dict:
        """Fetch a filing and parse+chunk it **in memory** — NO embedding, NO
        upsert. For the ephemeral path: the chunks are ranked
        in-memory against the question and fed to synthesis, so we never grow or
        persist the index. Returns ``{"ok", "ticker", "form", "chunks": [...]}``
        where each chunk is ``{text, company, ticker, year, source, ...}``.
        """
        from finagent.ingestion.ingest import CorpusIngester

        ok, used_form = self._download(ticker, company, filing_type, n)
        if not ok:
            return {"ok": False, "ticker": ticker, "chunks": [],
                    "error": "no filing downloaded"}

        ingester = CorpusIngester(
            corpus_dir=self.corpus_dir,
            collection_name=self.collection_name, market=self.market,
            embedding_model=self.embedding_model,
        )
        chunks: list[dict] = []
        for rec in ok:
            for doc in ingester.documents_from_record(rec):
                m = doc.metadata
                chunks.append({
                    "text": doc.page_content,
                    "company": m.get("company") or m.get("ticker", ticker),
                    "ticker": m.get("ticker", ticker),
                    "year": m.get("year", ""),
                    "page": m.get("page", "—"),
                    "source_url": m.get("source_url", ""),
                })
        return {"ok": bool(chunks), "ticker": ticker, "company": company or ticker,
                "form": used_form, "chunks": chunks, "filings": len(ok)}

    def fetch_and_ingest(self, ticker: str, company: str = "",
                         filing_type: str = "10-K", n: int = 1) -> dict:
        """Download the latest `n` filings for `ticker` and ingest into the live
        collection (persistent path). Returns counts + the source URLs added."""
        from finagent.ingestion.ingest import CorpusIngester

        ok, used_form = self._download(ticker, company, filing_type, n)
        if not ok:
            return {"ok": False, "ticker": ticker, "chunks_added": 0,
                    "error": "no filing downloaded", "source_urls": []}

        # Write a dedicated manifest so we never clobber the baseline sec_manifest.
        manifest_path = self.corpus_dir / f"dynamic_fetch_{ticker}.json"
        manifest_path.write_text(json.dumps(ok, indent=2))

        ingester = CorpusIngester(
            corpus_dir=self.corpus_dir,
            collection_name=self.collection_name, market=self.market,
            embedding_model=self.embedding_model,
        )
        stats = ingester.ingest_all(manifest_path=manifest_path,
                                    skip_if_already_indexed=True)
        return {
            "ok": stats.total_chunks > 0,
            "ticker": ticker,
            "company": company or ticker,
            "form": used_form,
            "chunks_added": stats.total_chunks,
            "filings": len(ok),
            "source_urls": [r.get("source_url", "") for r in ok],
            "years": [r.get("year") for r in ok],
        }

    # --- dated event-filing fetch (8-K by date) --------------------------------

    # The submissions index + archive documents both require the SEC identity
    # header; a small timeout keeps a dead SEC endpoint from stalling the turn.
    _SEC_TIMEOUT = 20

    def _sec_get(self, url: str):
        import requests

        name, email = _sec_identity()
        return requests.get(url, headers={"User-Agent": f"{name} {email}"},
                            timeout=self._SEC_TIMEOUT)

    def fetch_dated_filing(self, company: str, form: str, date,
                           window_days: int = 7, max_chunks: int = 40) -> dict:
        """Fetch the text of the SPECIFIC filing `company` made on (or nearest
        to) `date` with the given `form` (e.g. an 8-K event disclosure).

        Event filings are never in the indexed corpus (only annual reports are),
        and web search can't reliably surface a years-old dated disclosure — but
        the SEC submissions API lists every filing with its exact date, so we
        can pull the named document directly. Returns
        ``{"ok", "form", "filing_date", "url", "chunks": [...]}`` with chunks
        shaped for the ephemeral `fetched_chunks` path.
        """
        from datetime import date as _date, timedelta

        r = self.resolver.resolve(company)
        cik = r.get("cik")
        if not cik:
            return {"ok": False, "error": f"could not resolve '{company}' to a CIK"}
        cik10 = str(cik).zfill(10)

        target = date if isinstance(date, _date) else _date.fromisoformat(str(date))
        form_norm = form.upper().replace(" ", "").replace("-", "")

        def _scan(block: dict) -> list[dict]:
            forms = block.get("form", []) or []
            dates = block.get("filingDate", []) or []
            accs = block.get("accessionNumber", []) or []
            docs = block.get("primaryDocument", []) or []
            out = []
            for i, f in enumerate(forms):
                if not (f or "").upper().replace(" ", "").replace("-", "").startswith(form_norm):
                    continue
                try:
                    d = _date.fromisoformat(dates[i])
                except Exception:
                    continue
                if abs((d - target).days) <= window_days:
                    out.append({"form": f, "date": d, "accession": accs[i],
                                "doc": docs[i] if i < len(docs) else ""})
            return out

        resp = self._sec_get(f"https://data.sec.gov/submissions/CIK{cik10}.json")
        resp.raise_for_status()
        sub = resp.json()
        recent = (sub.get("filings", {}) or {}).get("recent", {}) or {}
        matches = _scan(recent)

        # Older than the ~1000-filing "recent" window → walk the paged archives
        # whose [filingFrom, filingTo] range covers the target date.
        if not matches:
            for page in (sub.get("filings", {}) or {}).get("files", []) or []:
                try:
                    lo = _date.fromisoformat(page.get("filingFrom", "9999-12-31"))
                    hi = _date.fromisoformat(page.get("filingTo", "0001-01-01"))
                except Exception:
                    continue
                if lo - timedelta(days=window_days) <= target <= hi + timedelta(days=window_days):
                    pr = self._sec_get(f"https://data.sec.gov/submissions/{page['name']}")
                    pr.raise_for_status()
                    matches = _scan(pr.json())
                    if matches:
                        break

        if not matches:
            return {"ok": False, "error": (f"no {form} filing within {window_days} days "
                                           f"of {target} for CIK {cik}")}

        best = min(matches, key=lambda m: abs((m["date"] - target).days))
        acc = best["accession"].replace("-", "")
        doc = best["doc"] or f"{best['accession']}.txt"
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
        dr = self._sec_get(url)
        dr.raise_for_status()

        # Parse + chunk through the shared ingestion pipeline (unstructured
        # chunk_by_title, tables rendered from text_as_html) — same fidelity
        # the 10-K path gets — instead of regex-stripping tables to whitespace.
        import tempfile

        from finagent.ingestion.ingest import CorpusIngester

        company_name = r.get("company") or sub.get("name") or company
        suffix = Path(doc).suffix or ".htm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(dr.content)
            tmp = Path(tf.name)
        try:
            ingester = CorpusIngester(
                corpus_dir=self.corpus_dir,
                collection_name=self.collection_name, market=self.market,
                embedding_model=self.embedding_model,
            )
            docs = ingester.documents_from_record({
                "local_path": str(tmp), "source_url": url,
                "company": company_name, "ticker": r.get("ticker", ""),
                "year": str(best["date"].year), "filing_type": best["form"],
            })
        finally:
            tmp.unlink(missing_ok=True)

        shaped = [{"text": d.page_content, "company": company_name,
                   "ticker": r.get("ticker", ""), "year": str(best["date"].year),
                   "page": d.metadata.get("page", i + 1),
                   "filing_type": best["form"], "source_url": url}
                  for i, d in enumerate(docs[:max_chunks])]
        return {"ok": True, "form": best["form"],
                "filing_date": best["date"].isoformat(), "url": url,
                "chunks": shaped}

    # --- orchestration -------------------------------------------------------

    def run(self, ticker: str, filing_type: str = "10-K", n: int = 1) -> dict:
        """Gate then fetch. Returns a status dict with `status` ∈
        {already_indexed, fetched, not_us_listed, error}."""
        g = self.gate(ticker)
        decision = g["decision"]
        if decision in ("not_us_listed", "unresolved"):
            return {"status": "not_us_listed", **g}
        if decision == "already_indexed":
            return {"status": "already_indexed", **g}
        # decision == "fetch"
        res = self.fetch_and_ingest(g["ticker"], company=g.get("company") or "",
                                    filing_type=filing_type, n=n)
        if not res.get("ok"):
            return {"status": "error", **g, **res}
        return {"status": "fetched", **g, **res}
