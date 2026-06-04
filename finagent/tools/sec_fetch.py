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

The first two are decided by a cheap Chroma metadata query (`where ticker=…`,
limit 1) and the Phase-2 resolver (a CIK means SEC-registered/US-listed).

Persistence note: locally this writes straight to the on-disk Chroma store and
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


def _sec_identity() -> tuple[str, str]:
    """SEC EDGAR requires a name + email in the User-Agent of every request."""
    name = (os.getenv("SEC_UA_NAME") or "").strip() or "FinAgent Research"
    email = (os.getenv("SEC_UA_EMAIL") or "").strip() or "finagent@example.com"
    return name, email


class SecFilingFetcher(BaseTool):
    """Fetch + ingest a company's latest SEC 10-K on demand (self-expanding corpus).

    Usage:
        f = SecFilingFetcher(collection_name="us_filings", chroma_dir="data/chroma")
        f.gate("AAPL")      # -> "already_indexed" | "fetch" | "not_us_listed"
        f.run("CRM")        # resolve -> fetch latest 10-K -> ingest -> report
    """

    name = "sec_fetch"
    description = "Fetch and ingest a company's latest SEC filing if not already indexed."

    def __init__(
        self,
        resolver: Optional[TickerCIKResolver] = None,
        collection_name: str = "us_filings",
        chroma_dir: str | Path = "data/chroma",
        corpus_dir: str | Path = "data/us/pdfs",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        market: str = "us",
    ) -> None:
        self.resolver = resolver or TickerCIKResolver()
        self.collection_name = collection_name
        self.chroma_dir = str(chroma_dir)
        self.corpus_dir = Path(corpus_dir)
        self.embedding_model = embedding_model
        self.market = market
        self._store = None

    # --- membership gate -----------------------------------------------------

    def _get_store(self):
        if self._store is None:
            from finagent.vectorstore import build_store
            self._store = build_store(self.collection_name, self.embedding_model,
                                      self.chroma_dir)
        return self._store

    def is_indexed(self, ticker: str) -> bool:
        """Cheap Chroma metadata check: is any chunk tagged with this ticker?"""
        t = (ticker or "").upper().strip()
        if not t:
            return False
        try:
            got = self._get_store()._collection.get(where={"ticker": t}, limit=1)
            return bool(got.get("ids"))
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
        Chroma write. For the cloud / per-session path: the chunks are ranked
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
            corpus_dir=self.corpus_dir, chroma_dir=self.chroma_dir,
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
            corpus_dir=self.corpus_dir, chroma_dir=self.chroma_dir,
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
