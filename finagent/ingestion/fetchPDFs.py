"""
fetch_pdfs.py

Download SEC filings (US) either from EDGAR or from a CSV of URLs:

    - from_sec():   SEC EDGAR API for US-listed companies
    - from_urls():  bulk download from a CSV of pre-collected URLs

Usage as a library
------------------
    from fetch_pdfs import FetchPDFs

    # SEC EDGAR (requires your name + email for SEC's User-Agent policy)
    fetcher = FetchPDFs(
        output_dir="data/us",
        sec_user_agent_name="Your Name",
        sec_user_agent_email="you@example.com",
    )
    records = fetcher.from_sec(["AAPL", "MSFT", "NVDA"], num_filings=3)

Usage as CLI
------------
    python fetch_pdfs.py sec  --tickers AAPL,MSFT,NVDA --num-filings 3 \\
                              --out data/us --name "Your Name" --email "you@example.com"
    python fetch_pdfs.py urls --csv urls.csv --out data/pdfs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd
import requests
from tqdm import tqdm


class FetchPDFs:
    """Download financial filings via SEC EDGAR or a URL list."""

    # Browser-like headers; required because most IR sites block requests
    # without a recognized User-Agent.
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    MIN_VALID_PDF_BYTES = 10_000  # below this we treat the file as junk / notice

    def __init__(
        self,
        output_dir: Union[str, Path] = "data/pdfs",
        polite_delay: float = 1.5,
        max_retries: int = 3,
        retry_backoff: float = 5.0,
        timeout: int = 60,
        sec_user_agent_name: Optional[str] = None,
        sec_user_agent_email: Optional[str] = None,
    ):
        """
        Args:
            output_dir: Where downloaded files go.
            polite_delay: Seconds between successful downloads.
            max_retries: HTTP retry count.
            retry_backoff: Backoff multiplier in seconds.
            timeout: Per-request timeout in seconds.
            sec_user_agent_name: Your name (required by SEC EDGAR for from_sec()).
                Can also be passed per-call. Not used by other methods.
            sec_user_agent_email: Your email (required by SEC EDGAR for from_sec()).
                Can also be passed per-call. Not used by other methods.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.polite_delay = polite_delay
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.timeout = timeout
        self.sec_user_agent_name = sec_user_agent_name
        self.sec_user_agent_email = sec_user_agent_email
        self._session = requests.Session()
        self._session.headers.update(self.DEFAULT_HEADERS)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def from_urls(self, csv_path: Union[str, Path]) -> list[dict]:
        """Download PDFs whose URLs are listed in a CSV.

        Required columns: company, year, pdf_url
        Optional columns: sector

        Writes manifest.json into output_dir. Returns the manifest records.
        """
        df = pd.read_csv(csv_path)
        required = {"company", "year", "pdf_url"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        df = df[df["pdf_url"].notna() & (df["pdf_url"].str.strip() != "")]
        if df.empty:
            raise ValueError(
                "No URLs to download. Fill in the pdf_url column first."
            )

        records: list[dict] = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="from_urls"):
            out_path = self.output_dir / row["company"] / f"{row['year']}.pdf"
            ok, error = self._download_pdf(row["pdf_url"], out_path)
            records.append({
                "source_method": "urls",
                "company": row["company"],
                "year": int(row["year"]),
                "sector": row.get("sector", ""),
                "source_url": row["pdf_url"],
                "local_path": str(out_path),
                "size_bytes": out_path.stat().st_size if out_path.exists() else 0,
                "status": "ok" if ok else "failed",
                "error": error,
            })

        self.write_manifest(records, "manifest.json")
        self._print_summary(records)
        return records

    def from_sec(
        self,
        tickers: Union[str, Iterable[str]],
        filing_type: str = "10-K",
        num_filings: int = 3,
        after: Optional[str] = None,
        before: Optional[str] = None,
        user_agent_name: Optional[str] = None,
        user_agent_email: Optional[str] = None,
    ) -> list[dict]:
        """Download US SEC filings via the EDGAR API.

        The SEC requires identification in every request's User-Agent header.
        Pass your real name and email — they are not registered or used for
        any account, only logged on SEC's side for compliance / abuse tracking.

        Args:
            tickers: Comma-separated string or iterable of US tickers,
                e.g. "AAPL,MSFT,NVDA" or ["AAPL", "MSFT", "NVDA"].
            filing_type: SEC form name. Common values:
                "10-K"  annual report (use this for RAG corpus)
                "10-Q"  quarterly report
                "8-K"   current events
                "DEF 14A" proxy statement
            num_filings: Number of most recent filings to download per ticker.
                Ignored if both `after` and `before` are supplied.
            after: Optional ISO date "YYYY-MM-DD" — only filings on/after this date.
            before: Optional ISO date "YYYY-MM-DD" — only filings on/before this date.
            user_agent_name: Override the class-level SEC identification name.
            user_agent_email: Override the class-level SEC identification email.

        Returns:
            List of manifest records. Each record has fields: ticker, year,
            filing_type, accession_number, source_url, local_path, status.

        Notes:
            - Files land at {output_dir}/{TICKER}/{YEAR}_{accession}.htm
              (we copy the primary 10-K HTML out of the library's nested folder
              structure so the corpus layout matches the from_urls() output).
            - The full nested download remains at
              {output_dir}/sec-edgar-filings/{TICKER}/{filing_type}/{accession}/
              in case you need other documents from the same filing.
            - The 10-K HTML is what you want to ingest. The .txt full submission
              is a concatenated dump that's much harder to parse.
        """
        try:
            from sec_edgar_downloader import Downloader
        except ImportError as e:
            raise ImportError(
                "The 'sec-edgar-downloader' package is required for from_sec(). "
                "Install it with:  pip install sec-edgar-downloader"
            ) from e

        name = user_agent_name or self.sec_user_agent_name
        email = user_agent_email or self.sec_user_agent_email
        if not name or not email:
            raise ValueError(
                "from_sec() requires SEC EDGAR identification (name + email). "
                "Pass them to FetchPDFs(...) or to from_sec(...) directly. "
                "SEC requires this in every request's User-Agent header."
            )

        tickers = self._normalize_tickers(tickers)
        records: list[dict] = []

        # The Downloader creates {output_dir}/sec-edgar-filings/{TICKER}/{form}/{accession}/
        # underneath whatever folder we hand it. We pass our class-level output_dir
        # so everything stays in one tree.
        dl = Downloader(name, email, str(self.output_dir))

        for ticker in tqdm(tickers, desc="from_sec"):
            records.extend(
                self._process_sec_ticker(
                    ticker, dl, filing_type, num_filings, after, before
                )
            )

        self.write_manifest(records, "sec_manifest.json")
        self._print_summary(records)
        return records

    def write_manifest(self, records: list[dict], name: str) -> Path:
        """Write a JSON manifest into output_dir."""
        manifest_path = self.output_dir / name
        with open(manifest_path, "w") as f:
            json.dump(records, f, indent=2)
        return manifest_path

    def close(self):
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ------------------------------------------------------------------ #
    # Internal helpers — used by all three public methods
    # ------------------------------------------------------------------ #

    def _download_pdf(
        self,
        url: str,
        out_path: Path,
        session: Optional[requests.Session] = None,
    ) -> tuple[bool, Optional[str]]:
        """Download a single PDF with retries.

        Returns (success, error_message). Success means the file already
        existed or was newly downloaded and looks like a PDF.
        """
        if out_path.exists() and out_path.stat().st_size > self.MIN_VALID_PDF_BYTES:
            return True, None

        out_path.parent.mkdir(parents=True, exist_ok=True)
        session = session or self._session

        for attempt in range(1, self.max_retries + 1):
            try:
                with session.get(
                    url,
                    timeout=self.timeout,
                    stream=True,
                    allow_redirects=True,
                ) as resp:
                    resp.raise_for_status()
                    content_type = resp.headers.get("Content-Type", "").lower()

                    first_chunk = next(resp.iter_content(chunk_size=8192), b"")
                    if not (
                        "pdf" in content_type
                        or first_chunk.startswith(b"%PDF")
                    ):
                        return False, f"not a PDF (content-type={content_type})"

                    with open(out_path, "wb") as f:
                        f.write(first_chunk)
                        for chunk in resp.iter_content(chunk_size=64 * 1024):
                            if chunk:
                                f.write(chunk)

                time.sleep(self.polite_delay)
                return True, None

            except (requests.RequestException, OSError) as e:
                if attempt == self.max_retries:
                    return False, f"failed after {self.max_retries} attempts: {e}"
                time.sleep(self.retry_backoff * attempt)

        return False, "unknown error"

    @staticmethod
    def _normalize_tickers(tickers: Union[str, Iterable[str]]) -> list[str]:
        if isinstance(tickers, str):
            tickers = tickers.split(",")
        return [t.strip().upper() for t in tickers if t and t.strip()]

    @staticmethod
    def _print_summary(records: list[dict]) -> None:
        if not records:
            print("\nNo records returned.")
            return
        ok = sum(1 for r in records if r["status"] == "ok")
        failed = [r for r in records if r["status"] != "ok"]
        print(f"\nDownloaded: {ok}/{len(records)}")
        if failed:
            print("Failures:")
            for r in failed[:10]:
                label = r.get("company") or r.get("ticker") or "?"
                year = r.get("year", "")
                err = r.get("error", "")
                print(f"  - {label} {year}: {err}")
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more")

    def _process_sec_ticker(
        self,
        ticker: str,
        dl,
        filing_type: str,
        num_filings: int,
        after: Optional[str],
        before: Optional[str],
    ) -> list[dict]:
        """Download SEC filings for one ticker and normalize the file layout.

        The sec-edgar-downloader library drops everything into a nested
        folder structure. We do the download, then walk the result and
        copy the primary 10-K HTML out to a flat layout that matches
        the rest of this class's output convention.
        """
        records: list[dict] = []

        try:
            # If date filters are given, use them; else use limit.
            kwargs = {"download_details": True}
            if after or before:
                if after:
                    kwargs["after"] = after
                if before:
                    kwargs["before"] = before
            else:
                kwargs["limit"] = num_filings
            dl.get(filing_type, ticker, **kwargs)
        except Exception as e:
            print(f"[SEC {ticker}] download failed: {e}")
            return records

        # Library writes to {output_dir}/sec-edgar-filings/{TICKER}/{FORM}/{accession}/
        sec_root = (
            self.output_dir
            / "sec-edgar-filings"
            / ticker
            / filing_type
        )
        if not sec_root.exists():
            print(f"[SEC {ticker}] no filings landed under {sec_root}")
            return records

        for filing_dir in sorted(sec_root.iterdir()):
            if not filing_dir.is_dir():
                continue

            primary_doc = self._find_primary_sec_document(filing_dir)
            if primary_doc is None:
                print(f"[SEC {ticker}] no primary document in {filing_dir.name}")
                continue

            accession = filing_dir.name  # e.g. "0000320193-23-000106"
            year = self._extract_sec_year(accession)

            # Normalized output: {output_dir}/{TICKER}/{YEAR}_{accession_short}.htm
            # We keep accession in the filename so multiple filings in same year
            # (e.g. 10-K + amendment) don't collide.
            short_accession = accession.split("-")[-1]  # last segment is unique
            out_path = (
                self.output_dir / ticker / f"{year}_{short_accession}.htm"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                import shutil
                if not out_path.exists() or out_path.stat().st_size < self.MIN_VALID_PDF_BYTES:
                    shutil.copy2(primary_doc, out_path)
                status = "ok"
                error = None
            except OSError as e:
                status = "failed"
                error = str(e)

            # Build EDGAR archive URL for traceability in the manifest.
            cik_dashes_removed = accession.replace("-", "")
            source_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_dashes_removed}/{primary_doc.name}"
            )

            records.append({
                "source_method": "sec",
                "ticker": ticker,
                "year": year,
                "filing_type": filing_type,
                "accession_number": accession,
                "source_url": source_url,
                "local_path": str(out_path),
                "size_bytes": out_path.stat().st_size if out_path.exists() else 0,
                "status": status,
                "error": error,
            })

        return records

    @staticmethod
    def _find_primary_sec_document(filing_dir: Path) -> Optional[Path]:
        """Pick the primary 10-K document from a SEC filing folder.

        Strategy: prefer .htm/.html files, since the .txt 'full submission'
        is a concatenated dump that's harder to parse. Among HTML files,
        the largest one is almost always the main 10-K (the others are
        small exhibits like certifications and consents).
        """
        html_files = (
            list(filing_dir.glob("*.htm"))
            + list(filing_dir.glob("*.html"))
        )
        # Filter out the tiny boilerplate files (certifications, consents).
        html_files = [f for f in html_files if f.stat().st_size > 50_000]
        if html_files:
            return max(html_files, key=lambda p: p.stat().st_size)
        # Last-resort fallback: .txt full submission.
        txt_files = list(filing_dir.glob("*.txt"))
        return txt_files[0] if txt_files else None

    @staticmethod
    def _extract_sec_year(accession_number: str) -> str:
        """Pull a 4-digit year out of an SEC accession number.

        Accession format is NNNNNNNNNN-YY-NNNNNN where YY is the 2-digit
        year the filing was submitted. We assume any YY <= current-year-2k
        is 2000s, else 1900s (this rule is fine through ~2030).
        """
        parts = accession_number.split("-")
        if len(parts) >= 2 and len(parts[1]) == 2 and parts[1].isdigit():
            yy = int(parts[1])
            return str(2000 + yy if yy <= 40 else 1900 + yy)
        return "unknown"


# ---------------------------------------------------------------------- #
# CLI entry point
# ---------------------------------------------------------------------- #

def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download financial filings via SEC EDGAR or a URL CSV.",
    )
    sub = parser.add_subparsers(dest="method", required=True)

    p_urls = sub.add_parser("urls", help="Bulk download from a CSV of URLs")
    p_urls.add_argument("--csv", required=True, help="Path to urls.csv")
    p_urls.add_argument("--out", default="data/pdfs")

    p_sec = sub.add_parser("sec", help="SEC EDGAR (US companies)")
    p_sec.add_argument("--tickers", required=True,
                       help="Comma-separated US tickers, e.g. AAPL,MSFT,NVDA")
    p_sec.add_argument("--filing-type", default="10-K",
                       help="SEC form type (default: 10-K)")
    p_sec.add_argument("--num-filings", type=int, default=3,
                       help="Most recent N filings per ticker (default: 3)")
    p_sec.add_argument("--after", default=None,
                       help="Optional ISO date filter YYYY-MM-DD")
    p_sec.add_argument("--before", default=None,
                       help="Optional ISO date filter YYYY-MM-DD")
    p_sec.add_argument("--name", required=True,
                       help="Your name (SEC EDGAR User-Agent requirement)")
    p_sec.add_argument("--email", required=True,
                       help="Your email (SEC EDGAR User-Agent requirement)")
    p_sec.add_argument("--out", default="data/us")

    return parser


def main():
    args = _build_cli().parse_args()

    if args.method == "sec":
        with FetchPDFs(
            output_dir=args.out,
            sec_user_agent_name=args.name,
            sec_user_agent_email=args.email,
        ) as fetcher:
            fetcher.from_sec(
                args.tickers,
                filing_type=args.filing_type,
                num_filings=args.num_filings,
                after=args.after,
                before=args.before,
            )
    else:
        with FetchPDFs(output_dir=args.out) as fetcher:
            if args.method == "urls":
                fetcher.from_urls(args.csv)


if __name__ == "__main__":
    main()