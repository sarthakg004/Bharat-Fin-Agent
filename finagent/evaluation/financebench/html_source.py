"""
html_source.py  ·  finagent/evaluation/financebench/html_source.py

Acquire the *original SEC HTML* filing for each FinanceBench document, so the
eval measures the SAME ingestion path production serves (unstructured HTML), not
the PDF path. FinanceBench ships PDFs; production serves EDGAR HTML — this closes
that gap.

Mapping (verified, not assumed): every FinanceBench doc carries
`company + doc_period + doc_type`. We resolve company→CIK with the production
`TickerCIKResolver`, then find the filing on EDGAR's submissions API by form +
fiscal period (+ quarter for 10-Qs), and download its primary document. Older
EDGAR primary docs are wrapped in an SEC SGML `<DOCUMENT>..<TEXT>` header that
makes `partition_html` return zero elements — we strip it to `<html>..</html>`,
exactly what `sec-edgar-downloader` hands the production ingester.

Scope: 10-K and 10-Q only. 8-K / earnings docs are excluded because their
evidence lives in an *exhibit* (EX-99.1 press release), not the primary filing,
so a single served HTML document cannot contain it (measured: 0.0 containment).

Files land at `data/us/eval/financebench/html/{doc_name}.htm`, so the existing
manifest/indexing/gold pipeline — which keys on the `local_path` stem — reuses
unchanged.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import requests

from finagent.tools.resolver import TickerCIKResolver, pad_cik, _build_user_agent

HTML_DIR = "data/us/eval/financebench/html"
DOC_INFO = "data/us/eval/financebench/data/financebench_document_information.jsonl"

# FinanceBench doc_type -> EDGAR form. 8-K / earnings deliberately absent: their
# evidence is in exhibits, not the primary document (see module docstring).
FORM = {"10k": "10-K", "10q": "10-Q", "10k_annualreport": "10-K"}

_ACC_RE = re.compile(r"(\d{10}-\d{2}-\d{6})")
_HDR = {"User-Agent": _build_user_agent(), "Accept-Encoding": "gzip, deflate"}
# FinanceBench "QN" quarter -> the calendar-month range a 10-Q's period-end
# falls in. Used to disambiguate the THREE 10-Qs a company files each year.
_Q_MONTHS = {1: (1, 5), 2: (4, 8), 3: (7, 11)}


def _get(url: str) -> requests.Response:
    """GET with EDGAR-friendly retries (SEC allows ~10 req/s)."""
    last = None
    for attempt in range(4):
        r = requests.get(url, headers=_HDR, timeout=30)
        if r.status_code == 200:
            return r
        last = r
        time.sleep(0.4 * (attempt + 1))
    last.raise_for_status()


def _submissions(cik) -> list[dict]:
    """Every filing (recent + older shards) for a CIK."""
    j = _get(f"https://data.sec.gov/submissions/CIK{pad_cik(cik)}.json").json()
    keys = ("accessionNumber", "form", "filingDate", "reportDate", "primaryDocument")
    out: list[dict] = []
    for block in [j["filings"]["recent"]]:
        for i in range(len(block["accessionNumber"])):
            out.append({k: block[k][i] for k in keys})
    for f in j["filings"].get("files", []):
        time.sleep(0.12)
        rj = _get(f"https://data.sec.gov/submissions/{f['name']}").json()
        for i in range(len(rj["accessionNumber"])):
            out.append({k: rj[k][i] for k in keys})
    return out


def _quarter_of(fb_id_or_name: str) -> Optional[int]:
    """Pull the fiscal quarter (1-3) out of a doc_name like 'X_2023Q2_10Q'."""
    m = re.search(r"\bQ([1-4])\b|(\d{4})Q([1-3])", fb_id_or_name or "")
    if not m:
        return None
    q = m.group(1) or m.group(3)
    return int(q) if q else None


def _find_filing(subs: list[dict], form: str, fiscal_year: int,
                 quarter: Optional[int], want_accession: Optional[str]) -> Optional[dict]:
    """Pick the right filing: exact accession if known, else form + fiscal
    period. For 10-Qs the quarter matters — a company files three a year, and
    matching year alone pulls the wrong one (wrong numbers → 0 containment)."""
    if want_accession:
        for s in subs:
            if s["accessionNumber"] == want_accession:
                return s
    cand = [s for s in subs if s["form"] == form]

    def score(s: dict) -> tuple:
        rd, fd = s.get("reportDate") or "", s.get("filingDate") or ""
        ry = int(rd[:4]) if rd[:4].isdigit() else None
        rm = int(rd[5:7]) if len(rd) >= 7 and rd[5:7].isdigit() else None
        fy = int(fd[:4]) if fd[:4].isdigit() else None
        # Year match: report-year == fiscal_year, or filed the next calendar year.
        year_ok = (ry == fiscal_year) or (fy == fiscal_year + 1 and ry is None)
        year_pen = 0 if ry == fiscal_year else (1 if fy == fiscal_year + 1 else 5)
        # Quarter match (10-Q only): period-end month inside the quarter window.
        q_pen = 0
        if quarter and rm is not None:
            lo, hi = _Q_MONTHS.get(quarter, (1, 12))
            q_pen = 0 if lo <= rm <= hi else 3
        return (year_pen + q_pen, abs((ry or 0) - fiscal_year))

    cand.sort(key=score)
    return cand[0] if cand and score(cand[0])[0] < 5 else None


def strip_sec_wrapper(html_text: str) -> str:
    """Drop the SEC SGML `<DOCUMENT>..<TEXT>` header older primary docs carry;
    keep only `<html>..</html>` (what production's downloader yields)."""
    m = re.search(r"<html[\s>].*?</html>", html_text, re.S | re.I)
    return m.group(0) if m else html_text


def _load_doc_info(path: str = DOC_INFO) -> dict:
    info = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            info[r["doc_name"]] = r
    return info


def fetch_html_for_docs(doc_names, html_dir=HTML_DIR, doc_info_path=DOC_INFO,
                        overwrite=False) -> dict:
    """Download the SEC HTML for each 10-K/10-Q doc_name into `html_dir`.

    Returns {doc_name: {status, path|reason, accession?}}. Non-10-K/10-Q docs
    are reported as skipped. Idempotent: existing files are reused.
    """
    html_dir = Path(html_dir)
    html_dir.mkdir(parents=True, exist_ok=True)
    info = _load_doc_info(doc_info_path)
    res = TickerCIKResolver()
    out: dict = {}

    for dn in sorted(set(doc_names)):
        meta = info.get(dn, {})
        form = FORM.get(meta.get("doc_type", ""))
        dest = html_dir / f"{dn}.htm"
        if form is None:
            out[dn] = {"status": "skipped", "reason": f"unsupported type "
                       f"{meta.get('doc_type')!r} (evidence in exhibit)"}
            continue
        if dest.exists() and not overwrite:
            out[dn] = {"status": "ok", "path": str(dest), "cached": True}
            continue
        try:
            company = meta.get("company", dn.split("_")[0])
            year = int(meta.get("doc_period", 0) or 0)
            cik = res.resolve(company).get("cik")
            if not cik:
                out[dn] = {"status": "error", "reason": f"no CIK for {company!r}"}
                continue
            want = None
            m = _ACC_RE.search(meta.get("doc_link", ""))
            if m:
                want = m.group(1)
            time.sleep(0.12)
            filing = _find_filing(_submissions(cik), form, year,
                                  _quarter_of(dn), want)
            if not filing:
                out[dn] = {"status": "error", "reason": "filing not found on EDGAR"}
                continue
            acc = filing["accessionNumber"].replace("-", "")
            url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                   f"{acc}/{filing['primaryDocument']}")
            time.sleep(0.12)
            html = strip_sec_wrapper(_get(url).text)
            dest.write_text(html, errors="ignore")
            out[dn] = {"status": "ok", "path": str(dest),
                       "accession": filing["accessionNumber"], "url": url}
        except Exception as e:
            out[dn] = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
    return out


if __name__ == "__main__":
    # Smoke: fetch the two previously-failing 10-Qs and confirm the quarter fix.
    r = fetch_html_for_docs(
        ["AMCOR_2023Q2_10Q", "MGMRESORTS_2023Q2_10Q", "3M_2022_10K"],
        overwrite=True)
    print(json.dumps(r, indent=2))
