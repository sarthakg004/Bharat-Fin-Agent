"""The un-indexed-company path: fetch the filing, don't search the wrong ones,
and don't call the answer confident when the filing never arrived.

Traced failure (Langfuse f7960494, "Applied Materials' primary business
segments … latest fiscal year"): the EDGAR fetch died on an ImportError for a
package the deployed image doesn't ship, corpus retrieval then returned
Walmart/Corning/General Mills/3M chunks for an AMAT question, and the answer —
written entirely from analyst web pages, quoting a QUARTER for a fiscal-year
question.

No network: the SEC submissions index is a fake.
"""

from __future__ import annotations

from finagent.graph.nodes.fetch import FetchNodes
from finagent.tools.sec_fetch import SecFilingFetcher

# Shape of data.sec.gov/submissions/CIK##########.json, trimmed to the keys read.
FAKE_SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["8-K", "10-K", "10-K/A", "10-Q", "10-K"],
            "filingDate": ["2025-12-20", "2025-12-12", "2025-12-15",
                           "2025-08-14", "2024-12-13"],
            "accessionNumber": ["0000000000-25-000001", "0001628280-25-056742",
                                "0000000000-25-000002", "0000000000-25-000003",
                                "0000006951-24-000044"],
            "primaryDocument": ["ev.htm", "amat-20251026.htm", "amend.htm",
                                "q3.htm", "amat-20241027.htm"],
        }
    }
}


class _FakeResponse:
    def __init__(self, payload=None, content=b""):
        self._payload, self.content = payload, content

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


def _fetcher(tmp_path, downloads: list) -> SecFilingFetcher:
    f = SecFilingFetcher(corpus_dir=tmp_path)

    def fake_get(url, timeout=None):
        if "data.sec.gov/submissions" in url:
            return _FakeResponse(payload=FAKE_SUBMISSIONS)
        downloads.append(url)
        return _FakeResponse(content=b"x" * 60_000)

    f._sec_get = fake_get                      # type: ignore[method-assign]
    f.resolver = type("R", (), {"resolve": staticmethod(
        lambda q: {"cik": "0000006951", "ticker": "AMAT",
                   "title": "APPLIED MATERIALS INC /DE"})})()
    return f


def test_list_filings_takes_exact_forms_newest_first(tmp_path):
    got = _fetcher(tmp_path, [])._list_filings("0000006951", ("10-K",))
    # 10-K/A is an amendment, not a 10-K — an exact form match only.
    assert [f["date"] for f in got] == ["2025-12-12", "2024-12-13"]
    assert got[0]["doc"] == "amat-20251026.htm"


def test_download_needs_no_ingest_extra(tmp_path):
    """The whole point: this path must run on the serve image, which installs
    neither sec-edgar-downloader nor pypdf."""
    downloads: list[str] = []
    records, form = _fetcher(tmp_path, downloads)._download(
        "AMAT", "APPLIED MATERIALS INC /DE", "10-K", 2)

    assert form == "10-K"
    assert [r["year"] for r in records] == ["2025", "2024"]
    assert downloads[0] == ("https://www.sec.gov/Archives/edgar/data/6951/"
                            "000162828025056742/amat-20251026.htm")
    # Manifest shape CorpusIngester consumes, unchanged from the old path.
    rec = records[0]
    assert rec["status"] == "ok" and rec["filing_type"] == "10-K"
    assert rec["company"] == "APPLIED MATERIALS INC /DE"
    assert open(rec["local_path"], "rb").read(1) == b"x"


def test_download_reports_failure_rather_than_raising(tmp_path):
    """A dead SEC endpoint must degrade to the web branch, not blow up the turn
    (the ImportError used to propagate as a fetch_status error string)."""
    f = _fetcher(tmp_path, [])

    def boom(url, timeout=None):
        raise OSError("SEC unreachable")

    f._sec_get = boom                          # type: ignore[method-assign]
    assert f._download("AMAT", "", "10-K", 1) == ([], None)


def test_corpus_retrieval_skipped_when_gate_proved_absence():
    """`decision == "fetch"` means the resolver found the company and the
    collection has nothing for it — so the corpus can only return SOMEONE
    ELSE'S filing."""
    lacks = FetchNodes._corpus_lacks_company
    # Cloud/ephemeral: parsed in memory, never written — corpus still empty.
    assert lacks({"fetch_status": {"decision": "fetch", "status": "fetched",
                                   "ephemeral": True}})
    assert lacks({"fetch_status": {"decision": "fetch", "status": "error"}})

    # Persistent ingest (PERSIST_DYNAMIC_FETCH=true, how the deploy runs) put
    # the filing IN the collection — skipping would discard it.
    assert not lacks({"fetch_status": {"decision": "fetch", "status": "fetched"}})
    for status in ({"decision": "already_indexed"}, {"decision": "not_us_listed"},
                   {}, None):
        assert not lacks({"fetch_status": status})
