"""The HTML ingest path must build the same parent geometry as the PDF path.

These drifted apart once: `PARENT_CHUNK_SIZE` moved 1500 -> 2500 (+0.070 cov@8,
results/RETRIEVAL_EXPERIMENTS.md §6a) but the HTML path never read that constant,
and a separate `[:1900]` truncation capped every parent below it anyway — so the
served corpus, which is 100% HTML, kept the losing geometry while the eval corpus
got the win. Both failure modes are asserted here.
"""
import importlib.util

import pytest

from finagent.ingestion.ingest import (
    EMBED_CHAR_CAP,
    PARENT_TEXT_CAP,
    CorpusIngester,
)

# `unstructured` IS a runtime dependency now: the dynamic-fetch path chunks a
# freshly downloaded 10-K inside the container, so the image that used to skip
# these tests is the image that runs this code. The guard stays for the bare
# environments that install neither (a docs-only checkout, a partial dev env).
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("unstructured") is None,
    reason="'unstructured' not installed",
)


def _filing_html(paragraphs: int = 60) -> str:
    body = "".join(
        f"<p>Item 7. Management's discussion paragraph {i} — "
        f"revenue grew on higher volumes and pricing across all segments. "
        f"Operating margin expanded year over year. {'filler ' * 12}</p>"
        for i in range(paragraphs)
    )
    return f"<html><body><h1>Annual Report</h1>{body}</body></html>"


def _ingest(tmp_path, html: str, **kwargs):
    path = tmp_path / "filing.htm"
    path.write_text(html)
    ing = CorpusIngester(collection_name="unused", **kwargs)
    return ing._html_documents(path, {"company": "TEST"}, parent_doc=True)


def test_html_parents_match_the_pdf_path_geometry(tmp_path):
    docs = _ingest(tmp_path, _filing_html())
    assert docs, "no documents produced"

    parents = {d.metadata["parent_id"]: d.metadata["parent_text"] for d in docs}
    longest = max(len(p) for p in parents.values())

    # The whole point of the change: parents must be able to exceed the old
    # 1500/1900 ceilings and reach PARENT_CHUNK_SIZE.
    assert longest > 1900, (
        f"longest parent is {longest} chars — a truncation is still capping "
        f"parents below PARENT_CHUNK_SIZE={CorpusIngester.PARENT_CHUNK_SIZE}")
    assert longest <= PARENT_TEXT_CAP


def test_embedded_text_stays_inside_the_encoder_window(tmp_path):
    # A table is kept whole (one child == the parent), so it is the case where a
    # big parent could leak into embedded text.
    rows = "".join(
        f"<tr><td>Line item {i}</td><td>{i * 1000}</td><td>{i * 900}</td></tr>"
        for i in range(200)
    )
    html = f"<html><body><h1>Balance sheet</h1><table>{rows}</table></body></html>"
    docs = _ingest(tmp_path, html)

    assert docs, "no documents produced"
    for d in docs:
        assert len(d.page_content) <= EMBED_CHAR_CAP, (
            f"{len(d.page_content)}-char chunk would be silently truncated by "
            f"the embedder")


def test_html_geometry_is_not_hardcoded(tmp_path):
    # The defaults must track the class constants, not a copy of their values.
    docs = _ingest(tmp_path, _filing_html(paragraphs=120),
                   html_max_chars=800, html_new_after=700, html_overlap=100)
    longest = max(len(d.metadata["parent_text"]) for d in docs)
    assert longest <= 900, f"html_max_chars ignored: {longest}-char parent"


if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path

    for fn in (test_html_parents_match_the_pdf_path_geometry,
               test_embedded_text_stays_inside_the_encoder_window,
               test_html_geometry_is_not_hardcoded):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
        print(f"ok  {fn.__name__}")
    sys.exit(0)
