"""Page-cap guard for uploads (cost control): cheap count, correct threshold."""

from io import BytesIO

from pypdf import PdfWriter

from finagent.api.main import _UPLOAD_MAX_PAGES, _pdf_page_count


def _blank_pdf(n_pages: int) -> bytes:
    w = PdfWriter()
    for _ in range(n_pages):
        w.add_blank_page(width=72, height=72)
    buf = BytesIO()
    w.write(buf)
    return buf.getvalue()


def test_page_count_and_threshold():
    assert _pdf_page_count(_blank_pdf(3)) == 3
    big = _pdf_page_count(_blank_pdf(_UPLOAD_MAX_PAGES + 5))
    assert big == _UPLOAD_MAX_PAGES + 5 and big > _UPLOAD_MAX_PAGES
    assert _pdf_page_count(b"not a pdf") is None   # unreadable → let Docling try


if __name__ == "__main__":
    test_page_count_and_threshold()
    print("ok")
