"""
upload.py  ·  finagent/ingestion/upload.py

Parse a user-uploaded document (PDF/DOCX) into ephemeral in-memory chunks with
Docling. The chunks mirror the shape of `sec_fetch.fetch_chunks` output, so
they ride the existing `fetched_chunks` lane through the agent: ranked in
memory against the question (`_rank_fetched_chunks`) and never written to the
persistent index — the stateless-safe pattern the dynamic-fetch path
already uses on Cloud Run.

Docling (layout model + TableFormer, no OCR) is a heavy import; it is loaded
lazily on first upload and the converter is cached process-wide.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

_converter = None


def _get_converter():
    """Process-wide DocumentConverter (the layout/table models are loaded once)."""
    global _converter
    if _converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
        artifacts = os.getenv("DOCLING_ARTIFACTS_PATH")
        if artifacts:
            opts.artifacts_path = artifacts
        _converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    return _converter


def _get_splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from finagent.ingestion.ingest import CorpusIngester

    return RecursiveCharacterTextSplitter(
        chunk_size=CorpusIngester.DEFAULT_CHUNK_SIZE,
        chunk_overlap=CorpusIngester.DEFAULT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def parse_upload(data: bytes, filename: str) -> dict:
    """Parse one uploaded document into ranked-ready chunks.

    Returns ``{ok, filename, pages, tables, chunks}`` where each chunk is
    ``{text, company, ticker, year, page, source_url, filing_type, filename}``
    — the `fetch_chunks` shape plus upload provenance.
    """
    from docling.datamodel.base_models import DocumentStream

    result = _get_converter().convert(DocumentStream(name=filename,
                                                     stream=BytesIO(data)))
    doc = result.document

    # Group extracted text per page; tables are appended to their page as
    # markdown so figures inside tables stay retrievable next to their labels.
    pages: dict[int, list[str]] = {}
    for item in doc.texts:
        page = item.prov[0].page_no if item.prov else 0
        if item.text and item.text.strip():
            pages.setdefault(page, []).append(item.text)
    n_tables = 0
    for tbl in doc.tables:
        n_tables += 1
        page = tbl.prov[0].page_no if tbl.prov else 0
        try:
            md = tbl.export_to_markdown(doc)
        except TypeError:                      # older docling signature
            md = tbl.export_to_markdown()
        if md.strip():
            pages.setdefault(page, []).append(md)

    company = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    splitter = _get_splitter()
    chunks: list[dict] = []
    for page in sorted(pages):
        for chunk in splitter.split_text("\n\n".join(pages[page])):
            chunks.append({
                "text": chunk,
                "company": company,
                "ticker": "",
                "year": "",
                "page": page or "—",
                "source_url": "",
                "filing_type": "uploaded",
                "filename": filename,
            })

    return {
        "ok": bool(chunks),
        "filename": filename,
        "pages": len(pages),
        "tables": n_tables,
        "chunks": chunks,
    }
