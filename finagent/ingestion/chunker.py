"""
chunker.py  ·  finagent/ingestion/chunker.py

The text-splitting step, exposed standalone so the notebook can show a page of
parsed text turning into retrievable chunks. Uses the SAME configuration as the
production `CorpusIngester`: `RecursiveCharacterTextSplitter`, chunk size 1000,
overlap 200, paragraph→sentence→word separator hierarchy.
"""

from __future__ import annotations

from finagent.ingestion.ingest import CorpusIngester

CHUNK_SIZE = CorpusIngester.DEFAULT_CHUNK_SIZE       # 1000
CHUNK_OVERLAP = CorpusIngester.DEFAULT_CHUNK_OVERLAP  # 200


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split `text` into overlapping chunks (production splitter config)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)
