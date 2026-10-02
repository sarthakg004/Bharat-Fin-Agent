"""Turn SEC HTML filings into searchable chunks in Qdrant.

Chunking strategy (parent-document retrieval):

    filing.htm -> elements (unstructured)
               -> PARENT passages: a section, up to 2,500 characters
               -> CHILD chunks: 600 characters with 100 overlap, cut from a parent
               -> every chunk prefixed with "<company> <year> · <section caption>"
               -> dense + BM25 vectors -> Qdrant

Only the small children are embedded, because a short chunk matches a query
precisely. Each child carries its parent's text, and the parent is what the
reranker scores and the writer reads, because the answer needs the context
around the match.

A table is never split: it is its own parent and its own child, so its rows
stay together. The prefix gives a chunk the identity it would otherwise lack: a
balance-sheet table holds no company name, year or statement title.

A chunk's point id is derived from its text, so the index stays valid only
while this module produces the same text for the same filing.
"""

from __future__ import annotations

import argparse
import html as _html
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from tqdm import tqdm

from finagent.vectorstore import DEFAULT_EMBED_MODEL, EmbeddingQuotaExhausted

# A 10-K section heading at the start of a line: "Item 1A. Risk Factors".
_ITEM_RE = re.compile(r"(?im)^\s*item\s+(\d{1,2}[ab]?)\s*[.:—–-]")

PARENT_CHARS = 2500         # the passage handed to the reranker and the model
PARENT_NEW_AFTER = 2000     # start a new parent after this many characters
PARENT_COMBINE_UNDER = 400  # merge sections shorter than this
PARENT_OVERLAP = 300
CHILD_CHARS = 600           # the chunk that is embedded and matched
CHILD_OVERLAP = 100
EMBED_CHAR_CAP = 7000       # what Gemini's embedder can read (2,048 tokens)
MIN_VALID_FILE_BYTES = 10_000   # smaller files are failed downloads

# Lines that look like a heading but name nothing: page numbers, "Table of Contents".
_NOT_A_CAPTION = re.compile(r"^(table of contents|page\s*\d+|\d{1,4}|[ivxlc]+|\W*)$", re.I)
_CAPTION_MAX_CHARS = 90
_CAPTION_LINES = 3


def _element_captions(elements) -> list[str]:
    """For each element, the recent short lines that name its section.

    The chunker separates a statement's heading from the table below it, so the
    table chunk would carry no company, year or statement name. This recovers
    the heading. SEC HTML has no heading tags, so a heading is recognised by
    being short; a long paragraph ends the current caption.
    """
    out: list[str] = []
    recent: list[str] = []
    for e in elements:
        text = (e.text or "").strip()
        if type(e).__name__ not in ("Table", "TableChunk") and text:
            if len(text) <= _CAPTION_MAX_CHARS and not _NOT_A_CAPTION.match(text):
                recent.append(text)
                del recent[:-_CAPTION_LINES]
            elif len(text) > _CAPTION_MAX_CHARS:
                recent.clear()
        out.append(" · ".join(recent))
    return out


def _tag_items(texts: list[str], start: str = "") -> list[str]:
    """The 10-K item ("1A", "7", ...) each text belongs to. A text containing a
    heading takes that heading; the last heading carries forward."""
    out, current = [], start
    for t in texts:
        found = _ITEM_RE.findall(t or "")
        if found:
            current = found[-1].upper()
            out.append(found[0].upper())
        else:
            out.append(current)
    return out


def _context_header(base_meta: dict, caption: str) -> str:
    """"3M 2022 · Consolidated Balance Sheet": the identity a chunk cannot supply itself."""
    head = " ".join(p for p in (str(base_meta.get(k) or "").strip()
                                for k in ("company", "year")) if p)
    if caption:
        head = f"{head} · {caption}" if head else caption
    return head[:200]


def _table_to_markdown(table_html: str) -> str:
    """An HTML table as a markdown table. Empty cells (SEC tables are padded
    with spacer cells and lone "$" columns) are dropped."""
    rows: list[list[str]] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.S | re.I):
        cells = [_html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S | re.I)]
        cells = [c for c in cells if c]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


@dataclass
class IngestionStats:
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    total_chunks: int = 0
    total_seconds: float = 0.0
    failures: list = field(default_factory=list)


class CorpusIngester:
    """Parse, chunk, embed and store SEC HTML filings."""

    def __init__(self, collection_name: str, embedding_model: str = DEFAULT_EMBED_MODEL,
                 market: str = "us"):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.market = market
        self._store = None

    # --- chunking -----------------------------------------------------------

    def _base_meta(self, record: dict, file_path: Path) -> dict:
        """Metadata every chunk of one filing shares."""
        return {
            "market": self.market,
            "source_url": record.get("source_url", ""),
            "local_path": str(file_path),
            "company": record.get("company", record.get("ticker", "")),
            "ticker": record.get("ticker", ""),
            "year": str(record.get("year", "")),
            "sector": record.get("sector", ""),
            "filing_type": record.get("filing_type", "annual_report"),
        }

    def documents(self, file_path: Path, record: dict) -> list:
        """One filing as LangChain Documents (children carrying their parent)."""
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from unstructured.chunking.title import chunk_by_title
        from unstructured.partition.html import partition_html

        base_meta = self._base_meta(record, file_path)
        elements = partition_html(filename=str(file_path))
        chunks = chunk_by_title(
            elements, max_characters=PARENT_CHARS, new_after_n_chars=PARENT_NEW_AFTER,
            combine_text_under_n_chars=PARENT_COMBINE_UNDER, overlap=PARENT_OVERLAP)
        captions = _element_captions(elements)
        position = {getattr(e, "id", None): i for i, e in enumerate(elements)}

        parents: list[tuple[str, dict, bool]] = []
        for idx, ch in enumerate(chunks):
            is_table = type(ch).__name__ in ("Table", "TableChunk")
            table_html = getattr(getattr(ch, "metadata", None), "text_as_html", None)
            if is_table and table_html:
                content, element_type = _table_to_markdown(table_html), "table"
            else:
                content, element_type = ch.text or "", "text"
            content = content.strip()
            if not content:
                continue
            meta = {**base_meta, "element_type": element_type, "element_index": idx}
            if is_table and table_html:
                meta["text_as_html"] = table_html[:6000]
            if captions:
                # The caption of the chunk's first element: where its heading was cut away.
                orig = getattr(getattr(ch, "metadata", None), "orig_elements", None)
                i = position.get(getattr(orig[0], "id", None)) if orig else None
                head = _context_header(base_meta, captions[i] if i is not None else "")
                if head:
                    meta["context_header"] = head
            parents.append((content, meta, is_table))

        def headed(text: str, meta: dict) -> str:
            head = meta.get("context_header")
            return (f"{head}\n{text}" if head else text)[:EMBED_CHAR_CAP]

        child_split = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHARS, chunk_overlap=CHILD_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""])
        items = _tag_items([c for c, _, _ in parents])
        docs: list = []
        for pid, ((content, meta, is_table), item) in enumerate(zip(parents, items)):
            pmeta = {**meta, "item": item, "parent_id": pid,
                     "parent_text": headed(content, meta)}
            if is_table:
                children = [content[:EMBED_CHAR_CAP]]
            else:
                children = [c.strip() for c in child_split.split_text(content)
                            if c.strip()] or [content[:EMBED_CHAR_CAP]]
            # Every child gets the header: the later children of a long
            # statement are the ones holding the numbers.
            docs += [Document(page_content=headed(child, meta), metadata=dict(pmeta))
                     for child in children]
        return docs

    # --- storing ------------------------------------------------------------

    def _get_store(self):
        if self._store is None:
            from finagent.vectorstore import build_store

            self._store = build_store(self.collection_name, self.embedding_model, create=True)
        return self._store

    def ingest_file(self, file_path: Path, record: dict) -> int:
        """Chunk, embed and upsert one filing. Returns the number of chunks."""
        from finagent.llm import collect_provider_keys
        from finagent.vectorstore import GEMINI_EMBED_BATCH, chunk_point_id, get_embeddings

        docs = self.documents(file_path, record)
        if not docs:
            return 0
        store = self._get_store()
        ids = [chunk_point_id(d.metadata, d.page_content) for d in docs]
        # Embed the whole file into the disk cache before writing anything. If
        # the quota runs out mid-file this raises with nothing written, so a
        # filing is never left half-indexed (and then skipped on the next run).
        get_embeddings(self.embedding_model).embed_documents([d.page_content for d in docs])
        batch = GEMINI_EMBED_BATCH * max(1, len(collect_provider_keys("gemini")))
        store.add_documents(docs, ids=ids, batch_size=batch)
        return len(docs)

    def ingest_all(self, manifest_path: Union[str, Path],
                   skip_if_already_indexed: bool = True) -> IngestionStats:
        """Ingest every record in a manifest (a JSON list of filing records).
        Filings whose `source_url` is already indexed are skipped."""
        from finagent.vectorstore import distinct_values

        stats = IngestionStats()
        t0 = time.time()
        records = json.loads(Path(manifest_path).read_text())
        indexed: set = set()
        if skip_if_already_indexed:
            try:
                indexed = distinct_values(self.collection_name, "source_url", limit=200_000)
            except Exception:
                indexed = set()

        for rec in tqdm(records, desc="ingest"):
            path = Path(rec["local_path"])
            source_url = rec.get("source_url", str(path))
            if (rec.get("status", "ok") != "ok" or not path.exists()
                    or path.stat().st_size < MIN_VALID_FILE_BYTES or source_url in indexed):
                stats.files_skipped += 1
                continue
            try:
                n = self.ingest_file(path, rec)
            except EmbeddingQuotaExhausted:
                # The day's quota is gone: stop instead of failing every
                # remaining file. The embedding cache makes the next run resume.
                stats.total_seconds = time.time() - t0
                self._print_summary(stats)
                raise
            except Exception as e:
                stats.files_failed += 1
                stats.failures.append((str(path), f"{type(e).__name__}: {e}"))
                continue
            if n == 0:
                stats.files_failed += 1
                stats.failures.append((str(path), "no extractable text"))
                continue
            stats.files_processed += 1
            stats.total_chunks += n

        stats.total_seconds = time.time() - t0
        self._print_summary(stats)
        return stats

    @staticmethod
    def _print_summary(stats: IngestionStats) -> None:
        print(f"\nIngested {stats.files_processed} file(s), {stats.total_chunks} chunks, "
              f"{stats.files_skipped} skipped, {stats.files_failed} failed, "
              f"{stats.total_seconds:.0f}s")
        for path, err in stats.failures[:10]:
            print(f"  ! {path}: {err}")


def main():
    ap = argparse.ArgumentParser(description="Ingest SEC HTML filings into Qdrant.")
    ap.add_argument("--manifest", required=True, help="JSON list of filing records")
    ap.add_argument("--collection", required=True, help="Qdrant collection name")
    ap.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--market", default="us")
    args = ap.parse_args()
    CorpusIngester(args.collection, args.embedding_model, args.market).ingest_all(args.manifest)


if __name__ == "__main__":
    main()
