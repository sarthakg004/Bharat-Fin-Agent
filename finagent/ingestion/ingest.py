"""Ingestion pipeline for financial filings. Reads the manifest `fetchPDFs.py`
writes, parses each document, chunks it, embeds the chunks and stores them in
Qdrant.

Both formats are handled: SEC HTML through unstructured's `partition()`, which
returns Element objects carrying section metadata, and PDFs per-page with pypdf.
pypdf rather than unstructured for PDFs because unstructured's "fast" strategy
silently returned zero elements on several born-digital annual reports that pypdf
reads fine. Page numbers are preserved in the payload so citations can point at
them.

The embedder is `vectorstore.DEFAULT_EMBED_MODEL`, the single source of truth — a
collection is sized to whatever model writes it, so two embedders must never be
mixed in one collection.

One cluster holds several collections, one per corpus; the agent queries a
specific one by name.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from finagent.vectorstore import DEFAULT_EMBED_MODEL, EmbeddingQuotaExhausted

from tqdm import tqdm

# Lazy imports for heavy ML deps happen inside methods so help-printing /
# argument-validation stays fast and doesn't require everything installed.

# 10-K section heading at line start: "Item 1A. Risk Factors", "ITEM 7 —".
# ponytail: line-start heuristic — a mid-sentence "see Item 8" on its own line
# would mislabel; upgrade to a TOC-anchored parser if tagging noise shows up.
import re as _re

_ITEM_RE = _re.compile(r"(?im)^\s*item\s+(\d{1,2}[ab]?)\s*[.:—–-]")

# Hard cap on text that is EMBEDDED, in characters: the BGE encoders take 512
# tokens (~2048 chars) and silently drop the rest, so anything longer would be
# indexed on content it cannot see. Applies to children, and to tables (kept
# whole, so the table is its own child) — never to parents, which are only
# reached through their children and are read by the reranker and the LLM.
EMBED_CHAR_CAP = 1900

# …and the same cap for an embedder with a bigger window. Gemini's embedding
# models take 2048 tokens against BGE's 512, so a financial statement that BGE
# could only see the top of is embedded whole.
#
# This is the cap that actually bites. MEASURED on the 44,542-chunk eval index:
# 671 chunks (1.5% of all chunks, but 6.8% of TABLES) sit at 1900 characters,
# i.e. they were cut. Children are ~600 chars and never reach it — it is only
# ever tables, which are indexed whole precisely so their rows stay together,
# and which are the chunks numeric questions depend on.
GEMINI_EMBED_CHAR_CAP = 7000

# Hard cap on `parent_text`, stored on every child. The reranker scores the
# parent at 1024 tokens (~4000 chars); beyond that the tail is invisible to
# ranking and just costs payload storage on each of the parent's children.
PARENT_TEXT_CAP = 4000


def embed_char_cap(embedding_model: str) -> int:
    """Characters an `embedding_model` can actually see in one chunk."""
    from finagent.vectorstore import GEMINI_EMBED_PREFIX

    return (GEMINI_EMBED_CHAR_CAP
            if embedding_model.startswith(GEMINI_EMBED_PREFIX)
            else EMBED_CHAR_CAP)


# Lines that look like a heading but name nothing: page numbers, running
# headers, the TOC link every SEC page carries.
_NOT_A_CAPTION = _re.compile(
    r"^(table of contents|page\s*\d+|\d{1,4}|[ivxlc]+|\W*)$", _re.I)
# A caption line is short. Financial-statement headings ("Consolidated Balance
# Sheet", "3M Company and Subsidiaries") sit well under this; body prose does not.
_CAPTION_MAX_CHARS = 90
_CAPTION_LINES = 3


def _element_captions(elements) -> list[str]:
    """Running caption for each element index: the recent short standalone lines
    that name the section a chunk belongs to.

    Why this is needed: `chunk_by_title` splits a financial statement's HEADING
    away from the table holding its numbers, so the chunk that actually carries
    "Total assets" reads `(Dollars in millions) | 2022 | 2021 | Assets | ...` —
    no company, no year, no statement name. Every company's balance sheet then
    embeds to nearly the same vector and "3M total assets 2022" has nothing to
    match on. Measured on FinanceBench: 18 of 28 questions whose evidence never
    reached the candidate pool were numeric, i.e. table-bearing.

    SEC HTML has no `Title` elements — unstructured categorises every heading as
    plain `Text` — so a heading is recognised by shape, not by category.
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
                # Real prose: the previous heading no longer describes what
                # follows, so stop carrying it forward into unrelated tables.
                recent.clear()
        out.append(" · ".join(recent))
    return out


def _tag_items(texts: list[str], start: str = "") -> list[str]:
    """For an in-order list of chunk/page texts, return the 10-K item each one
    belongs to ("1A", "7", …; "" = before any heading). A text containing a
    heading is tagged with that (last) heading; the state carries forward."""
    out, current = [], start
    for t in texts:
        found = _ITEM_RE.findall(t or "")
        if found:
            current = found[-1].upper()
            out.append(found[0].upper())
        else:
            out.append(current)
    return out


@dataclass
class IngestionStats:
    """Summary of an ingestion run."""
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    total_chunks: int = 0
    total_seconds: float = 0.0
    failures: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "files_failed": self.files_failed,
            "total_chunks": self.total_chunks,
            "total_seconds": round(self.total_seconds, 1),
            "failures": self.failures,
        }


class CorpusIngester:
    """Parse, chunk, embed, and index a corpus of financial filings into Qdrant.

    Works identically for annual-report PDFs and US SEC 10-K HTML files.
    The market parameter is stored as document metadata so the agent can
    filter by market at query time.
    """

    # Sane defaults for financial-document chunking.
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200

    # Parent-document retrieval: embed SMALL children (precise lexical/dense
    # match, fit the bge-small ~2048-char window), return the LARGER parent for
    # context. The chunk ablation showed a single 1000-char chunk captures only
    # ~47% of gold evidence vs ~74% at 1500 — but 1500 dilutes match precision,
    # so we match on children and hand the parent to synthesis.
    PARENT_CHUNK_SIZE = 2500
    PARENT_CHUNK_OVERLAP = 300
    CHILD_CHUNK_SIZE = 600
    CHILD_CHUNK_OVERLAP = 100

    # Files smaller than this are likely junk (download failures, empty stubs).
    MIN_VALID_FILE_BYTES = 10_000

    def __init__(
        self,
        corpus_dir: Union[str, Path] = "data/us",
        state_dir: Union[str, Path] = "data",
        collection_name: str = "financial_filings",
        market: str = "us",
        embedding_model: str = DEFAULT_EMBED_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        unstructured_strategy: str = "fast",
        html_max_chars: int = PARENT_CHUNK_SIZE,
        html_new_after: int = 2000,
        html_combine_under: int = 400,
        html_overlap: int = PARENT_CHUNK_OVERLAP,
        parent_doc: bool = True,
        context_headers: bool = True,
        table_format: str = "md",
    ):
        """Build an ingester over `corpus_dir`, writing into `collection_name`.

        `market` is stored as document metadata. `unstructured_strategy` is
        "fast" (no OCR) or "hi_res" (slow, better at tables); "fast" is correct
        for the main text corpus. Ingestion stats are written to `state_dir`.
        """
        self.corpus_dir = Path(corpus_dir)
        self.state_dir = Path(state_dir)
        self.collection_name = collection_name
        self.market = market
        self.embedding_model = embedding_model
        # What this embedder can see in one chunk, and the matching floor on
        # `parent_text`: the parent must never be shorter than the child we
        # embedded from it, or the reranker and the LLM read LESS of a table
        # than the index matched on.
        self.embed_char_cap = embed_char_cap(embedding_model)
        self.parent_text_cap = max(PARENT_TEXT_CAP, self.embed_char_cap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unstructured_strategy = unstructured_strategy
        # HTML (US SEC filings) section-aware chunking. Sizes are in characters.
        # A section chunk is the PARENT under parent-document retrieval, so it is
        # sized from PARENT_CHUNK_SIZE — the two paths drifted apart once already
        # (HTML capped at 1500 while the PDF path moved to 2500), which quietly
        # kept the served corpus on the losing geometry. Only the text actually
        # embedded has to fit the embedder's window; parents do not.
        self.html_max_chars = html_max_chars
        self.html_new_after = html_new_after
        self.html_combine_under = html_combine_under
        self.html_overlap = html_overlap
        # Parent-document retrieval for the INDEXED path (ephemeral fetch keeps
        # flat parent-sized chunks — it ranks in memory with no parent swap).
        self.parent_doc = parent_doc
        # Prefix every chunk with "<company> <year> <form> · <section caption>".
        # `chunk_by_title` cuts a financial statement's heading away from its
        # numbers, leaving the answer-bearing table with no company, year or
        # statement name to match a query against. Kept as a flag so the
        # measurement in RETRIEVAL_EXPERIMENTS.md §12 stays reproducible.
        self.context_headers = context_headers
        # How an HTML <table> is rendered into the text that gets embedded:
        # "md" (GitHub-flavoured markdown) or "pipe" (bare `a | b | c` rows, the
        # pre-existing behaviour). Kept as a flag so the A/B stays reproducible,
        # exactly like `context_headers`.
        self.table_format = table_format

        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Resources initialized lazily on first use.
        self._splitter = None
        self._parent_splitter = None
        self._child_splitter = None
        self._vector_store = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def ingest_all(
        self,
        manifest_path: Optional[Union[str, Path]] = None,
        skip_if_already_indexed: bool = True,
    ) -> IngestionStats:
        """Ingest every successful record from the manifest into Qdrant.

        Args:
            manifest_path: Path to the JSON manifest written by fetch_pdfs.py.
                If None, walks corpus_dir for .pdf and .htm files directly.
            skip_if_already_indexed: If True, files whose source_url is already
                already indexed are skipped (idempotent re-runs).

        Returns:
            IngestionStats summarizing the run. Also written to
            {state_dir}/ingestion_stats_{market}.json.
        """
        stats = IngestionStats()
        t0 = time.time()

        records = self._load_records(manifest_path)
        print(f"Found {len(records)} records to consider")

        # Pre-check which sources are already indexed (for idempotent re-runs).
        already_indexed = set()
        if skip_if_already_indexed:
            already_indexed = self._existing_source_urls()
            if already_indexed:
                print(f"Found {len(already_indexed)} already-indexed sources")

        for rec in tqdm(records, desc=f"ingest [{self.market}]"):
            local_path = Path(rec["local_path"])
            source_url = rec.get("source_url", str(local_path))

            # Skip anything that never downloaded cleanly (e.g. 403/404 in the
            # fetch step leaves a failed record with no file on disk).
            if rec.get("status", "ok") != "ok":
                stats.files_skipped += 1
                continue

            if not local_path.exists():
                stats.files_skipped += 1
                continue

            if local_path.stat().st_size < self.MIN_VALID_FILE_BYTES:
                stats.files_skipped += 1
                continue

            if source_url in already_indexed:
                stats.files_skipped += 1
                continue

            try:
                n_chunks = self.ingest_file(local_path, rec)
            except EmbeddingQuotaExhausted:
                # Not a per-file failure: the budget for the whole run is gone,
                # and continuing would record 65 identical "failures" while each
                # retry spends more of a quota that is already empty. Stop, and
                # let the caller see how far the run actually got — the
                # embedding cache makes the next attempt resume, not restart.
                stats.total_seconds = time.time() - t0
                self._print_summary(stats)
                raise
            except Exception as e:
                # Corrupt/unreadable file (e.g. broken PDF xref table).
                stats.files_failed += 1
                stats.failures.append((str(local_path), f"{type(e).__name__}: {e}"))
                print(f"  ! failed {local_path.name}: {e}")
                continue

            if n_chunks == 0:
                stats.files_failed += 1
                stats.failures.append((str(local_path), "no extractable text"))
                continue

            stats.files_processed += 1
            stats.total_chunks += n_chunks

        stats.total_seconds = time.time() - t0

        # Persist stats for reference.
        stats_path = self.state_dir / f"ingestion_stats_{self.market}.json"
        with open(stats_path, "w") as f:
            json.dump(stats.as_dict(), f, indent=2)

        self._print_summary(stats)
        return stats

    def ingest_file(self, file_path: Path, record: dict) -> int:
        """Parse one file, chunk it, embed the chunks, upsert them.

        Args:
            file_path: Path to a single PDF or HTML file.
            record: The manifest record for this file. We propagate selected
                fields (company, ticker, year, sector) as point metadata
                so the retriever can show citations and the router can filter.

        Returns:
            Number of chunks upserted.
        """
        docs = self._documents_for(file_path, self._base_meta(record, file_path),
                                   parent_doc=self.parent_doc)
        if not docs:
            return 0

        # Upsert with DETERMINISTIC ids: the same chunk of the same filing
        # always lands on the same point, so re-ingesting (or two writers racing
        # on the same filing) overwrites rather than duplicating.
        from finagent.vectorstore import (GEMINI_EMBED_PREFIX, chunk_point_id,
                                          get_embeddings)

        store = self._get_vector_store()
        ids = [chunk_point_id(d.metadata, d.page_content) for d in docs]

        # Embed the WHOLE file before a single point is written. `add_documents`
        # embeds and upserts one batch at a time, so a metered embedder hitting
        # its daily wall mid-file left the filing HALF-indexed — and because its
        # source_url was then present, the next day's resumed build skipped it
        # and truncated that filing permanently. Warming the disk cache first
        # makes the file all-or-nothing: either this raises before anything is
        # upserted, or every batch below is a cache hit.
        #
        # Gated on the cached API embedder, because a local encoder has no disk
        # cache and cannot run out of quota — there it would just embed twice.
        if self.embedding_model.startswith(GEMINI_EMBED_PREFIX):
            get_embeddings(self.embedding_model).embed_documents(
                [d.page_content for d in docs])

        store.add_documents(docs, ids=ids, batch_size=self._upsert_batch())

        return len(docs)

    def _upsert_batch(self) -> int:
        """Texts handed to the embedder per call.

        LangChain's default is 64, which is right for a local GPU encoder and
        badly wrong for the Gemini API: that embedder fans a call out across the
        key pool 32 texts at a time, so 64 only ever keeps TWO of the seven keys
        busy. Handing it a full pool's worth per call saturates all seven and
        took the measured corpus build from ~59 minutes to ~15.
        """
        from finagent.llm import collect_provider_keys
        from finagent.vectorstore import GEMINI_EMBED_BATCH, GEMINI_EMBED_PREFIX

        if not self.embedding_model.startswith(GEMINI_EMBED_PREFIX):
            return 64
        return GEMINI_EMBED_BATCH * max(1, len(collect_provider_keys("gemini")))

    def _base_meta(self, record: dict, file_path: Path) -> dict:
        """Metadata fields shared by all sources. Some are empty depending on
        the source (e.g. ticker is only meaningful for SEC filings)."""
        return {
            "market": self.market,
            "source_url": record.get("source_url", ""),
            "local_path": str(file_path),
            "company": record.get("company", record.get("ticker", "")),
            "ticker": record.get("ticker", record.get("nse_symbol", "")),
            "year": str(record.get("year", "")),
            "sector": record.get("sector", ""),
            "filing_type": record.get("filing_type", "annual_report"),
        }

    def _documents_for(self, file_path: Path, base_meta: dict,
                       parent_doc: bool = False) -> list:
        """Parse + chunk one file into LangChain Documents (no embedding / no
        upsert). Shared by `ingest_file` and by the ephemeral dynamic-fetch
        path that ranks a freshly-fetched filing in memory without indexing it.

        Branch by format. HTML (US SEC filings) uses structure-aware chunking
        that gives each table its own chunk; PDFs use per-page pypdf text +
        char chunking.

        `parent_doc` (indexed path): split each parent-sized unit into SMALL
        children and stash the parent on each child (`parent_id`/`parent_text`)
        for parent-document retrieval. The ephemeral path passes False and keeps
        flat parent-sized chunks (it ranks in memory with no parent swap).
        """
        from langchain_core.documents import Document

        suffix = file_path.suffix.lower()
        if suffix in (".htm", ".html"):
            return self._html_documents(file_path, base_meta, parent_doc)

        page_texts = self._extract_pages(file_path)
        if not page_texts:
            return []

        if not parent_doc:
            splitter = self._get_splitter()
            pairs: list[tuple[int, str]] = []
            for page_num in sorted(page_texts):
                for chunk in splitter.split_text(page_texts[page_num]):
                    pairs.append((page_num, chunk))
            items = _tag_items([c for _, c in pairs])
            return [
                Document(page_content=chunk,
                         metadata={**base_meta, "page": page_num, "item": item})
                for (page_num, chunk), item in zip(pairs, items)
            ]

        # Parent-document: page text -> parent chunks -> small children.
        parent_split = self._get_parent_splitter()
        parents: list[tuple[str, dict]] = []
        for page_num in sorted(page_texts):
            for parent in parent_split.split_text(page_texts[page_num]):
                parents.append((parent, {"page": page_num}))
        return self._children_from_parents(parents, base_meta)

    def _children_from_parents(self, parents: list[tuple[str, dict]],
                               base_meta: dict) -> list:
        """Turn an in-order list of (parent_text, extra_meta) into child
        Documents: each parent is item-tagged (heading state carried across
        parents), split into small children, and every child carries its
        `parent_id` + `parent_text` so retrieval can return the parent."""
        from langchain_core.documents import Document

        items = _tag_items([p for p, _ in parents])
        child_split = self._get_child_splitter()
        docs: list = []
        for pid, ((ptext, extra), item) in enumerate(zip(parents, items)):
            stored_parent = ptext[:self.parent_text_cap]
            children = [c.strip() for c in child_split.split_text(ptext) if c.strip()]
            for child in children or [ptext.strip()]:
                if not child:
                    continue
                docs.append(Document(
                    page_content=child[:self.embed_char_cap],
                    metadata={**base_meta, **extra, "item": item,
                              "parent_id": pid, "parent_text": stored_parent},
                ))
        return docs

    def documents_from_record(self, record: dict) -> list:
        """Public: parse + chunk a single manifest record into Documents,
        without touching the database. Used by the ephemeral fetch path (flat chunks,
        no parent-document metadata — it ranks in memory)."""
        file_path = Path(record["local_path"])
        if not file_path.exists():
            return []
        return self._documents_for(file_path, self._base_meta(record, file_path),
                                   parent_doc=False)

    # ------------------------------------------------------------------ #
    # HTML (US SEC filings) — structure-aware extraction
    # ------------------------------------------------------------------ #

    def _html_documents(self, file_path: Path, base_meta: dict,
                        parent_doc: bool = False) -> list:
        """Build section-aware Documents for one HTML filing.

        Uses unstructured's ``chunk_by_title`` so chunks respect section
        boundaries and each table gets its OWN chunk rather than being sliced
        mid-row by a fixed-character splitter. A table is kept whole unless it
        exceeds ``max_characters``, in which case it is divided into
        ``TableChunk``s — measured on a 3M 10-K, 4 of 45 tables hit that, so the
        primary statements survive intact but the claim is not absolute.

        Note the cost of that isolation, which §12 of RETRIEVAL_EXPERIMENTS.md
        measures: a table never shares a chunk with the heading that names it,
        so the chunk holding "Total assets" carries no company, year or
        statement title. ``context_headers`` puts that identity back.

        For table chunks we render ``text_as_html`` into pipe-delimited rows so
        the row/column structure survives into the embedded text, and stash the
        raw HTML in metadata for downstream use.

        Page numbers are meaningless for a single HTML document, so each chunk
        carries ``element_index`` + ``element_type`` instead of a fake page.

        `parent_doc`: each section chunk becomes a PARENT; text sections are
        split into small children (tables are kept whole so rows survive) and
        every child carries `parent_id`/`parent_text` for parent-document
        retrieval.
        """
        from langchain_core.documents import Document
        from unstructured.chunking.title import chunk_by_title
        from unstructured.partition.html import partition_html

        elements = partition_html(filename=str(file_path))
        chunks = chunk_by_title(
            elements,
            max_characters=self.html_max_chars,
            new_after_n_chars=self.html_new_after,
            combine_text_under_n_chars=self.html_combine_under,
            overlap=self.html_overlap,
        )
        captions = _element_captions(elements) if self.context_headers else []
        at = {getattr(e, "id", None): i for i, e in enumerate(elements)}

        prepared: list[tuple[str, dict, bool]] = []
        for idx, ch in enumerate(chunks):
            is_table = type(ch).__name__ in ("Table", "TableChunk")
            table_html = getattr(getattr(ch, "metadata", None), "text_as_html", None)
            if is_table and table_html:
                content = self._html_table_to_text(table_html)
                element_type = "table"
            else:
                content = ch.text or ""
                element_type = "text"

            content = content.strip()
            if not content:
                continue
            meta = {**base_meta, "element_type": element_type, "element_index": idx}
            if is_table and table_html:
                meta["text_as_html"] = table_html[:6000]
            if captions:
                # The caption of the chunk's FIRST original element: that is
                # where the heading that names this chunk was cut away.
                orig = getattr(getattr(ch, "metadata", None), "orig_elements", None)
                i = at.get(getattr(orig[0], "id", None)) if orig else None
                head = self._context_header(base_meta, captions[i] if i is not None else "")
                if head:
                    meta["context_header"] = head
            prepared.append((content, meta, is_table))

        items = _tag_items([c for c, _, _ in prepared])

        def headed(text: str, meta: dict, cap: int) -> str:
            """Prefix the chunk's context header, then cap. EVERY child needs it
            — a header on only the first child would leave the rest anonymous,
            and it is the later children of a split statement that hold the
            numbers."""
            head = meta.get("context_header")
            return (f"{head}\n{text}" if head else text)[:cap]

        if not parent_doc:
            # Flat path: the chunk itself is what gets embedded, so it must stay
            # inside the embedder's 512-token (~2048 char) window.
            return [Document(page_content=headed(c, m, self.embed_char_cap),
                             metadata={**m, "item": item})
                    for (c, m, _), item in zip(prepared, items)]

        # Parent-document: the section chunk is the parent; split text sections
        # into children, keep tables intact (one child == the parent).
        child_split = self._get_child_splitter()
        docs: list = []
        for pid, ((content, meta, is_table), item) in enumerate(zip(prepared, items)):
            # The header goes on the parent too: the reranker and the LLM both
            # read `parent_text`, and both were seeing an untitled table.
            pmeta = {**meta, "item": item, "parent_id": pid,
                     "parent_text": headed(content, meta, self.parent_text_cap)}
            if is_table:
                # A table is never split, so the child IS the embedded text and
                # has to fit the embed window — unlike a text parent, which is
                # only ever reached through its (small) children.
                children = [content[:self.embed_char_cap]]
            else:
                children = [c.strip() for c in child_split.split_text(content)
                            if c.strip()] or [content[:self.embed_char_cap]]
            for child in children:
                docs.append(Document(page_content=headed(child, meta, self.embed_char_cap),
                                     metadata=dict(pmeta)))
        return docs

    @staticmethod
    def _context_header(base_meta: dict, caption: str) -> str:
        """`3M 2022 10-K · Consolidated Balance Sheet · At December 31`.

        Identity the chunk cannot supply itself. Company/year come from the
        manifest and are always present; the caption is best-effort.
        """
        parts = [str(base_meta.get(k) or "").strip()
                 for k in ("company", "year", "doc_type")]
        head = " ".join(p for p in parts if p)
        if caption:
            head = f"{head} · {caption}" if head else caption
        return head[:200]

    def _html_table_to_text(self, table_html: str) -> str:
        """Render an HTML ``<table>`` as a GitHub-flavoured **markdown** table.

        Keeps cell adjacency (and therefore row/column relationships) that a flat
        ``element.text`` rendering destroys — far better for both embedding and
        for an LLM reading the chunk.

        Markdown rather than raw HTML, and the reason is the character cap.
        Everything embedded has to fit `EMBED_CHAR_CAP`, and the same table as
        HTML is 3-5x the characters — `<td style="...">46,455</td>` against
        `46,455`. Re-rendering as HTML would therefore fit far FEWER rows of a
        financial statement under the cap, which makes the truncation this
        pipeline already suffers strictly worse. Markdown costs 2 characters a
        row over the bare pipe rows it replaces and buys the header delimiter,
        which is what marks row 1 as column headings (the fiscal years) rather
        than as data.

        Empty cells are dropped before joining: SEC tables are padded with
        spacer cells and lone `$` columns, and keeping them produced rows like
        ``| Total assets | $ | 46,455 | | $ | 47,072 |``.
        """
        import html as _html
        import re

        rows: list[list[str]] = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.S | re.I):
            cells = [
                _html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S | re.I)
            ]
            cells = [c for c in cells if c]
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        if self.table_format == "pipe":
            return "\n".join(" | ".join(r) for r in rows)
        header, body = rows[0], rows[1:]
        out = ["| " + " | ".join(header) + " |",
               "|" + "|".join("---" for _ in header) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in body]
        return "\n".join(out)

    def reset_collection(self) -> None:
        """Delete the collection. Useful when re-ingesting from scratch."""
        from finagent.vectorstore import delete_collection

        delete_collection(self.collection_name)
        self._vector_store = None          # recreated on next use
        print(f"Collection '{self.collection_name}' reset")

    def query(self, question: str, k: int = 5) -> list:
        """Convenience method for quick sanity-checking after ingestion.

        Returns top-k chunks with metadata. Not used by the agent — the
        agent uses the hybrid retriever — this is just a CLI smoke test.
        """
        store = self._get_vector_store()
        return store.similarity_search(question, k=k)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_records(self, manifest_path) -> list[dict]:
        """Load records from a manifest, or build synthetic ones by walking
        corpus_dir when no manifest is supplied."""
        if manifest_path is not None:
            with open(manifest_path) as f:
                return json.load(f)

        # Fallback: walk the corpus directory and synthesize minimal records.
        records = []
        for path in self.corpus_dir.rglob("*"):
            if path.suffix.lower() not in {".pdf", ".htm", ".html"}:
                continue
            if not path.is_file():
                continue
            # Infer ticker/company from parent folder name.
            company = path.parent.name
            # Try to pull a year out of the filename.
            year = ""
            for token in path.stem.split("_"):
                if token.isdigit() and len(token) == 4:
                    year = token
                    break
            records.append({
                "local_path": str(path),
                "source_url": str(path),
                "company": company,
                "ticker": company,
                "year": year,
            })
        return records

    def _extract_pages(self, file_path: Path) -> dict:
        """Extract text grouped by page number: {page_number: text}.

        PDFs use pypdf; HTML uses unstructured. Raises on unreadable/corrupt
        files so the caller records them as failures instead of silently
        producing zero chunks.
        """
        if file_path.suffix.lower() == ".pdf":
            return self._pdf_pages(file_path)
        return self._html_pages(file_path)

    @staticmethod
    def _pdf_pages(file_path: Path) -> dict:
        """Per-page text via pypdf. Reliable on born-digital annual reports
        where unstructured's fast strategy returns nothing."""
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))  # raises on corrupt PDFs
        pages: dict = {}
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages[i] = text
        return pages

    def _html_pages(self, file_path: Path) -> dict:
        """Group unstructured Element objects by page number.

        For HTML files that have no explicit pages, everything ends up under
        page 1.
        """
        from unstructured.partition.auto import partition

        elements = partition(
            filename=str(file_path),
            strategy=self.unstructured_strategy,
        )
        pages: dict = {}
        for el in elements:
            text = (getattr(el, "text", "") or "").strip()
            if not text:
                continue
            page = 1
            md = getattr(el, "metadata", None)
            if md is not None:
                page = getattr(md, "page_number", None) or 1
            pages.setdefault(int(page), []).append(text)
        return {p: "\n\n".join(parts) for p, parts in pages.items()}

    _SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # paragraphs → sentences → words

    def _get_splitter(self):
        if self._splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self._SEPARATORS,
            )
        return self._splitter

    def _get_parent_splitter(self):
        if self._parent_splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self._parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.PARENT_CHUNK_SIZE,
                chunk_overlap=self.PARENT_CHUNK_OVERLAP,
                separators=self._SEPARATORS,
            )
        return self._parent_splitter

    def _get_child_splitter(self):
        if self._child_splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self._child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.CHILD_CHUNK_SIZE,
                chunk_overlap=self.CHILD_CHUNK_OVERLAP,
                separators=self._SEPARATORS,
            )
        return self._child_splitter


    def _get_vector_store(self):
        if self._vector_store is None:
            from finagent.vectorstore import build_store

            self._vector_store = build_store(
                self.collection_name, self.embedding_model, create=True)
        return self._vector_store

    def _existing_source_urls(self) -> set:
        """source_urls already indexed, for skip-if-indexed."""
        from finagent.vectorstore import distinct_values

        try:
            return distinct_values(self.collection_name, "source_url", limit=200_000)
        except Exception:
            return set()

    @staticmethod
    def _print_summary(stats: IngestionStats) -> None:
        print()
        print("=" * 50)
        print("Ingestion complete")
        print("=" * 50)
        print(f"  Files processed: {stats.files_processed}")
        print(f"  Files skipped:   {stats.files_skipped}")
        print(f"  Files failed:    {stats.files_failed}")
        print(f"  Total chunks:    {stats.total_chunks}")
        print(f"  Duration:        {stats.total_seconds:.1f}s")
        if stats.failures:
            print(f"\n  Failures ({len(stats.failures)}):")
            for path, err in stats.failures[:10]:
                print(f"    - {path}: {err}")


# ---------------------------------------------------------------------- #
# CLI entry point
# ---------------------------------------------------------------------- #

def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest a corpus of financial filings into Qdrant.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest JSON written by fetch_pdfs.py. Omit to walk --corpus-dir.",
    )
    parser.add_argument(
        "--corpus-dir",
        default="data/us",
        help="Source files directory (default: data/us)",
    )
    parser.add_argument(
        "--collection",
        default="financial_filings",
        help="Qdrant collection name (default: financial_filings).",
    )
    parser.add_argument(
        "--market",
        choices=["us"],
        default="us",
        help="Stored as document metadata (default: us)",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBED_MODEL,
        help="sentence-transformers model name",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CorpusIngester.DEFAULT_CHUNK_SIZE,
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CorpusIngester.DEFAULT_CHUNK_OVERLAP,
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the collection before ingesting (start fresh)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="After ingestion, run a smoke-test similarity search with this string",
    )
    return parser


def main():
    args = _build_cli().parse_args()

    ing = CorpusIngester(
        corpus_dir=args.corpus_dir,
        collection_name=args.collection,
        market=args.market,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.reset:
        ing.reset_collection()

    ing.ingest_all(manifest_path=args.manifest)

    if args.query:
        print(f"\nSmoke-test query: {args.query!r}")
        results = ing.query(args.query, k=3)
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"  Source: {doc.metadata.get('company') or doc.metadata.get('ticker')} "
                  f"({doc.metadata.get('year')}), page {doc.metadata.get('page')}")
            print(f"  Snippet: {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()