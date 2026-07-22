"""
ingest.py

Unified ingestion pipeline for financial filings. Reads the manifest produced
by fetch_pdfs.py, parses each document (PDF or SEC HTML),
chunks the text, embeds the chunks, and stores everything in Qdrant.

Handles both formats — PDFs are parsed with pypdf, HTML (US SEC
filings) with unstructured.io.

Usage as a library
------------------
    from ingest import CorpusIngester

    ing = CorpusIngester(
        corpus_dir="data/us/pdfs",
        collection_name="us_filings",
        market="us",
    )
    stats = ing.ingest_all(manifest_path="data/us/pdfs/sec_manifest.json")
    print(stats)

Usage as CLI
------------
    python ingest.py --manifest data/us/pdfs/sec_manifest.json \\
                     --corpus-dir data/us/pdfs \\
                     --collection us_filings \\
                     --market us

    python ingest.py --manifest data/us/sec_manifest.json \\
                     --corpus-dir data/us \\
                     --collection us_filings \\
                     --market us

Design notes
------------
* The same cluster holds multiple collections — one per corpus. The
  agent can query a specific market by passing the right collection name.
* Embeddings model: BAAI/bge-small-en-v1.5 by default (runs on CPU in a few
  minutes for ~40 docs; bge-large is better quality but 4x slower).
  Swap to OpenAI text-embedding-3-small if you have an API key — that's
  cheaper than you'd think (~$0.50 for the whole 40-doc corpus) and faster.
* Chunk size 1000 with 200 overlap is a sane default for financial prose.
  Numeric-table-heavy sections benefit from larger chunks; we use the
  RecursiveCharacterTextSplitter so chunks respect paragraph and sentence
  boundaries when possible.
* PDFs are extracted per-page with pypdf. unstructured.io's "fast" strategy
  silently returned zero elements on several born-digital annual-report
  PDFs that pypdf reads fine, so pypdf is the reliable choice here. HTML (SEC
  filings) still goes through unstructured's `partition()`, which gives Element
  objects with section metadata. Page numbers are preserved in the payload so the
  retriever can show citations.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

from tqdm import tqdm

# Lazy imports for heavy ML deps happen inside methods so help-printing /
# argument-validation stays fast and doesn't require everything installed.

# 10-K section heading at line start: "Item 1A. Risk Factors", "ITEM 7 —".
# ponytail: line-start heuristic — a mid-sentence "see Item 8" on its own line
# would mislabel; upgrade to a TOC-anchored parser if tagging noise shows up.
import re as _re

_ITEM_RE = _re.compile(r"(?im)^\s*item\s+(\d{1,2}[ab]?)\s*[.:—–-]")


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
    PARENT_CHUNK_SIZE = 1500
    PARENT_CHUNK_OVERLAP = 200
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
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        embedding_provider: str = "huggingface",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        unstructured_strategy: str = "fast",
        html_max_chars: int = 1500,
        html_new_after: int = 1200,
        html_combine_under: int = 400,
        html_overlap: int = 200,
        parent_doc: bool = True,
    ):
        """
        Args:
            corpus_dir: Where the source files live (output_dir from fetch_pdfs.py).
            state_dir: Where the ingestion-stats JSON is written.
            collection_name: Qdrant collection to write into (e.g. "us_filings").
            market: stored as document metadata (default "us").
            embedding_model: Model name. For HuggingFace: any sentence-transformers
                model. For OpenAI: "text-embedding-3-small" or similar.
            embedding_provider: "huggingface" (free, local) or "openai" (faster, paid).
            chunk_size: Characters per chunk before splitting.
            chunk_overlap: Overlapping characters between adjacent chunks.
            unstructured_strategy: "fast" (default, no OCR) or "hi_res" (slow,
                better at tables — needed later in Week 4 for the table agent).
                For Week 1 baseline, "fast" is correct.
        """
        self.corpus_dir = Path(corpus_dir)
        self.state_dir = Path(state_dir)
        self.collection_name = collection_name
        self.market = market
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unstructured_strategy = unstructured_strategy
        # HTML (US SEC filings) section-aware chunking. Sizes are in characters;
        # html_max_chars stays well under the bge-small 512-token (~2048 char)
        # window so chunks are never silently truncated at embed time.
        self.html_max_chars = html_max_chars
        self.html_new_after = html_new_after
        self.html_combine_under = html_combine_under
        self.html_overlap = html_overlap
        # Parent-document retrieval for the INDEXED path (ephemeral fetch keeps
        # flat parent-sized chunks — it ranks in memory with no parent swap).
        self.parent_doc = parent_doc

        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Resources initialized lazily on first use.
        self._embeddings = None
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
        from finagent.vectorstore import chunk_point_id

        store = self._get_vector_store()
        ids = [chunk_point_id(d.metadata, d.page_content) for d in docs]
        store.add_documents(docs, ids=ids)

        return len(docs)

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
        that keeps tables intact; PDFs use per-page pypdf text + char chunking.

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
            stored_parent = ptext[:4000]
            children = [c.strip() for c in child_split.split_text(ptext) if c.strip()]
            for child in children or [ptext.strip()]:
                if not child:
                    continue
                docs.append(Document(
                    page_content=child[:1900],
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
        boundaries and keep tables intact (instead of a fixed-character splitter
        slicing through them). For table chunks we render ``text_as_html`` into
        pipe-delimited rows so the row/column structure survives into the
        embedded text, and stash the raw HTML in metadata for downstream use.

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
            # Stay within the embedder's 512-token (~2048 char) window so a long
            # rendered table is never silently truncated mid-content.
            content = content[:1900]

            meta = {**base_meta, "element_type": element_type, "element_index": idx}
            if is_table and table_html:
                meta["text_as_html"] = table_html[:6000]
            prepared.append((content, meta, is_table))

        items = _tag_items([c for c, _, _ in prepared])
        if not parent_doc:
            return [Document(page_content=c, metadata={**m, "item": item})
                    for (c, m, _), item in zip(prepared, items)]

        # Parent-document: the section chunk is the parent; split text sections
        # into children, keep tables intact (one child == the parent).
        child_split = self._get_child_splitter()
        docs: list = []
        for pid, ((content, meta, is_table), item) in enumerate(zip(prepared, items)):
            pmeta = {**meta, "item": item, "parent_id": pid,
                     "parent_text": content[:4000]}
            if is_table:
                children = [content]
            else:
                children = [c.strip() for c in child_split.split_text(content)
                            if c.strip()] or [content]
            for child in children:
                docs.append(Document(page_content=child[:1900], metadata=dict(pmeta)))
        return docs

    @staticmethod
    def _html_table_to_text(table_html: str) -> str:
        """Render an HTML ``<table>`` as pipe-delimited rows, one row per line.

        Keeps cell adjacency (and therefore row/column relationships) that a flat
        ``element.text`` rendering destroys — far better for both embedding and
        for an LLM reading the chunk.
        """
        import html as _html
        import re

        rows: list[str] = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.S | re.I):
            cells = [
                _html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S | re.I)
            ]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)

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

    def _get_embeddings(self):
        if self._embeddings is None:
            if self.embedding_provider == "openai":
                from langchain_openai import OpenAIEmbeddings
                self._embeddings = OpenAIEmbeddings(model=self.embedding_model)
            else:
                # Reuse the process-wide, lru_cached HuggingFace embedder so a
                # dynamic fetch (Phase 5) ingesting inside the running agent does
                # NOT load a second ~130MB copy of the BGE model. Same config
                # (normalized, device-aware) as the retrieval path.
                from finagent.vectorstore import get_embeddings
                self._embeddings = get_embeddings(self.embedding_model)
        return self._embeddings

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
        default="BAAI/bge-small-en-v1.5",
        help="HuggingFace ST model or OpenAI model name",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["huggingface", "openai"],
        default="huggingface",
        help="huggingface = free local; openai = paid + faster",
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
        embedding_provider=args.embedding_provider,
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