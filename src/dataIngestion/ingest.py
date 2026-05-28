"""
ingest.py

Unified ingestion pipeline for financial filings. Reads the manifest produced
by fetch_pdfs.py, parses each document (PDFs from India, HTML from SEC),
chunks the text, embeds the chunks, and stores everything in Chroma.

Handles both markets — PDFs (India) are parsed with pypdf, HTML (US SEC
filings) with unstructured.io.

Usage as a library
------------------
    from ingest import CorpusIngester

    ing = CorpusIngester(
        corpus_dir="data/us/pdfs",
        chroma_dir="data/chroma",
        collection_name="us_filings",
        market="us",
    )
    stats = ing.ingest_all(manifest_path="data/us/pdfs/sec_manifest.json")
    print(stats)

Usage as CLI
------------
    python ingest.py --manifest data/us/pdfs/sec_manifest.json \\
                     --corpus-dir data/us/pdfs \\
                     --chroma-dir data/chroma \\
                     --collection us_filings \\
                     --market us

    python ingest.py --manifest data/India/pdfs/manifest.json \\
                     --corpus-dir data/India/pdfs \\
                     --chroma-dir data/chroma \\
                     --collection india_filings \\
                     --market india

Design notes
------------
* The same Chroma store can hold multiple collections — one per market. The
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
  silently returned zero elements on several born-digital Indian annual-report
  PDFs that pypdf reads fine, so pypdf is the reliable choice here. HTML (SEC
  filings) still goes through unstructured's `partition()`, which gives Element
  objects with section metadata. Page numbers are preserved in Chroma so the
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
    """Parse, chunk, embed, and index a corpus of financial filings into Chroma.

    Works identically for Indian annual report PDFs and US SEC 10-K HTML files.
    The market parameter is stored as document metadata so the agent can
    filter by market at query time.
    """

    # Sane defaults for financial-document chunking.
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200

    # Files smaller than this are likely junk (download failures, empty stubs).
    MIN_VALID_FILE_BYTES = 10_000

    def __init__(
        self,
        corpus_dir: Union[str, Path] = "data/India/pdfs",
        chroma_dir: Union[str, Path] = "data/chroma",
        collection_name: str = "financial_filings",
        market: str = "india",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        embedding_provider: str = "huggingface",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        unstructured_strategy: str = "fast",
    ):
        """
        Args:
            corpus_dir: Where the source files live (output_dir from fetch_pdfs.py).
            chroma_dir: Persistent Chroma directory. Can be shared across collections.
            collection_name: One per market is recommended (e.g. "us_filings",
                "india_filings"). The agent picks which one to query.
            market: "india" or "us" — stored as document metadata.
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
        self.chroma_dir = Path(chroma_dir)
        self.collection_name = collection_name
        self.market = market
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unstructured_strategy = unstructured_strategy

        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        # Resources initialized lazily on first use.
        self._embeddings = None
        self._splitter = None
        self._vector_store = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def ingest_all(
        self,
        manifest_path: Optional[Union[str, Path]] = None,
        skip_if_already_indexed: bool = True,
    ) -> IngestionStats:
        """Ingest every successful record from the manifest into Chroma.

        Args:
            manifest_path: Path to the JSON manifest written by fetch_pdfs.py.
                If None, walks corpus_dir for .pdf and .htm files directly.
            skip_if_already_indexed: If True, files whose source_url is already
                in Chroma are skipped (idempotent re-runs).

        Returns:
            IngestionStats summarizing the run. Also written to
            {chroma_dir}/ingestion_stats_{market}.json.
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

        # Persist stats next to the Chroma store for reference.
        stats_path = self.chroma_dir / f"ingestion_stats_{self.market}.json"
        with open(stats_path, "w") as f:
            json.dump(stats.as_dict(), f, indent=2)

        self._print_summary(stats)
        return stats

    def ingest_file(self, file_path: Path, record: dict) -> int:
        """Parse one file, chunk it, embed the chunks, push to Chroma.

        Args:
            file_path: Path to a single PDF or HTML file.
            record: The manifest record for this file. We propagate selected
                fields (company, ticker, year, sector) as Chroma metadata
                so the retriever can show citations and the router can filter.

        Returns:
            Number of chunks added to Chroma.
        """
        # 1. Extract text grouped by page. PDFs go through pypdf (reliable on
        # born-digital annual reports where unstructured's fast strategy
        # silently returns zero elements); HTML goes through unstructured.
        # We keep page numbers so the retriever can cite "Reliance AR 2023,
        # page 102". Raises on unreadable/corrupt files.
        page_texts = self._extract_pages(file_path)
        if not page_texts:
            return 0

        # 3. Chunk each page's text.
        splitter = self._get_splitter()

        # 4. Build LangChain Documents with rich metadata.
        from langchain_core.documents import Document

        # Metadata fields that exist for both markets. Some are empty depending
        # on the source (e.g. ticker is only meaningful for SEC; company is
        # only meaningful for India).
        base_meta = {
            "market": self.market,
            "source_url": record.get("source_url", ""),
            "local_path": str(file_path),
            "company": record.get("company", record.get("ticker", "")),
            "ticker": record.get(
                "ticker", record.get("nse_symbol", "")
            ),
            "year": str(record.get("year", "")),
            "sector": record.get("sector", ""),
            "filing_type": record.get("filing_type", "annual_report"),
        }

        docs: list[Document] = []
        for page_num, text in page_texts.items():
            for chunk in splitter.split_text(text):
                meta = {**base_meta, "page": page_num}
                docs.append(Document(page_content=chunk, metadata=meta))

        if not docs:
            return 0

        # 5. Push to Chroma. add_documents handles embedding internally.
        store = self._get_vector_store()
        store.add_documents(docs)

        return len(docs)

    def reset_collection(self) -> None:
        """Delete the collection. Useful when re-ingesting from scratch."""
        store = self._get_vector_store()
        store.delete_collection()
        # Re-create empty by clearing the cached handle.
        self._vector_store = None
        print(f"Collection '{self.collection_name}' reset")

    def query(self, question: str, k: int = 5) -> list:
        """Convenience method for quick sanity-checking after ingestion.

        Returns top-k chunks with metadata. Not used by the agent — the
        agent will use Chroma directly via langchain_chroma — this is just
        for a CLI smoke test.
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

    def _get_splitter(self):
        if self._splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                # Separator priority: paragraphs first, then sentences, then words.
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        return self._splitter

    def _get_embeddings(self):
        if self._embeddings is None:
            if self.embedding_provider == "openai":
                from langchain_openai import OpenAIEmbeddings
                self._embeddings = OpenAIEmbeddings(model=self.embedding_model)
            else:
                # HuggingFace sentence-transformers, runs locally.
                from langchain_huggingface import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    # Normalize for cosine similarity.
                    encode_kwargs={"normalize_embeddings": True},
                )
        return self._embeddings

    def _get_vector_store(self):
        if self._vector_store is None:
            from langchain_chroma import Chroma
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._get_embeddings(),
                persist_directory=str(self.chroma_dir),
            )
        return self._vector_store

    def _existing_source_urls(self) -> set:
        """Return the set of source_urls already in Chroma, for skip-if-indexed."""
        try:
            store = self._get_vector_store()
            # Chroma's .get() with no IDs returns all entries. For very large
            # collections this gets slow — for ~40 documents it's instant.
            existing = store.get(include=["metadatas"])
            return {
                m.get("source_url", "")
                for m in (existing.get("metadatas") or [])
                if m
            }
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
        description="Ingest a corpus of financial filings into Chroma.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest JSON written by fetch_pdfs.py. Omit to walk --corpus-dir.",
    )
    parser.add_argument(
        "--corpus-dir",
        default="data/India/pdfs",
        help="Source files directory (default: data/India/pdfs)",
    )
    parser.add_argument(
        "--chroma-dir",
        default="data/chroma",
        help="Where to persist Chroma (default: data/chroma)",
    )
    parser.add_argument(
        "--collection",
        default="financial_filings",
        help="Chroma collection name (default: financial_filings). "
             "Use separate names for US and India.",
    )
    parser.add_argument(
        "--market",
        choices=["india", "us"],
        default="india",
        help="Stored as document metadata (default: india)",
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
        chroma_dir=args.chroma_dir,
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