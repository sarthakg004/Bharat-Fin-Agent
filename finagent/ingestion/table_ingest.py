"""
table_ingest.py  ·  finagent/ingestion/table_ingest.py

Extract tables from financial filings (PDFs + 10-K HTML), persist each as a
parquet DataFrame, and write a master `data/tables/index.json` so the next
stage (`table_embed.py`) can build the `tables` Chroma collection.

Per table we save:
    - company, year, page_num, market
    - title (LLM-extracted from surrounding context; heuristic fallback)
    - HTML (the unstructured `text_as_html`)
    - parquet_path (the pandas DataFrame)
    - shape, columns, first row, last row (used by the embed step)

Usage as a library
------------------
    from finagent.ingestion.table_ingest import TableExtractor

    tx = TableExtractor(tables_dir="data/tables", market="india")
    tx.ingest_corpus("data/India/pdfs/manifest.json")

CLI
---
    python -m finagent.ingestion.table_ingest \\
        --manifest data/India/pdfs/manifest.json \\
        --tables-dir data/tables --market india

Heads-up
--------
This is a slow step. `unstructured` with `strategy="hi_res"` runs layout
detection + OCR. Plan on multiple minutes per ~100-page PDF on CPU. Use
`--limit N` for a smoke run before unleashing it on the full corpus.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from io import StringIO
from pathlib import Path
from typing import Optional, Union

from tqdm import tqdm


TITLE_PROMPT = """\
You are reading a financial filing. From the context immediately above a table
and the table's column headers, give the table a short, descriptive title
(maximum 12 words). Examples of good titles:

  - "Consolidated balance sheet (FY2023, in ₹ crore)"
  - "Segment revenue by geography"
  - "Five-year financial summary"

Context above the table:
\"\"\"{context}\"\"\"

Table column headers: {cols}

Return only the title."""


class TableExtractor:
    """Extract Table elements from filings into parquet + a JSON index."""

    DEFAULT_STRATEGY = "hi_res"
    MIN_VALID_FILE_BYTES = 10_000

    def __init__(
        self,
        tables_dir: Union[str, Path] = "data/tables",
        market: str = "india",
        provider: str = "groq",
        title_model: Optional[str] = None,
        use_llm_titles: bool = True,
        strategy: str = DEFAULT_STRATEGY,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            tables_dir: Output directory (parquet files + index.json).
            market: "india" or "us" — stored on every row.
            provider: LLM provider for title extraction (rotates if multi-key).
            title_model: Fast model. Default: per-provider planner-tier model.
            use_llm_titles: If False, skip the LLM call and use a heuristic
                title — much faster (and free). Recommended for first-pass runs.
            strategy: `unstructured` partition strategy. `"hi_res"` finds tables
                in scanned/born-digital PDFs but is slow. `"fast"` is rarely
                useful for table extraction.
            api_key: Override the env-collected key list.
        """
        self.tables_dir = Path(tables_dir)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.market = market
        self.provider = provider
        self.title_model = title_model
        self.use_llm_titles = use_llm_titles
        self.strategy = strategy
        self.api_key = api_key

        self.index_path = self.tables_dir / "index.json"
        self._llm = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def ingest_corpus(
        self,
        manifest_path: Union[str, Path],
        limit: Optional[int] = None,
    ) -> list[dict]:
        """Walk every successful record in the manifest, extract tables, persist."""
        with open(manifest_path) as f:
            records = json.load(f)

        existing = self._load_index()
        seen_files = {row["local_path"] for row in existing}

        all_tables = list(existing)
        processed = 0
        for rec in tqdm(records, desc="extract tables"):
            if limit is not None and processed >= limit:
                break

            if rec.get("status", "ok") != "ok":
                continue
            local_path = Path(rec["local_path"])
            if not local_path.exists():
                continue
            if local_path.stat().st_size < self.MIN_VALID_FILE_BYTES:
                continue
            if str(local_path) in seen_files:
                continue

            try:
                rows = self.extract_from_file(local_path, rec)
                all_tables.extend(rows)
                processed += 1
            except Exception as e:
                print(f"  ! failed {local_path.name}: {type(e).__name__}: {e}")

            self._save_index(all_tables)

        print(f"\n{len(all_tables)} table(s) total across {len(set(t['local_path'] for t in all_tables))} files")
        return all_tables

    def extract_from_file(self, file_path: Path, record: dict) -> list[dict]:
        """Extract every table in `file_path`. Returns the rows we'll index."""
        elements = self._partition(file_path)
        table_elements = [el for el in elements if getattr(el, "category", "") == "Table"]
        if not table_elements:
            return []

        company = record.get("company") or record.get("ticker", "unknown")
        year = str(record.get("year", "unknown"))

        rows: list[dict] = []
        for i, table_el in enumerate(table_elements):
            html = self._html_for(table_el)
            df = self._html_to_df(html) if html else None
            if df is None or df.empty:
                continue

            page = self._page_for(table_el)
            context = self._context_above(elements, table_el, n_before=4)
            title = (
                self._llm_title(context, df) if self.use_llm_titles
                else self._heuristic_title(df, context)
            )

            tid = self._table_id(company, year, page, i)
            parquet_path = self.tables_dir / f"{tid}.parquet"
            try:
                df.to_parquet(parquet_path)
            except Exception:
                # Fall back to CSV if pyarrow/fastparquet can't handle the schema.
                parquet_path = parquet_path.with_suffix(".csv")
                df.to_csv(parquet_path, index=False)

            rows.append({
                "table_id": tid,
                "company": company,
                "year": year,
                "page": int(page),
                "title": title,
                "html": html,
                "parquet_path": str(parquet_path),
                "n_rows": int(len(df)),
                "n_cols": int(len(df.columns)),
                "columns": [str(c) for c in df.columns],
                "first_row": self._row_to_dict(df.iloc[0]) if len(df) > 0 else {},
                "last_row": self._row_to_dict(df.iloc[-1]) if len(df) > 0 else {},
                "context_above": context[:600],
                "source_url": record.get("source_url", ""),
                "local_path": str(file_path),
                "market": self.market,
                "filing_type": record.get("filing_type", "annual_report"),
                "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
        return rows

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _partition(self, file_path: Path):
        """Run unstructured with table-structure inference on."""
        from unstructured.partition.auto import partition

        return partition(
            filename=str(file_path),
            strategy=self.strategy,
            infer_table_structure=True,
        )

    @staticmethod
    def _html_for(el) -> str:
        md = getattr(el, "metadata", None)
        if md is None:
            return ""
        return getattr(md, "text_as_html", "") or ""

    @staticmethod
    def _page_for(el) -> int:
        md = getattr(el, "metadata", None)
        if md is None:
            return 1
        return int(getattr(md, "page_number", None) or 1)

    @staticmethod
    def _html_to_df(html: str):
        import pandas as pd

        try:
            tables = pd.read_html(StringIO(html))
            return tables[0] if tables else None
        except Exception:
            return None

    @staticmethod
    def _context_above(elements, table_el, n_before: int = 4) -> str:
        """Concatenate text from the few elements immediately before this table."""
        try:
            idx = elements.index(table_el)
        except ValueError:
            return ""
        chunks: list[str] = []
        for el in elements[max(0, idx - n_before):idx]:
            text = (getattr(el, "text", "") or "").strip()
            if text and getattr(el, "category", "") != "Table":
                chunks.append(text)
        return "\n".join(chunks)[:1500]

    @staticmethod
    def _table_id(company: str, year: str, page: int, idx: int) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{company}_{year}_p{page}_{idx}")
        return slug.strip("_")[:120]

    @staticmethod
    def _row_to_dict(row) -> dict:
        # JSON-safe: stringify keys, coerce values
        return {str(k): (None if v is None else (v.item() if hasattr(v, "item") else v))
                for k, v in row.to_dict().items()}

    @staticmethod
    def _heuristic_title(df, context: str) -> str:
        """Fallback when LLM titles are off. Try the last sentence above the
        table, then the column headers."""
        if context:
            last_line = context.strip().splitlines()[-1].strip()
            if 6 <= len(last_line) <= 140:
                return last_line
        cols = ", ".join(str(c) for c in df.columns[:5])
        return f"Table — columns: {cols}"

    def _llm_title(self, context: str, df) -> str:
        try:
            from finagent.graph.state import TableTitle

            llm = self._get_llm().with_structured_output(TableTitle)
            cols = [str(c) for c in df.columns[:8]]
            out: TableTitle = llm.invoke(TITLE_PROMPT.format(context=context[:1500], cols=cols))
            title = (out.title or "").strip().strip('"').strip("'")
            return title[:200] if title else self._heuristic_title(df, context)
        except Exception:
            return self._heuristic_title(df, context)

    def _get_llm(self):
        if self._llm is None:
            from finagent.llm import build_llm

            model = self.title_model or {
                "groq": "llama-3.1-8b-instant",
                "gemini": "gemini-2.5-flash",
                "openai": "gpt-4o-mini",
                "anthropic": "claude-haiku-4-5",
            }[self.provider]
            self._llm = build_llm(self.provider, model, self.api_key, temperature=0.0)
        return self._llm

    # ------------------------------------------------------------------ #
    # Index persistence
    # ------------------------------------------------------------------ #

    def _load_index(self) -> list[dict]:
        if self.index_path.exists():
            try:
                return json.load(open(self.index_path))
            except Exception:
                return []
        return []

    def _save_index(self, rows: list[dict]) -> None:
        with open(self.index_path, "w") as f:
            json.dump(rows, f, indent=2, default=str)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract tables into parquet + index.json.")
    p.add_argument("--manifest", required=True,
                   help="manifest.json or sec_manifest.json produced by fetchPDFs.py")
    p.add_argument("--tables-dir", default="data/tables")
    p.add_argument("--market", choices=["india", "us"], default="india")
    p.add_argument("--provider", choices=["groq", "gemini", "openai", "anthropic"],
                   default="groq")
    p.add_argument("--title-model", default=None)
    p.add_argument("--no-llm-titles", action="store_true",
                   help="Skip the LLM title call (faster; use heuristic title)")
    p.add_argument("--strategy", default="hi_res", choices=["hi_res", "fast", "auto"])
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after extracting tables from N files (smoke test)")
    return p


def main():
    args = _build_cli().parse_args()
    tx = TableExtractor(
        tables_dir=args.tables_dir,
        market=args.market,
        provider=args.provider,
        title_model=args.title_model,
        use_llm_titles=not args.no_llm_titles,
        strategy=args.strategy,
    )
    tx.ingest_corpus(args.manifest, limit=args.limit)


if __name__ == "__main__":
    main()
