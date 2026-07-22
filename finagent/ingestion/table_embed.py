"""
table_embed.py  ·  finagent/ingestion/table_embed.py

Offline, run-once CLI (not on the serve path): it built the `tables` Qdrant
collection consumed at runtime by `graph/table_agent.py`.

Build a second Qdrant collection (`tables`) keyed on extracted-table titles +
first/last row + columns, so a question like "Reliance balance sheet 2023"
retrieves the actual balance-sheet table rather than narrative chunks.

Reads `data/tables/index.json` (produced by `table_ingest.py`) and embeds one
document per table.

Usage as a library
------------------
    from finagent.ingestion.table_embed import TableEmbedder

    emb = TableEmbedder(collection_name="tables")
    emb.embed_all("data/tables/index.json")

CLI
---
    python -m finagent.ingestion.table_embed \\
        --index data/tables/index.json \\
        --collection tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Union

from tqdm import tqdm


class TableEmbedder:
    """Embed table titles (plus structural hints) into a Qdrant collection."""

    def __init__(
        self,
        collection_name: str = "tables",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 128,
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def embed_all(
        self,
        index_path: Union[str, Path] = "data/tables/index.json",
        skip_if_already_indexed: bool = True,
    ):
        """Add every table in the index to the `tables` collection (idempotent)."""
        with open(index_path) as f:
            tables = json.load(f)
        print(f"Index has {len(tables)} tables")

        store = self._get_store()
        already = self._existing_ids(store) if skip_if_already_indexed else set()
        if already:
            print(f"{len(already)} already indexed; appending the rest")

        from langchain_core.documents import Document

        docs: list[Document] = []
        ids: list[str] = []
        for t in tables:
            tid = t["table_id"]
            if tid in already:
                continue
            docs.append(Document(page_content=self.doc_text(t), metadata=self._safe_metadata(t)))
            ids.append(tid)

        if not docs:
            print("Nothing new to embed.")
            return store

        for i in tqdm(range(0, len(docs), self.batch_size), desc="embed tables"):
            store.add_documents(
                docs[i: i + self.batch_size],
                ids=ids[i: i + self.batch_size],
            )
        print(f"Embedded {len(docs)} table(s) into '{self.collection_name}'")
        return store

    # ------------------------------------------------------------------ #
    # Document text + metadata
    # ------------------------------------------------------------------ #

    @staticmethod
    def doc_text(t: dict) -> str:
        """Embedding text per table: title + columns + first/last row."""
        title = t.get("title", "").strip()
        company = t.get("company", "")
        year = t.get("year", "")
        cols = ", ".join(map(str, t.get("columns", [])[:12]))
        first = TableEmbedder._row_to_str(t.get("first_row") or {})
        last = TableEmbedder._row_to_str(t.get("last_row") or {})
        return (
            f"{title}\n"
            f"Company: {company} ({year}), page {t.get('page', '?')}\n"
            f"Columns: {cols}\n"
            f"First row: {first}\n"
            f"Last row:  {last}"
        )

    @staticmethod
    def _row_to_str(row: dict) -> str:
        # Compact, comma-separated; truncate huge values.
        parts: list[str] = []
        for k, v in row.items():
            if v is None or v == "":
                continue
            s = str(v)
            if len(s) > 60:
                s = s[:60] + "…"
            parts.append(f"{k}={s}")
        return ", ".join(parts)

    @staticmethod
    def _safe_metadata(t: dict) -> dict:
        """Payload values must be scalars; strip out nested dicts."""
        keep = {
            "table_id", "company", "year", "page", "title",
            "parquet_path", "local_path", "market", "filing_type",
            "n_rows", "n_cols", "source_url",
        }
        out: dict = {}
        for k, v in t.items():
            if k not in keep:
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
            else:
                out[k] = str(v)
        return out

    # ------------------------------------------------------------------ #
    # Vector store
    # ------------------------------------------------------------------ #

    def _get_store(self):
        from finagent.vectorstore import build_store

        return build_store(self.collection_name, self.embedding_model, create=True)

    def _existing_ids(self, store=None) -> set:
        from finagent.vectorstore import distinct_values

        try:
            return distinct_values(self.collection_name, "table_id", limit=200_000)
        except Exception:
            return set()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Embed extracted tables into Qdrant.")
    p.add_argument("--index", default="data/tables/index.json")
    p.add_argument("--collection", default="tables")
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--reindex", action="store_true",
                   help="Embed every table even if already indexed")
    args = p.parse_args()

    emb = TableEmbedder(
        collection_name=args.collection,
        embedding_model=args.embedding_model,
    )
    emb.embed_all(args.index, skip_if_already_indexed=not args.reindex)


if __name__ == "__main__":
    main()
