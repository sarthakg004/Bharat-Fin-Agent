"""
table_embed.py  ·  finagent/ingestion/table_embed.py

Build a second Chroma collection (`tables`) keyed on extracted-table titles +
first/last row + columns, so a question like "Reliance balance sheet 2023"
retrieves the actual balance-sheet table rather than narrative chunks.

Reads `data/tables/index.json` (produced by `table_ingest.py`) and embeds one
document per table.

Usage as a library
------------------
    from finagent.ingestion.table_embed import TableEmbedder

    emb = TableEmbedder(chroma_dir="data/chroma", collection_name="tables")
    emb.embed_all("data/tables/index.json")

CLI
---
    python -m finagent.ingestion.table_embed \\
        --index data/tables/index.json --chroma-dir data/chroma \\
        --collection tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Union

from tqdm import tqdm


class TableEmbedder:
    """Embed table titles (plus structural hints) into a Chroma collection."""

    def __init__(
        self,
        chroma_dir: Union[str, Path] = "data/chroma",
        collection_name: str = "tables",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 128,
    ):
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
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
        """Chroma requires scalar metadata values; strip out nested dicts."""
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
    # Chroma
    # ------------------------------------------------------------------ #

    def _get_store(self):
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        from finagent.chroma_client import chroma_kwargs_for_langchain

        emb = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=emb,
            **chroma_kwargs_for_langchain(self.chroma_dir),
        )

    @staticmethod
    def _existing_ids(store) -> set:
        try:
            got = store._collection.get(include=["metadatas"])
            return {m.get("table_id") for m in (got.get("metadatas") or []) if m}
        except Exception:
            return set()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Embed extracted tables into Chroma.")
    p.add_argument("--index", default="data/tables/index.json")
    p.add_argument("--chroma-dir", default="data/chroma")
    p.add_argument("--collection", default="tables")
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--reindex", action="store_true",
                   help="Embed every table even if already indexed")
    args = p.parse_args()

    emb = TableEmbedder(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
    )
    emb.embed_all(args.index, skip_if_already_indexed=not args.reindex)


if __name__ == "__main__":
    main()
