"""
news_ingest.py  ·  src/dataIngestion/news_ingest.py

Index a financial-news dataset (default: Hugging Face `kdave/Indian_Financial_News`)
into a third Chroma collection (`news`). When Tavily is unavailable (no key,
no outbound web allowed, network sandboxed) the `external` route falls back to
this collection instead of failing.

The dataset has columns like `Date`, `Title`, `Content` (sometimes `Description`).
This script auto-detects the columns and embeds `Title + Content` per article.

Usage as a library
------------------
    from src.dataIngestion.news_ingest import NewsIngester

    ing = NewsIngester(chroma_dir="data/chroma", collection_name="news")
    ing.ingest_hf("kdave/Indian_Financial_News", split="train", limit=10000)

CLI
---
    python -m src.dataIngestion.news_ingest \\
        --hf-dataset kdave/Indian_Financial_News --limit 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

from tqdm import tqdm


class NewsIngester:
    """Embed financial-news articles into a Chroma collection for the
    `external` fallback retriever."""

    TEXT_COLS = ("Content", "content", "Description", "description", "Body", "body")
    TITLE_COLS = ("Title", "title", "Headline", "headline")
    DATE_COLS = ("Date", "date", "PublishedAt", "published_at", "Datetime")
    URL_COLS = ("URL", "url", "Link", "link")

    def __init__(
        self,
        chroma_dir: Union[str, Path] = "data/chroma",
        collection_name: str = "news",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 256,
    ):
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def ingest_hf(
        self,
        dataset_name: str = "kdave/Indian_Financial_News",
        split: str = "train",
        limit: Optional[int] = None,
        config: Optional[str] = None,
    ):
        """Stream from a Hugging Face dataset and embed up to `limit` articles."""
        from datasets import load_dataset

        kwargs = {"split": split}
        if config is not None:
            kwargs["name"] = config
        ds = load_dataset(dataset_name, **kwargs)
        rows = ds.select(range(min(limit, len(ds)))) if limit else ds
        return self._ingest_iterable(rows, source=dataset_name)

    def ingest_parquet(self, parquet_path: Union[str, Path], limit: Optional[int] = None):
        """Embed from a local parquet/csv file (same column heuristics)."""
        import pandas as pd

        p = str(parquet_path)
        df = (pd.read_csv(p) if p.endswith(".csv") else pd.read_parquet(p))
        if limit:
            df = df.head(limit)
        return self._ingest_iterable(df.to_dict(orient="records"), source=str(parquet_path))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _ingest_iterable(self, rows, source: str):
        store = self._get_store()
        from langchain_core.documents import Document

        title_col = text_col = date_col = url_col = None
        docs: list[Document] = []

        for i, r in enumerate(tqdm(rows, desc=f"index news [{source}]")):
            row = dict(r)  # works for both HF datasets and DataFrame records
            if title_col is None:
                title_col = next((c for c in self.TITLE_COLS if c in row), None)
                text_col = next((c for c in self.TEXT_COLS if c in row), None)
                date_col = next((c for c in self.DATE_COLS if c in row), None)
                url_col = next((c for c in self.URL_COLS if c in row), None)
                if not text_col:
                    raise ValueError(
                        f"Could not find a text column among {self.TEXT_COLS} in {list(row)}"
                    )

            title = str(row.get(title_col, "") or "").strip() if title_col else ""
            body = str(row.get(text_col, "") or "").strip()
            if not body:
                continue
            date = str(row.get(date_col, "") or "") if date_col else ""
            url = str(row.get(url_col, "") or "") if url_col else ""

            content = f"{title}\n\n{body}" if title else body
            meta = {
                "source": "news",
                "dataset": source,
                "title": title[:200],
                "date": date[:32],
                "url": url[:300],
            }
            docs.append(Document(page_content=content[:4000], metadata=meta))

            if len(docs) >= self.batch_size:
                store.add_documents(docs)
                docs = []

        if docs:
            store.add_documents(docs)
        print(f"\nDone. Collection '{self.collection_name}' is ready.")
        return store

    def _get_store(self):
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        emb = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=emb,
            persist_directory=str(self.chroma_dir),
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Index financial-news into Chroma.")
    p.add_argument("--hf-dataset", default="kdave/Indian_Financial_News")
    p.add_argument("--split", default="train")
    p.add_argument("--config", default=None)
    p.add_argument("--parquet", default=None,
                   help="Use this local parquet/csv instead of the HF dataset")
    p.add_argument("--chroma-dir", default="data/chroma")
    p.add_argument("--collection", default="news")
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    args = p.parse_args()

    ing = NewsIngester(chroma_dir=args.chroma_dir, collection_name=args.collection,
                       embedding_model=args.embedding_model)
    if args.parquet:
        ing.ingest_parquet(args.parquet, limit=args.limit)
    else:
        ing.ingest_hf(args.hf_dataset, split=args.split,
                      limit=args.limit, config=args.config)


if __name__ == "__main__":
    main()
