#!/usr/bin/env python
"""Seed the served Qdrant collection with a single filing.

The full Gemini index is a multi-day build against a metered free tier (roughly
1000 embedded chunks per key per day). This puts ONE filing in the new
collection so production can be cut over and validated now, and lets the corpus
grow afterwards from two places: dynamic EDGAR fetch, which upserts whatever it
pulls, and further runs of this script.

Idempotent. Point ids are derived from (filing, position, content), so
re-running overwrites rather than duplicates, and the sqlite embed cache means a
re-run costs no quota at all.

    python scripts/seed_collection.py --ticker AAPL
    python scripts/seed_collection.py --ticker MSFT --collection us_filings_v5_gemini
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from finagent.config import settings
from finagent.ingestion.ingest import CorpusIngester
from finagent.vectorstore import (DEFAULT_EMBED_MODEL, EmbeddingQuotaExhausted,
                                  count)

CORPUS_DIR = Path("data/us/pdfs")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", required=True,
                    help="Sub-directory of data/us/pdfs to ingest, e.g. AAPL")
    ap.add_argument("--collection", default=settings.us_collection)
    ap.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    args = ap.parse_args()

    src = CORPUS_DIR / args.ticker
    if not src.is_dir():
        available = sorted(p.name for p in CORPUS_DIR.iterdir() if p.is_dir())
        raise SystemExit(f"{src} not found. Available: {', '.join(available)}")

    print(f"seeding {args.collection} from {src} "
          f"with {args.embedding_model}", flush=True)
    ing = CorpusIngester(
        corpus_dir=str(src),
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        market="us",
        parent_doc=True,
    )
    started = time.time()
    try:
        ing.ingest_all(manifest_path=None, skip_if_already_indexed=True)
    except EmbeddingQuotaExhausted as e:
        # Partial progress is kept: whatever was embedded is already upserted
        # and cached, so the next run resumes rather than restarts.
        print(f"\nDAILY EMBEDDING QUOTA EXHAUSTED: {e}\n"
              f"Re-run this command after the quota resets; cached vectors "
              f"make the completed part free.", file=sys.stderr)
        return 1

    print(f"done in {time.time() - started:.0f}s; "
          f"{args.collection} now holds {count(args.collection)} points",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
