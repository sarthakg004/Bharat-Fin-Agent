"""Build the next-generation served index alongside the live one (blue/green).

Writes a NEW collection (default `us_filings_v2`) with the measured-best
geometry and embedder, while production keeps serving the old collection from
the old code. Nothing is mutated in place, so there is never a window where a
384-dim service queries a 1024-dim collection.

    embedder  BAAI/bge-large-en-v1.5   (1024-dim)  +0.042 cov@8
    parent    2500 / 300                           +0.070 cov@8
    child      600 / 100               (unchanged — no measurable effect)

See results/RETRIEVAL_EXPERIMENTS.md §6b for the numbers and §7 for the storage
arithmetic that makes this fit the 1 GB free tier.

Cutover after this finishes:
    1. push the constants (DEFAULT_EMBED_MODEL / DENSE_DIM / PARENT_CHUNK_SIZE)
    2. gcloud run services update finagent --set-env-vars US_COLLECTION=us_filings_v2
    3. verify, then delete the old collection

Rollback is the same env var pointed back at `us_filings`.

Usage
-----
    python scripts/reindex_us_filings.py                 # build
    python scripts/reindex_us_filings.py --status        # progress / counts
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from dotenv import load_dotenv

load_dotenv()

TARGET = os.getenv("REINDEX_COLLECTION", "us_filings_v2")
SOURCE = "us_filings"
EMBED = "BAAI/bge-large-en-v1.5"
PARENT, PARENT_OVERLAP = 2500, 300
CHILD, CHILD_OVERLAP = 600, 100
# Dynamic-fetch downloads live here; they are not part of the corpus.
EXCLUDE = "sec-edgar-filings"


def status() -> None:
    from finagent.vectorstore import get_client
    c = get_client()
    for name in (SOURCE, TARGET):
        if c.collection_exists(name):
            info = c.get_collection(name)
            dim = list(info.config.params.vectors.values())[0].size
            print(f"  {name:16} {info.points_count:>8,} points  dim={dim}  "
                  f"status={info.status}")
        else:
            print(f"  {name:16} (does not exist)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--collection", default=TARGET)
    args = ap.parse_args()

    if args.status:
        status()
        return

    # The eval corpus must be off the managed cluster first, or the old and new
    # served collections cannot both fit in the free tier during the swap.
    from finagent.vectorstore import get_client
    if get_client().collection_exists("financebench_eval"):
        raise SystemExit(
            "financebench_eval is still on the cluster. Run "
            "scripts/move_eval_to_local.py (then --drop-source) first; the old "
            "and new served collections need the room during cutover.")

    from finagent.ingestion.ingest import CorpusIngester

    # Override the class-level chunk geometry for this run only. These are the
    # values that will be committed to ingest.py at cutover; setting them here
    # keeps the running production code untouched while the new index builds.
    CorpusIngester.PARENT_CHUNK_SIZE = PARENT
    CorpusIngester.PARENT_CHUNK_OVERLAP = PARENT_OVERLAP
    CorpusIngester.CHILD_CHUNK_SIZE = CHILD
    CorpusIngester.CHILD_CHUNK_OVERLAP = CHILD_OVERLAP

    print(f"building {args.collection}: {EMBED} · parent {PARENT}/{PARENT_OVERLAP} "
          f"· child {CHILD}/{CHILD_OVERLAP}")
    print("production keeps serving", SOURCE, "throughout\n")

    # data/us/pdfs, NOT data/us: the parent also holds
    # data/us/eval/financebench/pdfs (368 benchmark filings), which must never
    # enter the served index.
    ing = CorpusIngester(
        corpus_dir="data/us/pdfs",
        collection_name=args.collection,
        embedding_model=EMBED,
        market="us",
        parent_doc=True,
    )
    # data/us/pdfs holds TWO layouts and only one of them is the corpus:
    #   TICKER/YEAR_ACCESSION.htm                     the curated 84 documents
    #   sec-edgar-filings/TICKER/10-K/.../*.html      dynamic-fetch scratch
    # The live us_filings contains ONLY the curated set (verified: 71,127/71,127
    # points, 84 distinct local_paths, zero sec-edgar-filings). Including the
    # scratch would (a) change the corpus, so a v1-vs-v2 comparison would no
    # longer isolate the geometry/embedder change, and (b) take the index from
    # ~407 MB to ~921 MB against a 1 GB free tier. Dynamically fetched filings
    # are re-fetched on demand at runtime; they are not corpus.
    _load = ing._load_records
    ing._load_records = lambda mp: [r for r in _load(mp)
                                    if EXCLUDE not in str(r.get("local_path", ""))]

    t0 = time.time()
    stats = ing.ingest_all(manifest_path=None, skip_if_already_indexed=True)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min: {stats}")
    status()


if __name__ == "__main__":
    main()
