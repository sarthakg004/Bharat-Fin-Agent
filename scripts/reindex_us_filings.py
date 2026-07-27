"""Build a fresh served index.

Writes a NEW collection with the CURRENT production chunk geometry and embedder
(taken straight from `CorpusIngester` / `vectorstore.DEFAULT_EMBED_MODEL` — the
single source of truth, so this script never carries its own copy that can drift
from what the code actually ships). Nothing is mutated in place.

BLUE/GREEN NO LONGER FITS. One collection is ~534 MB (80k points: ~329 MB of
1024-d vectors + ~205 MB of payload, since parent-doc retrieval stores
`parent_text` on every child) and the Qdrant free tier is 1024 MB. Two served
collections do not fit, so the old one must be DELETED BEFORE the new one is
built, and there is a window with no index at all. The real backup is
`data/us/pdfs` on disk — the corpus is always rebuildable from it.

If the tier is ever upgraded past ~1.1 GB, go back to build-then-cut-over: it
is strictly safer.

Run this whenever the chunk geometry or embedder changes, since deterministic
point ids are keyed on the chunker — a geometry change needs a rebuild, not a
re-ingest into the existing collection (see `vectorstore.chunk_point_id`).

    --source       the collection currently served (for status display + the
                   cutover reminder). Default: US_COLLECTION env, else us_filings_v4.
    --collection   the NEW collection to build. MUST be a fresh name — this is
                   the guard against writing into the live one. No default.

Cutover after this finishes: change `US_COLLECTION` in
`.github/workflows/deploy.yml` and push. The deploy sets it with
`--update-env-vars`, so the collection and the code that matches it land in one
revision.

    DO NOT run `gcloud run services update --set-env-vars US_COLLECTION=...`.

`--set-env-vars` REPLACES the service's entire non-secret environment. This
docstring used to recommend it, and running it at the v3 cutover silently wiped
QDRANT_URL, ALLOWED_ORIGINS, FORCE_IPV4, LANGFUSE_* and PERSIST_DYNAMIC_FETCH
from revision 00037 — every retrieval query then failed with "QDRANT_URL is not
set" until it was noticed several revisions later. If you must set it by hand,
use `--update-env-vars`, which only touches the key you name. The deploy
workflow now also re-asserts QDRANT_URL and ALLOWED_ORIGINS on every deploy so
the same mistake self-heals.

Rollback is the same env var pointed back at the old collection.

See results/RETRIEVAL_EXPERIMENTS.md for the geometry/embedder numbers and the
storage arithmetic.

Usage
-----
    python scripts/reindex_us_filings.py --collection us_filings_v5   # build
    python scripts/reindex_us_filings.py --status                     # counts
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

# The collection currently served — status/cutover text only, never written to.
SOURCE = os.getenv("US_COLLECTION", "us_filings_v4")
# Dynamic-fetch downloads live here; they are not part of the corpus.
EXCLUDE = "sec-edgar-filings"


def status(target: str | None = None) -> None:
    from finagent.vectorstore import get_client
    c = get_client()
    for name in (SOURCE, target):
        if not name:
            continue
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
    ap.add_argument("--collection",
                    help="NEW collection to build (must not already exist)")
    args = ap.parse_args()

    if args.status:
        status(args.collection)
        return

    if not args.collection:
        raise SystemExit("--collection is required: name the NEW collection to "
                         "build. It must not be the live one.")

    from finagent.vectorstore import get_client as _gc
    if _gc().collection_exists(args.collection):
        raise SystemExit(
            f"{args.collection!r} already exists. Blue/green needs a FRESH "
            f"target so the live collection is never touched — pick a new name "
            f"or delete this one first if it is a failed build.")

    # The eval corpus must be off the managed cluster first, or the old and new
    # served collections cannot both fit in the free tier during the swap.
    from finagent.vectorstore import get_client
    if get_client().collection_exists("financebench_eval"):
        raise SystemExit(
            "financebench_eval is still on the cluster. Run "
            "scripts/move_eval_to_local.py (then --drop-source) first; the old "
            "and new served collections need the room during cutover.")

    from finagent.vectorstore import DEFAULT_EMBED_MODEL
    from finagent.ingestion.ingest import CorpusIngester

    # No geometry overrides here: the build uses whatever CorpusIngester and
    # DEFAULT_EMBED_MODEL currently ship, so the new index is exactly what
    # production would write. (This is why ingest.py must be the single source of
    # truth — a private copy of the constants here is how the paths drift apart.)
    print(f"building {args.collection}: {DEFAULT_EMBED_MODEL} · "
          f"parent {CorpusIngester.PARENT_CHUNK_SIZE}/"
          f"{CorpusIngester.PARENT_CHUNK_OVERLAP} · "
          f"child {CorpusIngester.CHILD_CHUNK_SIZE}/"
          f"{CorpusIngester.CHILD_CHUNK_OVERLAP}")
    print("production keeps serving", SOURCE, "throughout\n")

    # data/us/pdfs, NOT data/us: the parent also holds
    # data/us/eval/financebench/pdfs (368 benchmark filings), which must never
    # enter the served index.
    ing = CorpusIngester(
        corpus_dir="data/us/pdfs",
        collection_name=args.collection,
        market="us",
        parent_doc=True,
    )
    # data/us/pdfs holds TWO layouts and only one of them is the corpus:
    #   TICKER/YEAR_ACCESSION.htm                     the curated documents
    #   sec-edgar-filings/TICKER/10-K/.../*.html      dynamic-fetch scratch
    # The served collection contains ONLY the curated set. Including the scratch
    # would change the corpus (so a version-to-version comparison would no longer
    # isolate the geometry/embedder change) and inflate the index past the free
    # tier. Dynamically fetched filings are re-fetched on demand at runtime; they
    # are not corpus.
    _load = ing._load_records
    ing._load_records = lambda mp: [r for r in _load(mp)
                                    if EXCLUDE not in str(r.get("local_path", ""))]

    t0 = time.time()
    stats = ing.ingest_all(manifest_path=None, skip_if_already_indexed=True)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min: {stats}")
    status(args.collection)


if __name__ == "__main__":
    main()
