"""Move the FinanceBench eval corpus off the managed cluster onto a local Qdrant.

`financebench_eval` is ~60% of all indexed points and is never queried when
answering a user — it exists only for `finagent/evaluation/**`. Keeping it in
the 1 GB free tier is what blocks the bge-large / parent-2500 upgrade of the
SERVED index.

Vectors are COPIED, not recomputed: dense and sparse both come across as stored,
so the eval measures exactly the same index before and after the move. Point ids
are preserved, so `results/financebench_gold_parentdoc.json` stays valid.

The source collection is NOT deleted here. Verify first, then drop it with
    --drop-source        (run again with this flag once counts match)

Usage
-----
    docker run -d --name qdrant-eval -p 6333:6333 \
        -v ~/qdrant_eval_storage:/qdrant/storage qdrant/qdrant

    python scripts/move_eval_to_local.py
    python scripts/move_eval_to_local.py --drop-source
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

from qdrant_client import QdrantClient, models  # noqa: E402

from finagent.vectorstore import (  # noqa: E402
    DENSE_VECTOR, META_KEY, SPARSE_VECTOR, _INDEXED_FIELDS, qdrant_url,
)

COLLECTION = "financebench_eval"
LOCAL_URL = os.getenv("QDRANT_EVAL_URL", "http://localhost:6333")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--drop-source", action="store_true",
                    help="delete the cloud copy (only after verifying)")
    args = ap.parse_args()

    cloud = QdrantClient(url=qdrant_url(),
                         api_key=(os.getenv("QDRANT_API_KEY") or "").strip() or None,
                         timeout=120)
    local = QdrantClient(url=LOCAL_URL, timeout=120)

    if not cloud.collection_exists(COLLECTION):
        # Already moved: verify the local copy and optionally finish the job.
        n = local.count(COLLECTION, exact=True).count if local.collection_exists(COLLECTION) else 0
        print(f"cloud copy is gone; local has {n:,} points")
        return

    src = cloud.get_collection(COLLECTION)
    total = src.points_count
    print(f"source: {total:,} points on the managed cluster")

    if args.drop_source:
        if not local.collection_exists(COLLECTION):
            raise SystemExit("refusing: no local collection to fall back on")
        n = local.count(COLLECTION, exact=True).count
        if n != total:
            raise SystemExit(f"refusing: local has {n:,} of {total:,} points")
        cloud.delete_collection(COLLECTION)
        print(f"dropped {COLLECTION} from the cluster ({n:,} points verified local)")
        return

    # Recreate locally with the SAME vector config (named dense + sparse w/ IDF).
    if local.collection_exists(COLLECTION):
        print(f"local already has {local.count(COLLECTION, exact=True).count:,} points; "
              f"re-copying (upsert is idempotent — ids are preserved)")
    else:
        params = src.config.params
        local.create_collection(
            COLLECTION,
            vectors_config={DENSE_VECTOR: models.VectorParams(
                size=params.vectors[DENSE_VECTOR].size,
                distance=params.vectors[DENSE_VECTOR].distance)},
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(
                modifier=models.Modifier.IDF)})
        for field in _INDEXED_FIELDS:
            local.create_payload_index(COLLECTION, f"{META_KEY}.{field}",
                                       field_schema=models.PayloadSchemaType.KEYWORD)
        print("created local collection + payload indexes")

    moved, offset, t0 = 0, None, time.time()
    while True:
        pts, offset = cloud.scroll(COLLECTION, limit=args.batch, with_payload=True,
                                   with_vectors=True, offset=offset)
        if not pts:
            break
        local.upsert(COLLECTION, points=[
            models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in pts], wait=False)
        moved += len(pts)
        print(f"  {moved:,}/{total:,}  ({moved/max(time.time()-t0,1e-9):,.0f}/s)",
              end="\r", flush=True)
        if offset is None:
            break

    time.sleep(2)
    got = local.count(COLLECTION, exact=True).count
    print(f"\ncopied {moved:,}; local now holds {got:,} points")
    if got != total:
        raise SystemExit(f"MISMATCH: cloud {total:,} vs local {got:,} — not dropping source")
    print("counts match. Re-run with --drop-source to free the cluster.")


if __name__ == "__main__":
    main()
