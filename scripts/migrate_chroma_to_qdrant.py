"""
migrate_chroma_to_qdrant.py

One-shot migration of the on-disk Chroma corpus into Qdrant.

Dense vectors are COPIED, not recomputed: they are already in Chroma and are
the same bge-small vectors we would produce again, so copying makes dense
retrieval provably identical after the move — if benchmark numbers shift, the
sparse/fusion change is the only possible cause. Re-embedding 177k chunks on
CPU would take ~40 minutes and prove nothing.

Sparse (BM25) vectors do not exist in Chroma and are computed here. That is
tokenisation and term weighting, no neural network, so it is cheap.

Point ids are deterministic (`finagent.vectorstore.chunk_point_id`), so re-running
this script overwrites the same points instead of duplicating them. The final
count is asserted against the source: an id scheme that collides would silently
lose chunks (an earlier key did exactly that, collapsing 71k into 34k).

Usage
-----
    python scripts/migrate_chroma_to_qdrant.py --collection us_filings
    python scripts/migrate_chroma_to_qdrant.py --all --chroma-dir data/chroma
"""

from __future__ import annotations

import argparse
import time

from qdrant_client import models

from finagent.vectorstore import (
    CONTENT_KEY, DENSE_VECTOR, META_KEY, SPARSE_VECTOR,
    chunk_point_id, ensure_collection, get_client, get_sparse_embeddings,
)


def _chroma_collection(chroma_dir: str, name: str):
    import chromadb

    return chromadb.PersistentClient(path=chroma_dir).get_collection(name)


def migrate(name: str, chroma_dir: str, batch: int = 512,
            dry_run: bool = False) -> int:
    col = _chroma_collection(chroma_dir, name)
    total = col.count()
    if total == 0:
        print(f"  {name}: empty, skipping")
        return 0
    print(f"  {name}: {total:,} chunks")
    if dry_run:
        return 0

    ensure_collection(name)
    client = get_client()
    sparse = get_sparse_embeddings()

    moved, offset, t0 = 0, 0, time.time()
    while True:
        got = col.get(limit=batch, offset=offset,
                      include=["embeddings", "documents", "metadatas"])
        docs = got.get("documents") or []
        if not docs:
            break
        metas = got.get("metadatas") or [{}] * len(docs)
        vecs = got.get("embeddings")

        # BM25 term weights for this batch (cheap: no neural net).
        sparse_vecs = sparse.embed_documents(list(docs))

        points = []
        for i, (text, meta) in enumerate(zip(docs, metas)):
            meta = meta or {}
            pid = chunk_point_id(meta, text)
            sv = sparse_vecs[i]
            points.append(models.PointStruct(
                id=pid,
                vector={
                    DENSE_VECTOR: [float(x) for x in vecs[i]],
                    SPARSE_VECTOR: models.SparseVector(
                        indices=list(sv.indices), values=list(sv.values)),
                },
                payload={CONTENT_KEY: text, META_KEY: meta},
            ))

        client.upsert(collection_name=name, points=points, wait=True)
        moved += len(points)
        offset += len(docs)
        rate = moved / max(time.time() - t0, 1e-6)
        print(f"    {moved:,}/{total:,}  ({rate:,.0f}/s)", end="\r", flush=True)

    print(f"    {moved:,}/{total:,}  done in {time.time() - t0:.0f}s" + " " * 12)

    stored = client.count(name, exact=True).count
    if stored < total:
        raise SystemExit(
            f"  {name}: FAILED — {total:,} chunks in but only {stored:,} points "
            f"stored. Point ids are colliding, which loses data.")
    return moved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chroma-dir", default="data/chroma")
    ap.add_argument("--collection", action="append", dest="collections")
    ap.add_argument("--all", action="store_true",
                    help="migrate every non-empty collection")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = args.collections or []
    if args.all or not names:
        import chromadb
        client = chromadb.PersistentClient(path=args.chroma_dir)
        names = [c if isinstance(c, str) else c.name
                 for c in client.list_collections()]

    print(f"Migrating {len(names)} collection(s) from {args.chroma_dir}")
    grand = 0
    for n in names:
        try:
            grand += migrate(n, args.chroma_dir, args.batch, args.dry_run)
        except Exception as e:
            print(f"  {n}: FAILED — {type(e).__name__}: {e}")
    print(f"\nMigrated {grand:,} points.")
    if not args.dry_run:
        c = get_client()
        for n in names:
            if c.collection_exists(n):
                print(f"  {n}: {c.count(n, exact=True).count:,} points in Qdrant")


if __name__ == "__main__":
    main()
