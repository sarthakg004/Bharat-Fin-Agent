"""Pre-reindex study: child size x parent size x embedder x reranker.

Reindexing 105k points is expensive and the chunk geometry is baked into every
point id, so this settles the geometry BEFORE we pay for it.

Ground truth
------------
The existing gold map picks "the chunk most similar to the evidence" using dense
similarity -- which is circular here: it would change with the chunk geometry AND
with the embedder we are trying to A/B. Instead we score against FinanceBench's
verified evidence span directly:

    coverage = fraction of the evidence span's content lines that appear
               somewhere in the union of the top-k parents handed to the LLM
    hit@k    = coverage >= HIT_THRESHOLD

That is geometry-independent, embedder-independent, and is the thing we actually
care about: how much of the answer-bearing text does the synthesizer receive.

Everything runs against a LOCAL in-memory Qdrant with the same named dense +
sparse vectors and the same server-side RRF fusion as production, so no cluster
writes and no cost.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

# run from the repo root regardless of cwd (this file lives in scripts/)
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from qdrant_client import QdrantClient, models

from finagent.evaluation.financebench.indexing import DEFAULT_PDF_DIR as _PDF_DIR

from pathlib import Path

PDF_DIR = Path(_PDF_DIR)

OUT = "results/geometry_study.json"
HIT_THRESHOLD = 0.5
POOL_DEPTH = 24          # measured best in results/funnel_sweep.json
FINAL_KS = (5, 8)
MIN_LINE = 25            # ignore trivial evidence lines ("2018", "$")

DENSE, SPARSE = "dense", "sparse"


# --------------------------------------------------------------------------- #
# Chunking — mirrors CorpusIngester's parent-document path
# --------------------------------------------------------------------------- #

def split(text: str, size: int, overlap: int) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]).split_text(text)


def chunk_doc(pages: list[str], child: int, parent: int) -> list[dict]:
    """Parent windows over page text; small children carrying parent_text."""
    c_over, p_over = max(child // 6, 50), max(parent // 7, 100)
    out, pid = [], 0
    for pno, ptext in enumerate(pages, 1):
        if not ptext.strip():
            continue
        for pchunk in split(ptext, parent, p_over):
            kids = [k.strip() for k in split(pchunk, child, c_over) if k.strip()]
            for k in kids or [pchunk.strip()]:
                out.append({"text": k[:1900], "parent_id": pid,
                            "parent_text": pchunk[:4000], "page": pno})
            pid += 1
    return out


# --------------------------------------------------------------------------- #
# Evidence coverage
# --------------------------------------------------------------------------- #

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def evidence_units(evidence: list) -> list[str]:
    """Content lines of the verified evidence span, normalised."""
    units = []
    for e in evidence or []:
        txt = e.get("evidence_text") if isinstance(e, dict) else str(e)
        for line in (txt or "").split("\n"):
            n = norm(line)
            if len(n) >= MIN_LINE:
                units.append(n)
    return units


def coverage(units: list[str], parents: list[str]) -> float:
    if not units:
        return 0.0
    blob = " || ".join(norm(p) for p in parents)
    return sum(1 for u in units if u in blob) / len(units)


# --------------------------------------------------------------------------- #
# Index + retrieve, all local
# --------------------------------------------------------------------------- #

def build_index(client, name, chunks, dense_vecs, sparse_vecs, dim):
    client.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=dim,
                                                   distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(
            modifier=models.Modifier.IDF)})
    pts = [models.PointStruct(
        id=i,
        vector={DENSE: dense_vecs[i].tolist(),
                SPARSE: models.SparseVector(indices=list(sparse_vecs[i].indices),
                                            values=list(sparse_vecs[i].values))},
        payload=chunks[i])
        for i in range(len(chunks))]
    for i in range(0, len(pts), 512):
        client.upsert(name, points=pts[i:i + 512])


def search(client, name, qdense, qsparse, depth, use_mmr):
    if use_mmr:
        r = client.query_points(
            name, query=models.NearestQuery(
                nearest=qdense.tolist(),
                mmr=models.Mmr(diversity=0.5, candidates_limit=depth * 2)),
            using=DENSE, limit=depth, with_payload=True)
    else:
        r = client.query_points(
            name,
            prefetch=[models.Prefetch(query=qdense.tolist(), using=DENSE,
                                      limit=depth),
                      models.Prefetch(query=models.SparseVector(
                          indices=list(qsparse.indices),
                          values=list(qsparse.values)), using=SPARSE, limit=depth)],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=depth, with_payload=True)
    return [p.payload for p in r.points]


def collapse(payloads):
    seen, out = set(), []
    for p in payloads:
        key = p.get("parent_id")
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=20)
    ap.add_argument("--stage", default="geometry",
                    choices=["geometry", "embedder", "reranker", "mmr",
                             "final", "large"])
    ap.add_argument("--child", type=int, default=600)
    ap.add_argument("--parent", type=int, default=1500)
    args = ap.parse_args()

    from pypdf import PdfReader
    from sentence_transformers import CrossEncoder
    from tqdm import tqdm

    from finagent.device import get_device
    from finagent.evaluation.dataset import load_eval_dataset
    from finagent.vectorstore import get_sparse_embeddings

    df = load_eval_dataset()
    counts = df.doc_name.value_counts()
    docs = [d for d in counts.index
            if (PDF_DIR / f"{d}.pdf").exists()][:args.docs]
    sub = df[df.doc_name.isin(docs)]
    print(f"{len(docs)} docs · {len(sub)} questions", flush=True)

    pages_by_doc = {d: [(p.extract_text() or "")
                        for p in PdfReader(str(PDF_DIR / f"{d}.pdf")).pages]
                    for d in tqdm(docs, desc="parsing")}

    sparse_embed = get_sparse_embeddings()
    device = get_device()

    if args.stage == "geometry":
        grid = [{"child": c, "parent": p, "embed": "BAAI/bge-small-en-v1.5",
                 "rerank": "BAAI/bge-reranker-v2-m3", "mmr": False}
                for p in (1500, 2500, 4000) for c in (300, 600, 900)]
    elif args.stage == "final":
        # Finalists only, on the FULL question set. Child is held at 600: the
        # 9-cell grid moved <=0.02 across 300/900 with hit@k ties and rank
        # flips, i.e. inside noise at n=74, so it is not a free variable --
        # it stays where production already is.
        grid = [{"child": 600, "parent": p, "embed": e,
                 "rerank": "BAAI/bge-reranker-v2-m3", "mmr": False}
                for e in ("BAAI/bge-small-en-v1.5", "BAAI/bge-large-en-v1.5")
                for p in (1500, 2500, 4000)]
    elif args.stage == "large":
        grid = [{"child": args.child, "parent": args.parent,
                 "embed": "BAAI/bge-large-en-v1.5",
                 "rerank": "BAAI/bge-reranker-v2-m3", "mmr": False}]
    elif args.stage == "embedder":
        grid = [{"child": args.child, "parent": args.parent, "embed": e,
                 "rerank": "BAAI/bge-reranker-v2-m3", "mmr": False}
                for e in ("BAAI/bge-small-en-v1.5", "BAAI/bge-large-en-v1.5")]
    elif args.stage == "reranker":
        grid = [{"child": args.child, "parent": args.parent,
                 "embed": "BAAI/bge-small-en-v1.5", "rerank": r, "mmr": False}
                for r in ("BAAI/bge-reranker-base", "BAAI/bge-reranker-v2-m3")]
    else:
        grid = [{"child": args.child, "parent": args.parent,
                 "embed": "BAAI/bge-small-en-v1.5",
                 "rerank": "BAAI/bge-reranker-v2-m3", "mmr": m}
                for m in (False, True)]

    embed_cache: dict = {}
    rerank_cache: dict = {}
    results = {}

    for cfg in grid:
        tag = (f"c{cfg['child']}_p{cfg['parent']}_"
               f"{cfg['embed'].split('-')[1]}_{cfg['rerank'].split('-')[-1]}"
               f"{'_mmr' if cfg['mmr'] else ''}")
        t0 = time.time()

        if cfg["embed"] not in embed_cache:
            import gc

            import torch
            from langchain_huggingface import HuggingFaceEmbeddings
            embed_cache.clear()          # free the previous embedder first
            gc.collect(); torch.cuda.empty_cache()
            # bge-large is 1024-d/335M; batch 128 at 512 tokens OOMs an 8GB card
            # once the v2-m3 reranker is also resident.
            bs = 32 if "large" in cfg["embed"] else 128
            embed_cache[cfg["embed"]] = HuggingFaceEmbeddings(
                model_name=cfg["embed"], model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True, "batch_size": bs})
        emb = embed_cache[cfg["embed"]]
        if cfg["rerank"] not in rerank_cache:
            rerank_cache[cfg["rerank"]] = CrossEncoder(
                cfg["rerank"], device=device,
                max_length=512 if "v2-m3" not in cfg["rerank"] else 1024)
        ce = rerank_cache[cfg["rerank"]]

        client = QdrantClient(":memory:")
        import numpy as np
        dim = None
        per_doc_col = {}
        for d in docs:
            chunks = chunk_doc(pages_by_doc[d], cfg["child"], cfg["parent"])
            texts = [c["text"] for c in chunks]
            dv = np.asarray(emb.embed_documents(texts))
            sv = list(sparse_embed.embed_documents(texts))
            dim = dv.shape[1]
            name = f"c_{abs(hash(d)) % 10**8}"
            build_index(client, name, chunks, dv, sv, dim)
            per_doc_col[d] = name

        cov5 = cov8 = hit5 = hit8 = 0
        n = 0
        by_type: dict = {}
        for _, row in sub.iterrows():
            units = evidence_units(row["evidence"])
            if not units:
                continue
            n += 1
            q = row["question"]
            qd = np.asarray(emb.embed_query(q))
            qs = list(sparse_embed.embed_documents([q]))[0]
            payloads = search(client, per_doc_col[row["doc_name"]], qd, qs,
                              POOL_DEPTH, cfg["mmr"])
            parents = collapse(payloads)
            if parents:
                scores = ce.predict([(q, p["parent_text"]) for p in parents])
                order = sorted(range(len(parents)), key=lambda i: -scores[i])
                parents = [parents[i] for i in order]
            t = by_type.setdefault(row["qtype"], {"n": 0, "cov8": 0.0, "hit8": 0})
            t["n"] += 1
            for k in (5, 8):
                c = coverage(units, [p["parent_text"] for p in parents[:k]])
                if k == 5:
                    cov5 += c; hit5 += c >= HIT_THRESHOLD
                else:
                    cov8 += c; hit8 += c >= HIT_THRESHOLD
                    t["cov8"] += c; t["hit8"] += c >= HIT_THRESHOLD

        results[tag] = {
            **{k: cfg[k] for k in ("child", "parent", "embed", "rerank", "mmr")},
            "dim": dim, "n": n,
            "mean_coverage@5": round(cov5 / n, 4), "hit@5": round(hit5 / n, 4),
            "mean_coverage@8": round(cov8 / n, 4), "hit@8": round(hit8 / n, 4),
            "elapsed_s": round(time.time() - t0, 1),
            "by_type": {k: {"n": v["n"], "cov@8": round(v["cov8"] / v["n"], 4),
                            "hit@8": round(v["hit8"] / v["n"], 4)}
                        for k, v in sorted(by_type.items())},
        }
        r = results[tag]
        print(f"{tag:44} cov@8={r['mean_coverage@8']:.4f} hit@8={r['hit@8']:.4f} "
              f"cov@5={r['mean_coverage@5']:.4f} hit@5={r['hit@5']:.4f} "
              f"({r['elapsed_s']}s)", flush=True)
        client.close()

    path = OUT.replace(".json", f"_{args.stage}.json")
    json.dump({"docs": len(docs), "pool_depth": POOL_DEPTH,
               "hit_threshold": HIT_THRESHOLD, "results": results},
              open(path, "w"), indent=2)
    print(f"\n→ {path}")


if __name__ == "__main__":
    main()
