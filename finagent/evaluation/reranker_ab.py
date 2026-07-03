"""
Reranker A/B harness — measure whether a smaller cross-encoder reranks our
*own* corpus as well as the current default, without any LLM at eval time.

Method (label-grounded, no relevance annotations needed):
  1. Sample N real chunks from our Chroma collections.
  2. For each, generate ONE natural question the chunk answers (Groq, cached to
     disk so reruns are free and deterministic).
  3. Build the BM25 ∪ dense candidate pool once per query (independent of the
     reranker), then rerank that *same* pool with each candidate reranker.
  4. Score how high the gold chunk lands: Hit@1/3/5 and MRR.

Because the candidate pool is identical across rerankers, this isolates the
reranker's contribution. Queries whose gold chunk never made it into the pool
(a retrieval-recall miss, not the reranker's fault) are reported separately and
excluded from the reranker metrics.

Usage:
  python -m finagent.evaluation.reranker_ab \
      --n 60 \
      --rerankers BAAI/bge-reranker-base,cross-encoder/ms-marco-MiniLM-L-6-v2
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GEN_MODEL = "llama-3.1-8b-instant"
CACHE = Path("data/eval_cache/reranker_questions.json")

_GEN_PROMPT = (
    "You are given an excerpt from a company's financial filing. Write ONE "
    "specific, natural question that a user might ask which this excerpt "
    "directly answers. The question must be answerable from the excerpt alone "
    "and should mention the company/metric so it is self-contained. Return ONLY "
    "the question, nothing else.\n\nExcerpt:\n{chunk}"
)


def _sample_chunks(collection: str, chroma_dir: str, n: int, seed: int):
    """Return n (text, metadata) chunks sampled from a collection."""
    from finagent.vectorstore import build_store

    store = build_store(collection, EMBED_MODEL, chroma_dir)
    # Page through the collection — a single .get() trips SQLite's variable cap.
    col = store._collection
    docs: list[tuple] = []
    offset = 0
    while True:
        batch = col.get(include=["documents", "metadatas"], limit=2000, offset=offset)
        ids = batch.get("documents") or []
        if not ids:
            break
        docs.extend(zip(batch["documents"], batch["metadatas"]))
        offset += len(ids)
    # Only chunks with enough text to ground a real question.
    docs = [(t, m) for t, m in docs if t and len(t) > 250]
    random.Random(seed).shuffle(docs)
    return docs[:n], store


def _generate_questions(samples: list[dict]) -> None:
    """Fill in `q` for any sample missing one, using Groq. Cached to disk."""
    from finagent.llm import build_llm

    todo = [s for s in samples if not s.get("q")]
    if not todo:
        return
    llm = build_llm("groq", GEN_MODEL, temperature=0.0, rotate=True)
    print(f"Generating {len(todo)} questions via {GEN_MODEL}…")
    for i, s in enumerate(todo, 1):
        try:
            resp = llm.invoke(_GEN_PROMPT.format(chunk=s["text"][:1200]))
            q = (getattr(resp, "content", None) or str(resp)).strip().strip('"')
            s["q"] = q.split("\n")[0][:300]
        except Exception as e:
            s["q"] = ""
            print(f"  [{i}] generation failed: {type(e).__name__}")
        if i % 10 == 0:
            print(f"  {i}/{len(todo)}")
            _save_cache(samples)
    _save_cache(samples)


def _save_cache(samples: list[dict]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(samples, indent=2))


def _gold_rank(ranked_texts: list[str], gold_text: str) -> int | None:
    """1-based rank of the gold chunk in a reranked list, or None if absent."""
    for i, t in enumerate(ranked_texts, 1):
        if t == gold_text:
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60, help="chunks sampled per collection")
    ap.add_argument("--collections", default="us_filings")
    ap.add_argument("--chroma-dir", default="data/chroma")
    ap.add_argument(
        "--rerankers",
        default="BAAI/bge-reranker-base,cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    ap.add_argument("--pool", type=int, default=16, help="BM25+dense candidates per side")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    collections = [c.strip() for c in args.collections.split(",") if c.strip()]
    rerankers = [r.strip() for r in args.rerankers.split(",") if r.strip()]

    # ---- 1+2. sample chunks and (re)generate questions, with a disk cache ----
    if CACHE.exists():
        samples = json.loads(CACHE.read_text())
        print(f"Loaded {len(samples)} cached samples from {CACHE}")
    else:
        samples = []
        for col in collections:
            chunks, _ = _sample_chunks(col, args.chroma_dir, args.n, args.seed)
            for text, meta in chunks:
                samples.append({"collection": col, "text": text, "meta": meta, "q": ""})
        print(f"Sampled {len(samples)} chunks across {collections}")
    _generate_questions(samples)
    samples = [s for s in samples if s.get("q")]

    # ---- build one retriever per collection (for the candidate pool) ----
    from finagent.retrieval import HybridRetriever
    from finagent.vectorstore import build_store

    retrievers = {
        col: HybridRetriever(
            build_store(col, EMBED_MODEL, args.chroma_dir),
            bm25_top_k=args.pool, dense_top_k=args.pool, final_top_k=args.pool * 2,
        )
        for col in collections
    }

    # ---- load each reranker once ----
    from sentence_transformers import CrossEncoder

    from finagent.device import get_device

    models = {name: CrossEncoder(name, device=get_device()) for name in rerankers}

    # ---- 3+4. for each query: shared pool, then rerank with each model ----
    # metrics[name] = list of gold ranks (within-pool queries only)
    metrics: dict[str, list[int]] = {name: [] for name in rerankers}
    latency: dict[str, list[float]] = {name: [] for name in rerankers}
    baseline_ranks: list[int] = []     # no-rerank (pool order)
    in_pool = 0

    for s in samples:
        col = s["collection"]
        union = retrievers[col]._bm25_then_dense(s["q"])
        texts = [t for t, _ in union]
        if s["text"] not in texts:
            continue                    # gold not retrieved into pool → skip
        in_pool += 1
        baseline_ranks.append(_gold_rank(texts, s["text"]))

        pairs = [(s["q"], t) for t in texts]
        for name, model in models.items():
            t0 = time.time()
            scores = model.predict(pairs)
            latency[name].append((time.time() - t0) * 1000)
            order = [texts[i] for i in sorted(range(len(texts)), key=lambda i: -scores[i])]
            r = _gold_rank(order, s["text"])
            if r:
                metrics[name].append(r)

    # ---- report ----
    def hit(ranks, k):
        return 100 * sum(1 for r in ranks if r <= k) / len(ranks) if ranks else 0.0

    def mrr(ranks):
        return sum(1 / r for r in ranks) / len(ranks) if ranks else 0.0

    n_eval = in_pool
    print("\n" + "=" * 78)
    print(f"Reranker A/B  ·  {len(samples)} queries  ·  {n_eval} gold-in-pool "
          f"(pool recall {100*n_eval/max(1,len(samples)):.0f}%)")
    print("=" * 78)
    print(f"{'model':<42}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>7}{'ms/q':>7}")
    print("-" * 78)
    print(f"{'(no rerank: BM25∪dense order)':<42}"
          f"{hit(baseline_ranks,1):>7.1f}{hit(baseline_ranks,3):>8.1f}"
          f"{hit(baseline_ranks,5):>8.1f}{mrr(baseline_ranks):>7.3f}{'—':>7}")
    for name in rerankers:
        r = metrics[name]
        ms = sum(latency[name]) / len(latency[name]) if latency[name] else 0
        short = name if len(name) <= 40 else "…" + name[-39:]
        print(f"{short:<42}{hit(r,1):>7.1f}{hit(r,3):>8.1f}"
              f"{hit(r,5):>8.1f}{mrr(r):>7.3f}{ms:>7.0f}")
    print("=" * 78)
    print("Higher Hit@k / MRR = better ranking; ms/q is reranker CPU time per query.")


if __name__ == "__main__":
    main()
