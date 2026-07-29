"""
evaluate_retrieval.py  ·  finagent/evaluation/evaluate_retrieval.py

Chunk-geometry x embedder x reranker sweep over the **served HTML pipeline** —
the evidence behind "why this parent/child size, why this embedder, why this
reranker".

Why this exists (the old numbers do not transfer)
-------------------------------------------------
`results/RETRIEVAL_EXPERIMENTS.md` settled those three choices on the *PDF*
pipeline. Production now serves SEC HTML, and that is a different chunker, not
just a different parser: the PDF path slides a `RecursiveCharacterTextSplitter`
over pypdf page text, while the HTML path groups `partition_html` elements with
`chunk_by_title(max_characters=..., overlap=...)`, which gives every table its
own chunk (whole unless it exceeds `max_characters` — measured: 4 of 45 tables
in a 3M 10-K do, so a table IS occasionally divided into `TableChunk`s).
"parent = 2500" means a different thing on each. The old geometry harness also
indexed **one filing per collection**, so no cross-company distractor could
occur — which is exactly what separates a good reranker from a weak one.

Ground truth
------------
NOT `results/financebench_gold.json`. That map picks "the chunk most similar to
the evidence" by dense similarity, so it moves with both the chunk geometry and
the embedder under test — it cannot compare configurations (it silently reported
recall@100 = 0.027 once for exactly this reason). We score against FinanceBench's
verified evidence span directly:

    coverage@k = shingle recall of the evidence span in the union of the top-k
                 PARENTS handed to the synthesizer (10-word shingles over
                 alphanumeric-normalised text — unstructured renders tables as
                 pipe-delimited rows, and a whitespace-only norm scores that
                 formatting difference as missing content)
    hit@k      = coverage >= 0.5

That is geometry-independent, embedder-independent, and is the thing we actually
care about: how much of the answer-bearing text reaches the LLM.

The parsing ceiling
-------------------
~23% of the served questions have evidence that `partition_html` never recovers
from the primary filing (mostly table layout). No retriever can return it, so
counting it as a retrieval miss would penalise every configuration by the same
constant and compress the differences we are trying to measure. Questions whose
evidence is not findable in the parsed document at all (`--ceiling`, default 0.5)
are excluded from the headline and reported separately — that exclusion count is
itself the parser's score.

Everything runs against the LOCAL eval Qdrant (`QDRANT_EVAL_URL`, default
http://localhost:6333) through the **real** `CorpusIngester` and the **real**
`HybridRetriever`, with all 72 filings in one collection, so distractors and the
company/year metadata filter are live.

Query modes (`--mode served`, the default)
-----------------------------------------
The planner still decomposes the question into 1-8 routed sub-queries, but since
§14 those route the tool lanes rather than the retriever. `hybrid_retrieve_node`
searches the filings with ONE retrieval query, plus the raw question at
`QUESTION_SLOTS`, and only when at least one sub-query is routed
`narrative`/`numeric` (a `market`/`external`/`cross_document` question is
answered by its own tool and retrieves nothing). `_cap_pool` then re-ranks the
merged pool against the ORIGINAL question down to `retrieve_cap=8`, reserving
each query's best passage first so one entity cannot crowd out the other in a
comparison. This harness reproduces that whole path, so what it scores is what
the synthesizer receives.

`--mode served` (the DEFAULT) is what production does since §14: retrieval runs
on ONE query per question written in the filing's own vocabulary (company +
fiscal year + statement caption + the line items a filing prints) plus the raw
question at `QUESTION_SLOTS`. Both the retrieval queries and the sub-query plans
are produced ONCE by the served nodes and cached
(`results/financebench_retrieval_queries.json`,
`results/financebench_subqueries.json`); re-generating per configuration would
fold the planner's LLM nondeterminism into the geometry deltas we are trying to
measure. They are upstream of retrieval, so they are held fixed, exactly like
the corpus.

Two controls, both LLM-free at measurement time:
  `--mode subquery`  retrieve on the planner's decomposition — what production
                     did before §14, kept so that regression stays visible.
  `--mode question`  retrieve on the bare question. Cheapest possible baseline.

Usage
-----
    export QDRANT_EVAL_URL=http://localhost:6333
    python -m finagent.evaluation.evaluate_retrieval --self-check
    python -m finagent.evaluation.evaluate_retrieval --stage parent --docs 3   # smoke
    python -m finagent.evaluation.evaluate_retrieval --stage parent
    python -m finagent.evaluation.evaluate_retrieval --stage child    --parent 2500
    python -m finagent.evaluation.evaluate_retrieval --stage embedder --parent 2500 --child 600
    python -m finagent.evaluation.evaluate_retrieval --table-only

Each run appends its rows to `results/html_retrieval_sweep.json` (keyed by
config, so a re-run overwrites that row) and regenerates the whole table into
`results/html_retrieval_sweep.md`. Stages do NOT auto-carry the winner: read the
table, then pass `--parent/--child` explicitly to the next stage.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# The eval corpus lives on the local Qdrant, never the served cluster.
os.environ.setdefault("QDRANT_EVAL_URL", "http://localhost:6333")
# Parent-4000 rerank batches are large and the models are swapped between
# passes, which fragments the allocator on a small card.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from finagent.evaluation.financebench.html_source import HTML_DIR
from finagent.graph.corrective import AgenticRAGv2
from finagent.retrieval.expansion import expand_query

OUT_JSON = Path("results/html_retrieval_sweep.json")
OUT_MD = Path("results/html_retrieval_sweep.md")
# A `--docs N` smoke writes here instead. The row key is (mode, config,
# reranker) and carries no corpus size, so sharing the real file lets a
# 3-filing smoke overwrite the 72-filing row of the same name — which is
# exactly what happened once. Keep them in separate files.
SMOKE_JSON = Path("results/html_retrieval_sweep_smoke.json")
SMOKE_MD = Path("results/html_retrieval_sweep_smoke.md")
PLANS_PATH = Path("results/financebench_subqueries.json")
RETRIEVAL_QUERIES_PATH = Path("results/financebench_retrieval_queries.json")
MANIFEST = Path("data/us/eval/financebench/eval_manifest.json")
SWEEP_MANIFEST = Path("data/us/eval/financebench/sweep_manifest.json")
ELEMENT_CACHE = Path("data/us/eval/financebench/elements")

POOL_DEPTH = 48             # production HybridRetriever.pool_top_k
FINAL_PER_SUB = 5           # production HybridRetriever.final_top_k
RETRIEVE_CAP = 8            # production AgenticRAGv2.retrieve_cap
QUESTION_SLOTS = AgenticRAGv2.QUESTION_SLOTS   # slots for the question query
# …and the wider budget a `narrative`-routed request gets (§14c). Both mirror
# production rather than being harness settings, so a sweep means editing the
# class the agent actually uses.
NARRATIVE_QUESTION_SLOTS = AgenticRAGv2.NARRATIVE_QUESTION_SLOTS
FINAL_KS = (5, RETRIEVE_CAP)
# Sub-query routes that reach the filings retriever. The rest (market /
# external / cross_document) are answered by yfinance / web / EDGAR, so
# production never retrieves chunks for them.
RETRIEVED_ROUTES = ("narrative", "numeric")
HIT_THRESHOLD = 0.5
SHINGLE_WORDS = 10
# Score chunks with the §12 context header removed. Off = score exactly what the
# synthesizer receives; on = score only text that retrieval actually found.
STRIP_HEADERS_IN_SCORING = False

LARGE = "BAAI/bge-large-en-v1.5"        # production embedder
RERANKERS = (
    None,                               # RRF order, no cross-encoder — the control
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-v2-m3",          # production reranker
    # Non-BGE arms. Both load through the same `CrossEncoder` API; the ones that
    # did not (jina-reranker-v2, gte-multilingual-reranker) ship remote code that
    # no longer imports against this transformers version, so they cannot be
    # measured here regardless of their published scores.
    "mixedbread-ai/mxbai-rerank-base-v2",     # 494M / 1976 MB
    "cross-encoder/ms-marco-MiniLM-L12-v2",   # 33M / 133 MB — the cheap floor
)


def arm_label(reranker_model: Optional[str]) -> str:
    """Row label for one reranker arm; also half of the row key."""
    return reranker_model or "none (RRF order)"


# --------------------------------------------------------------------------- #
# Evidence coverage
# --------------------------------------------------------------------------- #

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Alphanumeric content only. Parsers disagree on formatting, not content:
    unstructured renders a table as `a | b | c` rows while the FinanceBench
    evidence carries the PDF's whitespace-joined cells."""
    return _NON_ALNUM.sub(" ", (text or "").lower()).strip()


def shingle_recall(gold: str, parsed_norm: str, n: int = SHINGLE_WORDS) -> float:
    """Fraction of `gold`'s n-word shingles present in already-normalised
    `parsed_norm`. Gold shorter than one shingle falls back to containment."""
    gold_n = normalize(gold)
    words = gold_n.split()
    if not words:
        return 0.0
    if len(words) < n:
        return 1.0 if gold_n in parsed_norm else 0.0
    shingles = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return sum(1 for s in shingles if s in parsed_norm) / len(shingles)


def strip_header(text: str, meta: dict) -> str:
    """Remove the injected context header from a chunk before SCORING it.

    §12 prefixes every chunk with "<company> <year> <form> · <caption>". That
    text is real and comes from the filing, but 67 of the 99 gold spans *open
    with the statement caption*, so leaving it in lets a chunk satisfy the
    evidence metric with text we pasted on rather than text we retrieved.
    Scoring with it stripped answers the honest question: did retrieval find
    better chunks, or did the header just hand the metric free shingles?
    """
    head = (meta or {}).get("context_header")
    if head and text.startswith(head):
        return text[len(head):].lstrip("\n")
    return text


def coverage(spans: list[str], texts: list[str]) -> float:
    """Mean coverage over a question's evidence spans. Mean, not max: a
    cross-document question is only answered when *every* span is present."""
    if not spans:
        return 0.0
    blob = normalize(" || ".join(texts))
    return statistics.mean(shingle_recall(s, blob) for s in spans)


# --------------------------------------------------------------------------- #
# Parsed-element cache — partition_html costs 0.8-17 s per filing and its output
# does not depend on any swept knob, so it is parsed once and reused by every
# configuration. Patching the module attribute works because `_html_documents`
# imports the symbol at call time.
# --------------------------------------------------------------------------- #

def score_text(text: str, meta: dict) -> str:
    """The text a chunk is SCORED on (see `STRIP_HEADERS_IN_SCORING`)."""
    return strip_header(text, meta) if STRIP_HEADERS_IN_SCORING else text


def install_element_cache(cache_dir: Path = ELEMENT_CACHE) -> None:
    import unstructured.partition.html as uh

    if getattr(uh.partition_html, "_finagent_cached", False):
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    real = uh.partition_html

    def cached(filename=None, **kwargs):
        if filename is None:
            return real(**kwargs)
        blob = cache_dir / f"{Path(filename).stem}.pkl"
        if blob.exists():
            return pickle.loads(blob.read_bytes())
        elements = real(filename=filename, **kwargs)
        blob.write_bytes(pickle.dumps(elements))
        return elements

    cached._finagent_cached = True
    uh.partition_html = cached


def document_blob(doc_name: str) -> str:
    """Normalised text of everything `partition_html` recovers from one filing.
    Config-independent: `chunk_by_title` groups elements, it never drops their
    text, so this is the ceiling on what ANY geometry can return."""
    from unstructured.partition.html import partition_html

    path = Path(HTML_DIR) / f"{doc_name}.htm"
    if not path.exists():
        return ""
    return normalize(" ".join(e.text or "" for e in partition_html(filename=str(path))))


# --------------------------------------------------------------------------- #
# Configurations
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Config:
    parent: int = 2500          # chunk_by_title max_characters (the PARENT)
    child: int = 600            # RecursiveCharacterTextSplitter size (embedded)
    embed: str = LARGE
    # Prefix every chunk with "<company> <year> <form> · <section caption>".
    # Production (`CorpusIngester.context_headers`) defaults ON since §12, so
    # this must too or a plain sweep measures a config that is not shipped.
    # Rows measured BEFORE §12 have no `_hdr` tag suffix and were built without.
    headers: bool = True

    @property
    def parent_overlap(self) -> int:
        return round(self.parent * 0.12)        # production: 300 at 2500

    @property
    def new_after(self) -> int:
        return round(self.parent * 0.8)         # production: 2000 at 2500

    @property
    def child_overlap(self) -> int:
        return round(self.child / 6)            # production: 100 at 600

    @property
    def tag(self) -> str:
        base = f"p{self.parent}_c{self.child}_{self.embed.split('/')[-1]}"
        return f"{base}_hdr" if self.headers else base


def grid(stage: str, parent: int, child: int) -> list[Config]:
    """Staged sweep, not a cross product: 3^n configurations would each cost a
    full re-index for knobs that prior work already showed are near-orthogonal."""
    if stage == "parent":
        return [Config(parent=p, child=child) for p in (1500, 2500, 4000)]
    if stage == "child":
        return [Config(parent=parent, child=c) for c in (300, 600, 900)]
    if stage == "embedder":
        return [Config(parent=parent, child=child, embed=e) for e in (
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            LARGE,
            # A fine-tune OF bge-base, so this isolates domain adaptation from
            # model size — the only fair way to ask "does finance-tuned win?".
            "FinLang/finance-embeddings-investopedia",
        )]
    if stage == "headers":
        # The §12 A/B: identical geometry and embedder, chunk context header
        # off then on. Run for both the incumbent and the §12 embedder winner
        # so the header effect is not confounded with the embedder change.
        return [Config(parent=parent, child=child, embed=e, headers=h)
                for e in (LARGE, "intfloat/e5-large-v2") for h in (False, True)]
    if stage == "embedder2":
        # Non-BGE families, sized for Cloud Run (bge-large is 1341 MB and the
        # reranker takes another 2271 of the 4 GiB). Each needs its own full
        # re-index, so this is the expensive stage. Prefix-trained models get
        # their query/document prefix from `vectorstore._EMBED_PROMPTS` —
        # without it e5/nomic score near random and the comparison is void.
        return [Config(parent=parent, child=child, embed=e) for e in (
            LARGE,                                      # control, re-measured
            "intfloat/e5-large-v2",                     # 335M / 1341 MB
            "mixedbread-ai/mxbai-embed-large-v1",       # 335M / 1341 MB
            "Snowflake/snowflake-arctic-embed-l-v2.0",  # 568M / 2271 MB
            "nomic-ai/nomic-embed-text-v1.5",           # 137M /  547 MB
        )]
    raise ValueError(f"unknown stage {stage!r}")


# --------------------------------------------------------------------------- #
# Question set
# --------------------------------------------------------------------------- #

@dataclass
class Question:
    id: str
    text: str
    qtype: str
    spans: list[str] = field(default_factory=list)
    ceiling: float = 0.0
    # The sub-queries production would actually retrieve on. Empty means the
    # planner routed every sub-query away from the filings (web/market/EDGAR),
    # so production retrieves nothing here — scored as a miss, because it is one.
    subs: list[str] = field(default_factory=list)
    # ONE retrieval query (`--mode served`): the question restated in
    # the filing's own vocabulary — company, fiscal year, statement caption and
    # the line items a filing prints. Authored offline and cached, exactly like
    # `subs`, so no LLM runs inside the measurement.
    retrieval_query: str = ""
    # Did the planner route any sub-query `narrative`? Drives the question's
    # slot budget: narrative evidence is prose, and the terse §14 keyword query
    # is a poor dense match for prose, so the raw question needs more room.
    narrative: bool = False


def load_eval_questions(doc_names: Optional[set] = None,
                        ceiling_min: float = HIT_THRESHOLD) -> tuple[list[Question], int]:
    """Served-HTML questions with their evidence spans, ceiling-filtered.

    Returns (kept, n_dropped_by_ceiling)."""
    from finagent.evaluation.financebench.dataset import (
        load_questions, restrict_to_served_docs, tag_question_types,
    )

    df = restrict_to_served_docs(tag_question_types(load_questions()))
    blobs: dict[str, str] = {}
    out, dropped = [], 0
    for _, row in df.iterrows():
        spans, docs = [], set()
        for ev in (row.get("evidence") or []):
            if not isinstance(ev, dict):
                continue
            doc, text = ev.get("doc_name", ""), (ev.get("evidence_text") or "")
            if not doc or not text or (doc_names is not None and doc not in doc_names):
                continue
            spans.append(text)
            docs.add(doc)
        if not spans:
            continue
        for d in docs:
            if d not in blobs:
                blobs[d] = document_blob(d)
        doc_blob = normalize(" || ".join(blobs[d] for d in sorted(docs)))
        ceiling = statistics.mean(shingle_recall(s, doc_blob) for s in spans)
        if ceiling < ceiling_min:
            dropped += 1
            continue
        out.append(Question(id=row["financebench_id"], text=row["question"],
                            qtype=row.get("qtype", "unknown"), spans=spans,
                            ceiling=round(ceiling, 4), subs=[row["question"]]))
    return out, dropped


# --------------------------------------------------------------------------- #
# Sub-query plans — the production decomposition, cached
# --------------------------------------------------------------------------- #

def attach_retrieval_queries(questions: list[Question],
                             path: Path = RETRIEVAL_QUERIES_PATH,
                             refresh: bool = False) -> int:
    """Write each question's retrieval query with the SERVED node and cache it.

    Same discipline as `attach_subqueries`, and for the same reason: the query
    is upstream of retrieval, so it is generated once and held fixed across
    configurations — regenerating per config would fold the model's sampling
    noise into every geometry delta measured downstream.

    Calls `AgenticRAGv3._retrieval_query` directly, so the harness scores the
    prompt production actually ships rather than a copy that can drift from it.
    Returns the number of questions with no usable query (those fall back to the
    raw question, scored as-is — which is what production does too)."""
    from tqdm import tqdm

    cache: dict = {}
    if path.exists() and not refresh:
        cache = json.loads(path.read_text())

    todo = [q for q in questions if q.id not in cache]
    if todo:
        from finagent.graph import FinAgent

        agent = FinAgent()
        for q in tqdm(todo, desc="retrieval queries"):
            state: dict = {"errors": []}
            # `numeric` mirrors a filings-routed question: the real gate is the
            # planner's routes, and every question in this set is answered from
            # the filings by construction.
            cache[q.id] = {"question": q.text,
                           "retrieval_query": agent._retrieval_query(
                               state, q.text, ["numeric"])}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, indent=2))   # resumable

    missing = 0
    for q in questions:
        q.retrieval_query = (cache.get(q.id) or {}).get("retrieval_query") or ""
        if not q.retrieval_query:
            missing += 1
    return missing


def attach_subqueries(questions: list[Question], path: Path = PLANS_PATH,
                      refresh: bool = False) -> int:
    """Plan every question with the SERVED planner and keep the sub-queries that
    production would retrieve on. Cached: the plan is upstream of retrieval, so
    it is held fixed across configurations — re-planning per config would fold
    the planner's nondeterminism into the geometry deltas.

    Returns the number of questions production would retrieve nothing for."""
    from tqdm import tqdm

    cache: dict = {}
    if path.exists() and not refresh:
        cache = json.loads(path.read_text())

    todo = [q for q in questions if q.id not in cache]
    if todo:
        from finagent.graph import FinAgent

        agent = FinAgent()
        for q in tqdm(todo, desc="planning"):
            try:
                out = agent.planner_node(
                    {"question": q.text, "chat_history": [], "errors": []})
                cache[q.id] = {"question": q.text,
                               "sub_queries": out.get("sub_queries") or [q.text],
                               "query_routes": out.get("query_routes") or []}
            except Exception as e:
                print(f"  ! planner failed on {q.id}: {type(e).__name__}: {e}")
                cache[q.id] = {"question": q.text, "sub_queries": [q.text],
                               "query_routes": [], "error": str(e)}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, indent=2))   # resumable

    n_unretrieved = 0
    for q in questions:
        plan = cache.get(q.id) or {}
        subs = plan.get("sub_queries") or [q.text]
        routes = plan.get("query_routes") or []
        # Mirror `hybrid_retrieve_node`: gate on the routes only when they align
        # with the sub-queries, else retrieve for all of them.
        if routes and len(routes) == len(subs):
            subs = [s for s, r in zip(subs, routes) if r in RETRIEVED_ROUTES]
        q.subs = subs
        q.narrative = "narrative" in routes
        if not subs:
            n_unretrieved += 1
    return n_unretrieved


# --------------------------------------------------------------------------- #
# Index one configuration
# --------------------------------------------------------------------------- #

def write_manifest(doc_names: Optional[set]) -> Path:
    records = json.loads(MANIFEST.read_text())
    if doc_names is not None:
        records = [r for r in records if r.get("doc_name") in doc_names]
    SWEEP_MANIFEST.write_text(json.dumps(records, indent=2))
    return SWEEP_MANIFEST


def _release_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def free_embedder() -> None:
    """Drop the cached dense embedder.

    `get_embeddings` is `lru_cache(maxsize=4)`, so the embedder stage would
    otherwise leave all four models resident — ~2.4 GB — next to the
    cross-encoders on an 8 GB card, and the next bge-large batch OOMs on top.
    """
    from finagent.vectorstore import get_embeddings

    get_embeddings.cache_clear()
    _release_gpu()


def free_reranker(model_name: Optional[str]) -> None:
    """Drop one cross-encoder from the process-wide cache.

    That cache exists so the SERVED agent never loads a second copy; here we
    want the opposite — the arms run sequentially, so keeping v2-m3 (2.3 GB)
    resident while base runs just costs headroom the next batch needs.
    """
    if model_name is None:
        return
    from finagent.retrieval import reranker

    reranker._SHARED_RERANKERS.pop(model_name, None)
    _release_gpu()


def build_index(cfg: Config, collection: str, manifest: Path) -> dict:
    """Ingest the eval corpus at `cfg` using the production ingester."""
    from finagent.ingestion.ingest import CorpusIngester

    ing = CorpusIngester(
        corpus_dir=HTML_DIR,
        collection_name=collection,
        market="us",
        embedding_model=cfg.embed,
        html_max_chars=cfg.parent,
        html_new_after=cfg.new_after,
        html_overlap=cfg.parent_overlap,
        parent_doc=True,
        context_headers=cfg.headers,
    )
    # ponytail: child geometry is a class constant, not a ctor argument — an
    # instance attribute shadows it, which is enough for a sweep. Promote it to
    # a real parameter if it ever needs to vary in production.
    ing.CHILD_CHUNK_SIZE = cfg.child
    ing.CHILD_CHUNK_OVERLAP = cfg.child_overlap

    ing.reset_collection()          # chunk geometry changes every point id
    t0 = time.time()
    stats = ing.ingest_all(manifest_path=str(manifest), skip_if_already_indexed=False)
    return {"chunks": stats.total_chunks, "files": stats.files_processed,
            "ingest_s": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------- #
# Score one configuration (all reranker arms share the index and the pool)
# --------------------------------------------------------------------------- #

def cap_pool(question: str, chunks: list[tuple], reranker_model: Optional[str],
             cap: int = RETRIEVE_CAP) -> list[tuple]:
    """Reproduce `AgenticRAGv2._cap_pool`: keep the `cap` best passages of the
    merged sub-query pool, reserving each sub-query's single best first —
    without that floor, one entity's chunks crowd out the other's on a
    comparison question.

    Ranking re-scores the whole pool against the ORIGINAL question. §11 ranked
    on the score each chunk carried from its own sub-query instead, having
    measured that as better (37 -> 41 of 99); §13 re-ran the A/B once chunks had
    context headers and it reversed (58 vs 61), so the re-score is back.

    `chunks` are (text, meta, sub_query) triples in merge order."""
    if len(chunks) <= cap:
        return chunks
    if reranker_model is None:
        return chunks[:cap]         # the no-reranker control head-truncates

    from finagent.retrieval.reranker import _get_shared_reranker

    scores = _get_shared_reranker(reranker_model).predict(
        [(question, t) for t, _, _ in chunks])
    ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])
    kept, seen_subs = [], set()
    for i in ranked:
        if chunks[i][2] not in seen_subs:
            seen_subs.add(chunks[i][2])
            kept.append(i)
        if len(kept) >= cap:
            break
    for i in ranked:
        if len(kept) >= cap:
            break
        if i not in kept:
            kept.append(i)
    return [chunks[i] for i in sorted(kept, key=ranked.index)]


def evaluate_index(cfg: Config, collection: str, questions: list[Question],
                   mode: str = "served") -> list[dict]:
    """Score one index, in two passes so only ONE model is on the GPU at a time.

    Pass 1 pulls every candidate pool (needs the embedder); pass 2 runs the
    reranker arms over those cached pools (needs one cross-encoder). Holding the
    embedder and both cross-encoders together — 1.3 + 1.1 + 2.3 GB — OOM'd an
    8 GB card on the first parent-4000 batch. Caching the pools also guarantees
    every arm is compared on byte-identical candidates."""
    from finagent.retrieval.hybrid import HybridRetriever
    from finagent.vectorstore import build_store, dim_for, get_client
    from tqdm import tqdm

    # In question mode there is one "sub-query" and no merge, so the single
    # reranked list is taken straight to the cap.
    per_sub_k = FINAL_PER_SUB if mode == "subquery" else max(FINAL_KS)

    # ---- Pass 1: candidate pools (embedder resident) ----------------------
    store = build_store(collection, cfg.embed)
    pooler = HybridRetriever(store, pool_top_k=POOL_DEPTH,
                             final_top_k=FINAL_PER_SUB)
    dim = dim_for(cfg.embed)                    # probes the model — before it is freed
    points = get_client().get_collection(collection).points_count

    cached: list[list[tuple[str, list]]] = []   # per question: [(sub, parents)]
    pool_hit, pool_ms, n_subs = 0, [], []
    for q in tqdm(questions, desc=f"{cfg.tag} pools", leave=False):
        if mode == "subquery":
            subs = q.subs
        elif mode == "served":
            subs = [q.retrieval_query or q.text]
        else:
            subs = [q.text]
        # Production also retrieves on the ORIGINAL question (§13/§14), gated on
        # the planner having routed at least one sub-query to the filings. The
        # retrieval query keeps the full per-sub budget; the question adds its
        # top QUESTION_SLOTS on top.
        if mode in ("subquery", "served") and subs and q.text not in subs:
            subs = subs + [q.text]
        n_subs.append(len(subs))
        pools: list[tuple[str, list]] = []
        for sub in subs:
            # `HybridRetriever.search` expands derived metrics before it does
            # anything else; the stages are called directly here, so do it here.
            sub = expand_query(sub)
            flt = pooler.infer_filter(sub)
            t0 = time.time()
            pool = pooler.get_pool(sub, flt)
            if not pool and flt and (flt.get("years") or flt.get("items")):
                pool = pooler.get_pool(sub, {"companies": flt["companies"]})
            pool_ms.append((time.time() - t0) * 1000)
            # Production collapses matched children to their parents before
            # ranking, so the parent is what the reranker scores AND what the
            # LLM receives.
            pools.append((sub, pooler._collapse_to_parents(pool)))
        cached.append(pools)
        pool_texts = [score_text(t, m) for _, ps in pools for t, m in ps]
        if coverage(q.spans, pool_texts) >= HIT_THRESHOLD:
            pool_hit += 1

    del pooler, store
    free_embedder()

    # ---- Pass 2: one reranker arm at a time -------------------------------
    n = len(questions)
    acc = {r: {"cov": {k: 0.0 for k in FINAL_KS}, "hit": {k: 0 for k in FINAL_KS},
               "ms": [], "ctx": [], "by_type": {}} for r in RERANKERS}
    for r in RERANKERS:
        # `rerank()` never touches the store — the embedder is freed by now.
        arm = None if r is None else HybridRetriever(
            None, reranker_model=r, pool_top_k=POOL_DEPTH,
            final_top_k=FINAL_PER_SUB)
        for q, pools in zip(tqdm(questions, desc=f"{cfg.tag} {r or 'none'}",
                                 leave=False), cached):
            t0 = time.time()
            merged, seen = [], set()
            question_query = expand_query(q.text)
            for sub, parents in pools:
                # The question query gets a smaller slot budget than a
                # sub-query — see `AgenticRAGv2.QUESTION_SLOTS`.
                is_question = (mode in ("subquery", "served")
                               and len(pools) > 1 and sub == question_query)
                k = ((NARRATIVE_QUESTION_SLOTS if q.narrative else QUESTION_SLOTS)
                     if is_question else per_sub_k)
                top = (parents[:k] if arm is None
                       else arm.rerank(sub, parents, top_k=k))
                for text, meta in top:
                    key = (meta.get("local_path", ""), meta.get("page", ""),
                           text[:80])
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append((text, meta, sub))
            final = cap_pool(q.text, merged, r)
            acc[r]["ms"].append((time.time() - t0) * 1000)

            texts = [score_text(t, m) for t, m, _ in final]
            acc[r]["ctx"].append(sum(len(t) for t in texts[:max(FINAL_KS)]))
            bt = acc[r]["by_type"].setdefault(q.qtype, {"n": 0, "cov": 0.0, "hit": 0})
            bt["n"] += 1
            for k in FINAL_KS:
                c = coverage(q.spans, texts[:k])
                acc[r]["cov"][k] += c
                acc[r]["hit"][k] += c >= HIT_THRESHOLD
                if k == max(FINAL_KS):
                    bt["cov"] += c
                    bt["hit"] += c >= HIT_THRESHOLD
        del arm
        free_reranker(r)

    points = get_client().get_collection(collection).points_count
    dim = dim_for(cfg.embed)
    denom = max(1, n)
    pool_recall = pool_hit / denom

    rows = []
    for r in RERANKERS:
        a = acc[r]
        top = max(FINAL_KS)
        rows.append({
            "config": cfg.tag, "parent": cfg.parent, "child": cfg.child,
            "embed": cfg.embed, "dim": dim,
            "reranker": arm_label(r),
            "mode": mode,
            "headers": cfg.headers,
            "scored": "header stripped" if STRIP_HEADERS_IN_SCORING else "as served",
            # Which rule ordered the merged pool. §11 replaced the re-score
            # against the original question with the per-sub-query score, so
            # rows either side of that change are not comparable.
            "rank_by": "sub-query score",
            "n": n,
            "subqueries": round(statistics.mean(n_subs), 2) if n_subs else 0,
            "pool_recall": round(pool_recall, 4),
            **{f"cov@{k}": round(a["cov"][k] / denom, 4) for k in FINAL_KS},
            **{f"hit@{k}": round(a["hit"][k] / denom, 4) for k in FINAL_KS},
            # Of the questions whose evidence WAS retrievable, what fraction
            # survives ranking — isolates the reranker from first-stage recall.
            "retention": round((a["hit"][top] / denom) / pool_recall, 4)
            if pool_recall else 0.0,
            "points": points,
            "vectors_mb": round(points * dim * 4 / 1e6, 1),
            f"ctx_chars@{top}": round(statistics.mean(a["ctx"])) if a["ctx"] else 0,
            "pool_ms": round(statistics.median(pool_ms)) if pool_ms else 0,
            "rerank_ms": round(statistics.median(a["ms"])) if a["ms"] else 0,
            "by_type": {t: {"n": v["n"], f"cov@{top}": round(v["cov"] / v["n"], 4),
                            f"hit@{top}": round(v["hit"] / v["n"], 4)}
                        for t, v in sorted(a["by_type"].items())},
        })
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

_COLUMNS = ["mode", "parent", "child", "embed", "dim", "reranker", "n",
            "subqueries", "pool_recall", "cov@5", "hit@5", "cov@8", "hit@8",
            "retention", "points", "vectors_mb", "ctx_chars@8", "pool_ms",
            "rerank_ms"]


def render_table(data: dict) -> str:
    rows = sorted(data["rows"].values(),
                  key=lambda r: (r.get("mode", ""), r["parent"], r["child"],
                                 r["embed"], r["reranker"]))
    meta = data.get("meta", {})
    out = [
        "# Retrieval sweep — served HTML pipeline",
        "",
        f"Corpus: {meta.get('files', '?')} SEC filings (FinanceBench 10-K/10-Q), "
        f"one Qdrant collection, production `CorpusIngester` + `HybridRetriever`, "
        f"pool depth {POOL_DEPTH}, {FINAL_PER_SUB} per sub-query, "
        f"capped to {RETRIEVE_CAP} against the original question.",
        "",
        f"Questions: **{meta.get('n_questions', '?')}** scored · "
        f"{meta.get('n_dropped', '?')} excluded because `partition_html` never "
        f"recovers their evidence from the primary filing (parsing ceiling "
        f"< {HIT_THRESHOLD}, no retriever can return it) · "
        f"{meta.get('n_unretrieved', '?')} where the planner routed every "
        f"sub-query away from the filings (scored as misses — production "
        f"retrieves nothing for them either).",
        "",
        "`mode` = `served` runs the production path (one retrieval query → "
        "per-sub-query retrieval → merge → global cap); `question` retrieves on "
        "the bare question and is a control, not what production does.",
        "",
        "`coverage@k` = shingle recall of the verified evidence span in the top-k "
        "parents given to the synthesizer; `hit@k` = coverage ≥ "
        f"{HIT_THRESHOLD}; `retention` = hit@8 / pool_recall.",
        "",
        "| " + " | ".join(_COLUMNS) + " |",
        "|" + "|".join("---" for _ in _COLUMNS) + "|",
    ]
    embed_col = _COLUMNS.index("embed")
    for r in rows:
        cells = [str(r.get(c, "")) for c in _COLUMNS]
        cells[embed_col] = cells[embed_col].split("/")[-1]
        out.append("| " + " | ".join(cells) + " |")

    out += ["", "## By question type (top-8, production reranker)", "",
            "| config | qtype | n | cov@8 | hit@8 |", "|---|---|---|---|---|"]
    for r in rows:
        if r["reranker"] != "BAAI/bge-reranker-v2-m3":
            continue
        for t, v in r.get("by_type", {}).items():
            out.append(f"| {r['config']} | {t} | {v['n']} | "
                       f"{v.get('cov@8', '')} | {v.get('hit@8', '')} |")
    return "\n".join(out) + "\n"


def save(rows: list[dict], meta: dict, smoke: bool = False) -> None:
    out_json, out_md = (SMOKE_JSON, SMOKE_MD) if smoke else (OUT_JSON, OUT_MD)
    data = {"meta": {}, "rows": {}}
    if out_json.exists():
        data = json.loads(out_json.read_text())
    data["meta"] = {**data.get("meta", {}), **meta}
    for r in rows:
        key = f"{r['mode']}|{r['config']}|{r['reranker']}"
        if r.get("scored") == "header stripped":
            key += "|stripped"
        data["rows"][key] = r
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2))
    out_md.write_text(render_table(data))
    print(f"\n→ {out_json}\n→ {out_md}")


# --------------------------------------------------------------------------- #

def self_check() -> None:
    """The coverage metric is the whole eval — if it is wrong, every number is."""
    doc = normalize("Total net sales were $394,328 million in 2022, up 8%.")
    assert shingle_recall("Total net sales were $394,328 million in 2022, up 8%.", doc) == 1.0
    assert shingle_recall("Total  NET   sales; were: 394,328 (million) in 2022 — up 8%", doc) == 1.0, \
        "punctuation/case/whitespace must not count as missing content"
    # Digit grouping is NOT normalised away — "394,328" and "394328" are
    # different token streams. Both parsers keep the filing's own grouping, so
    # this is only a limitation for evidence that reformats numbers.
    assert shingle_recall("total net sales were 394328 million in 2022 up 8", doc) == 0.0
    assert shingle_recall("The board declared a quarterly dividend of $0.23", doc) == 0.0
    assert shingle_recall("net sales", doc) == 1.0, "short gold falls back to containment"
    assert shingle_recall("", doc) == 0.0
    # Table formatting: unstructured pipes vs the PDF's spacing.
    table = normalize("Revenue | 394,328 | 365,817")
    assert shingle_recall("Revenue 394,328 365,817", table) == 1.0
    assert coverage(["net sales", "no such text here at all"], [doc]) == 0.5
    assert coverage([], [doc]) == 0.0

    # A §12 header must not be able to satisfy the evidence metric by itself:
    # 67 of the 99 gold spans open with the statement caption the header adds.
    head = "3M 2022 10-K \u00b7 Consolidated Balance Sheet"
    meta = {"context_header": head}
    chunk = f"{head}\nTotal assets | 46,455 | 47,072"
    assert strip_header(chunk, meta) == "Total assets | 46,455 | 47,072"
    assert strip_header(chunk, {}) == chunk, "no header recorded = leave text alone"
    assert strip_header("Total assets", meta) == "Total assets", "prefix must match"
    assert shingle_recall(head, normalize(chunk)) == 1.0
    assert shingle_recall(head, normalize(strip_header(chunk, meta))) == 0.0

    # cap_pool: the per-sub-query floor must reserve sub B's best passage even
    # when sub A owns every top score — that is what stops one entity of a
    # comparison question from crowding out the other.
    from finagent.retrieval import reranker as _rr

    scores = {"a1": 9.0, "a2": 8.0, "a3": 7.0, "b1": 1.0}
    _rr._SHARED_RERANKERS["__stub__"] = type(
        "Stub", (), {"predict": staticmethod(lambda pairs: [scores[t] for _, t in pairs])})()
    chunks = [("a1", {}, "subA"), ("a2", {}, "subA"), ("a3", {}, "subA"),
              ("b1", {}, "subB")]
    kept = [t for t, _, _ in cap_pool("q", chunks, "__stub__", cap=2)]
    assert kept == ["a1", "b1"], kept
    assert [t for t, _, _ in cap_pool("q", chunks, None, cap=2)] == ["a1", "a2"]
    assert cap_pool("q", chunks, "__stub__", cap=9) == chunks

    del _rr._SHARED_RERANKERS["__stub__"]

    # expansion: a derived metric gains the line items that carry the numbers,
    # and a query naming none is left byte-identical (§13).
    assert "total current liabilities" in expand_query("Verizon quick ratio FY2022")
    assert expand_query("3M total revenue 2022") == "3M total revenue 2022"
    print("self-check OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--stage", choices=["parent", "child", "embedder", "embedder2", "headers"])
    ap.add_argument("--mode",
                    choices=["served", "subquery", "question"],
                    default="served",
                    help="served = production since §14 (one retrieval query "
                         "in the filing's vocabulary + the question); subquery "
                         "= the pre-§14 decomposition; question = the bare "
                         "question (LLM-free control)")
    ap.add_argument("--refresh-plans", action="store_true",
                    help="re-plan the sub-queries instead of reusing the cache")
    ap.add_argument("--parent", type=int, default=2500, help="held parent size")
    ap.add_argument("--child", type=int, default=600, help="held child size")
    ap.add_argument("--docs", type=int, help="smoke test: only the N filings "
                                            "with the most questions")
    ap.add_argument("--ceiling", type=float, default=HIT_THRESHOLD,
                    help="drop questions whose evidence is less findable than "
                         "this in the parsed filing")
    ap.add_argument("--keep", action="store_true", help="keep the sweep collections")
    ap.add_argument("--resume", action="store_true",
                    help="skip configs already present in the results file")
    ap.add_argument("--strip-headers-scoring", action="store_true",
                    help="score chunks with the §12 context header removed, so "
                         "the injected caption cannot satisfy the evidence "
                         "metric on its own")
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.strip_headers_scoring:
        globals()["STRIP_HEADERS_IN_SCORING"] = True
    if args.self_check:
        return self_check()
    if args.table_only:
        data = json.loads(OUT_JSON.read_text())
        OUT_MD.write_text(render_table(data))
        return print(f"→ {OUT_MD}")
    if not args.stage:
        return ap.error("--stage is required (or --table-only / --self-check)")

    self_check()
    install_element_cache()

    records = json.loads(MANIFEST.read_text())
    doc_names = None
    if args.docs:
        from finagent.evaluation.financebench.dataset import load_questions
        counts = load_questions().doc_name.value_counts()
        available = {r["doc_name"] for r in records}
        doc_names = set(list(d for d in counts.index if d in available)[:args.docs])

    manifest = write_manifest(doc_names)
    print(f"Corpus: {len(json.loads(manifest.read_text()))} filings")

    questions, dropped = load_eval_questions(doc_names, args.ceiling)
    print(f"Questions: {len(questions)} scored, {dropped} dropped "
          f"(evidence not recoverable from the parsed filing)")
    if not questions:
        raise SystemExit("no questions survived the ceiling filter")

    unretrieved = 0
    if args.mode == "subquery":
        unretrieved = attach_subqueries(questions, refresh=args.refresh_plans)
        mean_subs = statistics.mean(len(q.subs) for q in questions)
        print(f"Plans: {mean_subs:.2f} retrieved sub-queries/question, "
              f"{unretrieved} routed entirely away from the filings")
    elif args.mode == "served":
        missing = attach_retrieval_queries(questions,
                                           refresh=args.refresh_plans)
        print(f"Retrieval queries: 1/question, {missing} missing "
              f"(fell back to the raw question)")

    from finagent.vectorstore import delete_collection

    # A config costs ~30 min, and rows are saved per config — so a run that dies
    # (or is killed) mid-sweep should not re-pay for the configs it finished.
    done: set = set()
    if args.resume:
        rp = SMOKE_JSON if args.docs else OUT_JSON
        if rp.exists():
            # Only rows measured on THIS question set count as done.
            done = {k for k, v in json.loads(rp.read_text()).get("rows", {}).items()
                    if v.get("n") == len(questions)}

    for cfg in grid(args.stage, args.parent, args.child):
        collection = f"sweep_{cfg.tag}"
        if all(f"{args.mode}|{cfg.tag}|{arm_label(r)}" in done for r in RERANKERS):
            print(f"\n=== {cfg.tag} === already in {OUT_JSON}, skipping", flush=True)
            continue
        print(f"\n=== {cfg.tag} ===", flush=True)
        stats = build_index(cfg, collection, manifest)
        rows = evaluate_index(cfg, collection, questions, mode=args.mode)
        for r in rows:
            r.update(ingest_s=stats["ingest_s"], chunks=stats["chunks"])
            print(f"  {r['reranker']:26} pool={r['pool_recall']:.4f} "
                  f"cov@8={r['cov@8']:.4f} hit@8={r['hit@8']:.4f} "
                  f"hit@5={r['hit@5']:.4f} ctx={r['ctx_chars@8']}")
        save(rows, smoke=bool(args.docs), meta={"files": stats["files"],
                    "n_questions": len(questions),
                    "n_dropped": dropped, "n_unretrieved": unretrieved,
                    "pool_depth": POOL_DEPTH,
                    "docs_subset": args.docs or "all"})
        if not args.keep:
            delete_collection(collection)
        free_embedder()


if __name__ == "__main__":
    main()
