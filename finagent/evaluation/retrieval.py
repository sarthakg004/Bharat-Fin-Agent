"""Retrieval evaluation on FinanceBench: does the evidence reach the writer?

For every question FinanceBench gives the exact passage that answers it. This
runs the production retrieval step and checks whether that passage is among the
8 handed to the writer. If it is not, the writer cannot answer correctly, so
this number is the ceiling on everything downstream.

    pool_recall  evidence is somewhere in the candidates Qdrant returned
    cov@k        share of the evidence text present in the top k passages
    hit@k        cov@k >= 0.5
    num@8        share of the evidence's FIGURES present in the top 8
    mrr          1 / rank of the first passage carrying the evidence
    retention    hit@8 / pool_recall: how much the reranker keeps of what was found

Three reranker arms are scored side by side on the same index: none (Qdrant's
fused order), Cohere, and the local cross-encoder.

Question set: 150 FinanceBench questions -> 127 whose filing exists as SEC HTML
-> 99 whose evidence survives HTML parsing (the rest no retriever could find).

Runs against the LOCAL eval Qdrant. The search queries are written once by the
production planner and cached, so re-runs cost no LLM calls.

    python -m finagent.evaluation.retrieval                 # all 99, all arms
    python -m finagent.evaluation.retrieval --sample 10 --rerankers none,cohere
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

# The eval corpus lives on the local Qdrant, never the served cluster.
os.environ.setdefault("QDRANT_EVAL_URL", "http://localhost:6333")
os.environ.setdefault("FINANCEBENCH_COLLECTION", "sweep_p2500_c600_gemini-embedding-2_hdr_tbl-md")

QUESTIONS = Path("data/us/eval/financebench/data/financebench_open_source.jsonl")
HTML_DIR = Path("data/us/eval/financebench/html")
ELEMENT_CACHE = Path("data/us/eval/financebench/elements")
OUT_JSON = Path("results/retrieval_eval.json")

COHERE = "cohere:rerank-v4.0-pro"
LOCAL = "BAAI/bge-reranker-v2-m3"
ARMS = {"none": "none", "cohere": COHERE, "local": LOCAL}
KS = (5, 8)
HIT_THRESHOLD = 0.5
SHINGLE_WORDS = 10


# --------------------------------------------------------------------------- #
# The evidence metric
# --------------------------------------------------------------------------- #

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_PAREN_NEG = re.compile(r"\(\s*\$?\s*(\d[\d,.]*)\s*\)")       # (1,234) is -1,234
_THOUSANDS = re.compile("(?<=\\d)[,\u00a0\u2009\u202f](?=\\d{3}(?!\\d))")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    """Lowercase letters and digits only. Parsers disagree on formatting (a
    table is `a | b | c` in one and spaces in another), not on content."""
    return _NON_ALNUM.sub(" ", (text or "").lower()).strip()


def shingle_recall(gold: str, parsed_norm: str, n: int = SHINGLE_WORDS) -> float:
    """Share of `gold`'s n-word sequences found in already-normalised text."""
    gold_n = normalize(gold)
    words = gold_n.split()
    if not words:
        return 0.0
    if len(words) < n:
        return 1.0 if gold_n in parsed_norm else 0.0
    shingles = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return sum(1 for s in shingles if s in parsed_norm) / len(shingles)


def coverage(spans: list[str], texts: list[str]) -> float:
    """Mean evidence coverage over a question's evidence spans."""
    if not spans:
        return 0.0
    blob = normalize(" || ".join(texts))
    return statistics.mean(shingle_recall(s, blob) for s in spans)


def figures(text: str) -> set:
    """The distinct numbers in `text`: "1,234", "1 234" and "(1,234)" all match.
    Single digits are ignored; they match by accident in any table."""
    s = _THOUSANDS.sub("", _PAREN_NEG.sub(r" -\1 ", text or ""))
    return {n for n in _NUMBER.findall(s) if len(n.lstrip("-0.")) >= 2}


def numeric_coverage(spans: list[str], texts: list[str]) -> float:
    """Share of the evidence's figures present, ignoring word order. Sees table
    evidence that word sequences miss when a parser reorders cells."""
    if not spans:
        return 0.0
    present = figures(" || ".join(texts))

    def recall(span):
        gold = figures(span)
        return sum(1 for n in gold if n in present) / len(gold) if gold else 1.0

    return statistics.mean(recall(s) for s in spans)


def first_hit_rank(spans: list[str], texts: list[str]) -> int:
    """Rank at which cumulative coverage reaches the threshold; 0 if never."""
    for k in range(1, len(texts) + 1):
        if coverage(spans, texts[:k]) >= HIT_THRESHOLD:
            return k
    return 0


def self_check() -> None:
    """The metric is the whole eval: if it is wrong, every number is."""
    doc = normalize("Total net sales were $394,328 million in 2022, up 8%.")
    assert shingle_recall("Total  NET   sales; were: 394,328 (million) in 2022 — up 8%", doc) == 1.0
    assert shingle_recall("The board declared a quarterly dividend of $0.23", doc) == 0.0
    assert shingle_recall("net sales", doc) == 1.0
    assert coverage(["net sales", "no such text here at all"], [doc]) == 0.5
    assert figures("46,455") == figures("46\u00a0455") == {"46455"}
    assert figures("(1,234)") == {"-1234"} and figures("46,455 | 47,072") == {"46455", "47072"}
    row = "Consolidated Balance Sheets | 46,455 | Total assets | 47,072"
    assert numeric_coverage(["Total assets 46,455 47,072"], [row]) == 1.0
    assert first_hit_rank(["Total net sales were $394,328 million"],
                          ["x", "y", "Total net sales were $394,328 million"]) == 3


# --------------------------------------------------------------------------- #
# Questions
# --------------------------------------------------------------------------- #

_COMPARISON_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|relative to|compared (?:to|with)|higher than|"
    r"lower than|greater than|less than|difference between|more than|change (?:in|from)|"
    r"year[- ]over[- ]year|yoy|growth)\b", re.I)


def question_type(row: dict) -> str:
    """comparison | numeric | narrative, from the question text and FinanceBench's tags."""
    if _COMPARISON_RE.search(str(row.get("question", ""))):
        return "comparison"
    if (row.get("question_type") == "metrics-generated"
            or re.search(r"numerical|logical reasoning", str(row.get("question_reasoning") or ""), re.I)):
        return "numeric"
    return "narrative"


def load_questions(served_only: bool = True) -> list[dict]:
    """FinanceBench questions with a `qtype`. `served_only` keeps those whose
    evidence filings exist as SEC HTML (drops the 8-K / earnings-release ones)."""
    rows = [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]
    out = []
    for r in rows:
        docs = {e.get("doc_name", "") for e in r.get("evidence") or []} or {r.get("doc_name", "")}
        docs.discard("")
        if served_only and not (docs and all((HTML_DIR / f"{d}.htm").exists() for d in docs)):
            continue
        out.append({**r, "qtype": question_type(r)})
    return out


def _install_element_cache() -> None:
    """Parse each filing once: `partition_html` takes up to 17 s per filing."""
    import unstructured.partition.html as uh

    if getattr(uh.partition_html, "_cached", False):
        return
    ELEMENT_CACHE.mkdir(parents=True, exist_ok=True)
    real = uh.partition_html

    def cached(filename=None, **kwargs):
        blob = ELEMENT_CACHE / f"{Path(filename).stem}.pkl"
        if blob.exists():
            return pickle.loads(blob.read_bytes())
        elements = real(filename=filename, **kwargs)
        blob.write_bytes(pickle.dumps(elements))
        return elements

    cached._cached = True
    uh.partition_html = cached


def _document_text(doc_name: str) -> str:
    """Everything the parser recovers from one filing, normalised."""
    from unstructured.partition.html import partition_html

    path = HTML_DIR / f"{doc_name}.htm"
    return normalize(" ".join(e.text or "" for e in partition_html(filename=str(path)))) if path.exists() else ""


@dataclass
class Question:
    id: str
    text: str
    qtype: str
    spans: list[str] = field(default_factory=list)
    retrieval_query: str = ""
    routes: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)


def load_eval_questions() -> tuple[list[Question], int]:
    """The questions retrieval can be scored on, and how many were dropped
    because the parser never recovers their evidence from the filing."""
    _install_element_cache()
    docs_text: dict[str, str] = {}
    kept, dropped = [], 0
    for row in load_questions():
        spans = [e["evidence_text"] for e in row.get("evidence") or [] if e.get("evidence_text")]
        docs = sorted({e["doc_name"] for e in row.get("evidence") or [] if e.get("doc_name")})
        if not spans:
            continue
        for d in docs:
            docs_text.setdefault(d, _document_text(d))
        blob = normalize(" || ".join(docs_text[d] for d in docs))
        if statistics.mean(shingle_recall(s, blob) for s in spans) < HIT_THRESHOLD:
            dropped += 1
            continue
        kept.append(Question(row["financebench_id"], row["question"], row["qtype"], spans))
    return kept, dropped


def attach_plans(questions: list[Question], refresh: bool = False) -> None:
    """Give each question the routes and the search query the production planner
    writes. Cached per planner model: the plan is an input to retrieval, so it
    is held fixed between runs."""
    from tqdm import tqdm

    from finagent.agent import FinAgent, plan
    from finagent.runtime import ROLES

    path = Path(f"results/financebench_plans_{ROLES['planner'][1].replace('/', '-')}.json")
    cache = json.loads(path.read_text()) if path.exists() and not refresh else {}
    todo = [q for q in questions if q.id not in cache]
    agent = FinAgent(collection="unused") if todo else None
    for q in tqdm(todo, desc="planning"):
        state = {"question": q.text, "log": [], "notices": []}
        out = plan.planner(agent, state)
        # Production only writes the search query for a narrative question (the
        # others skip filing search). The eval scores retrieval for every
        # question, so it asks for the query either way.
        rq = out["retrieval_query"] or plan.retrieval_query(agent, state, q.text, ["narrative"])
        cache[q.id] = {"question": q.text, "sub_queries": out["sub_queries"],
                       "query_routes": out["query_routes"], "retrieval_query": rq}
        path.write_text(json.dumps(cache, indent=2))            # resumable
    for q in questions:
        p = cache[q.id]
        q.retrieval_query, q.routes, q.sub_queries = p["retrieval_query"], p["query_routes"], p["sub_queries"]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

class _NoRerank:
    """Keeps the order Qdrant returned: the control arm."""

    @staticmethod
    def predict(pairs):
        return [-i for i in range(len(pairs))]


def _agent_for(arm: str):
    """A production agent that reranks with `arm`."""
    from finagent.agent import FinAgent
    from finagent.config import settings
    from finagent.retrieval import reranker

    if arm == "none":
        reranker._shared["none"] = _NoRerank()
    elif arm.startswith(reranker.COHERE_PREFIX):
        # The bare Cohere adapter: production falls back to the local model on a
        # failure, which would silently blend two rerankers into one row.
        reranker._shared[arm] = reranker.CohereReranker(arm[len(reranker.COHERE_PREFIX):])
    return FinAgent(collection=settings.financebench_collection, reranker_model=arm)


def _pool_texts(agent, q: Question) -> list[str]:
    """The candidate parents for a question's two searches, before reranking."""
    from finagent.retrieval.expansion import expand_query

    texts = []
    for query in dict.fromkeys([q.retrieval_query or q.text, q.text]):
        query = expand_query(query)
        flt = agent.retriever.infer_filter(query)
        pool = agent.retriever.get_pool(query, flt)
        if not pool and flt and (flt.get("years") or flt.get("items")):
            pool = agent.retriever.get_pool(query, {"companies": flt["companies"]})
        texts += [t for t, _ in agent.retriever.collapse_to_parents(pool)]
    return texts


def evaluate(questions: list[Question], arms: list[str]) -> list[dict]:
    """Score each reranker arm by running the production retrieve step."""
    from tqdm import tqdm

    from finagent.agent import retrieve

    rows, pool_hits = [], None
    for arm in arms:
        agent = _agent_for(arm)
        if pool_hits is None:               # the pool is the same for every arm
            pool_hits = [coverage(q.spans, _pool_texts(agent, q)) >= HIT_THRESHOLD
                         for q in tqdm(questions, desc="candidate pools")]
        cov = {k: [] for k in KS}
        num, rr, ms, by_type = [], [], [], {}
        for q in tqdm(questions, desc=f"rerank: {arm}"):
            state = {"question": q.text, "sub_queries": q.sub_queries, "query_routes": q.routes,
                     "retrieval_query": q.retrieval_query, "log": [], "notices": []}
            t0 = time.time()
            texts = [c["text"] for c in retrieve._search(agent, state)]
            ms.append((time.time() - t0) * 1000)
            for k in KS:
                cov[k].append(coverage(q.spans, texts[:k]))
            num.append(numeric_coverage(q.spans, texts[:8]))
            rank = first_hit_rank(q.spans, texts[:8])
            rr.append(1 / rank if rank else 0.0)
            by_type.setdefault(q.qtype, []).append(cov[8][-1] >= HIT_THRESHOLD)
        n = len(questions)
        pool_recall = sum(pool_hits) / n
        hit8 = sum(c >= HIT_THRESHOLD for c in cov[8]) / n
        rows.append({
            "reranker": arm, "n": n, "pool_recall": round(pool_recall, 4),
            **{f"cov@{k}": round(statistics.mean(cov[k]), 4) for k in KS},
            **{f"hit@{k}": round(sum(c >= HIT_THRESHOLD for c in cov[k]) / n, 4) for k in KS},
            "num@8": round(statistics.mean(num), 4), "mrr": round(statistics.mean(rr), 4),
            "retention": round(hit8 / pool_recall, 4) if pool_recall else 0.0,
            "ms_per_question": round(statistics.median(ms)),
            "hit@8_by_type": {t: f"{sum(v)}/{len(v)}" for t, v in sorted(by_type.items())},
        })
    return rows


def write_report(rows: list[dict], meta: dict, sample: bool = False) -> None:
    # A --sample run gets its own files, so it can never be read as the real result.
    out_json = OUT_JSON.with_name("retrieval_eval_sample.json") if sample else OUT_JSON
    out_md = out_json.with_suffix(".md")
    out_json.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    cols = ["reranker", "n", "pool_recall", "cov@5", "hit@5", "cov@8", "hit@8", "num@8",
            "mrr", "retention", "ms_per_question"]
    lines = [
        "# Retrieval evaluation", "",
        f"Collection `{meta['collection']}`, embedder `{meta['embedder']}`, planner "
        f"`{meta['planner']}`. {meta['n']} questions scored; {meta['dropped']} dropped "
        f"because the HTML parser never recovers their evidence.", "",
        "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols),
        *("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows), "",
        "hit@8 by question type:", "",
        *(f"- `{r['reranker']}`: {r['hit@8_by_type']}" for r in rows), "",
    ]
    out_md.write_text("\n".join(lines))
    print(f"\n-> {out_json}\n-> {out_md}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval evaluation on FinanceBench.")
    ap.add_argument("--sample", type=int, help="score only the first N questions")
    ap.add_argument("--rerankers", default="none,cohere,local",
                    help="comma-separated arms: none, cohere, local")
    ap.add_argument("--refresh-plans", action="store_true", help="re-run the planner")
    args = ap.parse_args()

    from finagent.config import settings
    from finagent.runtime import ROLES
    from finagent.vectorstore import DEFAULT_EMBED_MODEL

    self_check()
    questions, dropped = load_eval_questions()
    if args.sample:
        questions = questions[:args.sample]
    attach_plans(questions, refresh=args.refresh_plans)
    rows = evaluate(questions, [ARMS[a.strip()] for a in args.rerankers.split(",")])
    for r in rows:
        print(f"{r['reranker']:28} pool={r['pool_recall']:.3f} hit@5={r['hit@5']:.3f} "
              f"hit@8={r['hit@8']:.3f} num@8={r['num@8']:.3f} mrr={r['mrr']:.3f}")
    write_report(rows, {"collection": settings.financebench_collection,
                        "embedder": DEFAULT_EMBED_MODEL, "planner": ROLES["planner"][1],
                        "n": len(questions), "dropped": dropped}, sample=bool(args.sample))


if __name__ == "__main__":
    main()
