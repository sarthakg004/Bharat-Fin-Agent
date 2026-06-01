"""
RAG service singletons.

We hold one NaiveRAG per market and one AgenticRAG per market, lazily
constructed so importing this module is cheap.

The `agentic` mode uses the **full v4 graph** (`AgenticRAGv4`):
planner → router → hybrid retrieve → grader → rewrite/proceed → table_agent →
**web_search** → synthesize → critic → numeric verification → refusal /
translate-out. Web search uses Tavily when `TAVILY_API_KEY` is set and falls
back to the local `news` Chroma collection otherwise; the web_search node
escalates automatically when text retrieval comes back empty or weakly graded,
so questions about companies not in the corpus (recent IPOs, foreign listings)
hit the web instead of returning "no information".

We default to the small reranker (`bge-reranker-base`) so backend startup is
quick; swap to `bge-reranker-large` once you don't mind the download.
"""

from __future__ import annotations

from threading import Lock
from typing import Optional

# Local imports of the existing RAG code.
from src.agents.naive_rag import NaiveRAG
from src.graph.final_rag import AgenticRAGv4

_COLLECTIONS = {
    "us": "us_filings",
    "india": "india_filings",
}

# Each combination of (market, mode) gets one instance — heavy Chroma + embedding
# + reranker loads are paid once at first use, not per request.
_lock = Lock()
_naive_cache: dict[str, NaiveRAG] = {}
_agentic_cache: dict[str, AgenticRAGv4] = {}


def get_naive(market: str) -> NaiveRAG:
    with _lock:
        if market not in _naive_cache:
            _naive_cache[market] = NaiveRAG(
                collection_name=_COLLECTIONS[market],
                llm_provider="groq",
            )
        return _naive_cache[market]


def get_agentic(market: str) -> AgenticRAGv4:
    with _lock:
        if market not in _agentic_cache:
            _agentic_cache[market] = AgenticRAGv4(
                collection_name=_COLLECTIONS[market],
                market=market,
                provider="groq",
                # Small reranker by default — large is great but downloads 1.3 GB.
                reranker_model="BAAI/bge-reranker-base",
                bm25_top_k=8, dense_top_k=8, final_top_k=5,
                # Keep loops tight so the user isn't waiting forever on bad
                # questions; the agent will refuse rather than spin.
                max_rewrites=2, max_critic_retries=1,
                # These collections are only consulted if the router classifies
                # sub-queries as numeric/external; missing collections just yield
                # empty `table_results` / `web_results` and the run continues.
                table_collection="tables",
                news_collection="news",
                web_top_k=10,        # ~7 trusted + ~3 general after the two-pass split
                table_top_k=3,
            )
        return _agentic_cache[market]


# --------------------------------------------------------------------------- #
# Output normalisation — both modes emit the same shape
# --------------------------------------------------------------------------- #

def _normalise_chunk(text: str, meta: dict, idx: int) -> dict:
    company = meta.get("company") or meta.get("ticker", "?")
    year = str(meta.get("year", "?"))
    page = meta.get("page", "?")
    market = meta.get("market", "")
    doc_kind = "10-K" if market == "us" else "AR"
    citation = f"[{company} {doc_kind} {year}, p. {page}]"
    return {
        "id": idx,
        "text": text,
        "company": company,
        "ticker": meta.get("ticker", ""),
        "year": year,
        "page": page,
        "market": market,
        "source_url": meta.get("source_url", ""),
        "citation": citation,
    }


def run_naive(market: str, question: str, top_k: int = 5,
              company_filter: Optional[list[str]] = None) -> dict:
    """One-shot synchronous call. Filters are honoured client-side after retrieve."""
    rag = get_naive(market)
    if top_k:
        rag.top_k = top_k
    res = rag.answer(question)
    chunks = []
    for i, (text, meta) in enumerate(zip(res["retrieved_chunks"], res["chunk_metadata"])):
        if company_filter:
            co = meta.get("company") or meta.get("ticker", "")
            if co not in company_filter:
                continue
        chunks.append(_normalise_chunk(text, meta, i))
    return {
        "answer": res["answer"],
        "chunks": chunks,
        "metadata": {
            "model": res.get("model", "llama-3.1-8b-instant"),
            "latency": res.get("latency", 0.0),
            "input_tokens": res.get("input_tokens", 0),
            "output_tokens": res.get("output_tokens", 0),
        },
    }


def run_agentic(market: str, question: str, top_k: int = 5,
                company_filter: Optional[list[str]] = None) -> dict:
    """Synchronous v4 agentic run. The graph handles its own loops + refusal.

    Web hits and table-derived computations are appended to the returned
    `chunks` list so the frontend's Citations panel can show them next to the
    text excerpts. The `metadata` carries the rest of the v4-specific run state
    (route classifications, verifier score, refused flag, ...).
    """
    rag = get_agentic(market)
    if top_k:
        rag.final_top_k = top_k
    state = rag.run(question)

    chunks: list[dict] = []
    next_id = 0

    # 1. Text excerpts (filtered by company if requested).
    for c in state.get("retrieved_chunks", []) or []:
        if company_filter:
            co = c.get("company") or c.get("ticker", "")
            if co not in company_filter:
                continue
        chunks.append({
            "id": next_id,
            "text": c.get("text", ""),
            "company": c.get("company", "?"),
            "ticker": c.get("ticker", ""),
            "year": str(c.get("year", "?")),
            "page": c.get("page", "?"),
            "market": market,
            "source_url": "",
            "citation": c.get("source", ""),
            "sub_query": c.get("sub_query", ""),
            "kind": "text",
        })
        next_id += 1

    # 2. Web search hits — appear in the same Citations panel alongside text.
    for h in state.get("web_results", []) or []:
        chunks.append({
            "id": next_id,
            "text": (h.get("content") or "")[:1500],
            "company": h.get("source", "web"),
            "ticker": "",
            "year": h.get("date", "")[:4] or "?",
            "page": "—",
            "market": market,
            "source_url": h.get("url", ""),
            "citation": f"<News: {h.get('title','')[:80]} — {h.get('source','web')}>",
            "sub_query": h.get("sub_query", ""),
            "kind": "web",
        })
        next_id += 1

    # 3. Table-agent computations — one synthetic chunk per numeric result.
    for t in state.get("table_results", []) or []:
        if t.get("error") and not t.get("answer"):
            continue
        used = t.get("tables_used", []) or []
        first = used[0] if used else {}
        chunks.append({
            "id": next_id,
            "text": (
                f"Computed: {t.get('answer','')[:600]}\n"
                f"Code:\n{(t.get('code','') or '')[:600]}"
            ),
            "company": first.get("company", "?"),
            "ticker": "",
            "year": str(first.get("year", "?")),
            "page": first.get("page", "—"),
            "market": market,
            "source_url": "",
            "citation": (
                f"(Table: {first.get('title','?')}, "
                f"{first.get('company','?')} {first.get('year','?')}, "
                f"p. {first.get('page','?')})"
            ),
            "sub_query": t.get("sub_query", ""),
            "kind": "table",
        })
        next_id += 1

    nv = state.get("numeric_verification") or {}

    return {
        "answer": state.get("final_answer") or "",
        "chunks": chunks,
        "metadata": {
            "model": rag.synth_model,
            "latency": 0.0,                          # filled in by main.py wrapper
            "language": state.get("language"),
            "sub_queries": state.get("sub_queries", []),
            "query_routes": state.get("query_routes", []),
            "grading_score": state.get("grading_score"),
            "avg_grade": state.get("avg_grade"),
            "rewrite_iterations": state.get("iteration_count", 0),
            "critic_iterations": state.get("critic_iterations", 0),
            "needs_retry": state.get("needs_retry"),
            "low_confidence": state.get("low_confidence"),
            "refused": state.get("refused", False),
            "numeric_verification_score": nv.get("score"),
            "unverified_count": len(nv.get("unverified", [])) if isinstance(nv, dict) else 0,
            "web_hits": len(state.get("web_results", []) or []),
            "table_computations": len(state.get("table_results", []) or []),
            "citations": state.get("citations", []),
        },
    }


def health() -> dict:
    # We don't probe the LLM (would burn quota). Just confirm imports + collections.
    return {
        "status": "ok",
        "collections": list(_COLLECTIONS.values()),
        "configs": ["naive", "agentic"],
    }
