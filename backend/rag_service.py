"""
RAG service singletons.

We hold one NaiveRAG per market and one AgenticRAG per market, lazily
constructed so importing this module is cheap. The agentic side uses
`AgenticRAGv2` (corrective: hybrid retrieve + grader + rewrite) with the
small reranker to keep startup fast — swap to `bge-reranker-large` when
you're ready to wait for the download.
"""

from __future__ import annotations

from threading import Lock
from typing import Optional

# Local imports of the existing RAG code.
from src.agents.naive_rag import NaiveRAG
from src.graph.corrective_rag import AgenticRAGv2

_COLLECTIONS = {
    "us": "us_filings",
    "india": "india_filings",
}

# Each combination of (market, mode) gets one instance — heavy Chroma + embedding
# + reranker loads are paid once at first use, not per request.
_lock = Lock()
_naive_cache: dict[str, NaiveRAG] = {}
_agentic_cache: dict[str, AgenticRAGv2] = {}


def get_naive(market: str) -> NaiveRAG:
    with _lock:
        if market not in _naive_cache:
            _naive_cache[market] = NaiveRAG(
                collection_name=_COLLECTIONS[market],
                llm_provider="groq",
            )
        return _naive_cache[market]


def get_agentic(market: str) -> AgenticRAGv2:
    with _lock:
        if market not in _agentic_cache:
            _agentic_cache[market] = AgenticRAGv2(
                collection_name=_COLLECTIONS[market],
                market=market,
                provider="groq",
                # Small reranker by default — large is great but downloads 1.3GB.
                reranker_model="BAAI/bge-reranker-base",
                bm25_top_k=8, dense_top_k=8, final_top_k=5,
                max_rewrites=2, max_critic_retries=1,
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
    """Synchronous agentic run. The corrective graph handles its own loops."""
    rag = get_agentic(market)
    if top_k:
        rag.final_top_k = top_k
    state = rag.run(question)

    chunks = []
    for i, c in enumerate(state.get("retrieved_chunks", []) or []):
        if company_filter:
            co = c.get("company") or c.get("ticker", "")
            if co not in company_filter:
                continue
        chunks.append({
            "id": i,
            "text": c.get("text", ""),
            "company": c.get("company", "?"),
            "ticker": c.get("ticker", ""),
            "year": str(c.get("year", "?")),
            "page": c.get("page", "?"),
            "market": market,
            "source_url": "",
            "citation": c.get("source", ""),
            "sub_query": c.get("sub_query", ""),
        })

    return {
        "answer": state.get("final_answer") or "",
        "chunks": chunks,
        "metadata": {
            "model": rag.synth_model,
            "latency": 0.0,           # filled in by main.py wrapper
            "sub_queries": state.get("sub_queries", []),
            "grading_score": state.get("grading_score"),
            "avg_grade": state.get("avg_grade"),
            "rewrite_iterations": state.get("iteration_count", 0),
            "critic_iterations": state.get("critic_iterations", 0),
            "needs_retry": state.get("needs_retry"),
            "low_confidence": state.get("low_confidence"),
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
