"""
RAG service singletons.

Single agentic pipeline (`AgenticRAGv4`): planner → router → hybrid retrieve →
grader → rewrite/proceed → table_agent → market_data (yfinance) → web_search
→ synthesize → critic → numeric verification → refusal / translate-out.

Web search uses Tavily when `TAVILY_API_KEY` is set and falls back to the
local `news` Chroma collection otherwise; web_search escalates automatically
when text retrieval comes back empty or weakly graded, so questions about
companies not in the corpus hit the web instead of returning "no information".

We default to the small reranker (`bge-reranker-base`) so backend startup is
quick; swap to `bge-reranker-large` once you don't mind the download.
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Callable, Optional

# Canonical agent namespace (the class chain itself lives in finagent.graph.*,
# re-exported through finagent.agents during the layout restructure).
from finagent.agents import AgenticRAGv4

# Phase 1 (US-only): active retrieval searches the US filings corpus exclusively.
# `india_filings` was dropped from the live pool — the eval is US-only and any
# non-US / non-corpus company now falls through to the web-search branch (the
# agent's web_search_node escalates automatically on empty/weak retrieval) with
# a graceful "searched the web instead" answer rather than off-market noise.
_COLLECTIONS = {
    "us": "us_filings",
}

# Each (market, provider, synth_model) combo gets its own instance. Most
# users will hit one of two cache keys (server-default Groq + their picked
# OpenAI/Anthropic/Gemini override) so the cache stays small.
_lock = Lock()
_agentic_cache: dict[tuple, AgenticRAGv4] = {}


def _build_agent(market: str, provider: str, synth_model: Optional[str],
                 api_key: Optional[str]) -> AgenticRAGv4:
    return AgenticRAGv4(
        collection_name="us_filings",
        # Phase 1: US-only active retrieval. Non-US / non-corpus questions get
        # empty/weak filing retrieval and escalate to web_search automatically.
        collections=["us_filings"],
        market="us",
        provider=provider,
        synth_model=synth_model,                # None → AgenticRAG picks per-provider default
        api_key=api_key,                        # None → reads env via build_llm()
        # Reranker is env-configurable so we can A/B smaller models without a
        # code change (default kept at bge-reranker-base). The image bakes
        # whatever this resolves to at build time.
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        bm25_top_k=8, dense_top_k=8, final_top_k=5,
        max_rewrites=2, max_critic_retries=1,
        table_collection="tables",
        news_collection="news",
        web_top_k=10,
        table_top_k=3,
    )


def get_agentic(market: str, provider: str = "groq",
                synth_model: Optional[str] = None,
                api_key: Optional[str] = None) -> AgenticRAGv4:
    """Return a singleton AgenticRAGv4 for (market, provider, synth_model).

    User-supplied keys are NOT used as part of the cache key — multiple users
    with the same provider + model share an instance. The instance's
    `self.api_key` is overwritten per-request below so the right key reaches
    the LLM client.
    """
    key = (market, provider, synth_model or "_default_")
    with _lock:
        if key not in _agentic_cache:
            _agentic_cache[key] = _build_agent(market, provider, synth_model, api_key)
        agent = _agentic_cache[key]
        # If the caller supplied an api_key, propagate it now and clear the
        # provider's LLM cache so the next call rebuilds with the new key.
        if api_key and agent.api_key != api_key:
            agent.api_key = api_key
            agent._llms.clear()
        return agent


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


def run_agentic(market: str, question: str, top_k: int = 5,
                company_filter: Optional[list[str]] = None,
                chat_history: Optional[list[dict]] = None,
                provider: str = "groq",
                synth_model: Optional[str] = None,
                api_key: Optional[str] = None,
                on_step: Optional[Callable[[str], None]] = None) -> dict:
    """Synchronous v4 agentic run with optional conversation memory + per-request
    LLM overrides.

    `provider` / `synth_model` / `api_key` come from the request's
    `provider_config` (set by the Settings UI). They override the server's
    default Groq config for THIS turn only. Keys are never persisted.

    `chat_history` is the last few (role, content) turns of the active chat —
    the agent uses it to resolve pronouns and follow-ups.

    `on_step(node_name)` — optional callback invoked as each graph node runs, so
    the API layer can stream live "thinking" progress to the UI. When omitted we
    fall back to a single blocking `invoke`.
    """
    rag = get_agentic(market, provider=provider, synth_model=synth_model, api_key=api_key)
    if top_k:
        rag.final_top_k = top_k

    # `chat_history` lives directly on AgentState (TypedDict, total=False) so
    # nodes can read it without changing graph signatures.
    initial_state: dict[str, object] = {
        "question": question,
        "iteration_count": 0, "errors": [],
        "table_results": [], "web_results": [], "xbrl_facts": [], "calc_results": [],
        "fetch_status": {}, "edgar_results": [],
    }
    if chat_history:
        initial_state["chat_history"] = chat_history

    # A hard recursion cap so a pathological retrieve→grade→rewrite or
    # critic→retrieve loop can never spin forever — it terminates with whatever
    # we have instead of hanging the request.
    config = {"recursion_limit": 50}

    if on_step is None:
        state = rag.graph.invoke(initial_state, config=config)
    else:
        # stream_mode=["updates","values"] yields ("updates", {node: delta})
        # for live progress and ("values", full_state) so we keep the final
        # accumulated state to build the response from.
        state: dict = {}
        for mode, data in rag.graph.stream(
            initial_state, stream_mode=["updates", "values"], config=config
        ):
            if mode == "values":
                state = data
            elif mode == "updates" and isinstance(data, dict):
                for node_name in data:
                    on_step(node_name)

    chunks: list[dict] = []
    next_id = 0

    # 0. XBRL facts FIRST — exact structured figures from SEC company-facts.
    # Listed first so the `[N]` numbering aligns with the synthesizer, which
    # presents XBRL facts as the leading authoritative evidence (Phase 3).
    for f in state.get("xbrl_facts", []) or []:
        if not f.get("ok"):
            continue
        chunks.append({
            "id": next_id,
            "text": (f"{f.get('concept','')} (FY{f.get('fy','?')}) = "
                     f"{f.get('value_str','')}\n"
                     f"Exact figure as filed — us-gaap:{f.get('tag','')}, "
                     f"{f.get('form','')} {f.get('end','')}."),
            "company": f.get("entity", f.get("ticker", "?")),
            "ticker": f.get("ticker", ""),
            "year": str(f.get("fy", "?")),
            "page": "—",
            "market": market,
            "source_url": "",
            "citation": f.get("source", ""),
            "sub_query": f.get("sub_query", ""),
            "kind": "xbrl",
        })
        next_id += 1

    # 0b. Derived metrics computed over XBRL inputs (margins, ratios, growth,
    # CAGR, trends) — listed right after the raw XBRL facts they're built from.
    for r in state.get("calc_results", []) or []:
        if not r.get("ok"):
            continue
        from finagent.graph.agent import AgenticRAGv4 as _Agent
        chunks.append({
            "id": next_id,
            "text": _Agent._format_calc_result(r),
            "company": r.get("ticker", "?"),
            "ticker": r.get("ticker", ""),
            "year": str(r.get("fy", r.get("end_period", "?"))),
            "page": "—",
            "market": market,
            "source_url": "",
            "citation": f"(Computed: {str(r.get('metric','')).replace('_',' ')} from XBRL)",
            "sub_query": r.get("sub_query", ""),
            "kind": "calc",
        })
        next_id += 1

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

    # 1c. EDGAR cross-document search — one chunk per matching company so the
    # user can click straight through to each filing.
    for r in state.get("edgar_results", []) or []:
        if not r.get("ok"):
            continue
        for comp in r.get("companies", []):
            chunks.append({
                "id": next_id,
                "text": (f"{comp.get('company','?')}"
                         f"{(' (' + comp['ticker'] + ')') if comp.get('ticker') else ''} — "
                         f"{comp.get('form','')} filed {comp.get('date','')} matches "
                         f"{r.get('query','')}."),
                "company": comp.get("company", "?"),
                "ticker": comp.get("ticker", ""),
                "year": str(comp.get("date", ""))[:4] or "?",
                "page": "—",
                "market": market,
                "source_url": comp.get("url", ""),
                "citation": f"<EDGAR: {comp.get('company','?')} {comp.get('form','')} {comp.get('date','')}>",
                "sub_query": r.get("sub_query", ""),
                "kind": "edgar",
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

    # 4. Live market data (yfinance tools) — one synthetic chunk per
    # successful call so the user can inspect the raw numbers next to the
    # other evidence in the right-hand panel.
    for m in state.get("market_data", []) or []:
        if not m.get("ok"):
            continue
        data = m.get("data") or {}
        tool = m.get("tool", "")
        sym = (
            data.get("symbol")
            or (data.get("summary") or {}).get("symbol")
            or "—"
        )
        chunks.append({
            "id": next_id,
            "text": str(data)[:1500],
            "company": sym,
            "ticker": sym,
            "year": "—",
            "page": "—",
            "market": market,
            "source_url": "",
            "citation": f"<Market: yfinance.{tool} {sym}>",
            "sub_query": m.get("sub_query", ""),
            "kind": "market",
        })
        next_id += 1

    nv = state.get("numeric_verification") or {}

    return {
        "answer": state.get("final_answer") or "",
        "chunks": chunks,
        # `charts` rides on its own channel — the frontend attaches them to
        # the assistant message and renders inline via lightweight-charts.
        "charts": list(state.get("charts", []) or []),
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
            "numbers_total": nv.get("numbers_total", 0) if isinstance(nv, dict) else 0,
            "hallucination_rate": nv.get("hallucination_rate", 0.0) if isinstance(nv, dict) else 0.0,
            "ungrounded_figures": [u.get("number") for u in nv.get("unverified", [])] if isinstance(nv, dict) else [],
            "web_hits": len(state.get("web_results", []) or []),
            "edgar_companies": sum(len(r.get("companies", []))
                                   for r in (state.get("edgar_results", []) or [])),
            "xbrl_facts": len(state.get("xbrl_facts", []) or []),
            "calc_results": len(state.get("calc_results", []) or []),
            "fetch_status": state.get("fetch_status") or {},
            "table_computations": len(state.get("table_results", []) or []),
            "market_calls": len(state.get("market_data", []) or []),
            "citations": state.get("citations", []),
        },
    }


def health() -> dict:
    # We don't probe the LLM (would burn quota). Just confirm imports + collections.
    return {
        "status": "ok",
        "collections": list(_COLLECTIONS.values()),
        "configs": ["agentic"],
    }
