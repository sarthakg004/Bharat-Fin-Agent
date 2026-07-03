"""
RAG service singletons.

Single agentic pipeline (`AgenticRAGv4`): planner(+routes) → router → hybrid
retrieve → grader → rewrite/proceed → xbrl → calculator → table_agent →
market_data (yfinance) ∥ web_search ∥ edgar_search → synthesize → critic →
numeric verification → confidence gate.

Web search uses Tavily when `TAVILY_API_KEY` is set; web_search escalates
automatically when text retrieval comes back empty or weakly graded, so
questions about companies not in the corpus hit the web instead of returning
"no information".

We default to the small reranker (`bge-reranker-base`) so backend startup is
quick; swap to `bge-reranker-large` once you don't mind the download.
"""

from __future__ import annotations

import os
import time
from threading import Lock
from typing import Callable, Optional

from finagent.graph import AgenticRAGv4
from finagent.config import settings

# Each (provider, synth_model, collection) combo gets its own instance. Most
# users will hit one of two cache keys (server-default Groq + their picked
# OpenAI/Anthropic/Gemini override) so the cache stays small.
_lock = Lock()
_agentic_cache: dict[tuple, AgenticRAGv4] = {}


def _build_agent(provider: str, synth_model: Optional[str],
                 api_key: Optional[str],
                 collection: Optional[str] = None) -> AgenticRAGv4:
    # Production serves `us_filings` (+ live SEC fetch for anything missing). The
    # FinanceBench eval overrides this to `financebench_eval` — the dedicated,
    # pre-ingested collection that actually contains the benchmark's filings at
    # the historical years the questions ask about. Without the override the eval
    # searches recent-only us_filings, finds nothing, and falls through to
    # web-search noise (the root cause of ~⅓ of "no information" non-answers).
    coll = collection or "us_filings"
    return AgenticRAGv4(
        collection_name=coll,
        # US-only active retrieval. Non-US / non-corpus questions get
        # empty/weak filing retrieval and escalate to web_search automatically.
        collections=[coll],
        provider=provider,
        synth_model=synth_model,                # None → AgenticRAG picks per-provider default
        api_key=api_key,                        # None → reads env via build_llm()
        # Reranker is env-configurable so we can A/B smaller models without a
        # code change (default kept at bge-reranker-base). The image bakes
        # whatever this resolves to at build time.
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        # Wider first-stage pool (was 8/8) so narrative/MD&A prose — where
        # first-stage recall was weakest (financebench narrative recall ~0.21) —
        # has more chances to surface before the reranker. The wider pool is
        # then trimmed back to a tight, globally-best set by `retrieve_cap`, so
        # the synthesizer still sees few high-precision passages, not 25 noisy ones.
        bm25_top_k=12, dense_top_k=12, final_top_k=5,
        retrieve_cap=8,
        max_rewrites=2, max_critic_retries=1,
        # Cloud: PERSIST_DYNAMIC_FETCH=false → fetched filings are used in-memory
        # for the session and never grow the baked index.
        persist_fetch=settings.persist_dynamic_fetch,
        table_collection="tables",
        web_top_k=10,
        table_top_k=3,
    )


def get_agentic(provider: str = "groq",
                synth_model: Optional[str] = None,
                api_key: Optional[str] = None,
                collection: Optional[str] = None) -> AgenticRAGv4:
    """Return a singleton AgenticRAGv4 for (provider, synth_model, collection).

    User-supplied keys are NOT used as part of the cache key — multiple users
    with the same provider + model share an instance. The instance's
    `self.api_key` is overwritten per-request below so the right key reaches
    the LLM client. `collection` is part of the key so the eval's
    `financebench_eval` instance never collides with the served `us_filings` one.
    """
    key = (provider, synth_model or "_default_", collection or "us_filings")
    with _lock:
        if key not in _agentic_cache:
            _agentic_cache[key] = _build_agent(provider, synth_model, api_key,
                                               collection=collection)
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

def _count_kinds(evidence: list[dict]) -> dict:
    """Per-source-kind tally of normalised evidence items, e.g. {'xbrl': 2, 'filing': 5}."""
    counts: dict[str, int] = {}
    for e in evidence:
        k = e.get("kind", "?")
        counts[k] = counts.get(k, 0) + 1
    return counts


def _sum_usage(usage_cb) -> dict:
    """Flatten a UsageMetadataCallbackHandler into total input/output tokens.

    Its `usage_metadata` is keyed by model name → {input_tokens, output_tokens,
    total_tokens, ...}; we sum across every model the run touched."""
    if usage_cb is None:
        return {}
    per_model = getattr(usage_cb, "usage_metadata", {}) or {}
    inp = sum((m.get("input_tokens") or 0) for m in per_model.values())
    out = sum((m.get("output_tokens") or 0) for m in per_model.values())
    if not (inp or out):
        return {}
    return {"input_tokens": inp, "output_tokens": out,
            "total_tokens": inp + out, "by_model": per_model}


def _tool_health(state: dict) -> dict:
    """Per-lane success/failure tally (#11 / #14): how many calls each tool
    lane made and how many succeeded, so a degraded lane is visible at a glance."""
    def tally(items, ok_key="ok"):
        items = items or []
        ok = sum(1 for x in items if isinstance(x, dict) and x.get(ok_key))
        return {"calls": len(items), "ok": ok, "failed": len(items) - ok}

    tables = state.get("table_results", []) or []
    return {
        "xbrl": tally(state.get("xbrl_facts")),
        "calc": tally(state.get("calc_results")),
        # tables carry an `error` instead of an `ok` flag.
        "table": {"calls": len(tables),
                  "ok": sum(1 for t in tables if not t.get("error")),
                  "failed": sum(1 for t in tables if t.get("error"))},
        "market": tally(state.get("market_data")),
        "edgar": tally(state.get("edgar_results")),
        # web hits have no ok flag — count presence.
        "web": {"calls": len(state.get("web_results", []) or []),
                "ok": len(state.get("web_results", []) or []), "failed": 0},
    }


def _retrieval_status(state: dict, max_attempts: int, grade_threshold: float) -> dict:
    """Retrieval-loop status (#7): how many rewrite attempts ran vs the cap, the
    retrieval confidence (mean grade), and whether the loop was EXHAUSTED without
    reaching the quality bar. The actual refuse decision is NOT made here — a
    numeric/market question legitimately skips retrieval — it's deferred to the
    confidence gate, which weighs ALL evidence, not just retrieval."""
    attempts = state.get("iteration_count", 0)
    avg = state.get("avg_grade")
    exhausted = attempts >= max_attempts and (avg is None or avg < grade_threshold)
    return {
        "attempts": attempts,
        "max_attempts": max_attempts,
        "avg_grade": avg,
        "grade_threshold": grade_threshold,
        "exhausted": bool(exhausted),
        "refusal_handled_by": "confidence_gate",
    }


def _build_audit(state: dict, retrieval: Optional[dict] = None) -> dict:
    """Audit trail (#13): a single self-contained record of HOW the answer was
    produced — sources used, calculations performed, verification status, and
    the confidence score. Assembled as a read-only VIEW over the final state at
    response time, so it never touches the answer-generation path.
    """
    ev = state.get("evidence") or []
    calc = state.get("calc_results") or []
    nv = state.get("numeric_verification") or {}
    vr = state.get("verification_report") or {}
    # Graceful-degradation summary (#14): lanes that ran but returned nothing
    # usable (all calls failed/missed). The run still completes — every lane
    # try/excepts and the agent answers from whatever else grounded — but we
    # record which lanes degraded so the answer's reliability is transparent.
    th = _tool_health(state)
    degraded_lanes = [k for k, v in th.items() if v["calls"] > 0 and v["ok"] == 0]

    return {
        "status": state.get("status"),
        "routes": state.get("query_routes") or [],
        # Retrieval-loop status (#7): attempts vs cap + retrieval confidence.
        "retrieval": retrieval or {},
        # Failure recovery (#14): did any lane degrade, and which.
        "degraded": {"is_degraded": bool(degraded_lanes), "lanes": degraded_lanes,
                     "tool_health": th},
        # Sources used — every normalised evidence item with its provenance.
        "sources_used": [
            {"kind": e.get("kind"), "source": e.get("source"),
             "citation": e.get("citation"), "confidence": e.get("confidence")}
            for e in ev
        ],
        # Calculations performed — auditable down to the exact XBRL inputs.
        "calculations": [
            {"metric": r.get("metric"), "ticker": r.get("ticker"),
             "formula": r.get("formula") or r.get("source"),
             "result": r.get("value_str", r.get("value")),
             "inputs": [
                 {"concept": i.get("concept"), "value": i.get("value_str"),
                  "period": i.get("fy")}
                 for i in (r.get("inputs") or []) if isinstance(i, dict)
             ]}
            for r in calc
        ],
        # Verification status — numeric grounding + the cross-source / unit /
        # citation report.
        "verification": {
            "numeric_score": nv.get("score"),
            "figures_total": nv.get("numbers_total"),
            "figures_grounded": nv.get("numbers_grounded"),
            "ungrounded_figures": [u.get("number") for u in (nv.get("unverified") or [])],
            "cross_source": vr.get("cross_source", {}),
            "units": vr.get("units", {}),
            "sources": vr.get("sources", {}),
        },
        # Confidence score + the sub-scores that produced it.
        "confidence": {
            "score": state.get("confidence"),
            "band": state.get("confidence_band"),
            "sub_scores": {
                "retrieval": state.get("retrieval_score"),
                "verification": state.get("verification_score"),
                "citation": state.get("citation_score"),
                "critic": state.get("critic_score"),
            },
        },
    }


def _step_detail(node: str, delta: dict) -> Optional[str]:
    """One short human-readable outcome line for a finished graph node, built
    from the partial state the node returned — e.g. "12 passages", "2 exact
    figures", "9/9 figures grounded". None = nothing worth showing."""
    try:
        if node == "planner":
            n = len(delta.get("sub_queries") or [])
            return f"{n} sub-quer{'y' if n == 1 else 'ies'}" if n else None
        if node == "router":
            routes = delta.get("query_routes") or []
            if not routes:
                return None
            counts: dict[str, int] = {}
            for r in routes:
                counts[r] = counts.get(r, 0) + 1
            return " · ".join(f"{k}×{v}" if v > 1 else k
                              for k, v in counts.items())
        if node == "fetch_filing":
            fs = delta.get("fetch_status") or {}
            if fs.get("status") == "fetched":
                n = fs.get("chunks_added") or fs.get("chunks_fetched")
                return f"fetched latest 10-K ({n} chunks)" if n else "fetched latest 10-K"
            if fs.get("decision") == "already_indexed":
                return "already in the index"
            return None
        if node == "retrieve":
            n = len(delta.get("retrieved_chunks") or [])
            return f"{n} passages" if n else "no relevant passages"
        if node == "grader":
            kept = len(delta.get("retrieved_chunks") or [])
            avg = delta.get("avg_grade")
            if avg is None:
                return None
            return f"kept {kept} · avg grade {avg:g}/5"
        if node == "rewrite":
            return "query refined"
        if node == "xbrl":
            n = len(delta.get("xbrl_facts") or [])
            return f"{n} exact figure{'s' if n != 1 else ''} from SEC XBRL" if n else None
        if node == "calculator":
            n = len(delta.get("calc_results") or [])
            return f"{n} derived metric{'s' if n != 1 else ''}" if n else None
        if node == "table_agent":
            n = sum(1 for t in delta.get("table_results") or []
                    if t.get("answer") and not t.get("error"))
            return f"{n} table computation{'s' if n != 1 else ''}" if n else None
        if node == "market_data":
            n = sum(1 for m in delta.get("market_data") or [] if m.get("ok"))
            chart = " + chart" if delta.get("charts") else ""
            return f"{n} market call{'s' if n != 1 else ''}{chart}" if n else None
        if node == "web_search":
            n = len(delta.get("web_results") or [])
            return f"{n} web result{'s' if n != 1 else ''}" if n else None
        if node == "edgar_search":
            n = sum(len(r.get("companies") or [])
                    for r in delta.get("edgar_results") or [])
            return f"{n} compan{'y' if n == 1 else 'ies'} matched" if n else None
        if node == "evidence_builder":
            n = len(delta.get("evidence") or [])
            return f"{n} evidence items" if n else None
        if node == "synthesize":
            words = len((delta.get("draft_answer") or "").split())
            return f"draft written ({words} words)" if words else None
        if node == "critic":
            gs = delta.get("grading_score")
            return f"{gs:.0%} of claims supported" if isinstance(gs, (int, float)) else None
        if node == "verify_numbers":
            nv = delta.get("numeric_verification") or {}
            total = nv.get("numbers_total") or 0
            if not total:
                return None
            return f"{nv.get('numbers_grounded', 0)}/{total} figures grounded"
        if node == "confidence":
            c = delta.get("confidence")
            return f"{c:.0%} confidence" if isinstance(c, (int, float)) else None
    except Exception:
        return None      # detail is decoration — never break the stream over it
    return None


def _normalise_chunk(text: str, meta: dict, idx: int) -> dict:
    company = meta.get("company") or meta.get("ticker", "?")
    year = str(meta.get("year", "?"))
    page = meta.get("page", "?")
    citation = f"[{company} 10-K {year}, p. {page}]"
    return {
        "id": idx,
        "text": text,
        "company": company,
        "ticker": meta.get("ticker", ""),
        "year": year,
        "page": page,
        "source_url": meta.get("source_url", ""),
        "citation": citation,
    }


def run_agentic(question: str, top_k: int = 5,
                company_filter: Optional[list[str]] = None,
                chat_history: Optional[list[dict]] = None,
                provider: str = "groq",
                synth_model: Optional[str] = None,
                api_key: Optional[str] = None,
                collection: Optional[str] = None,
                on_step: Optional[Callable[[str], None]] = None,
                on_step_done: Optional[Callable[[str, Optional[str]], None]] = None) -> dict:
    """Synchronous v4 agentic run with optional conversation memory + per-request
    LLM overrides.

    `provider` / `synth_model` / `api_key` come from the request's
    `provider_config` (set by the Settings UI). They override the server's
    default Groq config for THIS turn only. Keys are never persisted.

    `chat_history` is the last few (role, content) turns of the active chat —
    the agent uses it to resolve pronouns and follow-ups.

    `on_step(node_name)` — optional callback invoked as each graph node STARTS
    (langgraph "tasks" stream events), so the API layer can stream live
    "thinking" progress to the UI. `on_step_done(node_name, detail)` fires as
    each node finishes, with a short outcome line ("12 passages", "2 exact
    figures"). When both are omitted we fall back to a single blocking `invoke`.
    """
    rag = get_agentic(provider=provider, synth_model=synth_model,
                      api_key=api_key, collection=collection)
    if top_k:
        rag.final_top_k = top_k

    # `chat_history` lives directly on AgentState (TypedDict, total=False) so
    # nodes can read it without changing graph signatures.
    initial_state: dict[str, object] = {
        "question": question,
        "iteration_count": 0, "errors": [],
        "table_results": [], "web_results": [], "xbrl_facts": [], "calc_results": [],
        "fetch_status": {}, "edgar_results": [], "fetched_chunks": [],
    }
    if chat_history:
        initial_state["chat_history"] = chat_history

    # #11 Observability (complements LangSmith traces with in-app metrics):
    # a token-usage callback aggregates input/output tokens across EVERY model
    # call in the run, across providers, with no extra deps. Node latencies are
    # timed from the update stream below.
    usage_cb = None
    try:
        from langchain_core.callbacks import UsageMetadataCallbackHandler
        usage_cb = UsageMetadataCallbackHandler()
    except Exception:
        usage_cb = None

    # A hard recursion cap so a pathological retrieve→grade→rewrite or
    # critic→retrieve loop can never spin forever — it terminates with whatever
    # we have instead of hanging the request.
    config: dict = {"recursion_limit": 50}
    if usage_cb is not None:
        config["callbacks"] = [usage_cb]

    node_latencies: dict[str, float] = {}
    if on_step is None and on_step_done is None:
        state = rag.graph.invoke(initial_state, config=config)
    else:
        # "tasks" events fire when a node STARTS (accurate current-stage label
        # — "updates" only fires on completion, so the old label lagged one
        # node), "updates" carries each node's delta (for the outcome detail +
        # latency attribution), and "values" keeps the final accumulated state.
        state: dict = {}
        last_ts = time.time()
        for mode, data in rag.graph.stream(
            initial_state, stream_mode=["updates", "values", "tasks"], config=config
        ):
            if mode == "values":
                state = data
            elif mode == "tasks" and isinstance(data, dict):
                # A start event has no "result"/"error" yet.
                if "result" not in data and "error" not in data and on_step:
                    on_step(data.get("name", ""))
            elif mode == "updates" and isinstance(data, dict):
                now = time.time()
                # Attribute the elapsed wall-clock since the previous update to
                # the node(s) that just produced one (rough but useful; parallel
                # lanes that land together share the interval).
                for node_name, delta in data.items():
                    node_latencies[node_name] = round(
                        node_latencies.get(node_name, 0.0) + (now - last_ts), 3)
                    if on_step_done:
                        on_step_done(node_name, _step_detail(node_name, delta or {}))
                last_ts = now

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
            "text": (f"{f.get('concept','')} ({f.get('period_label','FY'+str(f.get('fy','?')))}) = "
                     f"{f.get('value_str','')}\n"
                     f"Exact figure as filed — us-gaap:{f.get('tag','')}, "
                     f"{f.get('form','')} {f.get('end','')}."),
            "company": f.get("entity", f.get("ticker", "?")),
            "ticker": f.get("ticker", ""),
            "year": str(f.get("fy", "?")),
            "page": "—",
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
            # #11 Observability: real token usage (was always null), per-node
            # latency, and per-lane tool health. LangSmith has the full trace;
            # these put the headline numbers in the answer's own metadata/footer.
            **_sum_usage(usage_cb),
            "node_latencies": node_latencies,
            "tool_health": _tool_health(state),
            "sub_queries": state.get("sub_queries", []),
            "query_routes": state.get("query_routes", []),
            "grading_score": state.get("grading_score"),
            "avg_grade": state.get("avg_grade"),
            "rewrite_iterations": state.get("iteration_count", 0),
            "critic_iterations": state.get("critic_iterations", 0),
            "needs_retry": state.get("needs_retry"),
            "low_confidence": state.get("low_confidence"),
            "refused": state.get("refused", False),
            # Confidence framework (#8/9): blended score, routing band, and the
            # four sub-scores that fed it (each null when not applicable).
            "confidence": state.get("confidence"),
            "confidence_band": state.get("confidence_band"),
            "answer_status": state.get("status"),
            "confidence_scores": {
                "retrieval": state.get("retrieval_score"),
                "verification": state.get("verification_score"),
                "citation": state.get("citation_score"),
                "critic": state.get("critic_score"),
            },
            # Normalised evidence (#3): one common shape across all lanes, capped
            # for payload size. Per-kind counts give a quick provenance summary.
            "evidence_count": len(state.get("evidence", []) or []),
            "evidence_kinds": _count_kinds(state.get("evidence", []) or []),
            "evidence": (state.get("evidence", []) or [])[:60],
            # Retrieval-loop status (#7): attempts vs cap, retrieval confidence,
            # whether exhausted (refusal is deferred to the confidence gate).
            "retrieval": _retrieval_status(state, rag.max_rewrites, rag.grade_threshold),
            # Audit trail (#13): full provenance of this answer.
            "audit": _build_audit(
                state, _retrieval_status(state, rag.max_rewrites, rag.grade_threshold)),
            "numeric_verification_score": nv.get("score"),
            # #5 verification report: cross-source corroboration, unit-shift
            # flags, and citation coverage of numeric evidence.
            "verification_report": state.get("verification_report") or {},
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
        "collections": ["us_filings"],
        "configs": ["agentic"],
    }
