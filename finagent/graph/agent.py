"""
agent.py  ·  finagent/graph/agent.py

Full agentic RAG (v4) — the deployed agent. Adds to v3:

  * **Fused plan+route** (inherited from v3) feeding a query-type dispatcher:
    purely numeric/market/cross-document/external questions skip retrieval
    and go straight to the tool chain.
  * **Structured numeric lanes** — XBRL facts (exact filed figures), a
    deterministic calculator over them (margins/ratios/growth/CAGR), and the
    table agent as a fallback.
  * **Web search** — `web_search_node` covers questions the corpus can't (post
    cut-off events, latest news) via Tavily (`TAVILY_API_KEY`). Escalates
    automatically when retrieval comes back empty/weak or the draft admits it
    can't answer.
  * **Numeric verification** — every figure in the draft is deterministically
    grounded against the evidence (with an LLM rescue pass only when figures
    fail to ground). Ungrounded figures re-route, then refuse.
  * **Confidence gate** — retrieval/verification/citation/critic sub-scores
    blend into one confidence; low bands answer with an explicit caveat.

Graph:

    START → planner(+routes) → router → {fetch_filing → retrieve → grader → rewrite↺ | xbrl}
          → xbrl → calculator → table_agent → (market_data ∥ web_search ∥ edgar_search)
          → evidence_builder → synthesize → critic → verify_numbers
          → confidence → {answer | answer_with_warning | low_confidence} → END
                       ↘ refuse → END

The node implementations live in `finagent/graph/nodes/` as topical mixins —
fetch (SEC fetch + retrieval), numeric (XBRL/calculator/table), external
(market/web/EDGAR), synthesis (evidence + drafting + critic), verification
(figure grounding + confidence gate). This module owns the constructor, the
lane resources, the routing functions, and the graph wiring.
"""

from __future__ import annotations

import argparse
from typing import Optional

from finagent.graph.full import AgenticRAGv3
from finagent.runtime import RuntimeContext
from finagent.graph.state import AgentState
from finagent.tools.web_search import WebSearcher
from finagent.tools.calculator import FinancialCalculator
from finagent.tools.edgar_search import EdgarFullTextSearch
from finagent.tools.sec_fetch import SecFilingFetcher
from finagent.tools.xbrl import XBRLClient
from finagent.graph.nodes import (
    FetchNodes, NumericNodes, ExternalNodes, SynthesisNodes, VerificationNodes,
)


# --------------------------------------------------------------------------- #
# AgenticRAGv4
# --------------------------------------------------------------------------- #

class AgenticRAGv4(FetchNodes, NumericNodes, ExternalNodes,
                   SynthesisNodes, VerificationNodes, AgenticRAGv3):
    """v3 + structured numeric lanes + web search + numeric verification + confidence gate."""

    def __init__(
        self,
        *args,
        # ~10 hits gives the synthesiser enough material for multi-faceted
        # questions ("performance over the last year"); a single article rarely
        # covers the full ground. Tavily allows up to 20 per call.
        web_top_k: int = 10,
        min_verify_score: float = 0.5,
        dispatch: bool = True,
        analyst_voice: bool = True,
        dedupe: bool = True,
        dedupe_threshold: float = 0.93,
        strict_numeric: bool = True,
        # Hard-refuse only when LESS than this share of the draft's figures is
        # grounded; a mostly-grounded answer falls through to the confidence
        # gate instead (warn band, or the withhold path that keeps the draft).
        refuse_below_grounding: float = 0.6,
        persist_fetch: bool = True,
        confidence_gating: bool = True,
        confidence_answer: float = 0.80,
        confidence_warn: float = 0.60,
        active_critic: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.web_top_k = web_top_k
        # Phase 7 query-type dispatcher. When True, non-narrative questions skip
        # retrieval/grading and go straight to the right tool. Set False to A/B
        # against the legacy "always retrieve everything" path.
        self.dispatch = dispatch
        # Phase 9 financial-analyst voice for the synthesizer + critic. Set False
        # to A/B against the prior generic prompts.
        self.analyst_voice = analyst_voice
        # Phase 10 near-duplicate filter: drop passages that are ~the same point
        # before synthesis so the answer doesn't repeat itself. Set False to A/B.
        self.dedupe = dedupe
        self.dedupe_threshold = dedupe_threshold
        # Phase 11 strict numeric verification: deterministically extract every
        # figure in the draft and confirm each traces to XBRL / evidence; an
        # ungrounded figure re-routes then refuses. Set False to A/B against the
        # prior LLM-only numeric check.
        self.strict_numeric = strict_numeric
        self.refuse_below_grounding = refuse_below_grounding
        # Dynamic fetch: persist the fetched filing into the on-disk index (True,
        # local) or use it ephemerally in memory for this session (False, cloud).
        self.persist_fetch = persist_fetch
        # Confidence framework (#8/9): blend retrieval/verification/citation/critic
        # sub-scores into a single confidence, then gate the answer on it. With
        # `confidence_gating` False the score is still computed (observability) but
        # the gate always answers — for A/B against the ungated path. Bands:
        #   confidence >= confidence_answer            → answer as-is
        #   confidence_warn <= confidence < answer     → answer + moderate caveat
        #   confidence <  confidence_warn              → answer + LOW-confidence
        #                                                caveat (full draft shown)
        self.confidence_gating = confidence_gating
        self.confidence_answer = confidence_answer
        self.confidence_warn = confidence_warn
        # Active-critic recovery (#6): when the critic finds unsupported claims
        # but the evidence is rich, route to a focused RE-DRAFT (cheap) instead
        # of a full re-retrieve — the draft over-claimed, not the evidence. Falls
        # back to the heavy re-gather path when evidence is thin. Bounded by the
        # existing critic_iterations cap. Set False to A/B against the prior
        # "critic always proceeds to verify" behaviour.
        self.active_critic = active_critic
        self.min_verify_score = min_verify_score
        self._web: Optional[WebSearcher] = None
        self._xbrl: Optional[XBRLClient] = None
        self._calc: Optional[FinancialCalculator] = None
        self._fetcher: Optional[SecFilingFetcher] = None
        self._edgar: Optional[EdgarFullTextSearch] = None

    # ------------------------------------------------------------------ #
    # Resources
    # ------------------------------------------------------------------ #

    @property
    def web(self) -> WebSearcher:
        if self._web is None:
            self._web = WebSearcher(top_k=self.web_top_k)
        return self._web

    @property
    def xbrl(self) -> XBRLClient:
        """SEC XBRL company-facts client (Phase 3). The tag-heterogeneity LLM
        fallback is wired to the router-tier model via `_xbrl_pick_tag`."""
        if self._xbrl is None:
            self._xbrl = XBRLClient(tag_resolver=self._xbrl_pick_tag)
        return self._xbrl

    @property
    def calc(self) -> FinancialCalculator:
        """Derived-metric calculator over XBRL inputs (Phase 4). Shares the XBRL
        client so it reuses the same on-disk company-facts cache."""
        if self._calc is None:
            self._calc = FinancialCalculator(xbrl=self.xbrl)
        return self._calc

    @property
    def edgar(self) -> EdgarFullTextSearch:
        """EDGAR full-text search client (Phase 6) for cross-document questions."""
        if self._edgar is None:
            self._edgar = EdgarFullTextSearch()
        return self._edgar

    @property
    def fetcher(self) -> SecFilingFetcher:
        """Dynamic SEC filing fetcher (Phase 5). Targets the *first* configured
        filings collection — the live US corpus we expand at runtime."""
        if self._fetcher is None:
            self._fetcher = SecFilingFetcher(
                resolver=self.xbrl.resolver,        # reuse the cached resolver
                collection_name=self.collections[0],
                embedding_model=self.embedding_model,
            )
        return self._fetcher



    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #




















    # ------------------------------------------------------------------ #
    # Evidence builder (#3)
    # ------------------------------------------------------------------ #









    # ------------------------------------------------------------------ #
    # #5 — financial verification report (cross-source / units / sources)
    # ------------------------------------------------------------------ #




    # ------------------------------------------------------------------ #
    # Confidence framework (#8/9)
    # ------------------------------------------------------------------ #








    # ------------------------------------------------------------------ #
    # Routers
    # ------------------------------------------------------------------ #


    def _dispatch_router(self, state: AgentState) -> str:
        """Phase 7 query-type dispatcher — pick the cheapest path that answers.

        * any narrative sub-query (or no routing at all) → the **retrieval
          path**: corpus-fetch gate → hybrid retrieve → grade → corrective loop.
          This path also flows into the tool chain afterwards, so a mixed
          question (narrative + numeric) still gets XBRL/calc/EDGAR.
        * purely non-narrative (numeric / derived-metric / market /
          cross-document / external) → straight into the **tool chain**, skipping
          the fetch gate, retrieval, and grading entirely. This is the
          efficiency win: a numeric question never touches retrieval.

        With `self.dispatch` False the router always takes the retrieval path —
        the legacy "retrieve everything" behaviour — for A/B measurement.
        """
        # A dated event-filing question MUST take the retrieval path: only
        # fetch_filing can pull the named 8-K/10-Q from EDGAR, and the tools
        # chain (XBRL/calc/market/web) has no lane that reads a specific dated
        # document.
        if self._dated_form_request(state["question"]):
            return "retrieval"
        # Pre-seeded ephemeral chunks (user-uploaded documents) are only ranked
        # inside the retrieval path, so never skip it when they're present.
        if state.get("fetched_chunks"):
            return "retrieval"
        routes = state.get("query_routes") or []
        has_narrative = (not routes) or any(r == "narrative" for r in routes)
        return "retrieval" if (not self.dispatch or has_narrative) else "tools"

    def _verify_router(self, state: AgentState) -> str:
        """After numeric verification: refuse / retry retrieval / continue to the confidence gate.

        The **verifier** is the source of truth for refusal: it does a precise
        per-number match against ALL evidence (text + tables + web). The
        critic is a cheaper retry trigger; if it disagrees with a passing
        verifier (e.g. it didn't see the web hits) we trust the verifier.
        """
        nv = state.get("numeric_verification") or {}
        score = nv.get("score")
        ungrounded = nv.get("unverified") or []
        # Bound the loop on BOTH counters: the critic increments
        # `critic_iterations` only when IT wants a retry, but the verifier can
        # re-route on its own, so `verify_iterations` is what actually caps the
        # verify→retrieve→…→verify cycle (otherwise it spins to the recursion
        # limit and the request errors out).
        out_of_retries = (
            state.get("critic_iterations", 0) >= self.max_critic_retries
            or state.get("verify_iterations", 0) > self.max_critic_retries
        )

        # Phase 11: an ungrounded figure is a potential hallucination. Under
        # strict numeric verification, re-route to try to ground it; once out of
        # retries, hard-refuse ONLY when most of the draft's figures failed to
        # ground (a substantially fabricated answer). A mostly-grounded draft
        # proceeds to the confidence gate, where the depressed verification
        # sub-score lands it in the warn band (answer + caveat) or the withhold
        # path (which preserves the draft for opt-in) — a refusal there would
        # throw away an answer whose figures overwhelmingly trace to sources.
        if self.strict_numeric:
            has_ungrounded = (nv.get("numbers_total", 0) > 0 and len(ungrounded) > 0)
            if state.get("needs_retry") and not out_of_retries:
                return "retrieve"
            if has_ungrounded and not out_of_retries:
                return "retrieve"          # re-route to re-ground the figure(s)
            if has_ungrounded and out_of_retries:
                severely_ungrounded = (
                    score is not None and score < self.refuse_below_grounding)
                return "refuse" if severely_ungrounded else "end"
            return "end"

        # Legacy LLM-only path.
        claims = nv.get("claims") if isinstance(nv, dict) else None
        if state.get("needs_retry"):
            return "retrieve"
        verify_failed = (score is not None and score < self.min_verify_score)
        if out_of_retries and verify_failed and claims:
            return "refuse"
        return "end"

    def _evidence_router(self, state: AgentState) -> str:
        """After evidence_builder: proceed to synthesis, or (one-shot) fall back
        to corpus retrieval when the tools lanes all came back empty."""
        return "retrieve" if state.get("corpus_fallback_pending") else "synthesize"

    def _confidence_gate(self, state: AgentState) -> str:
        """Route on the band `confidence_node` already chose: answer / warn /
        refuse. Kept as a pure read so the policy lives in one place."""
        return state.get("confidence_band") or "answer"

    def _critic_router(self, state: AgentState) -> str:
        """Active-critic recovery (#6). After the critic:

        * no unsupported claims (or active_critic off) → proceed to verify;
        * unsupported claims AND we already have substantial evidence → a focused
          RE-DRAFT (`resynthesize`): the draft over-claimed, so re-writing against
          the same evidence is cheap and usually fixes it;
        * unsupported claims AND evidence is thin → proceed to verify, whose
          router then re-routes to retrieve (the heavier re-gather path).

        Bounded by the existing `critic_iterations` cap: the critic only keeps
        `needs_retry` True while under the cap, so this can re-draft at most
        `max_critic_retries` times before it must proceed.
        """
        # Insufficient-draft escalation outranks a re-draft: when the draft
        # admits the evidence can't answer, re-writing over the SAME evidence
        # can't help — gather web evidence first, then re-synthesize.
        if state.get("web_fallback_pending"):
            return "websearch"
        if not self.active_critic or not state.get("needs_retry"):
            return "verify"
        has_evidence = bool(
            state.get("evidence") or state.get("retrieved_chunks")
            or state.get("xbrl_facts") or state.get("calc_results")
            or state.get("market_data") or state.get("web_results")
            or state.get("edgar_results")
        )
        return "resynthesize" if has_evidence else "verify"

    # ------------------------------------------------------------------ #
    # Graph
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(AgentState)
        g.add_node("planner", self.planner_node)
        g.add_node("router", self.router_node)
        g.add_node("fetch_filing", self.fetch_filing_node)
        g.add_node("retrieve", self.hybrid_retrieve_node)
        g.add_node("grader", self.grader_node)
        g.add_node("rewrite", self.rewrite_node)
        g.add_node("xbrl", self.xbrl_node)
        g.add_node("calculator", self.calculator_node)
        g.add_node("table_agent", self.table_agent_node)
        g.add_node("market_data", self.market_data_node)
        g.add_node("web_search", self.web_search_node)
        g.add_node("edgar_search", self.edgar_search_node)
        g.add_node("evidence_builder", self.evidence_builder_node)
        g.add_node("synthesize", self.synthesize_node)
        g.add_node("critic", self.critic_node)
        g.add_node("verify_numbers", self.verify_numbers_node)
        g.add_node("confidence", self.confidence_node)
        g.add_node("answer_with_warning", self.answer_with_warning_node)
        g.add_node("low_confidence", self.withhold_low_confidence_node)
        g.add_node("refuse", self.refuse_node)

        g.add_edge(START, "planner")
        g.add_edge("planner", "router")
        # Phase 7 dispatcher: route to the cheapest path. Narrative (and the
        # default) take the retrieval path; purely non-narrative questions skip
        # straight to the tool chain (no fetch gate, no retrieval, no grading).
        g.add_conditional_edges(
            "router", self._dispatch_router,
            {"retrieval": "fetch_filing", "tools": "xbrl"},
        )
        # Phase 5: corpus-membership gate before retrieval — fetch + ingest a
        # missing US company's 10-K so retrieval can find it on this turn.
        g.add_edge("fetch_filing", "retrieve")
        g.add_edge("retrieve", "grader")
        # Phase 3-4: numeric sub-queries hit XBRL first (exact structured facts),
        # then the calculator (derived metrics over those facts), then the table
        # agent supplements. `_grade_router` still returns "table_agent" as its
        # proceed verdict; we send that to `xbrl` and chain
        # xbrl → calculator → table_agent so all three run.
        g.add_conditional_edges(
            "grader", self._grade_router,
            {"rewrite": "rewrite", "table_agent": "xbrl"},
        )
        # Re-route after a rewrite: the rewritten query replaces sub_queries, so
        # it must be re-classified or query_routes goes stale (which silently
        # dropped the table/market lanes on the retry pass).
        g.add_edge("rewrite", "router")
        # Numeric chain stays SEQUENTIAL: xbrl→calculator→table_agent share the
        # XBRL client (facts + resolver cache) and table_agent reads the
        # `tables` collection. Keeping them ordered avoids the concurrent-client
        # segfault class this codebase already hit once (commit 71e6bda).
        g.add_edge("xbrl", "calculator")
        g.add_edge("calculator", "table_agent")
        # #2: the three NETWORK-bound lanes are mutually independent (distinct
        # state keys, different services) and dominate tool latency, so fan them
        # out to run in PARALLEL after the numeric chain. Only web_search's
        # news-fallback touches the store, and it can't race table_agent (which
        # has already finished) — so no two store readers ever overlap.
        g.add_edge("table_agent", "market_data")
        g.add_edge("table_agent", "web_search")
        g.add_edge("table_agent", "edgar_search")
        # Fan-in: evidence_builder runs once, after all three parallel lanes land,
        # and normalises every lane's output into one `evidence` list before synth.
        g.add_edge("market_data", "evidence_builder")
        g.add_edge("web_search", "evidence_builder")
        g.add_edge("edgar_search", "evidence_builder")
        # Corpus fallback (one-shot): a tools-path run whose lanes all came back
        # empty re-enters at fetch_filing → retrieve instead of synthesising
        # from nothing.
        g.add_conditional_edges(
            "evidence_builder", self._evidence_router,
            {"retrieve": "fetch_filing", "synthesize": "synthesize"},
        )
        g.add_edge("synthesize", "critic")
        # #6: the critic can actively route to a focused re-draft (resynthesize)
        # when the draft over-claimed against otherwise-good evidence, instead of
        # always deferring recovery to the verify→retrieve path.
        g.add_conditional_edges(
            "critic", self._critic_router,
            {"resynthesize": "synthesize", "verify": "verify_numbers",
             # Insufficient-draft escalation: gather web evidence, then the
             # existing web_search → evidence_builder → synthesize edges
             # re-draft with it.
             "websearch": "web_search"},
        )
        # Verifier settles hard refusals (ungrounded figures) and retries; an
        # answerable draft ("end") then flows into the confidence gate (#8/9).
        g.add_conditional_edges(
            "verify_numbers", self._verify_router,
            {"retrieve": "retrieve", "refuse": "refuse", "end": "confidence"},
        )
        # Confidence gate: high → answer, moderate → answer + caveat, low →
        # answer + a stronger low-confidence caveat (the full draft is always
        # shown; only hallucinated-figure refusals suppress an answer).
        g.add_conditional_edges(
            "confidence", self._confidence_gate,
            {"answer": END, "warn": "answer_with_warning", "refuse": "low_confidence"},
        )
        g.add_edge("answer_with_warning", END)
        g.add_edge("low_confidence", END)
        g.add_edge("refuse", END)
        return g.compile()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #



    # ------------------------------------------------------------------ #
    # Phase 11 — deterministic numeric grounding
    # ------------------------------------------------------------------ #

















# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the full agentic RAG (v4).")
    p.add_argument("--collection", default="us_filings")
    p.add_argument("--table-collection", default="tables")
    p.add_argument("--provider", choices=["groq", "gemini", "openai", "anthropic"],
                   default="groq")
    p.add_argument("--planner-model", default=None)
    p.add_argument("--synth-model", default=None)
    p.add_argument("--critic-model", default=None)
    p.add_argument("--grader-model", default=None)
    p.add_argument("--router-model", default=None)
    p.add_argument("--code-model", default=None)
    p.add_argument("--verifier-model", default=None)
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--reranker-model", default="BAAI/bge-reranker-large")
    p.add_argument("--pool-top-k", type=int, default=48)
    p.add_argument("--final-top-k", type=int, default=5)
    p.add_argument("--table-top-k", type=int, default=3)
    p.add_argument("--web-top-k", type=int, default=3)
    p.add_argument("--grade-threshold", type=float, default=3.0)
    p.add_argument("--max-rewrites", type=int, default=3)
    p.add_argument("--max-critic-retries", type=int, default=2)
    p.add_argument("--min-verify-score", type=float, default=0.5)
    p.add_argument("--no-confidence-gating", dest="confidence_gating",
                   action="store_false", help="compute confidence but never gate on it")
    p.add_argument("--confidence-answer", type=float, default=0.80)
    p.add_argument("--confidence-warn", type=float, default=0.60)
    p.add_argument("--question", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--question-col", default="question")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--output", default="results/final_rag_outputs.json")
    return p


def _load_dataset(path: str):
    import pandas as pd

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith((".jsonl", ".json")):
        return pd.read_json(path, lines=path.endswith(".jsonl"))
    raise ValueError(f"Unsupported dataset format: {path}")


def main():
    args = _build_cli().parse_args()

    # LLM choice is per-run, not a property of the agent — build the context
    # from the CLI flags and inject it at run().
    ctx = RuntimeContext(provider=args.provider, synth_model=args.synth_model,
                         top_k=args.final_top_k)

    agent = AgenticRAGv4(
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        pool_top_k=args.pool_top_k,
        final_top_k=args.final_top_k,
        table_collection=args.table_collection,
        table_top_k=args.table_top_k,
        web_top_k=args.web_top_k,
        grade_threshold=args.grade_threshold,
        max_rewrites=args.max_rewrites,
        max_critic_retries=args.max_critic_retries,
        min_verify_score=args.min_verify_score,
        confidence_gating=args.confidence_gating,
        confidence_answer=args.confidence_answer,
        confidence_warn=args.confidence_warn,
    )

    if args.dataset:
        df = _load_dataset(args.dataset)
        if args.sample:
            df = df.head(args.sample)
        agent.run_dataset(df, output_path=args.output, question_col=args.question_col)
        return

    if not args.question:
        raise SystemExit("Provide --question or --dataset.")

    state = agent.run(args.question, ctx)
    print("\n" + "=" * 60)
    print(f"Question:         {state['question']}")
    print(f"Sub-queries:      {state.get('sub_queries')}")
    print(f"Routes:           {state.get('query_routes')}")
    ev = state.get("evidence") or []
    kinds: dict = {}
    for e in ev:
        kinds[e.get("kind")] = kinds.get(e.get("kind"), 0) + 1
    print(f"Evidence:         {len(ev)} items {kinds}")
    print(f"Grades:           {state.get('grades')} (avg {state.get('avg_grade')})")
    print(f"Rewrites:         {state.get('iteration_count', 0)}")
    print(f"Critic retries:   {state.get('critic_iterations', 0)}")
    nv = state.get("numeric_verification") or {}
    print(f"Numeric verify:   score={nv.get('score')}  unverified={len(nv.get('unverified', []))}")
    print(f"Refused:          {state.get('refused', False)}")
    print(f"Low confidence:   {state.get('low_confidence', False)}")
    print(
        f"Confidence:       {state.get('confidence')} "
        f"(band={state.get('confidence_band')}, status={state.get('status')}) "
        f"[retr={state.get('retrieval_score')} verif={state.get('verification_score')} "
        f"cite={state.get('citation_score')} crit={state.get('critic_score')}]"
    )
    print(f"\nAnswer:\n{state.get('final_answer')}")
    print(f"\nCitations:        {state.get('citations')}")
    if state.get("table_results"):
        print(f"\nTable computations ({len(state['table_results'])}):")
        for t in state["table_results"]:
            print(f"  - {t['sub_query']}: {t.get('answer', '')[:120]}")
    if state.get("web_results"):
        print(f"\nWeb hits ({len(state['web_results'])}):")
        for h in state["web_results"]:
            print(f"  - [{h.get('source')}] {h.get('title', '')[:90]}")
    if state.get("errors"):
        print(f"\nErrors:           {state['errors']}")


if __name__ == "__main__":
    main()
