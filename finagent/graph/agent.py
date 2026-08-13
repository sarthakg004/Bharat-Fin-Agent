"""The deployed agent. Owns the constructor, the tool resources, the routing
functions, and the graph wiring.

    START → planner(+routes) → router → {fetch_filing → retrieve | xbrl}
          → xbrl → calculator → (market_data ∥ web_search ∥ edgar_search)
          → evidence_builder → synthesize → critic → END
                                             ↘ refuse → END

The node bodies live in `finagent/graph/nodes/` as topical mixins: fetch (SEC
fetch + retrieval), numeric (XBRL/calculator), external (market/web/EDGAR),
synthesis (evidence + drafting + critic + refusal).
"""

from __future__ import annotations

from typing import Optional

from finagent.graph.full import AgenticRAGv3
from finagent.graph.state import AgentState
from finagent.tools.web_search import WebSearcher
from finagent.tools.calculator import FinancialCalculator
from finagent.tools.edgar_search import EdgarFullTextSearch
from finagent.tools.sec_fetch import SecFilingFetcher
from finagent.tools.xbrl import XBRLClient
from finagent.graph.nodes import (
    FetchNodes, NumericNodes, ExternalNodes, SynthesisNodes,
)


class AgenticRAGv4(FetchNodes, NumericNodes, ExternalNodes,
                   SynthesisNodes, AgenticRAGv3):
    """Retrieval + structured numeric lanes + web search."""

    # Hard-refuse only when the critic could support LESS than this share of
    # the draft's claims after every retry was spent; a mostly-supported
    # answer is shown as drafted.
    REFUSE_BELOW_SUPPORT = 0.5

    def __init__(
        self,
        *args,
        # ~10 hits gives the synthesiser enough material for multi-faceted
        # questions ("performance over the last year"); a single article rarely
        # covers the full ground. Tavily allows up to 20 per call.
        web_top_k: int = 10,
        dispatch: bool = True,
        analyst_voice: bool = True,
        dedupe: bool = True,
        dedupe_threshold: float = 0.93,
        persist_fetch: bool = True,
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
        # Dynamic fetch: persist the fetched filing into the on-disk index (True,
        # local) or use it ephemerally in memory for this session (False, cloud).
        self.persist_fetch = persist_fetch
        # Active-critic recovery (#6): when the critic finds unsupported claims
        # but the evidence is rich, route to a focused RE-DRAFT (cheap) instead
        # of a full re-retrieve — the draft over-claimed, not the evidence. Falls
        # back to the heavy re-gather path when evidence is thin. Bounded by the
        # existing critic_iterations cap. Set False to A/B against the prior
        # "critic always proceeds" behaviour.
        self.active_critic = active_critic
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
    # Routers
    # ------------------------------------------------------------------ #

    def _dispatch_router(self, state: AgentState) -> str:
        """Phase 7 query-type dispatcher — pick the cheapest path that answers.

        * any narrative sub-query (or no routing at all) → the **retrieval
          path**: corpus-fetch gate → hybrid retrieve → corrective loop.
          This path also flows into the tool chain afterwards, so a mixed
          question (narrative + numeric) still gets XBRL/calc/EDGAR.
        * purely non-narrative (numeric / derived-metric / market /
          cross-document / external) → straight into the **tool chain**, skipping
          the fetch gate and retrieval entirely. This is the
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

    def _evidence_router(self, state: AgentState) -> str:
        """After evidence_builder: proceed to synthesis, or (one-shot) fall back
        to corpus retrieval when the tools lanes all came back empty."""
        return "retrieve" if state.get("corpus_fallback_pending") else "synthesize"

    def _critic_router(self, state: AgentState) -> str:
        """One recovery attempt, chosen by the critic, then answer or refuse.

        There are three ways to recover and the agent may use exactly ONE of
        them per question, because a second pass over the same evidence with
        the same models produces the same answer at twice the quota:

        * `websearch`  the draft ADMITS the evidence cannot answer and the web
                       lane has not run. No amount of re-drafting fixes missing
                       evidence, so this outranks the other two.
        * `retrieve`   the critic says `gather`: the evidence genuinely lacks
                       the fact. `AgenticRAGv2.critic_node` has already pointed
                       `sub_queries` at the flagged claims and blanked
                       `retrieval_query`, so the retry searches something new.
        * `resynthesize` the critic says `redraft`: the evidence is sufficient
                       and the draft overstated it. The synthesizer gets the
                       flagged claims quoted back and rewrites against the same
                       evidence.

        All three count against `critic_iterations`, and each is gated on that
        budget by the node that RAISES it, not here: `AgenticRAGv2.critic_node`
        only leaves `needs_retry` True while under the cap, and
        `_web_fallback_signal` only sets `web_fallback_pending` under the same
        cap. Both increment the counter when they fire. Re-checking the budget
        in this router would double-count, because by the time it runs the
        counter already includes the recovery it is being asked to dispatch.
        (The web-fallback pass used to skip the counter entirely, so a question
        could run three synthesize passes while the API reported one.)

        With the budget spent, a draft the critic still cannot mostly support
        (`REFUSE_BELOW_SUPPORT`) is refused rather than shipped.
        """
        if state.get("web_fallback_pending"):
            return "websearch"
        if state.get("needs_retry"):
            if state.get("critic_remedy") == "gather" or not self.active_critic:
                return "retrieve"
            return "resynthesize"

        score = state.get("grading_score")
        if (score is not None and score < self.REFUSE_BELOW_SUPPORT
                and state.get("critic_iterations", 0) >= self.max_critic_retries):
            return "refuse"
        return "end"

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
        g.add_node("xbrl", self.xbrl_node)
        g.add_node("calculator", self.calculator_node)
        g.add_node("market_data", self.market_data_node)
        g.add_node("web_search", self.web_search_node)
        g.add_node("edgar_search", self.edgar_search_node)
        g.add_node("evidence_builder", self.evidence_builder_node)
        g.add_node("synthesize", self.synthesize_node)
        g.add_node("critic", self.critic_node)
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
        # Phase 3-4: numeric sub-queries hit XBRL next (exact structured facts),
        # then the calculator (derived metrics over those facts).
        g.add_edge("retrieve", "xbrl")
        # Numeric chain stays SEQUENTIAL: xbrl→calculator share the XBRL client
        # (facts + resolver cache). Keeping them ordered avoids the concurrent-
        # client segfault class this codebase already hit once (commit 71e6bda).
        g.add_edge("xbrl", "calculator")
        # #2: the three NETWORK-bound lanes are mutually independent (distinct
        # state keys, different services) and dominate tool latency, so fan them
        # out to run in PARALLEL after the numeric chain. Only web_search's
        # news-fallback touches the store, and by here the sequential numeric
        # chain has finished, so no two store readers ever overlap.
        g.add_edge("calculator", "market_data")
        g.add_edge("calculator", "web_search")
        g.add_edge("calculator", "edgar_search")
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
        # when the draft over-claimed against otherwise-good evidence, fall back
        # to a full re-gather (retrieve) when evidence is thin, and — as the
        # only gate left — refuse a draft that stayed mostly unsupported after
        # every retry.
        g.add_conditional_edges(
            "critic", self._critic_router,
            {"resynthesize": "synthesize", "retrieve": "retrieve",
             # Insufficient-draft escalation: gather web evidence, then the
             # existing web_search → evidence_builder → synthesize edges
             # re-draft with it.
             "websearch": "web_search",
             "refuse": "refuse", "end": END},
        )
        g.add_edge("refuse", END)
        return g.compile()
