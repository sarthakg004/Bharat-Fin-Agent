"""The agent: one class that owns the tools and wires the steps into a graph.

    planner → fetch_filing → retrieve ─┐
            └──────(no narrative part)─┴→ xbrl → calculator
                → market_data | web_search | edgar_search → gather
                → synthesize → critic → END
                                  ├→ synthesize   (redraft)
                                  ├→ retrieve     (gather more, then synthesize)
                                  ├→ fetch_filing (filings skipped, draft admits a gap)
                                  ├→ web_search   (then synthesize)
                                  └→ refuse → END

The steps themselves are plain functions in the sibling modules; each takes the
agent and the state and returns the state keys it changed.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

from finagent.agent import answer, external, numeric, plan, retrieve
from finagent.agent.state import AgentState
from finagent.config import settings
from finagent.llm import classify_error
from finagent.runtime import create_llm, current_context
from finagent.vectorstore import DEFAULT_EMBED_MODEL

# Shown to the user when a step could not run because of its model provider.
_REASONS = {
    "rate_limit": "the model provider is rate limiting requests",
    "quota": "today's free quota for that model is used up",
    "busy": "the model provider is busy",
    "too_large": "the request was too large for that model",
    "auth": "the API key was rejected",
    "not_found": "that model is no longer available",
}


class FinAgent:
    """Holds the shared resources. Nothing request-specific is stored here, so
    one instance serves every request."""

    MAX_RECOVERIES = 1          # recovery passes the critic may trigger per question
    REFUSE_BELOW_SUPPORT = 0.5  # refuse when under half the claims are supported

    def __init__(self, collection: str = settings.us_collection,
                 embedding_model: str = DEFAULT_EMBED_MODEL,
                 reranker_model: str = settings.reranker_model,
                 pool_top_k: int = 48, final_top_k: int = 5,
                 retrieve_cap: int = 8, web_top_k: int = 10):
        self.collection = collection
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.pool_top_k = pool_top_k        # candidates pulled from Qdrant per search
        self.final_top_k = final_top_k      # passages kept per sub-query
        self.retrieve_cap = retrieve_cap    # passages handed to the writer
        self.web_top_k = web_top_k
        self._retriever = self._web = self._xbrl = self._calc = None
        self._edgar = self._fetcher = self._graph = None

    # --- resources, built on first use --------------------------------------

    @property
    def retriever(self):
        if self._retriever is None:
            from finagent.retrieval.hybrid import HybridRetriever

            self._retriever = HybridRetriever.from_collection(
                self.collection, self.embedding_model, reranker_model=self.reranker_model,
                pool_top_k=self.pool_top_k, final_top_k=self.final_top_k)
        return self._retriever

    def reset_retriever(self) -> None:
        """Forget the cached company list after a new filing is ingested."""
        self._retriever = None

    @property
    def xbrl(self):
        if self._xbrl is None:
            from finagent.tools.xbrl import XBRLClient

            self._xbrl = XBRLClient(tag_resolver=partial(numeric.pick_xbrl_tag, self))
        return self._xbrl

    @property
    def calc(self):
        if self._calc is None:
            from finagent.tools.calculator import FinancialCalculator

            self._calc = FinancialCalculator(xbrl=self.xbrl)
        return self._calc

    @property
    def fetcher(self):
        if self._fetcher is None:
            from finagent.tools.sec_fetch import SecFilingFetcher

            self._fetcher = SecFilingFetcher(resolver=self.xbrl.resolver,
                                             collection_name=self.collection,
                                             embedding_model=self.embedding_model)
        return self._fetcher

    @property
    def web(self):
        if self._web is None:
            from finagent.tools.web_search import WebSearcher

            self._web = WebSearcher(top_k=self.web_top_k)
        return self._web

    @property
    def edgar(self):
        if self._edgar is None:
            from finagent.tools.edgar_search import EdgarFullTextSearch

            self._edgar = EdgarFullTextSearch()
        return self._edgar

    # --- helpers the steps use ----------------------------------------------

    def llm(self, role: str):
        """The chat model for a role, under the running request's context."""
        return create_llm(current_context(), role)

    @staticmethod
    def log(state: AgentState, msg: str) -> None:
        state.setdefault("log", []).append(msg)

    @staticmethod
    def notice(state: AgentState, msg: str) -> None:
        """A message the user should see under the answer."""
        notices = state.setdefault("notices", [])
        if msg not in notices:
            notices.append(msg)

    def llm_failed(self, state: AgentState, step: str, exc: Exception,
                   always_notice: bool = False) -> None:
        """Record a failed model call. The step is skipped and the answer is
        built from whatever else was gathered; a provider problem is also shown
        to the user."""
        kind = classify_error(exc).kind
        self.log(state, f"{step} failed ({kind}): {type(exc).__name__}: {str(exc)[:160]}")
        if kind in _REASONS:
            self.notice(state, f"{step} was skipped: {_REASONS[kind]}.")
        elif always_notice:
            self.notice(state, f"{step} was skipped.")

    # --- routing -------------------------------------------------------------

    @staticmethod
    def _after_plan(state: AgentState) -> str:
        """Filing search only when a part of the question needs prose from a filing."""
        routes = state.get("query_routes") or []
        return "filings" if (not routes or "narrative" in routes) else "tools"

    @staticmethod
    def _after_retrieve(state: AgentState) -> str:
        """First pass: on to the tools. On a retry the tools already ran."""
        retry = state.get("recoveries") or state.get("corpus_fallback_used")
        return "synthesize" if retry else "xbrl"

    @staticmethod
    def _after_gather(state: AgentState) -> str:
        return "filings" if state.get("corpus_fallback_pending") else "synthesize"

    def _after_critic(self, state: AgentState) -> str:
        if state.get("corpus_fallback_pending"):
            return "filings"
        if state.get("web_fallback_pending"):
            return "web_search"
        if state.get("needs_retry"):
            return "retrieve" if state.get("retry_queries") else "synthesize"
        score = state.get("support_score")
        if (score is not None and score < self.REFUSE_BELOW_SUPPORT
                and state.get("recoveries", 0) >= self.MAX_RECOVERIES):
            return "refuse"
        return "end"

    # --- graph ---------------------------------------------------------------

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(AgentState)
        for name, step in (
            ("planner", plan.planner),
            ("fetch_filing", retrieve.fetch_filing),
            ("retrieve", retrieve.retrieve),
            ("xbrl", numeric.xbrl),
            ("calculator", numeric.calculator),
            ("market_data", external.market_data),
            ("web_search", external.web_search),
            ("edgar_search", external.edgar_search),
            ("gather", answer.gather),
            ("synthesize", answer.synthesize),
            ("critic", answer.critic),
            ("refuse", answer.refuse),
        ):
            g.add_node(name, partial(step, self))

        g.add_edge(START, "planner")
        g.add_conditional_edges("planner", self._after_plan,
                                {"filings": "fetch_filing", "tools": "xbrl"})
        g.add_edge("fetch_filing", "retrieve")
        g.add_conditional_edges("retrieve", self._after_retrieve,
                                {"xbrl": "xbrl", "synthesize": "synthesize"})
        # XBRL then calculator run in order (they share the XBRL client); the
        # three network lanes then run side by side and meet at `gather`.
        g.add_edge("xbrl", "calculator")
        for lane in ("market_data", "web_search", "edgar_search"):
            g.add_edge("calculator", lane)
            g.add_edge(lane, "gather")
        g.add_conditional_edges("gather", self._after_gather,
                                {"filings": "fetch_filing", "synthesize": "synthesize"})
        g.add_edge("synthesize", "critic")
        g.add_conditional_edges("critic", self._after_critic, {
            "synthesize": "synthesize", "retrieve": "retrieve", "filings": "fetch_filing",
            "web_search": "web_search", "refuse": "refuse", "end": END})
        g.add_edge("refuse", END)
        return g.compile()

    def run(self, question: str, ctx=None, chat_history: Optional[list[dict]] = None) -> AgentState:
        """Run one question and return the final state."""
        from finagent.runtime import RuntimeContext

        state: AgentState = {"question": question, "log": [], "notices": []}
        if chat_history:
            state["chat_history"] = chat_history
        return self.graph.invoke(state, config={
            "recursion_limit": 50,
            "configurable": {"runtime_context": ctx or RuntimeContext()}})
