"""
full.py  ·  finagent/graph/full.py

Table-augmented agentic RAG (v3). Adds a query **router** and a **table agent**
to the corrective-RAG (v2) graph:

    START → planner → router → retrieve → grader ── conditional ──┐
                                  ▲                               │
                                  │                               ▼
                                  └── rewrite ◄── (avg_grade low,
                                                    rewrites < cap)
                                                                  │
                                                                  ▼
                                            table_agent → synthesize → critic ── conditional
                                                                          │            │
                                                                          └── retrieve ◄┘
                                                                  (needs_retry,
                                                                   critic_retries < cap)
                                                                          │
                                                                          ▼
                                                                         END

The **router** classifies every sub-query as `narrative`, `numeric`, or
`external` (web). The narrative ones go through the hybrid retriever +
grader/rewrite loop from v2; the numeric ones are answered by the
**TableAgent** (retrieve relevant tables → write pandas → safe exec). The
synthesizer merges both streams and the critic checks the final answer
against text excerpts AND table-derived results.

Usage as a library
------------------
    from finagent.graph.full import AgenticRAGv3

    agent = AgenticRAGv3(collection_name="india_filings", market="india")
    state = agent.run("What was HDFC Bank's net interest margin in FY23?")
    print(state["final_answer"])
    print(state["query_routes"])

CLI
---
    python -m finagent.graph.full \\
        --collection india_filings --market india \\
        --question "What was HDFC Bank's net interest margin in FY23?"
"""

from __future__ import annotations

import argparse
import re
from typing import Optional

from finagent.graph.base import append_comparison_row  # noqa: F401 (re-export)
from finagent.graph.corrective import AgenticRAGv2, HybridRetriever
from finagent.graph.state import AgentState, QueryPlan, RouterReport
from finagent.graph.table_agent import TableAgent


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

ROUTER_SYSTEM = """\
You route financial-QA sub-queries to one of FIVE retrievers:

- narrative: text retrieval over filings. Prose-style questions (strategy,
  risks, segment overviews, MD&A commentary) about ONE named company.
- numeric:   table-based answer FROM THE FILINGS. Specific figures, ratios,
  margins, growth %, multi-year financial comparisons that the company has
  reported in a 10-K or annual report.
- market:    live market-data tools (yfinance). Anything about a listed
  company's MARKET BEHAVIOUR — current/premarket/intraday price, OHLC
  history, 52-week range, charts, ticker-level news headlines. Lean `market`
  whenever the question is *in the direction of* the stock: "how is X doing",
  "how has X performed", "is X a good stock", "X stock", "X share price",
  recent returns/trend. When in doubt between market and narrative for a
  stock-flavoured question, pick `market` (a price chart is shown).
- cross_document: EDGAR full-text search across MANY companies' filings. Use
  when the answer is a SET OF COMPANIES rather than facts about one named
  company: "which companies disclosed/mentioned X", "list firms that report Y",
  "how many 10-Ks discuss Z". The defining signal is no single subject company.
- external:  general web search. Macro news, corporate events, post-cutoff
  developments that aren't specifically about market data.

Numeric markers: "what was", "how much", "ratio", "margin", "growth %",
specific fiscal years (FY23, FY2024), currency amounts (₹, $), "crore", "billion".

Market markers: "current price", "premarket", "intraday", "today's move",
"stock chart", "52-week", "OHLC", "candlestick", "compare X and Y stock".

Cross-document markers: "which companies", "what companies", "list companies/firms",
"how many filings/companies", "across filings" — when no single company is the subject.

Examples:
  - "What is HDFC Bank's net interest margin in FY23?"           → numeric
  - "How much did Reliance earn from oil-to-chemicals in FY23?"  → numeric
  - "Describe Infosys' AI strategy"                              → narrative
  - "What were the principal risks listed in TCS' annual report?" → narrative
  - "EBITDA margin growth from FY21 to FY23?"                    → numeric
  - "What is Wipro's current share price?"                       → market
  - "Show me Apple's 1-year stock chart"                         → market
  - "How is Tesla doing as a stock?"                             → market
  - "Which companies disclosed a material weakness in FY2023?"   → cross_document
  - "List firms that mention quantum computing in their 10-K"    → cross_document
  - "Latest macro headlines from India today"                    → external
"""

ROUTER_PROMPT = """\
Classify each sub-query below. Return one verdict per sub-query in the SAME
order. Copy the sub-query text verbatim into the `sub_query` field.

Sub-queries:
{sub_queries}
"""

# Fused planner+router: ONE structured call both decomposes the question into
# sub-queries AND routes each to its lane. This replaces two sequential LLM
# calls (planner, then router) on every question — the single largest fixed
# latency cost on the hot path — without changing what either step produces.
PLAN_ROUTE_SYSTEM = """\
You are the query planner for a financial-filings question-answering system.
You do TWO things in one pass: decompose the user's question into 1-8 focused,
self-contained sub-queries, and route each sub-query to the lane that answers it.

Decomposition rules
-------------------
- Simple, single-fact questions → return ONE sub-query (often the original).
- Comparison or multi-hop questions ("compare X and Y", "growth from A to B")
  → FULLY ENUMERATE one sub-query per (entity × period × metric) combination so
  nothing is dropped. "Compare Apple and Microsoft R&D as % of revenue over
  2020-2022" → SIX sub-queries (each company × each of the 3 years).
- Each sub-query must stand on its own (no pronouns referring to the question).
- Use precise analyst terms: name the exact line item or metric ("operating
  margin", "R&D as % of revenue", "diluted EPS") and the exact fiscal period
  ("FY2022"), so each can be answered from a single XBRL concept or calculation.
- FOLLOW-UPS: if the question relies on the conversation above ("show me the
  chart", "what about last year"), rewrite it into a self-contained sub-query
  naming the company/ticker discussed just before.

Routing lanes (per sub-query)
-----------------------------
- narrative: text retrieval over filings. Prose questions (strategy, risks,
  segment overviews, MD&A commentary) about ONE named company.
- numeric: exact figures FROM THE FILINGS — reported line items, ratios,
  margins, growth %, multi-year financial comparisons (answered from XBRL
  facts, a deterministic calculator, and extracted tables).
- market: live market data (yfinance). Anything about a listed company's
  MARKET behaviour — current/premarket/intraday price, OHLC history, charts,
  52-week range, ticker news. Lean `market` whenever the question is in the
  direction of the stock ("how is X doing", "is X a good stock", "X stock").
- cross_document: EDGAR full-text search across MANY companies' filings — the
  answer is a SET of companies ("which companies disclosed X"), not facts
  about one named company.
- external: general web search. Macro news, corporate events, M&A timelines,
  post-cutoff developments not specifically about market data.

Examples:
  - "What is HDFC Bank's net interest margin in FY23?"            → numeric
  - "Describe Infosys' AI strategy"                               → narrative
  - "What is Wipro's current share price?"                        → market
  - "Which companies disclosed a material weakness in FY2023?"    → cross_document
  - "Latest macro headlines from India today"                     → external
"""

PLAN_ROUTE_PROMPT = """\
Question: {question}

Return the routed sub-queries (1-8, each with its lane).
"""

SYNTH_V3_SYSTEM = """\
You are a meticulous financial analyst. Write a clear answer using ONLY the
numbered evidence supplied below.

Citations
---------
Cite by **number** only. After every factual claim, append the index of the
evidence item(s) that support it in ASCII square brackets, e.g.
"Reliance's FY24 revenue was ₹9.74 lakh crore [1]."
Multiple sources for one claim: `[1,3]`. Use `[N]` — NOT `【N】`, `(N)`, or
any other bracket style. NEVER write out the source title, the URL, or any
tag in prose — the user sees those in a sidebar already.

Formatting (markdown)
---------------------
- Open with a short overview paragraph that directly answers the question.
- Use **bold** for the key figures and entity names.
- Use bullet lists when summarising 3+ points.
- Use a GitHub-flavoured markdown table when comparing two or more items
  across the same metrics (e.g. revenue / margin / growth for two companies).
- Use `## sub-headings` only when the answer has 2+ logical sections.
- Keep paragraphs short (2-4 sentences); leave a blank line between them.

Aim for a thorough, well-structured answer — long enough to fully address the
question but with no filler.

Time-sensitive questions
------------------------
Each web / news item has a publication date printed in its header
(`[N] TRUSTED PRESS (published 2026-04-15) — ...`). For time-sensitive
questions — premarket prices, "today's", "current", "this week's" — you must:

- Use the MOST RECENT item by publication date.
- State the as-of date in the answer ("As of <date>, ...").
- Do NOT lump older datapoints together with the most recent one as if they
  were equivalent. If you mention historical context, label it explicitly
  ("Earlier, on <date>, the stock had moved ...").

When the evidence is thin or only partially answers the question:
- STILL give the most useful answer you can from what IS provided, and cite each
  fact with [N]. Web / news items are valid evidence — use them. A partial,
  caveated answer is far better than a flat refusal.
- Then add a short caveat in italics noting the limitation, e.g.
  *"Note: this is drawn from news coverage, not the company's filings, and the
  figures should be verified."* or *"The available sources only cover FY2023, so
  the 2022→2024 trend is incomplete."*
- Only when there is genuinely NO relevant evidence at all (nothing on the
  companies/topic asked) should you say so — in one short sentence, with no
  citations.
- Never invent figures, companies, page numbers, or table titles. Use only what
  the evidence supports.
"""

SYNTH_V3_PROMPT = """\
Question: {question}

Sub-queries researched:
{sub_queries}

Text excerpts (each begins with its citation tag):
{text_context}

Numeric / table-derived results (computed answer + the source tables):
{table_context}

---
Write the final answer now, with an inline citation after every fact.
"""


# --------------------------------------------------------------------------- #
# Heuristic router (fallback when the LLM call fails)
# --------------------------------------------------------------------------- #

_NUMERIC_MARKERS = (
    "what was", "what is", "how much", "how many",
    "ratio", "margin", "growth", "yield", "return on",
    "%", "₹", "$", "crore", "lakh", "billion", "million",
    "revenue", "profit", "earnings", "income", "ebitda",
    "interest", "net interest", "tax", "capex", "asset",
    "liability", "equity", "share", "dividend",
)
_NUMERIC_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b|fy ?\d{2,4}", re.IGNORECASE)
# Anything about live market behaviour goes to `market` (yfinance tools). Volume
# and price-performance/trend phrasing is included so a "rapid increase in
# volume" or "how has the stock performed" question reliably reaches yfinance
# (which is the only source with OHLCV + volume), even when the LLM router
# mislabels it as narrative/external.
_MARKET_MARKERS = (
    "current price", "share price", "stock price", "premarket", "pre-market",
    "intraday", "today's move", "today's price", "live price",
    "1-year", "5-year", "ytd", "52-week", "52 week",
    "ohlc", "candlestick", "chart", "stock chart",
    "compare stock", "vs stock", "stock comparison",
    "volume", "trading volume", "share volume", "volume surge", "volume spike",
    "how is it doing", "how has it performed", "how is it performing",
    "stock performance", "price performance", "how has the stock",
)
# Fallback news/web markers.
_EXTERNAL_MARKERS = (
    "latest news", "press release", "macro", "geopolitical",
    "this week", "yesterday", "breaking",
)


# Cross-document: the answer is a SET of companies, not facts about one.
_CROSSDOC_MARKERS = (
    "which companies", "what companies", "which firms", "what firms",
    "list companies", "list of companies", "list firms", "companies that",
    "firms that", "how many companies", "how many filings", "across filings",
    "companies disclos", "companies mention", "companies report",
)


def _heuristic_route(query: str) -> str:
    q = (query or "").lower()
    # Cross-document is unambiguous when its markers appear ("which companies…").
    if any(m in q for m in _CROSSDOC_MARKERS):
        return "cross_document"
    # Market beats external when both keywords appear — "today's stock chart"
    # is market, not news.
    if any(m in q for m in _MARKET_MARKERS):
        return "market"
    if any(m in q for m in _EXTERNAL_MARKERS):
        return "external"
    hits = sum(1 for m in _NUMERIC_MARKERS if m in q)
    if hits >= 2 or _NUMERIC_YEAR_RE.search(q):
        return "numeric"
    return "narrative"


# --------------------------------------------------------------------------- #
# AgenticRAGv3
# --------------------------------------------------------------------------- #

class AgenticRAGv3(AgenticRAGv2):
    """Corrective RAG + router + table agent.

    Only the parts that change from v2 are overridden:
      * `router_node` — new (per-sub-query classification with structured output)
      * `hybrid_retrieve_node` — filters to narrative sub-queries only
      * `table_agent_node` — new (per numeric sub-query, via TableAgent)
      * `synthesize_node` — merges text excerpts + table results
      * `critic_node` — checks claims against BOTH evidence streams
      * `_build_graph`, `_grade_router` — wire the new nodes
    """

    def __init__(
        self,
        *args,
        table_collection: str = "tables",
        table_top_k: int = 3,
        code_model: Optional[str] = None,
        router_model: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.table_collection = table_collection
        self.table_top_k = table_top_k
        # Code generation needs the strong tier.
        self.code_model = code_model or self.synth_model
        # Tool selection + structured extraction (router, XBRL/calc/EDGAR/gate
        # extractors, market planner all share this tier) is accuracy-critical:
        # a weak model mis-routes ("acquisitions" not sent to web) and mis-maps
        # metrics. Default it to the STRONG synth tier, not the fast planner.
        # Override `router_model` for a cheaper/faster tool tier if desired.
        self.router_model = router_model or self.synth_model
        self._table_agent: Optional[TableAgent] = None

    # ------------------------------------------------------------------ #
    # Resources
    # ------------------------------------------------------------------ #

    @property
    def table_agent(self) -> TableAgent:
        if self._table_agent is None:
            self._table_agent = TableAgent(
                chroma_dir=self.chroma_dir,
                collection_name=self.table_collection,
                embedding_model=self.embedding_model,
                provider=self.provider,
                code_model=self.code_model,
                top_k=self.table_top_k,
                api_key=self.api_key,
            )
        return self._table_agent

    def _get_router_llm(self):
        if "router" not in self._llms:
            from finagent.llm import build_llm

            self._llms["router"] = build_llm(
                self.provider, self.router_model, self.api_key, temperature=0.0
            )
        return self._llms["router"]

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

    def planner_node(self, state: AgentState) -> dict:
        """Fused planner+router: ONE structured call decomposes the question
        AND routes each sub-query, replacing the two sequential LLM calls the
        v1/v2 path makes. Runs on the router/tool tier (strong model) because
        both decomposition and routing sit on the quality path. On failure it
        falls back to the base planner (router_node then classifies)."""
        from langchain_core.messages import HumanMessage, SystemMessage

        question = state["question"]
        history = state.get("chat_history") or []
        history_block = ""
        if history:
            lines = [
                f"{('User' if t.get('role') == 'user' else 'Assistant')}: "
                f"{(t.get('content') or '')[:400]}"
                for t in history[-6:]
            ]
            history_block = (
                "Recent conversation (most recent last):\n" + "\n".join(lines) + "\n\n"
            )

        llm = self._get_router_llm().with_structured_output(QueryPlan)
        try:
            out: QueryPlan = llm.invoke([
                SystemMessage(content=PLAN_ROUTE_SYSTEM),
                HumanMessage(content=history_block
                             + PLAN_ROUTE_PROMPT.format(question=question)),
            ])
            pairs = [(q.query.strip(), q.route)
                     for q in (out.queries or []) if (q.query or "").strip()][:8]
        except Exception as e:
            self._log(state, f"fused planner failed ({e}); "
                             f"falling back to plan-then-route")
            pairs = []
        if not pairs:
            # Degrade to the legacy two-call path: base planner decomposes,
            # router_node classifies (it sees empty routes and runs).
            out_state = super().planner_node(state)
            out_state["query_routes"] = []
            return out_state
        return {"sub_queries": [q for q, _ in pairs],
                "query_routes": [r for _, r in pairs]}

    def router_node(self, state: AgentState) -> dict:
        """Classify each sub-query as narrative / numeric / external.

        The fused planner usually routed the plan already — when routes are
        present and aligned with the sub-queries, this is a no-op (no LLM
        call). It still classifies after a rewrite (which clears routes) or
        when the fused call fell back to the bare planner."""
        from langchain_core.messages import HumanMessage, SystemMessage

        sub_queries = state.get("sub_queries") or [state["question"]]
        existing = state.get("query_routes") or []
        if existing and len(existing) == len(sub_queries):
            return {}
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sub_queries))

        # Conversation context so follow-ups ("show me the chart") route to the
        # same lane the discussed company would — e.g. `market` for a chart.
        history = state.get("chat_history") or []
        history_block = ""
        if history:
            lines = [
                f"{('User' if t.get('role') == 'user' else 'Assistant')}: "
                f"{(t.get('content') or '')[:400]}"
                for t in history[-6:]
            ]
            history_block = (
                "Recent conversation (most recent last):\n" + "\n".join(lines) + "\n\n"
            )

        llm = self._get_router_llm().with_structured_output(RouterReport)
        try:
            out: RouterReport = llm.invoke([
                SystemMessage(content=ROUTER_SYSTEM),
                HumanMessage(content=history_block + ROUTER_PROMPT.format(sub_queries=numbered)),
            ])
            verdicts = [v.route for v in out.routes][: len(sub_queries)]
            while len(verdicts) < len(sub_queries):
                verdicts.append("narrative")
        except Exception as e:
            verdicts = [_heuristic_route(s) for s in sub_queries]
            self._log(state, f"router failed ({e}); used heuristic")
        return {"query_routes": verdicts}

    def hybrid_retrieve_node(self, state: AgentState) -> dict:
        """Always retrieve from the filings for EVERY sub-query.

        Earlier this skipped 'numeric' sub-queries and left them to the table
        agent — but the 10-K / annual-report prose contains the figures too, and
        when the tables collection is empty that left numeric questions with no
        grounding at all (→ "I don't have information"). The table agent now
        supplements retrieval, it does not replace it.
        """
        return super().hybrid_retrieve_node(state)

    def table_agent_node(self, state: AgentState) -> dict:
        """Run the table agent once per numeric sub-query."""
        sub_queries = state.get("sub_queries") or [state["question"]]
        routes = state.get("query_routes") or ["narrative"] * len(sub_queries)
        numeric_subs = [s for s, r in zip(sub_queries, routes) if r == "numeric"]
        if not numeric_subs:
            return {"table_results": []}

        results: list[dict] = []
        for sub_q in numeric_subs:
            try:
                res = self.table_agent.answer(sub_q)
                results.append({"sub_query": sub_q, **res})
            except Exception as e:
                results.append({
                    "sub_query": sub_q, "answer": "", "code": "",
                    "explanation": "", "tables_used": [], "stdout": "",
                    "error": f"{type(e).__name__}: {e}",
                })
                self._log(state, f"table_agent failed for {sub_q!r}: {e}")
        return {"table_results": results}

    def synthesize_node(self, state: AgentState) -> dict:
        """Merge text excerpts + table results into the final answer."""
        from langchain_core.messages import HumanMessage, SystemMessage

        chunks = state.get("retrieved_chunks", [])
        table_res = state.get("table_results", [])
        text_context = (
            "\n\n".join(f"{c['source']}\n{c['text']}" for c in chunks) or "(none)"
        )
        table_context = self._format_table_results(table_res) or "(none)"
        sub_queries = "\n".join(f"- {q}" for q in state.get("sub_queries", []))

        llm = self._get_llm("synth")
        prompt = SYNTH_V3_PROMPT.format(
            question=state["question"],
            sub_queries=sub_queries,
            text_context=text_context,
            table_context=table_context,
        )
        response = llm.invoke([
            SystemMessage(content=SYNTH_V3_SYSTEM),
            HumanMessage(content=prompt),
        ])
        answer = response.content

        # Citation extraction: text tags [ … ] AND table tags (Table: … ).
        citations = sorted(set(
            re.findall(r"\[[^\]]+\]|\(Table:[^)]+\)", answer)
        ))
        low_conf = (
            state.get("avg_grade", 0.0) < self.grade_threshold
            and state.get("iteration_count", 0) >= self.max_rewrites
            and not table_res  # tables can rescue a low text-retrieval grade
        )
        return {
            "draft_answer": answer,
            "final_answer": answer,
            "citations": citations,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "low_confidence": bool(low_conf),
        }

    def critic_node(self, state: AgentState) -> dict:
        """Critic checks claims against text chunks AND table-derived evidence."""
        table_res = state.get("table_results", [])
        augmented = list(state.get("retrieved_chunks", []))
        for t in table_res:
            if t.get("error") or not t.get("answer"):
                continue
            sources = ", ".join(
                f"(Table: {tu.get('title', '?')}, {tu.get('company', '?')} "
                f"{tu.get('year', '?')}, p. {tu.get('page', '?')})"
                for tu in t.get("tables_used", [])[:3]
            ) or "(Table)"
            augmented.append({
                "text": f"Computed from tables: {t.get('answer', '')}\n"
                        f"Code: {t.get('code', '')[:300]}",
                "source": sources,
                "company": "?", "year": "?", "page": "?",
                "sub_query": t.get("sub_query", ""),
            })

        proxy = dict(state)
        proxy["retrieved_chunks"] = augmented
        return super().critic_node(proxy)

    # ------------------------------------------------------------------ #
    # Graph
    # ------------------------------------------------------------------ #

    def _grade_router(self, state: AgentState) -> str:
        """Route to `table_agent` when text retrieval is good enough (or there
        is no narrative sub-query, or rewriting clearly won't help) — otherwise
        rewrite and retry."""
        routes = state.get("query_routes") or []
        has_narrative = any(r != "numeric" for r in routes)
        avg = state.get("avg_grade", 0.0)
        kept = state.get("retrieved_chunks") or []

        if not has_narrative:
            return "table_agent"
        # Grader filtered every chunk away AND the average was very low →
        # the entity isn't in the corpus; rewriting won't fix that. Skip ahead
        # to table/web so any out-of-corpus evidence can still answer.
        if not kept and avg < self.very_poor_grade:
            return "table_agent"
        if avg >= self.grade_threshold:
            return "table_agent"
        if state.get("iteration_count", 0) >= self.max_rewrites:
            return "table_agent"
        return "rewrite"

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(AgentState)
        g.add_node("planner", self.planner_node)
        g.add_node("router", self.router_node)
        g.add_node("retrieve", self.hybrid_retrieve_node)
        g.add_node("grader", self.grader_node)
        g.add_node("rewrite", self.rewrite_node)
        g.add_node("table_agent", self.table_agent_node)
        g.add_node("synthesize", self.synthesize_node)
        g.add_node("critic", self.critic_node)

        g.add_edge(START, "planner")
        g.add_edge("planner", "router")
        g.add_edge("router", "retrieve")
        g.add_edge("retrieve", "grader")
        g.add_conditional_edges(
            "grader", self._grade_router,
            {"rewrite": "rewrite", "table_agent": "table_agent"},
        )
        g.add_edge("rewrite", "retrieve")
        g.add_edge("table_agent", "synthesize")
        g.add_edge("synthesize", "critic")
        g.add_conditional_edges(
            "critic", self._critic_router,
            {"retrieve": "retrieve", "end": END},
        )
        return g.compile()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_table_results(table_res: list[dict]) -> str:
        parts: list[str] = []
        for t in table_res:
            if t.get("error") and not t.get("answer"):
                parts.append(f"- ({t['sub_query']}): error: {t['error']}")
                continue
            sources = ", ".join(
                f"(Table: {tu.get('title', '?')}, {tu.get('company', '?')} "
                f"{tu.get('year', '?')}, p. {tu.get('page', '?')})"
                for tu in t.get("tables_used", [])[:3]
            ) or "(no source)"
            parts.append(
                f"- Sub-query: {t['sub_query']}\n"
                f"  Computed: {t.get('answer', '')[:600]}\n"
                f"  Sources:  {sources}"
            )
        return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the table-augmented RAG (v3) graph.")
    p.add_argument("--collection", default="us_filings")
    p.add_argument("--chroma-dir", default="data/chroma")
    p.add_argument("--market", choices=["india", "us"], default="us")
    p.add_argument("--table-collection", default="tables")
    p.add_argument("--provider", choices=["groq", "gemini", "openai", "anthropic"],
                   default="groq")
    p.add_argument("--planner-model", default=None)
    p.add_argument("--synth-model", default=None)
    p.add_argument("--critic-model", default=None)
    p.add_argument("--grader-model", default=None)
    p.add_argument("--router-model", default=None)
    p.add_argument("--code-model", default=None)
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--reranker-model", default=HybridRetriever.DEFAULT_RERANKER)
    p.add_argument("--bm25-top-k", type=int, default=10)
    p.add_argument("--dense-top-k", type=int, default=10)
    p.add_argument("--final-top-k", type=int, default=5)
    p.add_argument("--table-top-k", type=int, default=3)
    p.add_argument("--grade-threshold", type=float, default=3.0)
    p.add_argument("--max-rewrites", type=int, default=3)
    p.add_argument("--max-critic-retries", type=int, default=2)
    p.add_argument("--question", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--question-col", default="question")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--output", default="results/full_rag_outputs.json")
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

    agent = AgenticRAGv3(
        collection_name=args.collection,
        chroma_dir=args.chroma_dir,
        market=args.market,
        embedding_model=args.embedding_model,
        provider=args.provider,
        planner_model=args.planner_model,
        synth_model=args.synth_model,
        critic_model=args.critic_model,
        grader_model=args.grader_model,
        router_model=args.router_model,
        code_model=args.code_model,
        reranker_model=args.reranker_model,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        final_top_k=args.final_top_k,
        table_collection=args.table_collection,
        table_top_k=args.table_top_k,
        grade_threshold=args.grade_threshold,
        max_rewrites=args.max_rewrites,
        max_critic_retries=args.max_critic_retries,
    )

    if args.dataset:
        df = _load_dataset(args.dataset)
        if args.sample:
            df = df.head(args.sample)
        agent.run_dataset(df, output_path=args.output, question_col=args.question_col)
        return

    if not args.question:
        raise SystemExit("Provide --question or --dataset.")

    state = agent.run(args.question)
    print("\n" + "=" * 60)
    print(f"Question:        {state['question']}")
    print(f"Sub-queries:     {state.get('sub_queries')}")
    print(f"Query routes:    {state.get('query_routes')}")
    print(f"Grades:          {state.get('grades')}  (avg {state.get('avg_grade')})")
    print(f"Rewrites used:   {state.get('iteration_count', 0)}")
    print(f"Critic retries:  {state.get('critic_iterations', 0)}")
    print(f"\nAnswer:\n{state.get('final_answer')}")
    print(f"\nCitations:       {state.get('citations')}")
    print(f"Critic grade:    {state.get('grading_score')}  needs_retry={state.get('needs_retry')}")
    print(f"Low confidence:  {state.get('low_confidence')}")
    if state.get("table_results"):
        print(f"\nTable computations: {len(state['table_results'])}")
        for t in state["table_results"]:
            print(f"  - {t['sub_query']}: {t.get('answer', '')[:120]}")
    if state.get("errors"):
        print(f"\nErrors:          {state['errors']}")


if __name__ == "__main__":
    main()
