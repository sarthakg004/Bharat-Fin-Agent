"""The query router and the retrieval query rewriter.

Adds to `corrective.py`: one fused planner+router call that decomposes the
question AND classifies each sub-query (`narrative` / `numeric` / `market` /
`external` / `cross_document`), plus the rewrite that turns the question into
one query in the filing's own vocabulary. `agent.py` builds on this.
"""

from __future__ import annotations

import re
from typing import Optional

from finagent.graph.corrective import AgenticRAGv2
from finagent.graph.state import AgentState, QueryPlan, RouterReport
from finagent.llm import text_of
from finagent.prompts.planner import (
    RETRIEVAL_QUERY_SHOTS, RETRIEVAL_QUERY_SYSTEM, ROUTER_PROMPT, ROUTER_SYSTEM,
)

# Below this, a "retrieval query" is a failed generation rather than a terse
# one: the model answered the question or returned nothing. The shortest useful
# real query is company + year + a caption, which clears this comfortably.
MIN_RETRIEVAL_QUERY_WORDS = 6

# …and the other failure shape, which the length floor cannot see. The model
# sometimes REFUSES at length instead of writing a query — one FinanceBench
# generation returned "I'm not able to retrieve the specific figures from
# Walmart's filings needed to compute the EBITDA…", which is 20+ words, passes
# the floor, and then gets searched verbatim. The prompt asks for keywords with
# no verbs and no question words, so first-person prose is never a valid query.
_NOT_A_QUERY = re.compile(
    r"\b(i am|i'm|i cannot|i can't|i am not|i'm not|unable to|not able to|"
    r"as an ai|sorry|apolog|there is no|cannot determine|insufficient "
    r"information|does not (?:provide|contain))\b", re.I)


def _first_line(text: str) -> str:
    """First non-empty line, stripped of the wrappers models add anyway."""
    for line in (text or "").splitlines():
        line = line.strip().strip('"').strip("`").strip()
        for prefix in ("query:", "search query:", "rewrite:"):
            if line.lower().startswith(prefix):
                line = line[len(prefix):].strip()
        if line:
            return line
    return ""


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
- TIME: you are given today's date. Name an absolute fiscal year ONLY when the
  question names one. When it doesn't — or says "latest", "current", "most
  recent", "year-over-year" — it means the newest data available NOW; write
  "latest fiscal year" / "prior fiscal year" in the sub-query ("ServiceNow
  operating margin, latest fiscal year vs prior fiscal year"). Resolve relative
  periods ("last year") against today's date. NEVER fill in a year from your
  training data — your memory of "recent" years is stale.
- DRIVER / CONTRIBUTION questions ("which expense categories contributed most
  to the operating-margin change?") → enumerate the components so they can be
  compared: one `numeric` sub-query per major income-statement line for BOTH
  periods (revenue, cost of revenue, S&M, R&D, G&A — "latest vs prior
  period"), plus ONE `narrative` sub-query for the MD&A's own explanation of
  the change. The synthesiser then shows which lines grew slower than revenue.
- Do NOT invent a metric the question didn't ask for. A qualitative question
  ("was the company able to retain customers?", "what drove the margin change?")
  must stay qualitative — do NOT rewrite it into a made-up ratio ("retention
  rate", "margin change %") that the filing won't report. Keep its wording and
  route it `narrative`.
- FOLLOW-UPS: if the question relies on the conversation above ("show me the
  chart", "what about last year"), rewrite it into a self-contained sub-query
  naming the company/ticker discussed just before.

Routing lanes (per sub-query)
-----------------------------
- numeric: the answer hinges on a SPECIFIC reported figure or COMPUTABLE ratio
  — a named line item, ratio, margin, growth %, or multi-year comparison. This
  INCLUDES yes/no or "healthy?" judgements that turn on a named metric ("does X
  have a healthy quick ratio?", "is X capital-intensive?", "is the current
  ratio adequate?") — route them `numeric` so the calculator computes the ratio
  from XBRL; the synthesiser adds the judgement on top. Examples: "FY2016 COGS",
  "FY2021 inventory turnover", "EBITDA margin FY21→FY23", "quick ratio FY2022".
  Answered from XBRL facts and a deterministic calculator (see the v4 subclass).
- narrative: text retrieval over filings, for PURELY QUALITATIVE questions with
  no single computable metric — strategy, risks, segment/MD&A commentary,
  drivers ("what drove the margin change", "why did revenue fall"), and
  qualitative judgements ("was X able to retain customers", "describe X's
  strategy") about ONE named company. When unsure between numeric and narrative
  and a concrete metric is nameable, prefer `numeric` (it retrieves the filing
  prose too, AND computes the figure).
- market: live market data (yfinance). Anything about a listed company's
  MARKET behaviour — current/premarket/intraday price, OHLC history, charts,
  52-week range, ticker news. Lean `market` whenever the question is in the
  direction of the stock ("how is X doing", "is X a good stock", "X stock").
- cross_document: EDGAR full-text search across MANY companies' filings — the
  answer is a SET of companies ("which companies disclosed X"), not facts
  about one named company.
- external: general web search. ONLY for macro news, corporate events, M&A
  timelines, or post-cutoff developments. Do NOT route a question about ONE
  named company's own filing here — that is `narrative`/`numeric` (the filing,
  or a fetched 10-K, is the authoritative source; the web is a last resort the
  retrieval lanes escalate to on their own when the corpus genuinely lacks it).

Examples:
  - "What is the FY2016 COGS for Microsoft?"                      → numeric
  - "What is Nike's FY2021 inventory turnover ratio?"            → numeric
  - "What drove the gross-margin change for J&J in FY2022?"      → narrative
  - "Was American Express able to retain card members in 2022?"  → narrative
  - "Does AMD have a healthy liquidity profile (quick ratio)?"   → narrative
  - "Describe Microsoft's AI strategy"                            → narrative
  - "What is Nvidia's current share price?"                      → market
  - "Which companies disclosed a material weakness in FY2023?"    → cross_document
  - "Latest macro headlines today"                                → external
"""

PLAN_ROUTE_PROMPT = """\
Today's date: {today}

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
"Apple's FY24 revenue was $391 billion [1]."
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

_NUMERIC_MARKERS = (
    "what was", "what is", "how much", "how many",
    "ratio", "margin", "growth", "yield", "return on",
    "%", "$", "billion", "million",
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


class AgenticRAGv3(AgenticRAGv2):
    """Hybrid retrieval + a router that classifies each sub-query."""

    def _get_router_llm(self):
        return self._get_llm("router")

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

    def planner_node(self, state: AgentState) -> dict:
        """Fused planner+router: ONE structured call decomposes the question
        AND routes each sub-query, replacing the two sequential LLM calls the
        v1/v2 path makes. Runs on the `planner` role — which is a STRONG model,
        because decomposition sits on the quality path. It used to ask for the
        `router` role instead, which meant `DEFAULTS[...]["planner"]` configured
        nothing and the UI's model picker re-pointed the planner as a side
        effect of overriding synth. On failure it falls back to the base planner
        (router_node then classifies)."""
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

        from datetime import date

        # Uploaded documents ride `fetched_chunks`; without naming them here the
        # planner can't anchor relative periods ("year-over-year") to the
        # document the user is actually asking about (e.g. a Q1 10-Q means
        # this-quarter vs same-quarter-last-year, not fiscal-year YoY).
        uploads = sorted({c.get("filename") for c in (state.get("fetched_chunks") or [])
                          if c.get("filing_type") == "uploaded" and c.get("filename")})
        upload_block = ""
        if uploads:
            upload_block = (
                "The user attached document(s) to this question: "
                + ", ".join(uploads)
                + ". Interpret the question — including relative periods like "
                "'latest' or 'year-over-year' — against these documents; a "
                "quarterly filing (10-Q) means the comparison is that quarter "
                "vs the same quarter a year earlier.\n\n")

        llm = self._get_llm("planner").with_structured_output(QueryPlan)
        try:
            out: QueryPlan = llm.invoke([
                SystemMessage(content=PLAN_ROUTE_SYSTEM),
                HumanMessage(content=history_block + upload_block
                             + PLAN_ROUTE_PROMPT.format(
                                 question=question, today=date.today().isoformat())),
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
                "query_routes": [r for _, r in pairs],
                "retrieval_query": self._retrieval_query(
                    state, question, [r for _, r in pairs],
                    history_block + upload_block)}

    def _retrieval_query(self, state: AgentState, question: str,
                         routes: list[str], context_block: str = "") -> str:
        """Write the ONE query that searches the filings — the §14 rewrite.

        A SECOND call, deliberately not fused into the plan above. The two jobs
        pull in opposite directions: the plan routes and decomposes in prose,
        this writes a terse keyword query whose rules are mostly prohibitions
        (no derived names, no filler, both fiscal years). Measured separately at
        72/99 hit@8 vs 67 for retrieving on the sub-queries.

        Skipped entirely — no call, no cost — when the planner routed every
        sub-query to yfinance/web/EDGAR, because then the filings are not the
        source. Returns "" on any failure, which degrades retrieval to the
        sub-query path rather than losing the turn.
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        if not any(r in ("narrative", "numeric") for r in routes):
            return ""
        msgs: list = [SystemMessage(content=RETRIEVAL_QUERY_SYSTEM)]
        for shot_q, shot_a in RETRIEVAL_QUERY_SHOTS:
            msgs += [HumanMessage(content=shot_q), AIMessage(content=shot_a)]
        msgs.append(HumanMessage(content=context_block + question))
        try:
            out = _first_line(text_of(self._get_llm("planner").invoke(msgs)))
        except Exception as e:
            self._log(state, f"retrieval-query rewrite failed ({e}); "
                             f"retrieving on the sub-queries")
            return ""
        # Two ways this comes back unusable, and BOTH have been observed on
        # FinanceBench. Too short means the model answered the question instead
        # of writing a query for it ("Approximately 1.33."). Refusal prose means
        # it declined at length — long enough to clear the floor, and then
        # searched verbatim. Either way the sub-queries are a better fallback
        # than a bad query, so return "" and let retrieval degrade to them.
        if len(out.split()) < MIN_RETRIEVAL_QUERY_WORDS or _NOT_A_QUERY.search(out):
            self._log(state, f"discarded a non-query rewrite ({out[:60]!r}); "
                             f"retrieving on the sub-queries")
            return ""
        return out

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

        Earlier this skipped 'numeric' sub-queries — but the 10-K / annual-report
        prose contains the figures too, and skipping them left numeric questions
        with no grounding at all (→ "I don't have information"). Retrieval runs
        for every sub-query; the structured numeric lanes (v4) supplement it.
        """
        return super().hybrid_retrieve_node(state)
