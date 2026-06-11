"""
state.py  ·  finagent/graph/state.py

Shared state for the agentic RAG LangGraph, plus the Pydantic schemas used
for structured LLM outputs (planner sub-queries, critic verdict).

`AgentState` is a TypedDict — the canonical LangGraph state schema. Every node
reads it and returns a partial dict that LangGraph merges back in. Each field
is written by exactly one node, so no custom reducers are needed.

Fields that aren't used yet (table_results, web_results) are placeholders for
later weeks (the table agent and web-search agent). They default to empty lists
so downstream nodes can treat them uniformly.
"""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
    """Shared state passed between every node in the graph."""

    question: str                      # the user's original question
    sub_queries: list[str]             # planner's decomposition (1-3 queries)
    retrieved_chunks: list[dict]       # [{text, company, year, page, source, sub_query}]
    table_results: list[Any]           # placeholder — future table agent
    web_results: list[Any]             # placeholder — future web-search agent
    draft_answer: str                  # synthesizer output (pre-critique)
    final_answer: str                  # answer returned to the caller
    citations: list[str]               # citation tags used, e.g. "[TCS AR FY23, p. 102]"
    grading_score: float               # critic: fraction of claims supported (0-1)
    iteration_count: int               # how many synthesize attempts so far
    errors: list[str]                  # non-fatal problems logged along the way
    needs_retry: bool                  # critic flag; drives the critic-retry loop

    # --- Corrective-RAG additions (graded retrieval + rewrite loop) -------- #
    grades: list[int]                  # grader: per-chunk relevance score (1-5)
    avg_grade: float                   # grader: mean of grades
    rewrite_history: list[str]         # past rewritten sub-queries
    critic_iterations: int             # critic-retry counter (cap = max_critic_retries)
    critic_feedback: list[str]         # unsupported claims, fed to an active re-draft
    low_confidence: bool               # synth flag when grader stayed below threshold

    # --- Table-agent additions (v3) ---------------------------------------- #
    # `table_results` is already declared above; the v3 graph fills it with
    # one entry per numeric sub-query.
    query_routes: list[str]            # one of "narrative" | "numeric" | "external" per sub-query

    # --- v4 additions: i18n, web search, numeric verification, refusal ----- #
    web_results: list[Any]             # already declared above; filled by web_search_node
    numeric_verification: dict         # {"claims": [...], "unverified": [...], "score": 0..1}
    verification_report: dict          # #5: {numeric, cross_source, units, sources}
    verify_iterations: int             # how many times verify_numbers has run (bounds re-route)
    refused: bool                      # set when the agent explicitly declines to answer

    # --- Market-data tool results ----------------------------------------- #
    market_data: list[dict]            # one entry per tool call (quote/history/...)
    charts: list[dict]                 # frontend-ready chart specs (lightweight-charts JSON)

    # --- XBRL structured facts (Phase 3) ---------------------------------- #
    xbrl_facts: list[dict]             # one exact reported figure per numeric sub-query

    # --- Derived metrics computed over XBRL (Phase 4) --------------------- #
    calc_results: list[dict]           # margins/ratios/growth/CAGR/trends from XBRL inputs

    # --- Dynamic SEC fetch (Phase 5) -------------------------------------- #
    fetch_status: dict                 # {"decision","ticker","chunks_added",...} or {}
    fetched_chunks: list[dict]         # ephemeral fetch: in-memory chunks (not indexed)

    # --- EDGAR full-text search (Phase 6) --------------------------------- #
    edgar_results: list[dict]          # one cross-document search result per sub-query

    # --- Conversation memory ---------------------------------------------- #
    chat_history: list[dict]           # last K turns: [{role: "user"|"assistant", content}]

    # --- Normalised evidence (#3) ----------------------------------------- #
    # Every lane's output (XBRL, calc, filing text, tables, market, web, EDGAR)
    # projected into one common shape by `evidence_builder_node`, so downstream
    # nodes, the audit trail, and cross-source checks read a single structure
    # instead of seven bespoke ones. Each item:
    #   {kind, fact, value, unit, source, citation, confidence, sub_query}
    evidence: list[dict]

    # --- Confidence framework (#8/9) -------------------------------------- #
    # Four sub-scores in [0,1], then their weighted blend, computed once after
    # verification by `confidence_node`. Any sub-score that doesn't apply to a
    # given question (e.g. retrieval on a pure-XBRL numeric query, or
    # verification on an answer with no figures) is left out and the remaining
    # weights are renormalised — so a fully-grounded numeric answer isn't
    # penalised for not having gone through retrieval.
    retrieval_score: float             # from avg_grade (1-5) → [0,1]
    verification_score: float          # numeric_verification["score"]
    citation_score: float              # fraction of figures carrying a [N] marker
    critic_score: float                # grading_score (critic claim-support rate)
    confidence: float                  # weighted blend of the applicable sub-scores
    confidence_band: str               # "answer" | "warn" | "refuse" (gate routing)
    status: str                        # "answered" | "answered_with_warning" | "refused"

    # --- Tools-lane corpus fallback ---------------------------------------- #
    # Set by `evidence_builder_node` when a tools-path question (which skipped
    # retrieval) produced NO evidence — routes back through fetch_filing →
    # retrieve so the corpus gets a chance before synthesis. `pending` is the
    # one-shot routing signal (cleared every evidence_builder pass); `used`
    # latches so the fallback can only fire once per run.
    corpus_fallback_pending: bool
    corpus_fallback_used: bool

    # --- Insufficient-draft web escalation --------------------------------- #
    # Set by `critic_node` when the draft ADMITS the evidence can't answer
    # ("not specified in the sources", "no information available") and the web
    # lane hasn't run — routes critic → web_search → re-synthesize so the agent
    # exhausts its tools before telling the user it doesn't know. Same
    # pending/used pattern as the corpus fallback above.
    web_fallback_pending: bool
    web_fallback_used: bool


# --------------------------------------------------------------------------- #
# Structured-output schemas (used with llm.with_structured_output(...))
# --------------------------------------------------------------------------- #

class SubQueries(BaseModel):
    """Planner output: the question decomposed into focused sub-queries."""

    queries: list[str] = Field(
        description=(
            "1 to 8 self-contained sub-queries. Use a SINGLE query for simple "
            "questions. For comparison / multi-hop questions, FULLY enumerate one "
            "query per (entity × period × metric) combination so nothing is "
            "dropped — e.g. 'compare Apple and Microsoft R&D % of revenue over "
            "2020-2022' becomes SIX queries (each company × each of the 3 years). "
            "Each sub-query must name its company, metric, and period explicitly."
        )
    )


class ClaimVerdict(BaseModel):
    """One factual claim from the draft answer and whether the context supports it."""

    claim: str = Field(description="A single factual claim extracted from the answer.")
    supported: bool = Field(description="True if the retrieved context supports the claim.")
    reason: str = Field(description="Brief justification for the verdict.")


class CriticReport(BaseModel):
    """Critic output: per-claim support verdicts for the draft answer."""

    verdicts: list[ClaimVerdict] = Field(
        description="One entry per factual claim found in the answer."
    )


# --------------------------------------------------------------------------- #
# Corrective-RAG schemas
# --------------------------------------------------------------------------- #

class ChunkScore(BaseModel):
    """Per-excerpt relevance score from the grader."""

    score: int = Field(
        ge=1, le=5,
        description="1 = irrelevant, 3 = related, 5 = directly answers the question",
    )
    reason: str = Field(description="Brief justification for the score.")


class GraderReport(BaseModel):
    """Grader output: one score per excerpt, in the same order as presented."""

    scores: list[ChunkScore] = Field(
        description="Per-excerpt scores, in input order."
    )


class RewrittenQuery(BaseModel):
    """Rewriter output: a single reformulated, retrieval-friendly question."""

    query: str = Field(
        description=(
            "The reformulated question. Self-contained, ideally with "
            "domain-specific synonyms (e.g. 'net profit attributable to "
            "shareholders' instead of 'earnings')."
        )
    )


# --------------------------------------------------------------------------- #
# Router schemas (table-augmented RAG)
# --------------------------------------------------------------------------- #

from typing import Literal


class QueryRoute(BaseModel):
    """Per-sub-query routing verdict."""

    sub_query: str = Field(description="The sub-query being classified, copied verbatim.")
    route: Literal["narrative", "numeric", "market", "external", "cross_document"] = Field(
        description=(
            "narrative = text retrieval over filings (default for prose-y questions); "
            "numeric   = table agent over extracted filing tables (ratios from 10-Ks, "
            "segment breakdowns, multi-year financial comparisons IN the filings); "
            "market    = live market data via yfinance (current price, intraday move, "
            "premarket, historical OHLC, charts, news headlines — anything about a "
            "listed company's market behaviour, not about its filings); "
            "cross_document = EDGAR full-text search across MANY companies' filings — "
            "'which companies disclosed/mentioned X', 'list firms that report Y'; the "
            "answer is a SET of companies, not facts about one named company; "
            "external  = web search (general news, events, post-cutoff)."
        )
    )
    reason: str = Field(description="One-line justification.")


class RouterReport(BaseModel):
    """Router output: one verdict per sub-query, in the same order."""

    routes: list[QueryRoute] = Field(description="One per sub-query, in input order.")


class PlannedQuery(BaseModel):
    """One sub-query plus the lane that should answer it (fused planner+router)."""

    query: str = Field(
        description=(
            "A self-contained sub-query naming its company, metric, and period "
            "explicitly (no pronouns referring back to the question)."
        )
    )
    route: Literal["narrative", "numeric", "market", "external", "cross_document"] = Field(
        description=(
            "narrative = text retrieval over filings; numeric = exact figures / "
            "ratios from filings (XBRL + tables); market = live market data "
            "(yfinance); cross_document = EDGAR full-text search across many "
            "companies; external = general web search."
        )
    )


class QueryPlan(BaseModel):
    """Fused planner+router output: routed sub-queries in one structured call."""

    queries: list[PlannedQuery] = Field(
        description=(
            "1 to 8 routed sub-queries. ONE for simple questions; for "
            "comparison / multi-hop questions FULLY enumerate one per "
            "(entity × period × metric) combination."
        )
    )


class TableTitle(BaseModel):
    """LLM-extracted title for a single table."""

    title: str = Field(description="Concise table title (≤ 120 chars).")


class PandasCode(BaseModel):
    """Pandas code emitted by the table agent to compute an answer."""

    code: str = Field(description="Python code; assigns the final answer to `result`.")
    explanation: str = Field(description="One sentence explaining what the code does.")


# --------------------------------------------------------------------------- #
# v4 schemas (translation, numeric verification)
# --------------------------------------------------------------------------- #

class NumericClaim(BaseModel):
    """One numeric claim extracted from the draft answer."""

    claim: str = Field(description="The full claim, including the figure and what it refers to.")
    number: str = Field(description="The numeric value as written (e.g. '₹9,14,472 crore', '23.4%').")
    matched: bool = Field(description="True if the figure appears verbatim or as a close paraphrase in the supplied evidence.")
    evidence: str = Field(description="Short quote from the evidence that supports the figure, or '' if matched is False.")


class NumericVerification(BaseModel):
    """Verifier output for all numeric claims in the draft answer."""

    claims: list[NumericClaim] = Field(description="One entry per distinct numeric claim.")


# --------------------------------------------------------------------------- #
# Market-data tool selection
# --------------------------------------------------------------------------- #

class XBRLQuery(BaseModel):
    """Structured extraction of a numeric sub-query for the XBRL facts tool.

    Turns "What was Apple's total revenue in FY2022?" into
    (ticker='AAPL', concept='revenue', period='FY2022') so the XBRL client can
    look up the exact reported figure. `answerable` is False when the sub-query
    isn't a single-company single-figure lookup (comparisons, derived ratios,
    or narrative questions) — those stay with retrieval / the table agent.
    """

    answerable: bool = Field(
        default=False,
        description="True only if this asks for ONE reported line-item figure for "
                    "ONE US-listed company and period (revenue, net income, total "
                    "assets, EPS, R&D, etc.). False for ratios/growth/comparisons.",
    )
    ticker: str = Field(
        default="",
        description="Ticker or company name to resolve (e.g. 'AAPL' or 'Apple').",
    )
    concept: str = Field(
        default="",
        description="The financial line item in plain words: 'revenue', 'net "
                    "income', 'total assets', 'gross profit', 'diluted EPS', etc.",
    )
    period: str = Field(
        default="",
        description="Fiscal YEAR like 'FY2022' or '2021' ONLY if the question names a "
                    "specific year. Leave EMPTY when no year is given — the tool then "
                    "returns the most recent data (do not guess a year).",
    )
    quarterly: bool = Field(
        default=False,
        description="True if the question asks for a QUARTERLY figure ('last quarter', "
                    "'most recent quarter', 'Q3', 'quarterly EPS'). The tool then returns "
                    "the latest quarter (10-Q) instead of the annual (10-K) figure.",
    )


class CalcQuery(BaseModel):
    """Structured extraction of a *derived-metric* numeric sub-query (Phase 4).

    Turns "What was Apple's operating margin trend over the last 3 years?" into
    (is_derived=True, ticker='AAPL', metric='operating_margin',
    periods=['FY2020','FY2021','FY2022']). Plain single-figure lookups
    (is_derived=False) are left to the XBRL facts node.
    """

    is_derived: bool = Field(
        default=False,
        description="True only if the sub-query asks for a DERIVED metric: a "
                    "margin, ratio, YoY growth, CAGR, or a multi-year trend of one "
                    "of those. False for a single reported figure (handled by XBRL).",
    )
    ticker: str = Field(default="", description="Ticker or company name (e.g. 'AAPL').")
    metric: str = Field(
        default="",
        description="Canonical metric: gross_margin, operating_margin, net_margin, "
                    "current_ratio, quick_ratio, cash_ratio, debt_to_equity, "
                    "return_on_equity, return_on_assets, asset_turnover, "
                    "interest_coverage, dio, dso, dpo, ccc, growth, cagr.",
    )
    concept: str = Field(
        default="",
        description="For 'growth'/'cagr' only: the underlying line item "
                    "(revenue, net income, …) whose change is measured.",
    )
    periods: list[str] = Field(
        default_factory=list,
        description="Fiscal periods involved, e.g. ['FY2020','FY2021','FY2022']. "
                    "Two periods for growth/cagr (earliest first); 2-5 for a trend.",
    )


class XBRLQueryBatch(BaseModel):
    """Batch form of `XBRLQuery`: one extraction per numbered sub-query, in the
    same order. Lets the agent extract ALL numeric sub-queries in ONE LLM call
    instead of one call each."""

    queries: list[XBRLQuery] = Field(
        default_factory=list,
        description="Exactly one XBRLQuery per numbered sub-query, in order.",
    )


class CalcQueryBatch(BaseModel):
    """Batch form of `CalcQuery`: one extraction per numbered sub-query, in order."""

    queries: list[CalcQuery] = Field(
        default_factory=list,
        description="Exactly one CalcQuery per numbered sub-query, in order.",
    )


class EdgarQuery(BaseModel):
    """Extraction of an EDGAR full-text search from a cross-document sub-query (Phase 6).

    Turns "Which companies disclosed a material weakness in internal controls?"
    into (phrase='material weakness in internal controls', forms='10-K'). The
    phrase is what gets full-text-searched across every company's filings.
    """

    phrase: str = Field(
        default="",
        description="The exact concept to full-text search across filings — the "
                    "distinctive words/phrase, NOT the whole question. e.g. "
                    "'material weakness in internal controls', 'going concern'.",
    )
    forms: str = Field(
        default="10-K",
        description="SEC form to restrict to (e.g. '10-K', '10-Q', '8-K'). "
                    "Default '10-K' for annual-report disclosures.",
    )


class CorpusGateQuery(BaseModel):
    """The primary company a question is about, for the dynamic-fetch gate (Phase 5).

    Extracts the single company/ticker the question concerns so the gate can
    decide whether to fetch its 10-K. Empty when the question names no specific
    company or spans many (e.g. a macro/market-wide question).
    """

    company: str = Field(
        default="",
        description="The one company the question is primarily about — a ticker "
                    "('CRM') or name ('Salesforce'). Empty if none or many.",
    )


class MarketIntent(BaseModel):
    """The market-data node's plan — a SINGLE tool call.

    Deliberately a FLAT schema (no nested list of objects): small models like
    gpt-oss reliably emit a flat object but choke on `list[NestedModel]`,
    failing the function call. One call covers virtually every market question
    (`compare` itself takes a list of tickers).
    """

    tool: Literal["none", "get_quote", "get_history", "get_company_info", "get_news", "compare"] = Field(
        default="none",
        description="Market tool to invoke; 'none' if the question isn't about market data.",
    )
    company: str = Field(
        default="",
        description="The company's NAME in words (e.g. 'Rocket Lab', 'Apple'). Used to "
                    "resolve the correct ticker from SEC data — fill this whenever you "
                    "know the company, even if you also guess a symbol.",
    )
    symbol: str = Field(
        default="",
        description="Yahoo ticker if you're confident (e.g. AAPL, TSLA). Empty for `compare`. "
                    "Don't append an exchange suffix like '.NASDAQ'.",
    )
    symbols: list[str] = Field(
        default_factory=list,
        description="Tickers for `compare` (2-6); ignored by other tools.",
    )
    period: str = Field(
        default="1y",
        description="History period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.",
    )
    interval: str = Field(
        default="1d",
        description="History interval: 1d, 1wk, 1mo.",
    )
