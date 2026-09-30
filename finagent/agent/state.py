"""The shared state every step reads and writes, and the schemas for
structured LLM output.

`AgentState` is a plain dict. Each step returns only the keys it changed and
LangGraph merges them back in.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
    question: str
    chat_history: list[dict]        # last turns: [{role, content}]

    # planner
    sub_queries: list[str]          # the question split into 1-8 parts
    query_routes: list[str]         # one lane per part: narrative | numeric | market | external | cross_document
    retrieval_query: str            # ONE keyword query in the filing's own words; "" = search on the parts

    # filings
    fetch_status: dict              # what the 10-K fetch gate decided and did
    retrieved_chunks: list[dict]    # [{text, company, year, page, source, sub_query}]
    retry_queries: list[str]        # set by the critic: what to search for on a "gather" retry

    # tools
    xbrl_facts: list[dict]          # exact filed figures
    calc_results: list[dict]        # ratios, margins, growth computed from XBRL
    market_data: list[dict]         # yfinance tool calls
    charts: list[dict]              # chart specs the frontend renders
    web_results: list[dict]
    edgar_results: list[dict]

    # answer
    evidence: list[dict]            # the numbered evidence the writer was given, in [N] order
    answer: str
    citations: list[str]
    support_score: float            # critic: share of the answer's claims the evidence supports
    unsupported_claims: list[str]
    remedy: str                     # critic's fix: "redraft" | "gather"
    needs_retry: bool
    recoveries: int                 # recovery passes used (the cap is 1)
    refused: bool

    # one-shot routing flags
    corpus_fallback_pending: bool   # tool lanes all empty -> try filing search once
    corpus_fallback_used: bool
    web_fallback_pending: bool      # the draft admits it cannot answer -> try the web once
    web_fallback_used: bool

    log: list[str]                  # what happened, for debugging
    notices: list[str]              # degraded steps the user should know about


# --------------------------------------------------------------------------- #
# Structured-output schemas (used with llm.with_structured_output(...))
# --------------------------------------------------------------------------- #

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
    remedy: Literal["redraft", "gather"] = Field(
        default="redraft",
        description=(
            "What would fix the unsupported claims. 'redraft' = the evidence "
            "IS sufficient but the answer overstated it, so rewriting against "
            "the same evidence fixes it. 'gather' = the evidence genuinely "
            "does not contain the fact, so more retrieval is needed. Ignored "
            "when every claim is supported."
        ),
    )


class PlannedQuery(BaseModel):
    """One sub-query plus the lane that should answer it."""

    query: str = Field(
        description=(
            "A self-contained sub-query naming its company, metric, and period "
            "explicitly (no pronouns referring back to the question)."
        )
    )
    route: Literal["narrative", "numeric", "market", "external", "cross_document"] = Field(
        description=(
            "narrative = text retrieval over filings; numeric = exact figures / "
            "ratios from filings (SEC XBRL + calculator); market = live market data "
            "(yfinance); cross_document = EDGAR full-text search across many "
            "companies; external = general web search."
        )
    )


class QueryPlan(BaseModel):
    """Planner output: the question split into routed sub-queries."""

    queries: list[PlannedQuery] = Field(
        description=(
            "1 to 8 routed sub-queries. ONE for simple questions; for "
            "comparison / multi-hop questions FULLY enumerate one per "
            "(entity × period × metric) combination."
        )
    )


# --------------------------------------------------------------------------- #
# Tool-extraction schemas (XBRL, calculator, EDGAR, market data)
# --------------------------------------------------------------------------- #

class XBRLQuery(BaseModel):
    """Structured extraction of a numeric sub-query for the XBRL facts tool.

    Turns "What was Apple's total revenue in FY2022?" into
    (ticker='AAPL', concept='revenue', period='FY2022') so the XBRL client can
    look up the exact reported figure. `answerable` is False when the sub-query
    isn't a single-company single-figure lookup (comparisons, derived ratios,
    or narrative questions) — those stay with retrieval and the calculator.
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
    """Structured extraction of a *derived-metric* numeric sub-query.

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
        description="ONLY fiscal years the sub-query itself names, e.g. "
                    "['FY2020','FY2021']. Two periods for growth/cagr (earliest "
                    "first); 2-5 for a trend. Leave EMPTY when the sub-query "
                    "names no absolute year ('latest', 'prior fiscal year', "
                    "'year-over-year') — the tool resolves the newest filed "
                    "periods itself; NEVER guess a year.",
    )
    quarterly: bool = Field(
        default=False,
        description="True if the metric is asked for a QUARTER ('last quarter', "
                    "'Q1', 'most recent 10-Q') rather than a fiscal year.",
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


class FormulaSpec(BaseModel):
    """An LLM-planned formula for a metric the calculator doesn't hardcode (or
    that the question redefines), expressed over CANONICAL XBRL concepts.

    The LLM supplies only the STRUCTURE (which concepts, added/subtracted, over
    what denominator); the numbers are exact XBRL facts and the arithmetic is
    done deterministically in `FinancialCalculator.ratio_from_spec`, so a planned
    metric is as auditable — and as faithful — as a hardcoded one. Flat lists of
    concept names (no nested objects) so structured output is reliable.
    """

    ok: bool = Field(
        default=True,
        description="False if the metric cannot be expressed from the available "
                    "concepts (then the lane falls back / skips).",
    )
    numerator_add: list[str] = Field(
        default_factory=list,
        description="Canonical concepts SUMMED in the numerator (e.g. "
                    "['operating_income','depreciation_amortization']).",
    )
    numerator_sub: list[str] = Field(
        default_factory=list, description="Canonical concepts SUBTRACTED in the numerator.")
    denominator_add: list[str] = Field(
        default_factory=list,
        description="Canonical concepts SUMMED in the denominator. LEAVE EMPTY "
                    "when the metric is a dollar amount (numerator only), not a ratio.")
    denominator_sub: list[str] = Field(
        default_factory=list, description="Canonical concepts SUBTRACTED in the denominator.")
    average_denominator: bool = Field(
        default=False,
        description="True when the denominator is a balance-sheet stock that "
                    "convention averages over (t-1, t) — turnover and return "
                    "ratios (revenue/avg assets, COGS/avg inventory, etc.).")
    is_percent: bool = Field(
        default=False,
        description="True if the result is conventionally shown as a percent "
                    "(margins, returns, %-of-revenue).")


class EdgarQuery(BaseModel):
    """Extraction of an EDGAR full-text search from a cross-document sub-query.

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
    """The primary company a question is about, for the dynamic-fetch gate.

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

    A flat schema on purpose: small models fill a flat object reliably and
    often fail on a nested list of objects. One call covers virtually every market question
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
