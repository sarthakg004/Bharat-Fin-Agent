"""
specialists.py  ·  finagent/research/specialists.py

The Deep Research specialist registry.

A specialist is DATA, not a class: an id, a UI label, the report section it
feeds, and a focus-question template. Every specialist executes through the
SAME production pipeline (`AgenticRAGv4`, injected into the orchestrator as a
runner) — which already provides hybrid retrieval, XBRL facts, the
deterministic calculator, the table agent, live market data, web search,
EDGAR full-text search, numeric verification, and `[N]` citations. What makes
a specialist a specialist is the scoped question it asks, so there is exactly
one execution engine and no duplicated agent logic.

Templates are phrased to steer the v4 planner into the right lanes: named
metrics route `numeric` (XBRL + calculator), filings prose routes `narrative`
(hybrid retrieval), and news/sentiment/insider/political topics route
`market`/`external` (yfinance + Tavily).
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class Specialist:
    """One research angle: a label for the UI, the report section it feeds,
    and the focus question handed to the production agent."""

    id: str
    label: str
    section: str
    template: str          # .format_map({company, question, competitors})


SPECIALISTS: dict[str, Specialist] = {s.id: s for s in [
    Specialist(
        "company", "Company Research", "Company Overview",
        "Provide a company overview of {company}: main business segments and "
        "products, revenue sources and mix, geographic exposure, key customers, "
        "competitive moat, and management's capital-allocation track record.",
    ),
    Specialist(
        "sec_filings", "SEC Filing Analysis", "Recent Filings",
        "Analyze {company}'s most recent 10-K and 10-Q filings: management "
        "discussion highlights, stated risk factors, legal proceedings, segment "
        "reporting, and notable changes from the prior filing.",
    ),
    Specialist(
        "financials", "Financial Statements", "Financial Statement Analysis",
        "Analyze {company}'s latest income statement, balance sheet and cash "
        "flow statement: revenue, gross margin, operating margin, net margin, "
        "EPS growth, free cash flow, current ratio, quick ratio, debt-to-equity, "
        "interest coverage, return on equity and return on assets — interpret "
        "each figure, do not just state it.",
    ),
    Specialist(
        "xbrl", "XBRL Facts & Ratios", "Key Ratios",
        "Report {company}'s exact filed figures for the latest two fiscal "
        "years: revenue, net income, total assets, total equity, operating "
        "cash flow and capital expenditure; plus asset turnover, inventory "
        "turnover and year-over-year revenue growth.",
    ),
    Specialist(
        "earnings", "Earnings Analysis", "Recent Quarter",
        "Analyze {company}'s latest quarterly earnings: revenue and EPS versus "
        "the prior year, forward guidance, management commentary, key "
        "surprises, and the main positive and negative signals.",
    ),
    Specialist(
        "forecasts", "Analyst Forecasts", "Future Expectations",
        "What are analyst expectations for {company}: consensus earnings and "
        "revenue estimates, price targets, and the expected catalysts and "
        "risks over the next 12 months?",
    ),
    Specialist(
        "insider", "Insider Activity", "Insider Activity",
        "Summarize recent insider activity at {company}: executive and "
        "director share purchases and sales, institutional ownership changes, "
        "share buybacks, and any share issuance or dilution.",
    ),
    Specialist(
        "political", "Political Trading", "Political Activity",
        "Have members of the US Congress or other politicians disclosed "
        "trades in {company} stock recently? Such disclosures are often "
        "delayed or incomplete — state the recency and reliability of "
        "whatever is found.",
    ),
    Specialist(
        "sentiment", "Market Sentiment", "Market Sentiment",
        "What is the current market sentiment around {company}: recent news "
        "and events, analyst commentary, the stock's recent performance, "
        "sector trends, and macro conditions affecting the company?",
    ),
    Specialist(
        "competitors", "Competitor Analysis", "Competitor Analysis",
        "Compare {company} against its main competitors{competitors}: revenue "
        "growth, profitability, margins, valuation, market share and overall "
        "competitive positioning — strengths and weaknesses of each.",
    ),
    Specialist(
        "valuation", "Valuation", "Valuation",
        "Assess {company}'s valuation: P/E, P/S, P/B and EV/EBITDA where "
        "available, compared with its own history and industry peers. Is the "
        "stock trading at a premium or a discount, and why?",
    ),
    Specialist(
        "risk", "Risk Assessment", "Risks",
        "What are the key risks facing {company}: financial, legal, "
        "regulatory, competitive, technological, supply-chain, customer "
        "concentration, and macroeconomic or geopolitical risks?",
    ),
    Specialist(
        "macro", "Macro Environment", "Industry & Macro",
        "How do current macroeconomic conditions affect {company}: interest "
        "rates, inflation, commodity prices, currency movements, government "
        "policy and sector trends?",
    ),
    Specialist(
        "custom", "Your Question", "Answer to Your Question",
        "{question}",
    ),
]}

# The final "Investment Thesis" step is the orchestrator's own report-writer
# call (it needs every finding), so it is not a runnable specialist — but it
# appears in the research plan/timeline under this id.
THESIS_TASK_ID = "thesis"
THESIS_LABEL = "Investment Thesis"

# Fallback selection when the scoping call fails: the core angles that make a
# useful report for any "should I invest in X" style request.
DEFAULT_SPECIALISTS = ["company", "financials", "earnings", "sentiment",
                       "valuation", "risk"]


class ResearchScope(BaseModel):
    """Structured output of the scoping call: what to research and with which
    specialists."""

    company: str = Field(
        default="",
        description="The primary company under research, e.g. 'Apple'. Empty "
                    "only if the request names no company at all.",
    )
    ticker: str = Field(
        default="", description="Its stock ticker if known, e.g. 'AAPL'.")
    objective: str = Field(
        default="",
        description="One sentence restating the research goal (e.g. 'assess "
                    "whether NVDA is a good long-term investment').",
    )
    competitors: list[str] = Field(
        default_factory=list,
        description="0-4 main competitor company names, if relevant.",
    )
    specialists: list[str] = Field(
        default_factory=list,
        description="The specialist ids to run, MOST relevant first, chosen "
                    "from the catalog provided. Include 'custom' only when the "
                    "user asked a specific question beyond a general "
                    "assessment.",
    )
