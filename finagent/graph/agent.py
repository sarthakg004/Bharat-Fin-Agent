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
    cut-off events, latest news). Tavily if `TAVILY_API_KEY` is set, otherwise
    the local `news` Chroma collection. Escalates automatically when retrieval
    comes back empty/weak or the draft admits it can't answer.
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
"""

from __future__ import annotations

import argparse
import re
from typing import Optional

from finagent.graph.full import (
    AgenticRAGv3,
    SYNTH_V3_SYSTEM,
    _MARKET_MARKERS,
    append_comparison_row,  # noqa: F401  re-export
)
from finagent.graph.market_tools import call_tool as call_market_tool
from finagent.graph.state import (
    AgentState, MarketIntent, NumericVerification, XBRLQuery, XBRLQueryBatch,
    CalcQuery, CalcQueryBatch, CorpusGateQuery, EdgarQuery,
)
from finagent.graph.web_search import WebSearcher
from finagent.tools.calculator import FinancialCalculator
from finagent.tools.edgar_search import EdgarFullTextSearch
from finagent.tools.sec_fetch import SecFilingFetcher
from finagent.tools.xbrl import XBRLClient

# News / outlook intent that should reach the web even without an `external`
# route. Kept news-specific (not bare "current"/"recent", which appear in
# "current ratio" etc.) to avoid firing web on filing/numeric questions.
_WEB_NEWS_MARKERS = (
    "news", "latest", "headline", "press release", "announcement",
    "outlook", "forecast", "guidance", "analyst", "expected to perform",
    "upcoming month", "coming month", "next quarter", "this week", "today's",
    "recently", "happening", "sentiment", "what's new",
    # Corporate-events / M&A: often span multiple years and may sit outside the
    # indexed filing text, so let the web supplement filing retrieval.
    "acquisition", "acquisitions", "acquire", "acquired", "merger", "merged",
    "takeover", "divestiture", "divest", "spin-off", "spinoff", "joint venture",
    "partnership", "deal", "buyout",
    # Event filings: only annual reports are indexed in the corpus, so a
    # question about a SPECIFIC 8-K (a dated event disclosure) needs the web
    # to supplement the 10-K text that merely references it.
    "8-k", "8k",
)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

MARKET_PLANNER_SYSTEM = """\
You are a market-data planner. Given a question routed to the `market` lane,
decide which yfinance-backed tools to invoke and with what arguments.

Available tools:
  - get_quote(symbol)                — latest price, day range, 52-week.
  - get_history(symbol, period, interval) — OHLCV + a candlestick chart.
  - get_company_info(symbol)         — sector, industry, summary.
  - get_news(symbol, limit)          — recent ticker-specific headlines.
  - compare(symbols)                 — quote snapshot for 2-6 tickers.

ALWAYS fill `company` with the company's name in words (e.g. "Rocket Lab",
"Apple") — the system resolves the correct ticker from authoritative SEC data,
which is more reliable than guessing a symbol. You may also fill `symbol` if
you're confident, but do NOT invent tickers or append exchange suffixes like
".NASDAQ". For India use .NS (NSE) like RELIANCE.NS. Common aliases (TCS, INFY,
RELIANCE, etc.) are auto-normalised.

Prefer get_history for almost anything stock-related — it returns a candlestick
CHART plus the latest price, so it answers "how is X doing", "how has X
performed", "is X a good stock", "show me X", "1-year/5-year/ytd", and bare
"X stock" alike. Default to a 1-year daily history when no period is given.
Only use get_quote for a bare "what is X trading at right now" with no interest
in the trend. Use compare (put tickers in `symbols`) for "X vs Y". get_news for
"latest news on X". Resolve follow-ups from the conversation: "show me its
chart" / "the last one" refers to the company discussed just before. Set
tool='none' only if the question genuinely isn't about a listed company's stock.
"""

MARKET_PLANNER_PROMPT = """\
Question: {question}

Resolve the ticker (using the conversation above if the question is a follow-up),
then return a single MarketIntent (tool + symbol/symbols + period/interval).
"""


NUM_VERIFY_SYSTEM = """\
You verify numeric claims in a draft financial answer. Given the draft answer
and the source evidence (text excerpts, table-derived computations, optional
web results), extract each distinct numeric claim and decide whether the
SAME figure appears in the evidence. Accept reasonable paraphrases (₹914 bn
vs ₹91,400 crore) but reject silent invention.
"""

NUM_VERIFY_PROMPT = """\
Draft answer:
\"\"\"{answer}\"\"\"

Evidence:
{evidence}

---
Extract every numeric claim from the answer (revenue figures, percentages,
ratios, counts) and report whether each is supported by the evidence.
"""

# --------------------------------------------------------------------------- #
# Phase 9 — financial-analyst voice (synthesizer + critic)
# --------------------------------------------------------------------------- #

SYNTH_ANALYST_SYSTEM = """\
You are a senior equity research analyst writing for a financial professional.
Answer using ONLY the numbered evidence supplied below. Write the way a sell-side
analyst would: precise, quantitative, and economical with words.

Voice and precision
--------------------
- LEAD WITH THE BOTTOM LINE: the first sentence states the direct answer / the
  headline figure. No preamble.
- EVERY figure carries its unit AND period — "$394.3 billion (FY2022)",
  "30.3% operating margin (FY2022)", "+7.8% YoY". Never write a bare number.
- Use precise terminology: operating margin, gross margin, YoY, CAGR, basis
  points (bps), fiscal year (FY), GAAP. Say "fell 120 bps" not "went down a bit".
- XBRL FACT / DERIVED METRIC items are exact figures as filed — state them
  precisely (you may round in prose to one decimal, but keep them accurate).
  When you cite a figure, the [N] points the reader to its exact source
  (filing page or XBRL concept) in the sidebar.
- Be concise: a tight, structured answer beats a long one. No filler, no
  restating the question.

Citations
---------
Cite by **number** only. After every factual claim append the supporting index
in ASCII square brackets — "Apple's FY2022 revenue was $394.3 billion [1]."
Multiple sources: `[1,3]`. Use `[N]` — NOT `【N】`, `(N)`, or any other style.
NEVER write out the source title, URL, or tag in prose — the user sees those in
a sidebar already.

Source priority and reconciliation
----------------------------------
When sources disagree on a figure, do NOT list both values. Use the most
authoritative one and state the figure ONCE, following this priority:
  XBRL FACT / DERIVED METRIC  >  filing excerpt / table  >  newer web/press  >  older web.
- If a number exists in an XBRL FACT or DERIVED METRIC item, use THAT exact value
  and ignore any conflicting web snippet entirely — the filing is ground truth.
- Among web/press sources, prefer the most recent by publication date.
- State each figure once. Only flag a discrepancy explicitly (one short italic
  note) when sources of *comparable* authority genuinely conflict and it matters;
  never silently present "44%… then 42%" for the same metric and period.
- Do not repeat the same point in multiple bullets/sentences.

Structure (markdown)
--------------------
- One-line bottom line first, then supporting detail.
- **Bold** the key figures and entity names.
- Use a GitHub-flavoured markdown table for any comparison across entities or
  periods (companies × metrics, or a metric across fiscal years).
- Bullets for 3+ discrete points; `## sub-headings` only for 2+ sections.
- Short paragraphs (2-3 sentences), blank line between them.

Time-sensitive questions
------------------------
Each web/news item has a publication date in its header. For "today's",
"current", "premarket", "this week" questions: use the MOST RECENT item, state
the as-of date ("As of <date>, ..."), and don't blend older datapoints in as if
current.

Thin or partial evidence
------------------------
- Still give the most useful answer the evidence supports, citing each fact [N].
  A precise, caveated partial answer beats a refusal.
- Add a one-line italic caveat on the limitation (e.g. *"Sources cover FY2023
  only, so the 2022→2024 trend is incomplete."*).
- Only when there is genuinely NO relevant evidence, say so in one short
  sentence with no citations.
- NEVER invent figures, periods, companies, page numbers, or XBRL concepts. Use
  only what the evidence supports.
"""

CRITIC_ANALYST_SYSTEM = """\
You are a fact-checking equity research editor. Given a draft answer and the
source excerpts it was based on, extract each distinct factual claim and decide
whether the excerpts SUPPORT it. Judge ONLY against the excerpts, not your own
knowledge.

Apply analyst rigor to numeric claims specifically:
- A figure is supported only if the SAME value appears in the evidence for the
  SAME period (a FY2022 figure cited against FY2021 evidence is NOT supported).
- Treat XBRL FACT / DERIVED METRIC / TABLE items as exact ground truth; a prose
  figure that contradicts them is not supported.
- Accept sensible rounding and unit paraphrases ($394,328 million ≈ $394.3
  billion); reject silently invented or mis-periodised numbers.
Mark each claim supported / not supported with a brief reason.
"""

XBRL_EXTRACT_SYSTEM = """\
You extract a single structured XBRL lookup from a numeric sub-query about a US
public company's financial statements. Decide whether the sub-query asks for ONE
exact reported line-item figure (revenue, net income, total assets, gross
profit, R&D expense, diluted EPS, cash, long-term debt, …) for ONE company — if
so set answerable=true and fill ticker and concept (plain words).

Period rules (important):
- Set `period` to a fiscal YEAR (e.g. 'FY2022') ONLY if the question names a
  specific year. If NO year is mentioned, leave period EMPTY — do NOT guess a
  year; the tool then returns the LATEST available data.
- Set `quarterly`=true if the question asks for a quarter ('last quarter', 'most
  recent quarter', 'Q3', 'quarterly EPS') — the tool returns the latest 10-Q
  figure instead of the annual one.

Set answerable=false for derived metrics (margins, growth, ratios, CAGR),
multi-company comparisons, or narrative questions — those are handled elsewhere.
Use the conversation context to resolve a follow-up's company/period.
"""

XBRL_EXTRACT_PROMPT = """\
Numeric sub-query: {sub_query}

Return the XBRL lookup (answerable, ticker, concept, period).
"""

XBRL_TAG_SYSTEM = """\
You map a plain-language financial concept to the single best US-GAAP XBRL tag
from a list of tags a company actually reports. Reply with EXACTLY one tag name
copied verbatim from the list (no explanation). If none fit, reply 'NONE'.
"""

CALC_EXTRACT_SYSTEM = """\
You extract a derived-metric computation from a numeric sub-query about a US
public company. A DERIVED metric is one computed from reported figures: a margin
(gross/operating/net), a liquidity ratio (current ratio, quick/acid-test ratio,
cash ratio, operating_cash_flow_ratio = CFO / current liabilities), a
leverage/return/efficiency ratio (debt-to-equity, ROE, ROA, asset turnover,
fixed_asset_turnover = revenue / net PP&E, inventory_turnover = COGS /
inventory, interest coverage), an intensity ratio (rd_to_revenue = R&D as % of
revenue, sga_to_revenue, capex_to_revenue), an EBITDA margin (ebitda_margin =
(operating income + D&A) / revenue), a working-capital days metric
(dio = days inventory outstanding, dso = days sales outstanding, dpo = days
payable outstanding, ccc = cash conversion cycle = DIO + DSO − DPO — each uses
the two-period average of its balance-sheet input, so a "FY2019 CCC averaging
FY2018-FY2019" question is metric='ccc' with periods ['FY2018','FY2019']),
period-over-period GROWTH, a CAGR,
or a multi-year TREND of any of those. Set is_derived=true and fill ticker, the
canonical metric name, periods (fiscal years, earliest first), and — for
growth/cagr only — the underlying concept. Set is_derived=false for a single reported figure (revenue,
net income, total assets, …); those are handled by the XBRL facts tool, not here.
Use the conversation context to resolve a follow-up's company/periods.
"""

CALC_EXTRACT_PROMPT = """\
Numeric sub-query: {sub_query}

Return the derived-metric computation (is_derived, ticker, metric, concept, periods).
"""

GATE_EXTRACT_SYSTEM = """\
You identify the single US public company a question is primarily about, so the
system can fetch its SEC filing if it isn't indexed yet. Return the company as a
ticker or name (e.g. 'CRM' or 'Salesforce'). Return an empty string if the
question names no specific company, spans many companies, or is a macro/market
question. Resolve follow-ups from the conversation context.
"""

GATE_EXTRACT_PROMPT = """\
Question: {question}

Return the one company this is about (ticker or name), or '' if none/many.
"""

EDGAR_EXTRACT_SYSTEM = """\
You turn a cross-document question ("which companies disclosed X") into an EDGAR
full-text search. Extract the distinctive PHRASE to search for across filings —
the specific concept, not the whole question and not generic words like
"companies"/"disclosed". Prefer the exact phrase a filing would use. Also pick
the SEC form (default '10-K'). E.g. "Which companies warned about a going-concern
doubt?" -> phrase="going concern", forms="10-K".
"""

EDGAR_EXTRACT_PROMPT = """\
Cross-document sub-query: {sub_query}

Return the EDGAR full-text search (phrase, forms).
"""

REFUSAL_TEMPLATE = (
    "I don't have enough information to answer this from the available "
    "filings{web_clause}. The retrieval and numeric verification steps could "
    "not ground the requested figures{detail}."
)


# --------------------------------------------------------------------------- #
# AgenticRAGv4
# --------------------------------------------------------------------------- #

class AgenticRAGv4(AgenticRAGv3):
    """v3 + structured numeric lanes + web search + numeric verification + confidence gate."""

    def __init__(
        self,
        *args,
        news_collection: str = "news",
        # ~10 hits gives the synthesiser enough material for multi-faceted
        # questions ("performance over the last year"); a single article rarely
        # covers the full ground. Tavily allows up to 20 per call.
        web_top_k: int = 10,
        verifier_model: Optional[str] = None,
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
        self.news_collection = news_collection
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
        self.verifier_model = verifier_model or self.critic_model
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
            self._web = WebSearcher(
                chroma_dir=self.chroma_dir,
                collection_name=self.news_collection,
                embedding_model=self.embedding_model,
                top_k=self.web_top_k,
            )
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
                chroma_dir=self.chroma_dir,
                embedding_model=self.embedding_model,
                market=self.market,
            )
        return self._fetcher

    def _xbrl_pick_tag(self, concept: str, available_tags: list[str]):
        """LLM fallback for the XBRL client: pick the best US-GAAP tag for a
        concept from the tags a company actually reports. Used only when the
        curated concept→tag map misses (keeps cost near zero on the common path).
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Cap the candidate list so the prompt stays small; the curated map
        # already covers the common tags, so this is a genuine long-tail fallback.
        listing = "\n".join(available_tags[:200])
        try:
            resp = self._get_router_llm().invoke([
                SystemMessage(content=XBRL_TAG_SYSTEM),
                HumanMessage(content=f"Concept: {concept}\n\nAvailable tags:\n{listing}"),
            ])
            choice = (resp.content or "").strip().strip("`").split()[0]
            return None if choice.upper() == "NONE" else choice
        except Exception:
            return None

    def _get_verifier_llm(self):
        if "verifier" not in self._llms:
            from finagent.llm import build_llm

            self._llms["verifier"] = build_llm(
                self.provider, self.verifier_model, self.api_key, temperature=0.0
            )
        return self._llms["verifier"]

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

    def _get_market_planner_llm(self):
        """Same LLM as the router/planner tier — structured output, fast."""
        if "market_planner" not in self._llms:
            from finagent.llm import build_llm

            self._llms["market_planner"] = build_llm(
                self.provider, self.planner_model, self.api_key, temperature=0.0,
            )
        return self._llms["market_planner"]

    # A question about a SPECIFIC dated event filing ("the 8-K dated 1st July
    # 2022"). These documents are never in the indexed corpus (only annual
    # reports are) and rarely on the news web — they must be pulled from EDGAR
    # by (form, date) directly.
    _FORM_REQ_RE = re.compile(r"\b(8[\s-]?K|10[\s-]?Q|6[\s-]?K|DEF\s?14A)\b", re.I)
    _MONTHS = {m: i + 1 for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"])}
    _DATE_RES = (
        re.compile(r"\b(?P<d>\d{1,2})(?:st|nd|rd|th)?\s+of\s+"
                   r"(?P<m>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+"
                   r"(?P<y>\d{4})\b", re.I),
        re.compile(r"\b(?P<d>\d{1,2})(?:st|nd|rd|th)?\s+"
                   r"(?P<m>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+"
                   r"(?P<y>\d{4})\b", re.I),
        re.compile(r"\b(?P<m>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+"
                   r"(?P<d>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<y>\d{4})\b", re.I),
        re.compile(r"\b(?P<y>\d{4})-(?P<mn>\d{2})-(?P<d>\d{2})\b"),
    )

    @classmethod
    def _dated_form_request(cls, question: str) -> Optional[tuple]:
        """(form, date) when the question names an event filing AND a date."""
        qm = cls._FORM_REQ_RE.search(question or "")
        if not qm:
            return None
        from datetime import date
        for rx in cls._DATE_RES:
            m = rx.search(question)
            if not m:
                continue
            g = m.groupdict()
            month = int(g["mn"]) if g.get("mn") else cls._MONTHS[g["m"][:3].lower()]
            try:
                dt = date(int(g["y"]), month, int(g["d"]))
            except ValueError:
                continue
            squash = re.sub(r"[\s-]", "", qm.group(1).upper())
            form = {"8K": "8-K", "10Q": "10-Q", "6K": "6-K",
                    "DEF14A": "DEF 14A"}.get(squash, squash)
            return form, dt
        return None

    def fetch_filing_node(self, state: AgentState) -> dict:
        """Corpus-membership gate + on-demand SEC fetch (Phase 5).

        Runs between routing and retrieval. It identifies the company the
        question is about and consults the gate:
          * already indexed   → nothing to do;
          * US-listed, missing → fetch the latest 10-K, ingest it into the live
            collection, and invalidate the cached hybrid retrievers so the new
            chunks are searchable on this very turn;
          * not US-listed      → leave it for the web-search branch.

        The result lands in `state["fetch_status"]` so the API/UX can show the
        "Fetching latest 10-K…" state and report what was added.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        question = state["question"]

        # Resolve the company (with conversation context for follow-ups).
        history = state.get("chat_history") or []
        history_block = ""
        if history:
            lines = [
                f"{('User' if t.get('role') == 'user' else 'Assistant')}: "
                f"{(t.get('content') or '')[:300]}"
                for t in history[-4:]
            ]
            history_block = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

        try:
            extractor = self._get_router_llm().with_structured_output(CorpusGateQuery)
            gq: CorpusGateQuery = extractor.invoke([
                SystemMessage(content=GATE_EXTRACT_SYSTEM),
                HumanMessage(content=history_block + GATE_EXTRACT_PROMPT.format(question=question)),
            ])
            company = (gq.company or "").strip()
        except Exception as e:
            self._log(state, f"corpus-gate extract failed: {e}")
            return {"fetch_status": {}}

        if not company:
            return {"fetch_status": {}}

        # Dated event-filing request ("AMCOR's 8-K dated 1st July 2022"): pull
        # the named document straight from EDGAR by (form, date) and inject its
        # text as ephemeral chunks — `hybrid_retrieve_node` ranks them first.
        # This runs regardless of the corpus gate: the company being indexed
        # only means its ANNUAL reports are, never its event filings.
        extra: dict = {}
        dated = self._dated_form_request(question)
        if dated:
            form, dt = dated
            try:
                res = self.fetcher.fetch_dated_filing(company, form, dt)
            except Exception as e:
                self._log(state, f"dated filing fetch failed ({form} {dt}): {e}")
                res = {"ok": False}
            if res.get("ok") and res.get("chunks"):
                self._log(state, f"fetched {res.get('form')} filed "
                                 f"{res.get('filing_date')} from EDGAR "
                                 f"({len(res['chunks'])} chunks)")
                extra["fetched_chunks"] = res["chunks"]

        try:
            gate = self.fetcher.gate(company)
        except Exception as e:
            self._log(state, f"corpus gate failed for {company!r}: {e}")
            return {"fetch_status": {}, **extra}

        # Year-aware depth: a question about FY2019 can't be answered from the
        # LATEST 10-K alone — walk back enough annual filings that the asked
        # year is covered (each 10-K carries the prior year's comparatives,
        # hence the −1). Capped to bound a one-time ingest.
        n_filings = 1
        yrs = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", question)]
        if yrs:
            from datetime import date
            n_filings = max(1, min(5, date.today().year - min(yrs) - 1))

        if gate["decision"] != "fetch":
            # "Indexed" is COMPANY-level. If the question names a year the
            # index doesn't cover (e.g. only the latest 10-K was dynamically
            # fetched but the question asks about FY2021), deepen the index by
            # walking back to the filing that carries that year. Without this,
            # retrieval returns nothing useful and the company looks "covered".
            if (gate["decision"] == "already_indexed" and yrs
                    and self.persist_fetch):
                covered = self._indexed_years(gate.get("ticker") or company)
                target = min(yrs)
                digit_years = {int(y) for y in covered if y.isdigit()}
                if digit_years and not ({target, target + 1} & digit_years):
                    n_deep = max(1, min(5, max(digit_years) - target))
                    self._log(state, f"index covers {sorted(digit_years)} for "
                                     f"{gate.get('ticker')} but the question needs "
                                     f"{target}; fetching {n_deep} older filing(s)")
                    try:
                        res = self.fetcher.fetch_and_ingest(
                            gate["ticker"], company=gate.get("company") or "",
                            n=n_deep)
                        if res.get("ok"):
                            self._hybrids = None    # re-index for this turn
                            self._log(state, f"deepened index with "
                                             f"{res.get('chunks_added')} chunks")
                    except Exception as e:
                        self._log(state, f"index deepening failed: {e}")
            # already_indexed → retrieval handles it; not_us_listed → web branch.
            return {"fetch_status": gate, **extra}

        self._log(state, f"dynamic fetch: pulling latest {n_filings} filing(s) "
                         f"for {gate['ticker']}…")

        # Ephemeral path (cloud / per-session): parse + chunk the filing in
        # memory and rank it against the question later — NOTHING is written to
        # the persistent index, so the corpus never grows and there's no
        # scale-to-zero persistence problem.
        if not self.persist_fetch:
            try:
                res = self.fetcher.fetch_chunks(
                    gate["ticker"], company=gate.get("company") or "", n=n_filings)
            except Exception as e:
                self._log(state, f"ephemeral fetch failed for {gate['ticker']}: {e}")
                return {"fetch_status": {**gate, "status": "error", "error": str(e)},
                        **extra}
            chunks = res.get("chunks", []) if res.get("ok") else []
            if chunks:
                self._log(state, f"fetched {len(chunks)} in-memory chunks for {gate['ticker']} "
                                 f"({res.get('form')})")
            return {
                "fetched_chunks": extra.get("fetched_chunks", []) + chunks,
                "fetch_status": {**gate, "status": "fetched" if chunks else "error",
                                 "ephemeral": True, "form": res.get("form"),
                                 "chunks_fetched": len(chunks)},
            }

        # Persistent path (local): fetch + ingest into the live collection.
        try:
            res = self.fetcher.fetch_and_ingest(
                gate["ticker"], company=gate.get("company") or "", n=n_filings)
        except Exception as e:
            self._log(state, f"dynamic fetch failed for {gate['ticker']}: {e}")
            return {"fetch_status": {**gate, "status": "error", "error": str(e)},
                    **extra}

        if res.get("ok"):
            self._hybrids = None       # invalidate cached BM25/dense over old corpus
            self._log(state, f"ingested {res['chunks_added']} chunks for {gate['ticker']}")
        return {"fetch_status": {**gate, "status": "fetched" if res.get("ok") else "error",
                                 **res}, **extra}

    def _indexed_years(self, ticker: str) -> set[str]:
        """Metadata `year` values indexed for `ticker` (empty when unknowable —
        e.g. baseline corpora keyed by company name rather than ticker)."""
        try:
            col = self.fetcher._get_store()._collection
            res = col.get(where={"ticker": ticker},
                          include=["metadatas"], limit=2000)
            return {str(m.get("year", "")) for m in (res.get("metadatas") or [])}
        except Exception:
            return set()

    def hybrid_retrieve_node(self, state: AgentState) -> dict:
        """Persistent-corpus retrieval, plus in-memory ranking of an ephemerally
        fetched filing (cloud path) so its chunks are usable without ever being
        indexed."""
        out = super().hybrid_retrieve_node(state)
        fetched = state.get("fetched_chunks") or []
        if fetched:
            ranked = self._rank_fetched_chunks(state, fetched)
            # Fetched filing is the authoritative source for an out-of-corpus
            # company → put its chunks first.
            out["retrieved_chunks"] = ranked + (out.get("retrieved_chunks") or [])
        return out

    def _rank_fetched_chunks(self, state: AgentState, fetched: list[dict]) -> list[dict]:
        """Rank ephemerally-fetched chunks against the question IN MEMORY (BGE
        cosine pre-filter → cross-encoder rerank), returning the top few per
        sub-query — the same two-stage quality as the persistent retriever, but
        over a single filing and with nothing written to disk."""
        import numpy as np

        from finagent.retrieval.reranker import _get_shared_reranker
        from finagent.vectorstore import get_embeddings

        texts = [c.get("text", "") for c in fetched]
        if not texts:
            return []
        sub_queries = state.get("sub_queries") or [state["question"]]
        routes = state.get("query_routes") or []
        if routes and len(routes) == len(sub_queries):
            subs = [s for s, r in zip(sub_queries, routes) if r in ("narrative", "numeric")] \
                or [state["question"]]
        else:
            subs = sub_queries

        try:
            embedder = get_embeddings(self.embedding_model)
            mat = np.asarray(embedder.embed_documents(texts))          # (N, d), normalized
            reranker = _get_shared_reranker(self.reranker_model)
        except Exception as e:
            self._log(state, f"ephemeral rank failed ({e}); using fetch order")
            return [{**c, "source": self._fetched_source(c), "sub_query": subs[0]}
                    for c in fetched[: self.final_top_k]]

        pre_k = max(self.final_top_k * 4, 20)
        out: list[dict] = []
        seen: set = set()
        for sq in subs:
            qv = np.asarray(embedder.embed_query(sq))
            sims = mat @ qv
            cand = list(np.argsort(-sims)[:pre_k])
            scores = reranker.predict([(sq, texts[i]) for i in cand])
            order = [cand[j] for j in np.argsort(-np.asarray(scores))][: self.final_top_k]
            for i in order:
                key = texts[i][:80]
                if key in seen:
                    continue
                seen.add(key)
                out.append({**fetched[i], "source": self._fetched_source(fetched[i]),
                            "sub_query": sq})
        return out

    @staticmethod
    def _fetched_source(c: dict) -> str:
        return f"[{c.get('company', c.get('ticker', '?'))} filing {c.get('year', '?')}]"

    def _extract_batch(self, state: AgentState, sub_queries: list[str],
                       batch_schema, system: str, single_prompt: str,
                       single_schema, history_block: str) -> list[tuple]:
        """Run ONE structured-output extraction over all `sub_queries`,
        returning [(sub_query, extraction-or-None), ...] aligned by position.

        The batch call replaces N per-sub-query LLM calls. If it fails or the
        model returns a misaligned list, fall back to the per-sub-query loop so
        behaviour degrades to the legacy path, never to silent skips.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        if len(sub_queries) > 1:
            numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sub_queries))
            batch_prompt = (
                f"Numbered sub-queries:\n{numbered}\n\n"
                f"Return EXACTLY one extraction per numbered sub-query, in the "
                f"same order ({len(sub_queries)} entries)."
            )
            try:
                out = self._get_router_llm().with_structured_output(batch_schema).invoke([
                    SystemMessage(content=system),
                    HumanMessage(content=history_block + batch_prompt),
                ])
                queries = list(out.queries or [])
                if len(queries) == len(sub_queries):
                    return list(zip(sub_queries, queries))
                self._log(state, f"batch extract misaligned "
                                 f"({len(queries)} for {len(sub_queries)}); "
                                 f"falling back to per-sub-query extraction")
            except Exception as e:
                self._log(state, f"batch extract failed ({e}); "
                                 f"falling back to per-sub-query extraction")

        extractor = self._get_router_llm().with_structured_output(single_schema)
        pairs: list[tuple] = []
        for sub_q in sub_queries:
            try:
                q = extractor.invoke([
                    SystemMessage(content=system),
                    HumanMessage(content=history_block + single_prompt.format(sub_query=sub_q)),
                ])
            except Exception as e:
                self._log(state, f"extract failed for {sub_q!r}: {e}")
                q = None
            pairs.append((sub_q, q))
        return pairs

    def xbrl_node(self, state: AgentState) -> dict:
        """Answer numeric sub-queries from SEC XBRL structured facts (Phase 3).

        For each sub-query the router flagged `numeric`, extract a single
        (ticker, concept, period) lookup with a cheap LLM call, then fetch the
        exact reported figure from `data.sec.gov` company-facts. The figure is
        authoritative — it comes straight from the company's filing — so it
        becomes the highest-priority evidence the synthesizer cites, and it
        anchors numeric verification (no LLM in the number path = nothing to
        hallucinate). Derived metrics / comparisons fall through to retrieval and
        the table agent (and, in Phase 4, the calculator-over-XBRL).
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        sub_queries = state.get("sub_queries") or [state["question"]]
        routes = state.get("query_routes") or ["narrative"] * len(sub_queries)
        numeric_subs = [s for s, r in zip(sub_queries, routes) if r == "numeric"]
        if not numeric_subs:
            return {"xbrl_facts": []}

        # Conversation context lets a follow-up ("and FY2023?") inherit company.
        history = state.get("chat_history") or []
        history_block = ""
        if history:
            lines = [
                f"{('User' if t.get('role') == 'user' else 'Assistant')}: "
                f"{(t.get('content') or '')[:300]}"
                for t in history[-4:]
            ]
            history_block = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

        # ONE batched extraction call for all numeric sub-queries (was one call
        # per sub-query — the dominant tool-lane latency on multi-input
        # questions). Falls back to per-sub-query extraction if the batch
        # call fails or comes back misaligned.
        extracted = self._extract_batch(
            state, numeric_subs, XBRLQueryBatch, XBRL_EXTRACT_SYSTEM,
            XBRL_EXTRACT_PROMPT, XBRLQuery, history_block)

        facts: list[dict] = []
        for sub_q, q in extracted:
            if q is None or not q.answerable or not (q.ticker and q.concept):
                continue
            try:
                res = self.xbrl.run(ticker=q.ticker, concept=q.concept,
                                    period=q.period or None, quarterly=q.quarterly)
            except Exception as e:
                self._log(state, f"xbrl lookup failed for {sub_q!r}: {e}")
                continue
            res["sub_query"] = sub_q
            if res.get("ok"):
                facts.append(res)
            else:
                self._log(state, f"xbrl miss for {sub_q!r}: {res.get('error')}")
        return {"xbrl_facts": facts}

    def calculator_node(self, state: AgentState) -> dict:
        """Compute derived metrics (margins, ratios, growth, CAGR, trends) from
        exact XBRL inputs (Phase 4).

        For each `numeric` sub-query, a cheap LLM call decides whether it asks
        for a *derived* metric and, if so, extracts (ticker, metric, periods).
        The calculator then pulls the exact inputs via the Phase-3 XBRL client
        and computes deterministically — so the result is auditable down to the
        filed figures it divided. Plain single-figure lookups are left to the
        XBRL node; this node simply skips them (is_derived=False).
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        sub_queries = state.get("sub_queries") or [state["question"]]
        routes = state.get("query_routes") or ["narrative"] * len(sub_queries)
        numeric_subs = [s for s, r in zip(sub_queries, routes) if r == "numeric"]
        if not numeric_subs:
            return {"calc_results": []}

        history = state.get("chat_history") or []
        history_block = ""
        if history:
            lines = [
                f"{('User' if t.get('role') == 'user' else 'Assistant')}: "
                f"{(t.get('content') or '')[:300]}"
                for t in history[-4:]
            ]
            history_block = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

        extracted = self._extract_batch(
            state, numeric_subs, CalcQueryBatch, CALC_EXTRACT_SYSTEM,
            CALC_EXTRACT_PROMPT, CalcQuery, history_block)

        results: list[dict] = []
        for sub_q, q in extracted:
            if q is None or not q.is_derived or not (q.ticker and q.metric):
                continue
            try:
                res = self.calc.run(
                    metric=q.metric, ticker=q.ticker, concept=q.concept,
                    periods=q.periods,
                    period=(q.periods[0] if q.periods else None),
                    period_from=(q.periods[0] if len(q.periods) >= 2 else None),
                    period_to=(q.periods[-1] if len(q.periods) >= 2 else None),
                    start_period=(q.periods[0] if len(q.periods) >= 2 else None),
                    end_period=(q.periods[-1] if len(q.periods) >= 2 else None),
                )
            except Exception as e:
                self._log(state, f"calc failed for {sub_q!r}: {e}")
                continue
            res["sub_query"] = sub_q
            if res.get("ok"):
                results.append(res)
            else:
                self._log(state, f"calc miss for {sub_q!r}: {res.get('error')}")
        return {"calc_results": results}

    def table_agent_node(self, state: AgentState) -> dict:
        """Phase 7: the table agent is the numeric *fallback*, not a duplicate.

        Skip any numeric sub-query the XBRL facts node or the calculator already
        answered — that trims an embedding search over the tables collection plus
        a code-generation LLM call for each already-answered sub-query. The table
        agent still runs for numeric sub-queries XBRL/calc couldn't serve.
        """
        answered = {f.get("sub_query") for f in state.get("xbrl_facts", []) or []}
        answered |= {r.get("sub_query") for r in state.get("calc_results", []) or []}
        if not answered:
            return super().table_agent_node(state)

        sub_queries = state.get("sub_queries") or [state["question"]]
        routes = state.get("query_routes") or ["narrative"] * len(sub_queries)
        remaining = [s for s, r in zip(sub_queries, routes)
                     if r == "numeric" and s not in answered]
        if not remaining:
            return {"table_results": []}
        # Restrict the table agent to the still-unanswered numeric sub-queries.
        proxy = dict(state)
        proxy["sub_queries"] = remaining
        proxy["query_routes"] = ["numeric"] * len(remaining)
        return super().table_agent_node(proxy)

    # US exchange suffixes the LLM sometimes wrongly appends; any OTHER dotted
    # suffix (.NS, .L, .TO, .HK, …) is a deliberate foreign listing we keep.
    _US_SUFFIX_RE = re.compile(r"\.(NASDAQ|NYSE|NYS|NMS|NASD|NAS|OQ|N|O|A|P|Z|BATS|ARCA)$", re.I)

    def _resolve_market_ticker(self, company: str, symbol: str) -> str:
        """Resolve to the authoritative ticker, generally (not per-stock).

        Prefer the SEC resolver on the company NAME (e.g. 'Rocket Lab' → RKLB),
        which is far more reliable than the LLM's symbol guess. The symbol is only
        used as an EXACT ticker (never a fuzzy name match, which could land on a
        different company), and a deliberate foreign-exchange suffix is preserved.
        """
        symbol = (symbol or "").strip()
        # Keep a non-US exchange ticker as-is (RELIANCE.NS, BARC.L, RY.TO).
        if "." in symbol and not self._US_SUFFIX_RE.search(symbol):
            return symbol.upper()

        # 1. Resolve the company NAME (fuzzy is fine for names).
        name = (company or "").strip()
        if name:
            try:
                r = self.xbrl.resolver.resolve(name)
            except Exception:
                r = {}
            if r.get("ticker"):
                return r["ticker"]

        # 2. The symbol: strip a US suffix, then accept the resolver ONLY on an
        #    exact ticker match (so a typo can't fuzzy-map to another company).
        raw = self._US_SUFFIX_RE.sub("", symbol).upper()
        if raw:
            try:
                r = self.xbrl.resolver.resolve(raw)
            except Exception:
                r = {}
            if r.get("match") == "ticker" and r.get("ticker"):
                return r["ticker"]
        return raw

    def market_data_node(self, state: AgentState) -> dict:
        """Call yfinance tools for sub-queries routed to `market`.

        Strategy:
          1. Take the market sub-queries (those classified as `market` by the router).
          2. Ask the planner LLM for a MarketIntent (list of tool calls).
          3. Dispatch each call to `market_tools.call_tool` and collect:
                - `market_data`: structured numeric facts the synthesizer cites
                - `charts`     : lightweight-charts JSON payloads the frontend renders
          4. Synth treats market_data as additional numbered evidence; charts ride
             on a separate `state.charts` channel so the UI can attach them to
             the assistant message.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Fire the market lane if the router flagged ANY sub-query as `market`.
        # (We gate on the route list, not zip(sub_queries, routes): a rewrite can
        # shrink sub_queries while query_routes still reflects the original plan,
        # which previously dropped the market intent entirely.)
        question = state["question"]
        history = state.get("chat_history") or []

        # Deterministic safety net: chart/price questions ("show me the chart")
        # must reach the market planner even if the LLM router mislabels them —
        # otherwise a follow-up like "show me the chart" falls through to web
        # search and returns generic index links instead of the ticker's chart.
        # The market planner resolves the ticker from history and returns
        # tool='none' when the question genuinely isn't market-related, so firing
        # on a false positive is cheap.
        routes = state.get("query_routes") or []
        sub_queries = state.get("sub_queries") or []
        market_keyword_hit = any(
            m in (text or "").lower()
            for text in [question, *sub_queries]
            for m in _MARKET_MARKERS
        )
        if "market" not in routes and not market_keyword_hit:
            return {"market_data": [], "charts": []}

        # Give the market planner the conversation so follow-ups like
        # "show me the chart for the last year" resolve to the right ticker.
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

        market_q = question
        planner = self._get_market_planner_llm().with_structured_output(MarketIntent)
        try:
            intent: MarketIntent = planner.invoke([
                SystemMessage(content=MARKET_PLANNER_SYSTEM),
                HumanMessage(content=history_block + MARKET_PLANNER_PROMPT.format(
                    question=question,
                )),
            ])
        except Exception as e:
            self._log(state, f"market planner failed ({e}); skipping market lane")
            return {"market_data": [], "charts": []}

        # Correct the ticker via the SEC resolver (authoritative) instead of
        # trusting the LLM's guess — fixes hallucinated symbols like RLAB→RKLB
        # for ANY company, and strips bogus exchange suffixes (".NASDAQ").
        if intent.tool == "compare":
            intent.symbols = [self._resolve_market_ticker("", s) for s in (intent.symbols or [])]
            intent.symbols = [s for s in intent.symbols if s]
        else:
            intent.symbol = self._resolve_market_ticker(intent.company, intent.symbol)

        if intent.tool == "none" or not (intent.symbol or intent.symbols):
            return {"market_data": [], "charts": []}
        calls = [intent]   # flat schema → the planner's primary call

        # Volume / performance questions NEED OHLCV — only get_history carries
        # volume and the price trend. The single-call planner may pick get_news
        # for a multi-intent question ("news … how will it perform … volume"),
        # so when a volume/performance intent is present we ALSO fire get_history
        # for the resolved symbol, guaranteeing the data is there.
        ql = (question + " " + " ".join(sub_queries)).lower()
        wants_history = any(
            m in ql for m in ("volume", "performance", "how has", "how is it doing",
                              "how is it performing", "trend", "perform")
        )
        sym = intent.symbol or (intent.symbols[0] if intent.symbols else "")
        if wants_history and sym and intent.tool not in ("get_history", "compare"):
            calls.append(MarketIntent(tool="get_history", symbol=sym,
                                      period="1y", interval="1d"))

        market_data: list[dict] = []
        charts: list[dict] = []
        for c in calls:
            kwargs: dict[str, object] = {}
            if c.tool == "compare":
                kwargs["symbols"] = c.symbols
            else:
                kwargs["symbol"] = c.symbol
            if c.tool == "get_history":
                kwargs["period"] = c.period
                kwargs["interval"] = c.interval
            if c.tool == "get_news":
                kwargs["limit"] = 5

            res = call_market_tool(c.tool, **kwargs)
            entry = {
                "tool": c.tool,
                "args": kwargs,
                "ok": res.get("ok", False),
                "data": res.get("data"),
                "error": res.get("error"),
                "sub_query": market_q,
            }
            market_data.append(entry)

            # `get_history` produces a chart spec we want to surface separately.
            if c.tool == "get_history" and res.get("ok") and res["data"].get("chart"):
                charts.append(res["data"]["chart"])
        return {"market_data": market_data, "charts": charts}

    def web_search_node(self, state: AgentState) -> dict:
        """Search the web for `external` sub-queries, AND escalate when text
        retrieval came back empty or poorly graded.

        The escalation path covers the case the router can't anticipate: the
        question is about a company that simply isn't in the ingested corpus
        (recent IPO, foreign listing, etc.). Without it, web search would only
        fire when the router happened to call the question "external" — but
        the router doesn't know what's in our Chroma collections.
        """
        sub_queries = state.get("sub_queries") or [state["question"]]
        routes = state.get("query_routes") or ["narrative"] * len(sub_queries)
        external_subs = [s for s, r in zip(sub_queries, routes) if r == "external"]

        # Deterministic news net: a question with explicit news / outlook /
        # forecast intent ("current market news", "expected to perform",
        # "analyst outlook") should hit the web even when the router didn't mark
        # any sub-query `external` — so multi-intent questions get web news
        # alongside market data, not one or the other. Add the matching
        # sub-queries (or the question) as external work.
        news_subs = [
            s for s in [*sub_queries, state["question"]]
            if any(m in (s or "").lower() for m in _WEB_NEWS_MARKERS)
        ]
        for s in news_subs:
            if s not in external_subs:
                external_subs.append(s)

        # Insufficient-draft escalation: the critic flagged that the draft
        # admits the gathered evidence can't answer — search the original
        # question regardless of routes/markers.
        if state.get("web_fallback_pending"):
            fq = state["question"]
            if fq not in external_subs:
                external_subs.append(fq)

        # Corrective dispatch (CRAG): no explicit external sub-queries, but the
        # corpus was tried and came back empty/poorly graded → escalate to web.
        # "Tried" means: a narrative route ran (retrieval is its primary lane),
        # OR a numeric route ran and EVERY structured lane (XBRL, calculator,
        # tables) came back empty — e.g. a non-US company like ICICI Bank,
        # where SEC facts don't exist and retrieval only finds off-entity
        # noise. Without the numeric clause those questions ended with no
        # answer while the web lane sat unused. A pure tools-path question
        # whose lanes DID answer is expected to have no chunks — not escalated.
        tools_answered = bool(
            state.get("xbrl_facts") or state.get("calc_results")
            or any(t.get("answer") and not t.get("error")
                   for t in state.get("table_results") or [])
        )
        corpus_attempted = (
            (not routes)
            or any(r == "narrative" for r in routes)
            or (any(r == "numeric" for r in routes) and not tools_answered)
        )
        if not external_subs and corpus_attempted:
            chunks = state.get("retrieved_chunks") or []
            avg_grade = state.get("avg_grade")
            retrieval_was_poor = (not chunks) or (
                avg_grade is not None and avg_grade < 2.0
            )
            if retrieval_was_poor:
                fallback = state["question"]
                self._log(
                    state,
                    f"web_search escalation: chunks={len(chunks)} "
                    f"avg_grade={avg_grade}; searching web for {fallback!r}",
                )
                external_subs = [fallback]

        if not external_subs:
            return {"web_results": []}

        hits: list[dict] = []
        for sub_q in external_subs:
            try:
                hits.extend({"sub_query": sub_q, **h} for h in self.web.search(sub_q))
            except Exception as e:
                self._log(state, f"web_search failed for {sub_q!r}: {e}")
        return {"web_results": hits}

    def edgar_search_node(self, state: AgentState) -> dict:
        """Answer cross-document sub-queries via EDGAR full-text search (Phase 6).

        For each sub-query the router flagged `cross_document` ("which companies
        disclosed X"), extract the search phrase and run it across every
        company's filings on EDGAR, returning the distinct companies that match —
        something single-company chunk retrieval structurally cannot do.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        sub_queries = state.get("sub_queries") or [state["question"]]
        routes = state.get("query_routes") or ["narrative"] * len(sub_queries)
        cross_subs = [s for s, r in zip(sub_queries, routes) if r == "cross_document"]
        if not cross_subs:
            return {"edgar_results": []}

        extractor = self._get_router_llm().with_structured_output(EdgarQuery)
        results: list[dict] = []
        for sub_q in cross_subs:
            try:
                eq: EdgarQuery = extractor.invoke([
                    SystemMessage(content=EDGAR_EXTRACT_SYSTEM),
                    HumanMessage(content=EDGAR_EXTRACT_PROMPT.format(sub_query=sub_q)),
                ])
                phrase = (eq.phrase or "").strip()
            except Exception as e:
                self._log(state, f"edgar extract failed for {sub_q!r}: {e}")
                continue
            if not phrase:
                continue
            # Default to a recent window so "which companies disclose X" surfaces
            # current filers, not arbitrary decade-old matches (EDGAR's default
            # sort is by relevance, which for common phrases returns stale hits).
            from datetime import date, timedelta
            today = date.today()
            startdt = (today - timedelta(days=3 * 365)).isoformat()
            quoted = f'"{phrase}"' if " " in phrase and '"' not in phrase else phrase
            # Try an exact-phrase match first; if the LLM over-qualified the
            # phrase (e.g. "quantum computing as a risk" → 0 hits) fall back to an
            # unquoted all-words search so we still surface the relevant filers.
            res = None
            for q in ([quoted, phrase] if quoted != phrase else [phrase]):
                try:
                    r = self.edgar.run(q, forms=eq.forms or "10-K", n=10,
                                       startdt=startdt, enddt=today.isoformat())
                except Exception as e:
                    self._log(state, f"edgar search failed for {q!r}: {e}")
                    continue
                if r.get("ok") and r.get("companies"):
                    res = r
                    break
            if res is None:
                self._log(state, f"edgar no hits for {phrase!r}")
                continue
            res["sub_query"] = sub_q
            results.append(res)
        return {"edgar_results": results}

    # ------------------------------------------------------------------ #
    # Evidence builder (#3)
    # ------------------------------------------------------------------ #

    # Per-source baseline confidence for a normalised evidence item. Ordered by
    # how authoritative the source is for a financial claim: exact filed XBRL is
    # ground truth; deterministic math over it is nearly as good; filing text and
    # tables are strong; live market data is fresh-but-point-in-time; EDGAR FTS
    # locates filers; web is the weakest. Retrieved-chunk items refine this with
    # their own grader score where available.
    _EVIDENCE_BASE_CONF = {
        "xbrl": 0.98, "calc": 0.95, "table": 0.85, "filing": 0.75,
        "market": 0.90, "edgar": 0.70, "web": 0.55,
    }

    def _normalize_evidence(self, state: AgentState) -> list[dict]:
        """Project every lane's output into one common evidence shape.

        Returns a list of
            {kind, fact, value, unit, source, citation, confidence, sub_query}
        items — `value`/`unit` are filled for the structured numeric lanes
        (XBRL, calc, market) and left None for narrative/text evidence. This is
        the single normalised view the audit trail and cross-source checks read;
        it does not replace the synthesizer's own numbered-evidence assembly.
        """
        ev: list[dict] = []

        def add(kind, fact, *, value=None, unit="", source="", citation="",
                confidence=None, sub_query=""):
            ev.append({
                "kind": kind,
                "fact": (fact or "").strip(),
                "value": value,
                "unit": unit,
                "source": source,
                "citation": citation or source,
                "confidence": round(
                    self._EVIDENCE_BASE_CONF.get(kind, 0.5)
                    if confidence is None else confidence, 3),
                "sub_query": sub_query or "",
            })

        # XBRL facts — exact filed figures.
        for f in state.get("xbrl_facts", []) or []:
            add("xbrl",
                f"{f.get('entity', f.get('ticker',''))} {f.get('concept','')} "
                f"{f.get('period_label','FY'+str(f.get('fy','?')))} = {f.get('value_str','')}",
                value=f.get("value"), unit=f.get("unit", ""),
                source=f.get("source", "<XBRL>"),
                citation=f"us-gaap:{f.get('tag','')}",
                sub_query=f.get("sub_query", ""))

        # Derived metrics — deterministic math over XBRL inputs.
        for r in state.get("calc_results", []) or []:
            add("calc", self._format_calc_result(r),
                value=r.get("value"),
                unit="%" if r.get("is_percent") else "",
                source=r.get("source", f"<Calc: {r.get('metric','')}>"),
                citation=f"<Calc: {r.get('metric','')}>",
                sub_query=r.get("sub_query", ""))

        # Filing text excerpts — per-chunk grader score becomes its confidence.
        chunks = state.get("retrieved_chunks", []) or []
        grades = state.get("grades", []) or []
        for i, c in enumerate(chunks):
            conf = ((grades[i] - 1) / 4.0) if i < len(grades) else \
                self._EVIDENCE_BASE_CONF["filing"]
            add("filing", c.get("text", "")[:300],
                source=c.get("source", ""),
                citation=c.get("source", ""),
                confidence=max(0.0, min(1.0, conf)),
                sub_query=c.get("sub_query", ""))

        # Table-agent computations.
        for t in state.get("table_results", []) or []:
            if t.get("error") or not t.get("answer"):
                continue
            srcs = ", ".join(
                f"{tu.get('title','?')} ({tu.get('company','?')} {tu.get('year','?')})"
                for tu in (t.get("tables_used") or [])[:3])
            add("table", f"Computed: {str(t.get('answer',''))[:200]}",
                source=f"<Table: {srcs}>", citation=f"<Table: {srcs}>",
                sub_query=t.get("sub_query", ""))

        # Live market data (yfinance) — one item per successful tool call.
        for m in state.get("market_data", []) or []:
            if not m.get("ok"):
                continue
            data = m.get("data") or {}
            sym = data.get("symbol") or (data.get("summary") or {}).get("symbol") or "?"
            add("market", f"yfinance.{m.get('tool','')} {sym}: {str(data)[:200]}",
                source=f"<Market: yfinance.{m.get('tool','')} {sym}>",
                citation=f"<Market: {sym}>",
                sub_query=m.get("sub_query", ""))

        # EDGAR full-text search — the matching filers.
        for r in state.get("edgar_results", []) or []:
            add("edgar", self._format_edgar_result(r)[:300],
                source=f"<EDGAR FTS: {r.get('query','')}>",
                citation=f"<EDGAR: {r.get('query','')}>",
                sub_query=r.get("sub_query", ""))

        # Web hits.
        for h in state.get("web_results", []) or []:
            add("web", (h.get("title", "") or "")[:200],
                source=f"<News: {h.get('source','web')}>",
                citation=h.get("url", "") or f"<News: {h.get('source','web')}>",
                sub_query=h.get("sub_query", ""))

        return ev

    def evidence_builder_node(self, state: AgentState) -> dict:
        """Normalise all lane outputs into one `evidence` list (#3).

        Runs after the tool lanes and before synthesis, so the structured view
        is available to the verifier, the confidence/citation scoring, the audit
        trail, and (later) cross-source validation — without disturbing the
        synthesizer's existing numbered-evidence assembly.
        """
        try:
            ev = self._normalize_evidence(state)
        except Exception as e:
            # Never let evidence normalisation break the run — synth still has
            # its own assembly to fall back on.
            self._log(state, f"evidence_builder failed ({e}); continuing without normalised evidence")
            ev = []
        else:
            self._log(state, f"evidence_builder: normalised {len(ev)} items across "
                             f"{len({e['kind'] for e in ev})} source kinds")

        # Corpus fallback: a tools-path question (which SKIPPED retrieval) whose
        # XBRL/calc/table/market/web/EDGAR lanes ALL came back empty would reach
        # the synthesizer with nothing and produce a content-free "no data
        # available" answer. Route it back through fetch_filing → retrieve ONCE
        # (`corpus_fallback_used` guards the loop), so the filing corpus gets a
        # chance before we give up. `corpus_fallback_pending` is the one-shot
        # routing signal `_evidence_router` consumes; it is re-cleared here on
        # every pass (this node is its only writer and always precedes the
        # router).
        out: dict = {"evidence": ev, "corpus_fallback_pending": False}
        if not ev and not state.get("retrieved_chunks") \
                and not state.get("corpus_fallback_used"):
            routes = state.get("query_routes") or []
            took_tools_path = (self.dispatch and routes
                               and not any(r == "narrative" for r in routes))
            if took_tools_path:
                self._log(state, "tools lanes produced no evidence; "
                                 "falling back to corpus retrieval")
                out["corpus_fallback_pending"] = True
                out["corpus_fallback_used"] = True
        return out

    def critic_node(self, state: AgentState) -> dict:
        """v3 critic with web-search hits AND market-data tool results added to
        the evidence pool. Without this, a claim grounded in a Tavily hit or
        a yfinance quote looks "unsupported" to the critic (which only sees
        text + table chunks by default), triggers a retry, and ends up refused
        even though the verifier accepts it.
        """
        pseudo_chunks: list[dict] = []

        # XBRL facts → pseudo chunks (authoritative structured figures).
        for f in state.get("xbrl_facts", []) or []:
            pseudo_chunks.append({
                "text": (f"{f.get('concept','')} {f.get('period_label','FY'+str(f.get('fy','?')))} = "
                         f"{f.get('value_str','')} ({f.get('value')})"),
                "source": f.get("source", "<XBRL>"),
                "company": f.get("entity", f.get("ticker", "?")),
                "year": str(f.get("fy", "?")),
                "page": "—",
                "sub_query": f.get("sub_query", ""),
            })

        # Derived metrics → pseudo chunks (deterministic math on XBRL inputs).
        for r in state.get("calc_results", []) or []:
            pseudo_chunks.append({
                "text": self._format_calc_result(r),
                "source": f"<Calc: {r.get('metric','')} from XBRL>",
                "company": r.get("ticker", "?"),
                "year": str(r.get("fy", r.get("end_period", "?"))),
                "page": "—",
                "sub_query": r.get("sub_query", ""),
            })

        # EDGAR cross-document results → pseudo chunks (the matching companies).
        for r in state.get("edgar_results", []) or []:
            pseudo_chunks.append({
                "text": self._format_edgar_result(r),
                "source": f"<EDGAR FTS: {r.get('query','')}>",
                "company": "EDGAR",
                "year": "—",
                "page": "—",
                "sub_query": r.get("sub_query", ""),
            })

        # Web hits → pseudo chunks
        for h in state.get("web_results", []) or []:
            pseudo_chunks.append({
                "text": (h.get("content") or "")[:1500],
                "source": (
                    f"<News: {h.get('title','')[:80]} — {h.get('source','web')}>"
                ),
                "company": h.get("source", "web"),
                "year": (h.get("published_date") or h.get("date") or "")[:4] or "?",
                "page": "—",
                "sub_query": h.get("sub_query", ""),
            })

        # Live market data → pseudo chunks (one per successful tool call).
        for m in state.get("market_data", []) or []:
            if not m.get("ok"):
                continue
            tool = m.get("tool", "")
            data = m.get("data") or {}
            sym = (
                data.get("symbol")
                or (data.get("summary") or {}).get("symbol")
                or "—"
            )
            pseudo_chunks.append({
                "text": str(data)[:1800],
                "source": f"<Market: yfinance.{tool} {sym}>",
                "company": sym,
                "year": "—",
                "page": "—",
                "sub_query": m.get("sub_query", ""),
            })

        if not pseudo_chunks:
            out = super().critic_node(state)
        else:
            proxy = dict(state)
            proxy["retrieved_chunks"] = list(state.get("retrieved_chunks", [])) + pseudo_chunks
            out = super().critic_node(proxy)
        out.update(self._web_fallback_signal(state))
        return out

    # The draft conceding the evidence can't answer ("not specified in the
    # provided sources", "no information is available"). Grades can't catch
    # this: the chunks may be topically relevant (a 10-K MENTIONING an 8-K)
    # yet still not contain the answer — the draft's own admission is the
    # reliable insufficiency signal.
    _INSUFFICIENT_RE = re.compile(
        r"not (?:explicitly )?(?:specified|stated|provided|available|disclosed"
        r"|described|outlined|mentioned)"
        r"|no (?:specific )?(?:information|data|details?|mention|evidence|figures?)"
        r"|not enough information"
        r"|cannot be (?:determined|calculated|computed|answered)"
        r"|unable to (?:determine|find|locate)"
        r"|do(?:es)? not (?:contain|state|specify|describe|disclose|provide"
        r"|discuss|mention|cover)",
        re.I,
    )

    def _web_fallback_signal(self, state: AgentState) -> dict:
        """One-shot escalation signal: the draft admits the gathered evidence
        can't answer, and the web lane hasn't run yet — so route to web search
        and re-synthesize instead of shipping an "I couldn't find it" answer
        while a whole tool lane sits unused. `used` latches so a question the
        web genuinely can't answer either doesn't loop."""
        draft = state.get("draft_answer", "") or ""
        if (not state.get("web_results")
                and not state.get("web_fallback_used")
                and self._INSUFFICIENT_RE.search(draft)):
            self._log(state, "draft admits the evidence can't answer; "
                             "escalating to web search")
            return {"web_fallback_pending": True, "web_fallback_used": True}
        return {"web_fallback_pending": False}

    def synthesize_node(self, state: AgentState) -> dict:
        """v3 synth + web results, presented as one unified numbered evidence list.

        Numbering order matches what the frontend sees (text → web → table),
        so a `[3]` in the answer points at the third card in the sidebar.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        chunks = state.get("retrieved_chunks", []) or []
        web_res = state.get("web_results", []) or []
        table_res = state.get("table_results", []) or []
        market_data = state.get("market_data", []) or []
        xbrl_facts = state.get("xbrl_facts", []) or []
        calc_res = state.get("calc_results", []) or []
        edgar_res = state.get("edgar_results", []) or []

        # Phase 10 — kill redundancy before synthesis. Cluster near-identical
        # passages and keep one representative each, separately within filing
        # excerpts and within web hits (so a filing and a web page making the
        # same point both survive — source prioritisation, below, reconciles
        # them). Then order web hits newest-first so "newer wins" is the default.
        chunks = self._dedupe_evidence(chunks, "text")
        web_res = self._dedupe_evidence(web_res, "content")
        web_res = sorted(
            web_res,
            key=lambda h: (h.get("published_date") or h.get("date") or ""),
            reverse=True,
        )

        evidence_items: list[str] = []
        idx = 1

        # 0. XBRL facts FIRST — exact, structured, filing-sourced figures. These
        # are authoritative: when a figure exists here, the synthesizer must use
        # this value verbatim and prefer it over any number paraphrased from
        # prose, tables, or the web. (Matches the frontend ordering in
        # rag_service, which lists XBRL chunks first.)
        for f in xbrl_facts:
            evidence_items.append(
                f"[{idx}] XBRL FACT (authoritative — exact figure as filed) — "
                f"{f.get('entity', f.get('ticker',''))} {f.get('concept','')} "
                f"{f.get('period_label', 'FY' + str(f.get('fy','?')))}: {f.get('value_str','')}\n"
                f"Source: {f.get('source','')} (us-gaap:{f.get('tag','')})."
            )
            idx += 1

        # 0b. Derived metrics computed over XBRL inputs (margins, ratios, growth,
        # CAGR, trends). Also authoritative — deterministic math on exact filed
        # figures — so the synthesizer should use these values verbatim.
        for r in calc_res:
            evidence_items.append(
                f"[{idx}] DERIVED METRIC (computed from exact XBRL inputs) — "
                f"{self._format_calc_result(r)}"
            )
            idx += 1

        # 1. Text excerpts
        for c in chunks:
            tag = c.get("source", "")
            text = c.get("text", "")
            evidence_items.append(f"[{idx}] FILING EXCERPT — {tag}\n{text[:1500]}")
            idx += 1

        # 1b. Live market data (yfinance) — fresh numbers the synth should
        # prefer when the question is market-flavoured.
        for m in market_data:
            if not m.get("ok"):
                continue
            tool = m.get("tool", "")
            data = m.get("data") or {}
            if tool == "get_history":
                s = data.get("summary", {})
                vol_line = ""
                if s.get("avg_volume") is not None:
                    vol_line = (
                        f"\nVolume: last={s.get('last_volume'):,}, "
                        f"avg={s.get('avg_volume'):,}, recent_avg={s.get('recent_avg_volume'):,} "
                        f"vs prior_avg={s.get('prior_avg_volume'):,} "
                        f"({s.get('volume_change_pct')}% change"
                        f"{', SURGE' if s.get('volume_surge') else ''})."
                    )
                evidence_items.append(
                    f"[{idx}] LIVE MARKET (yfinance · get_history) — "
                    f"{s.get('symbol','?')} {s.get('period','?')} {s.get('interval','?')}\n"
                    f"Range {s.get('start','?')} → {s.get('end','?')}; "
                    f"first_close={s.get('first_close')}, last_close={s.get('last_close')}, "
                    f"high={s.get('high')}, low={s.get('low')}, "
                    f"pct_change={s.get('pct_change')}%."
                    f"{vol_line}"
                )
            elif tool == "compare":
                rows = "\n".join(
                    f"  - {r.get('symbol')}: last={r.get('lastPrice')} "
                    f"prevClose={r.get('previousClose')} yearChange={r.get('yearChange')}"
                    for r in (data.get("rows") or [])
                )
                evidence_items.append(
                    f"[{idx}] LIVE MARKET (yfinance · compare)\n{rows}"
                )
            elif tool == "get_news":
                arts = "\n".join(
                    f"  - {a.get('title','')} ({a.get('publisher','')})"
                    for a in (data.get("articles") or [])
                )
                evidence_items.append(
                    f"[{idx}] LIVE MARKET (yfinance · get_news) — {data.get('symbol','?')}\n{arts}"
                )
            else:
                # get_quote / get_company_info — flatten dict
                kv = ", ".join(
                    f"{k}={v}" for k, v in (data or {}).items() if v not in (None, "")
                )
                evidence_items.append(
                    f"[{idx}] LIVE MARKET (yfinance · {tool})\n{kv[:1200]}"
                )
            idx += 1

        # 2. Web hits (trusted first thanks to WebSearcher's two-pass).
        # The publication date is surfaced in the header so the synthesizer
        # can reason about recency — essential for "premarket", "today's
        # price", "this week's news" style questions.
        for h in web_res:
            tier = "TRUSTED PRESS" if h.get("tier") == "trusted" else "WEB"
            pub = h.get("published_date") or ""
            pub_str = f" (published {pub})" if pub else ""
            evidence_items.append(
                f"[{idx}] {tier}{pub_str} — {h.get('title','')[:120]} "
                f"({h.get('url','')})\n{(h.get('content') or '')[:1000]}"
            )
            idx += 1

        # 3. Table computations
        for t in table_res:
            if t.get("error") and not t.get("answer"):
                continue
            used = (t.get("tables_used") or [])[:1]
            first = used[0] if used else {}
            evidence_items.append(
                f"[{idx}] TABLE COMPUTATION — {first.get('title','?')} "
                f"({first.get('company','?')} {first.get('year','?')}, "
                f"p. {first.get('page','?')})\n"
                f"Computed: {t.get('answer','')[:600]}"
            )
            idx += 1

        # 4. EDGAR full-text cross-document results — the set of companies whose
        # filings match, which single-company retrieval can't produce.
        for r in edgar_res:
            evidence_items.append(
                f"[{idx}] EDGAR CROSS-DOCUMENT SEARCH — {self._format_edgar_result(r)}"
            )
            idx += 1

        evidence_block = (
            "\n\n".join(evidence_items) if evidence_items else "(no evidence retrieved)"
        )
        sub_queries = "\n".join(f"- {q}" for q in state.get("sub_queries", []))

        # Conversation memory — only include if we actually have prior turns
        # so single-turn questions stay short.
        history = state.get("chat_history") or []
        history_block = ""
        if history:
            lines = []
            for turn in history[-6:]:                       # cap at last 6
                role = "User" if turn.get("role") == "user" else "Assistant"
                body = (turn.get("content") or "")[:600]
                lines.append(f"{role}: {body}")
            history_block = (
                "Earlier in this conversation (most recent last):\n"
                + "\n".join(lines)
                + "\n\n"
            )

        # Active-critic re-draft (#6): if a prior critic pass flagged claims it
        # couldn't support, tell the synth to fix or drop exactly those — using
        # the SAME evidence (the issue was over-claiming, not missing evidence).
        feedback = state.get("critic_feedback") or []
        feedback_block = ""
        if feedback and self.active_critic:
            bullet = "\n".join(f"  - {c}" for c in feedback[:5])
            feedback_block = (
                "\nA reviewer flagged these claims as NOT supported by the evidence "
                "above — remove them, hedge them, or re-ground them in a cited "
                "[N] item; do not repeat an unsupported figure:\n" + bullet + "\n"
            )

        prompt = f"""{history_block}Question: {state['question']}

Sub-queries researched:
{sub_queries}

Numbered evidence (cite with `[N]`):
{evidence_block}
{feedback_block}
---
Write your answer now in well-structured markdown with [N] citations after
every factual claim. Treat the conversation history above as context for
resolving pronouns / follow-ups ("it", "that company", "what about FY24")
but do NOT cite items from prior turns — only cite the numbered evidence in
this turn. If the current evidence contains usable material — including
web / news items — USE IT; don't fall back to "no information" unless every
single item is irrelevant."""

        llm = self._get_llm("synth")
        synth_system = SYNTH_ANALYST_SYSTEM if self.analyst_voice else SYNTH_V3_SYSTEM
        response = llm.invoke([
            SystemMessage(content=synth_system),
            HumanMessage(content=prompt),
        ])
        answer = response.content
        # Citation extraction: only `[N]` / `[N, M]` markers, no verbose tags.
        citations = sorted(set(re.findall(r"\[\d+(?:\s*,\s*\d+)*\]", answer)))
        low_conf = (
            state.get("avg_grade", 0.0) < self.grade_threshold
            and state.get("iteration_count", 0) >= self.max_rewrites
            and not table_res and not web_res
        )
        return {
            "draft_answer": answer,
            "final_answer": answer,
            "citations": citations,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "low_confidence": bool(low_conf),
        }

    def verify_numbers_node(self, state: AgentState) -> dict:
        """Phase 11 fact-checking critic — every figure traces to a source.

        Deterministically extract EVERY number in the draft, then ground each
        against XBRL facts / derived metrics (exact) and numbers parsed from the
        retrieved chunks, tables, web, and market data. The LLM verifier runs as
        a *rescue* (its matched figures supplement the evidence) and to produce
        human-readable claims. A figure grounded by neither is a hallucination;
        `_verify_router` then re-routes or refuses. Tracks the hallucination rate
        explicitly. With `strict_numeric` False this falls back to the prior
        LLM-only check.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Count every verify pass so `_verify_router` can bound the re-route loop
        # independently of the critic (the verifier may keep finding an
        # ungrounded figure even when the critic is happy — without this counter
        # that loops until LangGraph's recursion limit aborts the request).
        vi = {"verify_iterations": state.get("verify_iterations", 0) + 1}

        answer = state.get("draft_answer", "")
        clean = {"claims": [], "unverified": [], "score": 1.0,
                 "numbers_total": 0, "numbers_grounded": 0, "hallucination_rate": 0.0}
        if not self._has_numbers(answer):
            return {"numeric_verification": clean, **vi}

        def _run_llm_verifier() -> list[dict]:
            evidence = self._build_evidence_block(state)
            llm = self._get_verifier_llm().with_structured_output(NumericVerification)
            try:
                report: NumericVerification = llm.invoke([
                    SystemMessage(content=NUM_VERIFY_SYSTEM),
                    HumanMessage(content=NUM_VERIFY_PROMPT.format(
                        answer=answer, evidence=evidence)),
                ])
                return [c.model_dump() for c in report.claims]
            except Exception as e:
                self._log(state, f"verifier failed ({e})")
                return []

        # Legacy LLM-only path (A/B baseline).
        if not self.strict_numeric:
            claims = _run_llm_verifier()
            llm_unverified = [c for c in claims if not c.get("matched")]
            score = ((len(claims) - len(llm_unverified)) / len(claims)) if claims else 1.0
            return {"numeric_verification": {"claims": claims, "unverified": llm_unverified,
                                             "score": round(score, 3),
                                             "numbers_total": len(claims),
                                             "numbers_grounded": len(claims) - len(llm_unverified),
                                             "hallucination_rate": round(1 - score, 3)}, **vi}

        # Deterministic, exhaustive grounding — runs FIRST and without any LLM.
        draft_nums = self._extract_numbers(answer)
        if not draft_nums:
            return {"numeric_verification": dict(clean), **vi}

        by_kind = self._evidence_numbers_by_kind(state)
        evidence_mags = [m for vals in by_kind.values() for m in vals]
        derive_base = self._derivation_base(by_kind)

        def _walk(extra_mags: list[float]) -> list[dict]:
            """Walk the draft's figures in order, letting each grounded figure
            feed the pool (`chained`) so a worked calculation grounds stepwise:
            the average grounds from its two inputs, the ratio from that
            average, the final metric from the ratios. Derivation (`_derivable`)
            rescues figures the draft legitimately computed from evidence."""
            pool = evidence_mags + extra_mags
            chained: list[float] = []
            missing: list[dict] = []
            for d in draft_nums:
                ok = (
                    self._grounded(d["magnitudes"], pool)
                    or self._grounded(d["magnitudes"], chained)
                    or self._derivable(d["magnitudes"], derive_base + chained)
                )
                if ok:
                    chained.extend(d["magnitudes"])
                else:
                    missing.append({"number": d["raw"], "claim": d["ctx"]})
            return missing

        ungrounded = _walk([])
        claims: list[dict] = []
        if ungrounded:
            # LLM rescue — invoked ONLY when deterministic grounding left
            # figures unaccounted for (skipping it on the fully-grounded common
            # path saves a strong-tier LLM call per answer). Figures the
            # verifier matched count as grounded too.
            claims = _run_llm_verifier()
            rescued: list[float] = []
            for c in claims:
                if c.get("matched"):
                    for n in self._extract_numbers(
                            str(c.get("number", "")) + " " + str(c.get("evidence", ""))):
                        rescued.extend(n["magnitudes"])
            if rescued:
                ungrounded = _walk(rescued)
        ungrounded_raws = {u["number"] for u in ungrounded}
        total = len(draft_nums)
        grounded = total - len(ungrounded)
        score = grounded / total if total else 1.0
        for u in ungrounded:
            self._log(state, f"ungrounded figure: {u['number']} — {u['claim'][:100]}")

        nv = {
            "claims": claims,
            "unverified": ungrounded,              # refuse_node reads this key
            "score": round(score, 3),
            "numbers_total": total,
            "numbers_grounded": grounded,
            "hallucination_rate": round(len(ungrounded) / total, 3) if total else 0.0,
        }
        try:
            report = self._build_verification_report(
                state, draft_nums, ungrounded_raws, by_kind)
        except Exception as e:
            # The report is advisory — never let it break the refusal-bearing
            # numeric verdict.
            self._log(state, f"verification report failed ({e})")
            report = {}
        report["numeric"] = nv
        return {"numeric_verification": nv, "verification_report": report, **vi}

    # ------------------------------------------------------------------ #
    # #5 — financial verification report (cross-source / units / sources)
    # ------------------------------------------------------------------ #

    def _exact_raw_values(self, state: AgentState) -> list[float]:
        """The exact, UN-expanded XBRL / calc figures (1× scale only) — used to
        tell whether a grounded draft number matched at its stated scale or only
        after a power-of-10 restatement (a possible unit slip)."""
        raw: list[float] = []
        for f in state.get("xbrl_facts", []) or []:
            v = f.get("value")
            if isinstance(v, (int, float)):
                raw.append(float(v))
        for r in state.get("calc_results", []) or []:
            for key_val in ([r.get("value")] + [s.get("value") for s in (r.get("series") or [])]
                            + [i.get("value") for i in (r.get("inputs") or []) if isinstance(i, dict)]):
                if isinstance(key_val, (int, float)):
                    raw.extend([float(key_val), float(key_val) * 100.0])
        return raw

    def _build_verification_report(self, state: AgentState, draft_nums: list[dict],
                                   ungrounded_raws: set, by_kind: dict) -> dict:
        """Assemble the financial verification report the spec (#5) asks for:
        cross-source corroboration, unit-shift flags, and source/citation
        coverage. This is a REPORT — it informs confidence and the audit trail;
        it does not itself trigger refusals (that stays with numeric grounding).
        """
        grounded_nums = [d for d in draft_nums if d["raw"] not in ungrounded_raws]
        source_kinds = [k for k in by_kind if k != "const"]

        # Cross-source: which independent lanes corroborate each grounded figure.
        details: list[dict] = []
        corroborated = single = web_only = 0
        for d in grounded_nums:
            matching = [
                k for k in source_kinds
                if any(self._num_close(m, ev) for m in d["magnitudes"] for ev in by_kind[k])
            ]
            if not matching:
                continue                            # grounded only by a constant
            if len(matching) >= 2:
                corroborated += 1
            else:
                single += 1
                if matching == ["web"]:
                    web_only += 1
            if len(details) < 12:
                details.append({"number": d["raw"], "kinds": matching})

        # Units: figures that grounded only after a scale restatement (the stated
        # magnitude didn't match an exact value at 1×) — a possible unit slip.
        exact_raw = self._exact_raw_values(state)
        scale_shifted = 0
        for d in grounded_nums:
            if any(self._num_close(m, v) for m in d["magnitudes"] for v in exact_raw):
                continue                            # matched at its stated scale
            if any(self._num_close(m, v * s) for m in d["magnitudes"]
                   for v in exact_raw for s in self._RESTATE_SCALES if s != 1.0):
                scale_shifted += 1

        # Sources: citation coverage of the structured numeric evidence items.
        ev = state.get("evidence") or []
        numeric_items = [e for e in ev if e.get("value") is not None]
        with_cit = sum(1 for e in numeric_items if (e.get("citation") or "").strip())

        return {
            "cross_source": {
                "corroborated": corroborated,       # ≥2 independent lanes agree
                "single_source": single,
                "web_only": web_only,               # weakest — web alone
                "details": details,
            },
            "units": {"scale_shifted_figures": scale_shifted},
            "sources": {
                "numeric_evidence_items": len(numeric_items),
                "with_citation": with_cit,
                "without_citation": len(numeric_items) - with_cit,
            },
        }

    def refuse_node(self, state: AgentState) -> dict:
        """Replace the final answer with an explicit refusal."""
        web_used = bool(state.get("web_results"))
        unverified = state.get("numeric_verification", {}).get("unverified", []) if isinstance(state.get("numeric_verification"), dict) else []
        detail = ""
        if unverified:
            nums = ", ".join(str(u.get("number")) for u in unverified[:3] if u.get("number"))
            if nums:
                detail = f" (unverified figures: {nums})"
        web_clause = "" if web_used else " or recent web sources"
        msg = REFUSAL_TEMPLATE.format(web_clause=web_clause, detail=detail)
        return {"final_answer": msg, "refused": True, "needs_retry": False,
                "status": "refused"}

    # ------------------------------------------------------------------ #
    # Confidence framework (#8/9)
    # ------------------------------------------------------------------ #

    # Weights from the spec. Applied only over the sub-scores that are present
    # for a given question; the denominator renormalises so a missing component
    # (e.g. retrieval on a pure-XBRL numeric query) doesn't drag the blend down.
    _CONF_WEIGHTS = {
        "retrieval": 0.25,
        "verification": 0.35,
        "citation": 0.25,
        "critic": 0.15,
    }

    # A citation marker: [1] / [1, 3] and the fullwidth 【1】 form some models
    # (gpt-oss, certain Llama builds) emit despite the prompt asking for ASCII.
    # Requiring a leading digit keeps it off prose brackets like "[note]".
    _CITE_RE = re.compile(r"[\[【]\s*\d[\d,\s]*\s*[\]】]")

    def _citation_score(self, state: AgentState) -> Optional[float]:
        """Fraction of material figures in the draft that carry a [N] citation.

        Returns None (component not applicable) when the draft is empty or there
        is no evidence to cite at all. For a figure-free answer it's 1.0 if any
        citation is present, else None — we don't penalise a purely qualitative
        answer for having no numbers to anchor.
        """
        answer = state.get("draft_answer", "") or ""
        if not answer.strip():
            return None
        has_evidence = any(state.get(k) for k in (
            "retrieved_chunks", "xbrl_facts", "calc_results", "table_results",
            "web_results", "market_data", "edgar_results",
        ))
        if not has_evidence:
            return None

        # Positions of citation markers in the ORIGINAL answer.
        markers = [m.start() for m in self._CITE_RE.finditer(answer)]

        # Mask out tokens that look numeric but aren't financial claims, keeping
        # length (and therefore offsets) identical so `markers` stays aligned.
        def _blank(m: re.Match) -> str:
            return " " * len(m.group(0))

        masked = self._CITE_RE.sub(_blank, answer)
        masked = re.sub(r"\bFY\s?\d{2,4}\b", _blank, masked, flags=re.I)
        masked = re.sub(r"\bQ[1-4]\b", _blank, masked, flags=re.I)
        masked = re.sub(r"\b10\s?-?\s?[KQ]\b|\b8\s?-?\s?K\b", _blank, masked, flags=re.I)
        masked = re.sub(r"\bp\.?\s*\d+\b", _blank, masked, flags=re.I)

        fig_ends = [m.end() for m in self._NUM_RE.finditer(masked) if m.group("num")]
        if not fig_ends:
            return 1.0 if markers else None

        # A figure is "cited" if a marker sits just after it (or a few chars
        # before, for "[1] revenue of $X" ordering).
        WINDOW = 80
        covered = sum(
            1 for p in fig_ends
            if any(-15 <= (mk - p) <= WINDOW for mk in markers)
        )
        return round(covered / len(fig_ends), 3)

    def _confidence_components(self, state: AgentState) -> dict:
        """The sub-scores that apply to this question, each in [0,1]."""
        comps: dict[str, float] = {}

        # Retrieval — normalise the mean grade (1-5) to [0,1]. Applicable only
        # when retrieval actually ran (graded chunks exist).
        grades = state.get("grades") or []
        avg = state.get("avg_grade")
        if grades and avg is not None:
            comps["retrieval"] = max(0.0, min(1.0, (avg - 1.0) / 4.0))
        elif state.get("retrieved_chunks"):
            # Chunks present but ungraded (e.g. ephemeral fetch on the tool path).
            comps["retrieval"] = 0.5

        # Verification — only when the answer actually made numeric claims.
        nv = state.get("numeric_verification") or {}
        if isinstance(nv, dict) and nv.get("numbers_total", 0) > 0:
            comps["verification"] = max(0.0, min(1.0, float(nv.get("score", 0.0))))

        # Citation coverage.
        cs = self._citation_score(state)
        if cs is not None:
            comps["citation"] = cs

        # Critic — claim-support rate (None when the critic couldn't score).
        gs = state.get("grading_score")
        if gs is not None:
            comps["critic"] = max(0.0, min(1.0, float(gs)))

        return comps

    def confidence_node(self, state: AgentState) -> dict:
        """Blend the applicable sub-scores into a single confidence and pick a band.

        Runs once, after numeric verification has settled, on the path that the
        verifier already deemed answerable (ungrounded-figure refusals are
        handled upstream by `_verify_router`). Writes the four sub-scores plus
        the blend and the routing band into state for the gate, the audit trail,
        and the UI.
        """
        try:
            comps = self._confidence_components(state)
            if comps:
                wsum = sum(self._CONF_WEIGHTS[k] for k in comps)
                conf = sum(self._CONF_WEIGHTS[k] * v for k, v in comps.items()) / wsum
            else:
                conf = 0.0
            conf = round(conf, 3)
        except Exception as e:
            # If scoring itself fails, don't block a verified answer — fail OPEN
            # (answer, no gate) rather than refusing a good answer on a math bug.
            self._log(state, f"confidence scoring failed ({e}); answering without the gate")
            return {"confidence": None, "confidence_band": "answer",
                    "status": "answered"}

        if not self.confidence_gating:
            band = "answer"
        elif conf >= self.confidence_answer:
            band = "answer"
        elif conf >= self.confidence_warn:
            band = "warn"
        else:
            band = "refuse"

        self._log(
            state,
            f"confidence={conf} band={band} "
            f"[{', '.join(f'{k}={v:.2f}' for k, v in comps.items())}]",
        )
        return {
            "retrieval_score": comps.get("retrieval"),
            "verification_score": comps.get("verification"),
            "citation_score": comps.get("citation"),
            "critic_score": comps.get("critic"),
            "confidence": conf,
            "confidence_band": band,
            "status": "answered" if band == "answer" else state.get("status"),
        }

    def withhold_low_confidence_node(self, state: AgentState) -> dict:
        """Low-confidence band: show the draft IN FULL with the confidence
        score appended — the same presentation as the warn band, with a
        stronger caveat. (Previously this hid the draft behind a "low-
        confidence answer available" notice, which buried answers that were
        often correct; hallucinated-figure refusals are handled separately by
        `refuse_node`, so the figures here are grounded — the uncertainty is
        about completeness/corroboration, which the caveat conveys.)
        """
        ans = state.get("draft_answer", "") or state.get("final_answer", "") or ""
        if "_Confidence:" in ans:                      # idempotent on a re-entry
            return {"status": "answered_low_confidence"}
        conf = state.get("confidence")
        pct = f"{conf:.0%}" if isinstance(conf, (int, float)) else "low"
        note = (
            f"\n\n*_Confidence: {pct} — low. Parts of this answer are weakly "
            f"corroborated by the available sources; verify against the "
            f"primary filing before relying on it._*"
        )
        return {
            "final_answer": ans + note,
            "refused": False,
            "needs_retry": False,
            "status": "answered_low_confidence",
        }

    def answer_with_warning_node(self, state: AgentState) -> dict:
        """Moderate-confidence band: keep the answer but append a one-line caveat."""
        ans = state.get("final_answer", "") or state.get("draft_answer", "")
        if "_Confidence:" in ans:                      # idempotent on a re-entry
            return {"status": "answered_with_warning"}
        conf = state.get("confidence")
        pct = f"{conf:.0%}" if isinstance(conf, (int, float)) else "moderate"
        note = (
            f"\n\n*_Confidence: {pct} — moderate. Some figures or claims are only "
            f"partially corroborated by the available sources; verify against the "
            f"primary filing before relying on them._*"
        )
        return {"final_answer": ans + note, "status": "answered_with_warning"}

    # ------------------------------------------------------------------ #
    # Routers
    # ------------------------------------------------------------------ #

    def _critic_system(self) -> str:
        """Phase 9: use the analyst-voice critic (period/unit-aware numeric
        rigor) when enabled, else the generic hallucination critic."""
        return CRITIC_ANALYST_SYSTEM if self.analyst_voice else super()._critic_system()

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
        # XBRL client (facts + resolver cache) and table_agent reads the Chroma
        # `tables` collection. Keeping them ordered avoids the concurrent-Chroma
        # segfault class this codebase already hit once (commit 71e6bda).
        g.add_edge("xbrl", "calculator")
        g.add_edge("calculator", "table_agent")
        # #2: the three NETWORK-bound lanes are mutually independent (distinct
        # state keys, different services) and dominate tool latency, so fan them
        # out to run in PARALLEL after the numeric chain. Only web_search's
        # news-fallback touches Chroma, and it can't race table_agent (which has
        # already finished) — so no two Chroma readers ever overlap.
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

    @staticmethod
    def _format_web_results(web_res: list[dict]) -> str:
        parts: list[str] = []
        for h in web_res:
            tier = h.get("tier", "web")
            label = "Trusted" if tier == "trusted" else "Web"
            tag = f"<News: {h.get('title', '?')[:90]} — {h.get('source', 'web')}>"
            url = h.get("url", "")
            body = (h.get("content") or "")[:1000]
            parts.append(f"[{label}] {tag} {url}\n{body}")
        return "\n\n".join(parts)

    @staticmethod
    def _has_numbers(text: str) -> bool:
        return bool(re.search(r"\d", text or ""))

    # ------------------------------------------------------------------ #
    # Phase 11 — deterministic numeric grounding
    # ------------------------------------------------------------------ #

    # Scale words → multiplier. Single-letter B/M/K/T handled with a word
    # boundary so they don't fire on stray capitals mid-word.
    _SCALES = {
        "trillion": 1e12, "tn": 1e12, "t": 1e12,
        "billion": 1e9, "bn": 1e9, "b": 1e9,
        "million": 1e6, "mn": 1e6, "m": 1e6,
        "thousand": 1e3, "k": 1e3,
        "crore": 1e7, "lakh": 1e5,
    }
    # Trailing `(?=\b)` lookahead guards ONLY the single-letter scales (so a bare
    # "B"/"M" must end a word, e.g. "$99.8B"); `%`/`bps`/word-scales match
    # directly (a trailing `\b` here would wrongly reject "7.8%").
    _NUM_RE = re.compile(
        r"(?P<sign>[-+]?)\$?\s*"
        r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
        r"(?P<scale>trillion|tn|billion|bn|million|mn|thousand|crore|lakh|bps|%|[bmkt](?=\b))?",
        re.IGNORECASE,
    )

    @classmethod
    def _extract_numbers(cls, text: str) -> list[dict]:
        """Extract EVERY material figure from `text` as {raw, magnitudes, ctx}.

        `magnitudes` is the set of plausible numeric values the figure could mean
        — e.g. "30.3%" → {30.3, 0.303}, "$394.3 billion" → {3.943e11}. Matching
        any magnitude against evidence (with tolerance) grounds the figure. We
        strip citation markers, page refs, fiscal-year tokens and bare years
        first so they aren't mistaken for financial claims.
        """
        if not text:
            return []
        # Normalise Unicode hyphens/dashes (‑ – — − …) to ASCII "-" so the
        # range/label scrubs below catch them regardless of which dash the model
        # emitted (the synth often uses a non-breaking hyphen in "FY2023‑24").
        scrubbed = text.translate({
            0x2010: 0x2d, 0x2011: 0x2d, 0x2012: 0x2d, 0x2013: 0x2d,
            0x2014: 0x2d, 0x2015: 0x2d, 0x2212: 0x2d,
        })
        # Remove things that look numeric but aren't financial claims.
        scrubbed = cls._CITE_RE.sub(" ", scrubbed)             # [1], [1, 3], 【1】
        # Fiscal-year RANGES first ("FY2023-24", "2023-2024", "FY2015 - FY2017"),
        # then single years. Without the optional second "FY" the range scrub
        # missed "FY2015 - FY2017", leaving a stray "- 3" style artifact.
        scrubbed = re.sub(r"\bFY\s?\d{4}\s?-\s?(?:FY\s?)?\d{2,4}\b", " ", scrubbed, flags=re.I)
        scrubbed = re.sub(r"\b(?:19|20)\d{2}\s?-\s?\d{2,4}\b", " ", scrubbed)
        scrubbed = re.sub(r"\bFY\s?\d{2,4}\b", " ", scrubbed, flags=re.I)
        # Window labels: "52-week", "52 wk", "200-day", "3 year average" — the
        # count names the metric/period, it isn't a reported figure.
        scrubbed = re.sub(r"\b\d{1,3}\s?-?\s?(?:week|wk|day|month|mo|year|yr|quarter|qtr)s?\b",
                          " ", scrubbed, flags=re.I)
        scrubbed = re.sub(r"\bQ[1-4]\b", " ", scrubbed, flags=re.I)
        # SEC form names in every spelling the models emit: 10-K, 10K, 8-K, 8k.
        scrubbed = re.sub(r"\b10\s?-?\s?[KQ]\b|\b8\s?-?\s?K\b", " ", scrubbed, flags=re.I)
        scrubbed = re.sub(r"\bp\.?\s*\d+\b", " ", scrubbed, flags=re.I)  # p. 12

        out: list[dict] = []
        for m in cls._NUM_RE.finditer(scrubbed):
            raw = m.group(0).strip()
            num = m.group("num").replace(",", "")
            try:
                base = float(num)
            except ValueError:
                continue
            if m.group("sign") == "-":
                # Distinguish a negative quantity (a $5.2bn loss) from a binary
                # subtraction operator inside a shown derivation ("a - b"): if the
                # text just before the '-' ends with a digit / ')' / '%', it's a
                # subtraction and the figure itself is positive. Misreading it as
                # negative would refuse a correct worked calculation.
                prefix = scrubbed[: m.start()].rstrip()
                if not (prefix and prefix[-1] in "0123456789)%"):
                    base = -base
            scale = (m.group("scale") or "").lower()

            if scale in ("%",):
                mags = {base, base / 100.0}
            elif scale == "bps":
                mags = {base, base / 100.0, base / 10000.0}
            elif scale in cls._SCALES:
                mags = {base * cls._SCALES[scale]}
            else:
                # No scale word. Skip bare integers that are almost certainly
                # years (1900-2099) — they're periods, not financial figures.
                if scale == "" and base.is_integer() and 1900 <= base <= 2099:
                    continue
                mags = {base}
            ctx = scrubbed[max(0, m.start() - 30): m.end() + 30].strip()
            out.append({"raw": raw, "magnitudes": mags, "ctx": ctx})
        return out

    @staticmethod
    def _num_close(a: float, b: float, rel_tol: float = 0.02) -> bool:
        """Scale-free closeness: within 2% relative (covers rounding like
        $394.3bn vs 394,328,000,000, or 30.3% vs 0.303)."""
        return abs(a - b) <= rel_tol * max(abs(a), abs(b), 1e-9)

    @classmethod
    def _grounded(cls, magnitudes: set, evidence_mags: list[float]) -> bool:
        return any(cls._num_close(mag, ev) for mag in magnitudes for ev in evidence_mags)

    # Cap on the deduped evidence pool used for derived grounding — keeps the
    # pairwise scan (~ cap² / 2 pairs per figure) cheap.
    _DERIVE_MAX_BASE = 160

    @classmethod
    def _derivation_base(cls, by_kind: dict[str, list[float]]) -> list[float]:
        """Deduped, capped pool of evidence magnitudes for `_derivable`.
        Exact structured lanes first so they survive the cap."""
        order = ("xbrl", "calc", "table", "filing", "market", "web", "const")
        seen: set[str] = set()
        base: list[float] = []
        for kind in order + tuple(k for k in by_kind if k not in order):
            for v in by_kind.get(kind, []):
                key = f"{v:.6g}"
                if key not in seen:
                    seen.add(key)
                    base.append(v)
                    if len(base) >= cls._DERIVE_MAX_BASE:
                        return base
        return base

    @classmethod
    def _derivable(cls, magnitudes: set, base: list[float]) -> bool:
        """A figure absent from the evidence VERBATIM may still be correct if it
        is a one-step combination of evidence values — a ratio/margin (a/b, with
        the ×100 percent and ×365 days-outstanding forms), a sum, a change
        (a−b), or a two-period average. Without this, every answer that does the
        arithmetic the question asked for (margins, averages, working-capital
        days) self-refutes: the derived figure is "ungrounded" even though all
        its inputs are in the evidence.
        """
        close = cls._num_close
        for t in magnitudes:
            n = len(base)
            for i in range(n):
                a = base[i]
                for j in range(i + 1, n):
                    b = base[j]
                    if close(t, a + b) or close(t, abs(a - b)) or close(t, (a + b) / 2.0):
                        return True
                    for num, den in ((a, b), (b, a)):
                        if den == 0.0:
                            continue
                        r = num / den
                        if close(t, r) or close(t, r * 100.0) or close(t, r * 365.0):
                            return True
        return False

    # Powers of ten by which an exact figure may be RESTATED in prose. A filed
    # value of 135,987,000,000 is the same fact whether the answer writes it as
    # "$135.987 billion" (×1), "135,987" million (×1e-6), or a scale-free
    # "135.987" inside a worked formula (×1e-9). Expanding each exact evidence
    # value across these scales makes grounding unit-insensitive, so a correct
    # derivation isn't refused just because it dropped the "billion" word.
    _RESTATE_SCALES = (1.0, 1e-3, 1e-6, 1e-9, 1e-12)

    # Structural constants that appear in growth / CAGR / margin DERIVATIONS
    # (×100 to render a percent, the 1 in `(b/a)^(1/n) − 1`, a leading 0), plus
    # time-period constants from working-capital formulas (365 in DSO/DIO/DPO,
    # 360-day conventions, 52 weeks, 12 months, 4 quarters, 2 in an average).
    # They are math scaffolding the synthesizer writes out, never financial
    # claims — so they must count as grounded or every shown calculation
    # self-refutes.
    _MATH_CONSTANTS = (0.0, 1.0, 2.0, 100.0, 365.0, 360.0, 52.0, 12.0, 4.0)

    def _evidence_numbers_by_kind(self, state: AgentState) -> dict[str, list[float]]:
        """Grounding magnitudes bucketed BY SOURCE KIND, so cross-source
        validation can tell which lanes corroborate a given figure.

        Exact structured lanes (xbrl, calc) are scale-expanded — see
        `_RESTATE_SCALES`; free-text lanes (filing, table, market, web) are taken
        at face value. The flat union (`_evidence_numbers`) is identical to the
        prior behaviour, so number GROUNDING is unchanged — this only adds the
        per-kind attribution used by the verification report.
        """
        buckets: dict[str, list[float]] = {"const": list(self._MATH_CONSTANTS)}

        def add_exact(kind: str, v: float) -> None:
            b = buckets.setdefault(kind, [])
            for scale in self._RESTATE_SCALES:
                b.append(v * scale)

        def add_text(kind: str, txt: str) -> None:
            b = buckets.setdefault(kind, [])
            for n in self._extract_numbers(txt):
                b.extend(n["magnitudes"])

        # XBRL — exact filed values (ground truth).
        for f in state.get("xbrl_facts", []) or []:
            v = f.get("value")
            if isinstance(v, (int, float)):
                add_exact("xbrl", float(v))

        # Derived metrics — result, percent form, trend series, AND the exact
        # XBRL inputs the metric was computed from (what a shown derivation uses).
        for r in state.get("calc_results", []) or []:
            v = r.get("value")
            if isinstance(v, (int, float)):
                add_exact("calc", float(v)); add_exact("calc", float(v) * 100.0)
            for s in r.get("series", []) or []:
                sv = s.get("value")
                if isinstance(sv, (int, float)):
                    add_exact("calc", float(sv)); add_exact("calc", float(sv) * 100.0)
            for inp in r.get("inputs", []) or []:
                iv = inp.get("value") if isinstance(inp, dict) else None
                if isinstance(iv, (int, float)):
                    add_exact("calc", float(iv))

        # Free-text lanes — face value.
        for c in state.get("retrieved_chunks", []) or []:
            add_text("filing", c.get("text", ""))
        for t in state.get("table_results", []) or []:
            add_text("table", str(t.get("answer", "")) + " " + str(t.get("stdout", "")))
        for mkt in state.get("market_data", []) or []:
            add_text("market", str(mkt.get("data", "")))
        for h in state.get("web_results", []) or []:
            add_text("web", h.get("content", "") or "")
        return buckets

    def _evidence_numbers(self, state: AgentState) -> list[float]:
        """Flat union of every grounding magnitude across all source kinds."""
        mags: list[float] = []
        for vals in self._evidence_numbers_by_kind(state).values():
            mags.extend(vals)
        return mags

    @staticmethod
    def _format_calc_result(r: dict) -> str:
        """One readable, auditable line (or block) for a derived-metric result."""
        ticker = r.get("ticker", "")
        metric = str(r.get("metric", "")).replace("_", " ")
        if r.get("series"):                       # trend
            pts = ", ".join(
                f"FY{s.get('fy', s.get('period'))}={s.get('value_str')}"
                for s in r["series"] if s.get("ok")
            )
            tail = f" — {r['summary']}" if r.get("summary") else ""
            return f"{ticker} {metric} trend: {pts}{tail}"
        # scalar (ratio / margin / growth / cagr)
        head = f"{ticker} {metric} = {r.get('value_str', '')}"
        return f"{head}\n{r.get('source', '')}"

    def _dedupe_evidence(self, items: list[dict], text_key: str) -> list[dict]:
        """Phase 10 near-duplicate filter: keep one representative per cluster of
        passages that say ~the same thing.

        Embeds each item's text with the shared (lru_cached) BGE model and
        greedily keeps an item only if its cosine similarity to every
        already-kept item is below `dedupe_threshold`. Order is preserved, so the
        highest-ranked passage in each cluster wins. This kills "five
        near-identical web snippets" before they reach the synthesizer.
        """
        if not self.dedupe or len(items) < 2:
            return items
        texts = [(it.get(text_key) or "").strip() for it in items]
        try:
            import numpy as np
            from finagent.vectorstore import get_embeddings
            vecs = np.asarray(get_embeddings(self.embedding_model).embed_documents(texts))
            # BGE embeddings are L2-normalized, so dot product is cosine sim.
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.clip(norms, 1e-9, None)
        except Exception:
            return items   # never let dedup break synthesis

        kept_idx: list[int] = []
        for i in range(len(items)):
            if not texts[i]:
                continue
            if all(float(vecs[i] @ vecs[j]) < self.dedupe_threshold for j in kept_idx):
                kept_idx.append(i)
        return [items[i] for i in kept_idx]

    @staticmethod
    def _format_edgar_result(r: dict) -> str:
        """A readable cross-document result: the matching companies + filing links."""
        lines = [
            f"  - {c.get('company', '?')}"
            f"{(' (' + c['ticker'] + ')') if c.get('ticker') else ''} — "
            f"{c.get('form', '')} {c.get('date', '')}  {c.get('url', '')}"
            for c in r.get("companies", [])
        ]
        total = r.get("total")
        head = (f"EDGAR full-text search {r.get('query','')} "
                f"({total:,} matching filings; showing {len(lines)} companies):"
                if isinstance(total, int) else
                f"EDGAR full-text search {r.get('query','')}:")
        return head + "\n" + "\n".join(lines)

    @staticmethod
    def _build_evidence_block(state: AgentState) -> str:
        parts: list[str] = []
        # XBRL facts are the strongest grounding for any numeric claim.
        for f in state.get("xbrl_facts", []) or []:
            parts.append(
                f"[XBRL] {f.get('entity', f.get('ticker',''))} {f.get('concept','')} "
                f"{f.get('period_label','FY'+str(f.get('fy','?')))} = {f.get('value_str','')} ({f.get('value')})\n"
                f"{f.get('source','')}"
            )
        # Derived metrics computed from those exact XBRL inputs.
        for r in state.get("calc_results", []) or []:
            parts.append(f"[Calc] {AgenticRAGv4._format_calc_result(r)}")
        for c in state.get("retrieved_chunks", []):
            parts.append(f"[Text] {c.get('source', '')}\n{c.get('text', '')[:1500]}")
        for t in state.get("table_results", []):
            if t.get("error") or not t.get("answer"):
                continue
            srcs = ", ".join(
                f"{tu.get('title', '?')} ({tu.get('company', '?')} {tu.get('year', '?')})"
                for tu in t.get("tables_used", [])[:3]
            )
            parts.append(f"[Table] {srcs}\nComputed: {t.get('answer', '')[:600]}\n"
                         f"Code: {t.get('code', '')[:400]}")
        for h in state.get("web_results", []):
            parts.append(f"[Web/{h.get('source', '')}] {h.get('title', '')}\n"
                         f"{(h.get('content') or '')[:600]}")
        # Live market data (yfinance) is structured but legitimate evidence
        # for any numeric claim about prices, ranges, returns.
        for m in state.get("market_data", []) or []:
            if not m.get("ok"):
                continue
            parts.append(f"[Market/{m.get('tool')}] {str(m.get('data',''))[:1200]}")
        return ("\n\n".join(parts))[:8000] or "(no evidence)"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the full agentic RAG (v4).")
    p.add_argument("--collection", default="us_filings")
    p.add_argument("--chroma-dir", default="data/chroma")
    p.add_argument("--market", choices=["india", "us"], default="us")
    p.add_argument("--table-collection", default="tables")
    p.add_argument("--news-collection", default="news")
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
    p.add_argument("--bm25-top-k", type=int, default=10)
    p.add_argument("--dense-top-k", type=int, default=10)
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

    agent = AgenticRAGv4(
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
        verifier_model=args.verifier_model,
        reranker_model=args.reranker_model,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        final_top_k=args.final_top_k,
        table_collection=args.table_collection,
        table_top_k=args.table_top_k,
        news_collection=args.news_collection,
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

    state = agent.run(args.question)
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
