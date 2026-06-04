"""
agent.py  ·  finagent/graph/agent.py

Full agentic RAG (v4). Adds to v3:

  * **Bilingual** — language detection at the entrance; if the user's question
    is not English, translate to English for retrieval, translate the final
    answer back at the exit.
  * **Web search** — `web_search_node` covers questions the corpus can't (post
    cut-off events, latest news). Tavily if `TAVILY_API_KEY` is set, otherwise
    the local `news` Chroma collection.
  * **Numeric verification** — after the critic, every numeric claim in the
    draft is matched against the supplied evidence (text excerpts, tables,
    web results). Claims with no match are recorded in
    `state["numeric_verification"]["unverified"]`.
  * **Refusal path** — if the critic + numeric verifier both fail at the cap,
    the final answer is replaced with a clear "I don't have enough information
    to answer this from the available filings."  rather than hallucinating.

Graph:

    START → detect_lang → translate_in → planner → router → retrieve
                                                      ↓ (numeric)        ↓ (external)
                                                  table_agent         web_search
                                                      └──────┬───────────┘
                                                             ▼
                                                  grader → {rewrite|continue}
                                                             ▼
                                                       synthesize → critic → verify_numbers
                                                                                   │
                                              {retrieve | refuse | translate_out}─┘
                                                             ▼
                                                       translate_out → END

`translate_out` is a no-op when language == "en".
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
    AgentState, MarketIntent, NumericVerification, XBRLQuery, CalcQuery,
    CorpusGateQuery, EdgarQuery,
)
from finagent.graph.translate import detect_language, language_name, translate_text
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

Use Yahoo ticker format: AAPL, MSFT (US listings have no suffix); for India use
.NS (NSE) like RELIANCE.NS or TCS.NS. Common aliases (TCS, INFY, RELIANCE,
HDFC, etc.) are auto-normalised — pass either form.

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
profit, R&D expense, diluted EPS, cash, long-term debt, …) for ONE company and
period — if so set answerable=true and fill ticker, concept (plain words), and
period (e.g. 'FY2022'). Set answerable=false for derived metrics (margins,
growth, ratios, CAGR), multi-company comparisons, or narrative questions — those
are handled elsewhere. Use the conversation context to resolve a follow-up's
company/period if the sub-query omits them.
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
(gross/operating/net), a ratio (current ratio, debt-to-equity, ROE, ROA, asset
turnover, interest coverage), an intensity ratio (rd_to_revenue = R&D as % of
revenue, sga_to_revenue, capex_to_revenue), period-over-period GROWTH, a CAGR,
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
    """v3 + language detection + web search + numeric verification + refusal."""

    def __init__(
        self,
        *args,
        news_collection: str = "news",
        # ~10 hits gives the synthesiser enough material for multi-faceted
        # questions ("performance over the last year"); a single article rarely
        # covers the full ground. Tavily allows up to 20 per call.
        web_top_k: int = 10,
        translator_model: Optional[str] = None,
        verifier_model: Optional[str] = None,
        min_verify_score: float = 0.5,
        dispatch: bool = True,
        analyst_voice: bool = True,
        dedupe: bool = True,
        dedupe_threshold: float = 0.93,
        strict_numeric: bool = True,
        persist_fetch: bool = True,
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
        # Dynamic fetch: persist the fetched filing into the on-disk index (True,
        # local) or use it ephemerally in memory for this session (False, cloud).
        self.persist_fetch = persist_fetch
        # Translation is sensitive to model quality (especially Indian languages
        # with their digit grouping and proper-noun handling). Default to the
        # strong tier; override via translator_model for cheap-tier runs.
        self.translator_model = translator_model or self.synth_model
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

    def _get_translator_llm(self):
        if "translator" not in self._llms:
            from finagent.llm import build_llm

            self._llms["translator"] = build_llm(
                self.provider, self.translator_model, self.api_key, temperature=0.0
            )
        return self._llms["translator"]

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

    def detect_language_node(self, state: AgentState) -> dict:
        """Detect the question's language; preserve the original for the exit."""
        question = state["question"]
        lang = detect_language(question)
        return {"language": lang, "query_original": question}

    def translate_in_node(self, state: AgentState) -> dict:
        """If the question isn't English, translate to English for retrieval."""
        lang = state.get("language", "en")
        if lang == "en":
            return {}
        translated = translate_text(
            state["question"], target_code="en",
            llm=self._get_translator_llm(), source_code=lang,
        )
        return {"question": translated}

    def _get_market_planner_llm(self):
        """Same LLM as the router/planner tier — structured output, fast."""
        if "market_planner" not in self._llms:
            from finagent.llm import build_llm

            self._llms["market_planner"] = build_llm(
                self.provider, self.planner_model, self.api_key, temperature=0.0,
            )
        return self._llms["market_planner"]

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

        question = state.get("query_original") or state["question"]

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

        try:
            gate = self.fetcher.gate(company)
        except Exception as e:
            self._log(state, f"corpus gate failed for {company!r}: {e}")
            return {"fetch_status": {}}

        if gate["decision"] != "fetch":
            # already_indexed → retrieval handles it; not_us_listed → web branch.
            return {"fetch_status": gate}

        self._log(state, f"dynamic fetch: pulling latest filing for {gate['ticker']}…")

        # Ephemeral path (cloud / per-session): parse + chunk the filing in
        # memory and rank it against the question later — NOTHING is written to
        # the persistent index, so the corpus never grows and there's no
        # scale-to-zero persistence problem.
        if not self.persist_fetch:
            try:
                res = self.fetcher.fetch_chunks(
                    gate["ticker"], company=gate.get("company") or "")
            except Exception as e:
                self._log(state, f"ephemeral fetch failed for {gate['ticker']}: {e}")
                return {"fetch_status": {**gate, "status": "error", "error": str(e)}}
            chunks = res.get("chunks", []) if res.get("ok") else []
            if chunks:
                self._log(state, f"fetched {len(chunks)} in-memory chunks for {gate['ticker']} "
                                 f"({res.get('form')})")
            return {
                "fetched_chunks": chunks,
                "fetch_status": {**gate, "status": "fetched" if chunks else "error",
                                 "ephemeral": True, "form": res.get("form"),
                                 "chunks_fetched": len(chunks)},
            }

        # Persistent path (local): fetch + ingest into the live collection.
        try:
            res = self.fetcher.fetch_and_ingest(
                gate["ticker"], company=gate.get("company") or "")
        except Exception as e:
            self._log(state, f"dynamic fetch failed for {gate['ticker']}: {e}")
            return {"fetch_status": {**gate, "status": "error", "error": str(e)}}

        if res.get("ok"):
            self._hybrids = None       # invalidate cached BM25/dense over old corpus
            self._log(state, f"ingested {res['chunks_added']} chunks for {gate['ticker']}")
        return {"fetch_status": {**gate, "status": "fetched" if res.get("ok") else "error",
                                 **res}}

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

        extractor = self._get_router_llm().with_structured_output(XBRLQuery)
        facts: list[dict] = []
        for sub_q in numeric_subs:
            try:
                q: XBRLQuery = extractor.invoke([
                    SystemMessage(content=XBRL_EXTRACT_SYSTEM),
                    HumanMessage(content=history_block + XBRL_EXTRACT_PROMPT.format(sub_query=sub_q)),
                ])
            except Exception as e:
                self._log(state, f"xbrl extract failed for {sub_q!r}: {e}")
                continue
            if not q.answerable or not (q.ticker and q.concept):
                continue
            try:
                res = self.xbrl.run(ticker=q.ticker, concept=q.concept,
                                    period=q.period or None)
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

        extractor = self._get_router_llm().with_structured_output(CalcQuery)
        results: list[dict] = []
        for sub_q in numeric_subs:
            try:
                q: CalcQuery = extractor.invoke([
                    SystemMessage(content=CALC_EXTRACT_SYSTEM),
                    HumanMessage(content=history_block + CALC_EXTRACT_PROMPT.format(sub_query=sub_q)),
                ])
            except Exception as e:
                self._log(state, f"calc extract failed for {sub_q!r}: {e}")
                continue
            if not q.is_derived or not (q.ticker and q.metric):
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

        # Corrective dispatch (CRAG): no explicit external sub-queries, but a
        # NARRATIVE retrieval was attempted and came back empty/poorly graded →
        # escalate to web. We gate on a narrative route having run: under the
        # Phase 7 dispatcher a purely numeric/cross-doc question skips retrieval
        # by design, so empty `retrieved_chunks` there is expected, not a failure
        # to correct — we must NOT escalate those to web.
        narrative_attempted = (not routes) or any(r == "narrative" for r in routes)
        if not external_subs and narrative_attempted:
            chunks = state.get("retrieved_chunks") or []
            avg_grade = state.get("avg_grade")
            retrieval_was_poor = (not chunks) or (
                avg_grade is not None and avg_grade < 2.0
            )
            if retrieval_was_poor:
                fallback = state.get("query_original") or state["question"]
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
                "text": (f"{f.get('concept','')} FY{f.get('fy','?')} = "
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
            return super().critic_node(state)

        proxy = dict(state)
        proxy["retrieved_chunks"] = list(state.get("retrieved_chunks", [])) + pseudo_chunks
        return super().critic_node(proxy)

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
                f"FY{f.get('fy','?')}: {f.get('value_str','')}\n"
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

        prompt = f"""{history_block}Question: {state['question']}

Sub-queries researched:
{sub_queries}

Numbered evidence (cite with `[N]`):
{evidence_block}

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

        evidence = self._build_evidence_block(state)
        llm = self._get_verifier_llm().with_structured_output(NumericVerification)
        claims: list[dict] = []
        try:
            report: NumericVerification = llm.invoke([
                SystemMessage(content=NUM_VERIFY_SYSTEM),
                HumanMessage(content=NUM_VERIFY_PROMPT.format(answer=answer, evidence=evidence)),
            ])
            claims = [c.model_dump() for c in report.claims]
        except Exception as e:
            self._log(state, f"verifier failed ({e})")

        # Legacy LLM-only path (A/B baseline).
        if not self.strict_numeric:
            llm_unverified = [c for c in claims if not c.get("matched")]
            score = ((len(claims) - len(llm_unverified)) / len(claims)) if claims else 1.0
            return {"numeric_verification": {"claims": claims, "unverified": llm_unverified,
                                             "score": round(score, 3),
                                             "numbers_total": len(claims),
                                             "numbers_grounded": len(claims) - len(llm_unverified),
                                             "hallucination_rate": round(1 - score, 3)}, **vi}

        # Deterministic, exhaustive grounding.
        draft_nums = self._extract_numbers(answer)
        if not draft_nums:
            return {"numeric_verification": {**clean, "claims": claims}, **vi}

        evidence_mags = self._evidence_numbers(state)
        # LLM rescue: figures the verifier matched count as grounded too — pull
        # their magnitudes from the matched claims and add them to the evidence.
        for c in claims:
            if c.get("matched"):
                for n in self._extract_numbers(str(c.get("number", "")) + " " + str(c.get("evidence", ""))):
                    evidence_mags.extend(n["magnitudes"])

        ungrounded = [
            {"number": d["raw"], "claim": d["ctx"]}
            for d in draft_nums if not self._grounded(d["magnitudes"], evidence_mags)
        ]
        total = len(draft_nums)
        grounded = total - len(ungrounded)
        score = grounded / total if total else 1.0
        for u in ungrounded:
            self._log(state, f"ungrounded figure: {u['number']} — {u['claim'][:100]}")
        return {
            "numeric_verification": {
                "claims": claims,
                "unverified": ungrounded,          # refuse_node reads this key
                "score": round(score, 3),
                "numbers_total": total,
                "numbers_grounded": grounded,
                "hallucination_rate": round(len(ungrounded) / total, 3) if total else 0.0,
            },
            **vi,
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
        return {"final_answer": msg, "refused": True, "needs_retry": False}

    def translate_out_node(self, state: AgentState) -> dict:
        """Translate the final answer back to the user's language."""
        lang = state.get("language", "en")
        if lang == "en":
            return {}
        translated = translate_text(
            state.get("final_answer", ""), target_code=lang,
            llm=self._get_translator_llm(), source_code="en",
        )
        return {"final_answer": translated}

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
        routes = state.get("query_routes") or []
        has_narrative = (not routes) or any(r == "narrative" for r in routes)
        return "retrieval" if (not self.dispatch or has_narrative) else "tools"

    def _verify_router(self, state: AgentState) -> str:
        """After numeric verification: refuse / retry retrieval / continue to translate-out.

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

        # Phase 11: an ungrounded figure is a hallucination — every number must
        # trace to a source. Under strict numeric verification, re-route once to
        # try to ground it, then refuse rather than ship a fabricated figure.
        if self.strict_numeric:
            has_ungrounded = (nv.get("numbers_total", 0) > 0 and len(ungrounded) > 0)
            if state.get("needs_retry") and not out_of_retries:
                return "retrieve"
            if has_ungrounded and not out_of_retries:
                return "retrieve"          # re-route to re-ground the figure(s)
            if has_ungrounded and out_of_retries:
                return "refuse"            # "not in the filings" beats fabrication
            return "end"

        # Legacy LLM-only path.
        claims = nv.get("claims") if isinstance(nv, dict) else None
        if state.get("needs_retry"):
            return "retrieve"
        verify_failed = (score is not None and score < self.min_verify_score)
        if out_of_retries and verify_failed and claims:
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
        g.add_node("grader", self.grader_node)
        g.add_node("rewrite", self.rewrite_node)
        g.add_node("xbrl", self.xbrl_node)
        g.add_node("calculator", self.calculator_node)
        g.add_node("table_agent", self.table_agent_node)
        g.add_node("market_data", self.market_data_node)
        g.add_node("web_search", self.web_search_node)
        g.add_node("edgar_search", self.edgar_search_node)
        g.add_node("synthesize", self.synthesize_node)
        g.add_node("critic", self.critic_node)
        g.add_node("verify_numbers", self.verify_numbers_node)
        g.add_node("refuse", self.refuse_node)

        # English-only: no language detection / translation nodes.
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
        g.add_edge("xbrl", "calculator")
        g.add_edge("calculator", "table_agent")
        g.add_edge("table_agent", "market_data")
        g.add_edge("market_data", "web_search")
        # Phase 6: cross-document sub-queries fan out to EDGAR full-text search.
        g.add_edge("web_search", "edgar_search")
        g.add_edge("edgar_search", "synthesize")
        g.add_edge("synthesize", "critic")
        g.add_edge("critic", "verify_numbers")
        g.add_conditional_edges(
            "verify_numbers", self._verify_router,
            {"retrieve": "retrieve", "refuse": "refuse", "end": END},
        )
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
        # Remove things that look numeric but aren't financial claims.
        scrubbed = re.sub(r"\[[\d,\s]+\]", " ", text)          # [1], [1, 3]
        scrubbed = re.sub(r"\bFY\s?\d{2,4}\b", " ", scrubbed, flags=re.I)
        scrubbed = re.sub(r"\bQ[1-4]\b", " ", scrubbed, flags=re.I)
        scrubbed = re.sub(r"\b10-[KQ]\b|\b8-K\b", " ", scrubbed, flags=re.I)
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

    def _evidence_numbers(self, state: AgentState) -> list[float]:
        """Every numeric value present in the grounding evidence: XBRL facts and
        derived metrics (exact), plus numbers parsed from retrieved chunks,
        tables, web hits, and market data."""
        mags: list[float] = []
        # XBRL — exact filed values (the ground truth).
        for f in state.get("xbrl_facts", []) or []:
            v = f.get("value")
            if isinstance(v, (int, float)):
                mags.append(float(v))
        # Derived metrics — value plus its percent form (0.303 and 30.3).
        for r in state.get("calc_results", []) or []:
            for key in ("value",):
                v = r.get(key)
                if isinstance(v, (int, float)):
                    mags.extend([float(v), float(v) * 100.0])
            for s in r.get("series", []) or []:
                v = s.get("value")
                if isinstance(v, (int, float)):
                    mags.extend([float(v), float(v) * 100.0])
        # Free-text evidence — parse any numbers out of the prose.
        texts: list[str] = []
        texts += [c.get("text", "") for c in state.get("retrieved_chunks", []) or []]
        for t in state.get("table_results", []) or []:
            texts.append(str(t.get("answer", "")) + " " + str(t.get("stdout", "")))
        for h in state.get("web_results", []) or []:
            texts.append(h.get("content", "") or "")
        for mkt in state.get("market_data", []) or []:
            texts.append(str(mkt.get("data", "")))
        for txt in texts:
            for n in self._extract_numbers(txt):
                mags.extend(n["magnitudes"])
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
                f"FY{f.get('fy','?')} = {f.get('value_str','')} ({f.get('value')})\n"
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
    p.add_argument("--translator-model", default=None)
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
        translator_model=args.translator_model,
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
    print(f"Question (orig):  {state.get('query_original')}")
    print(f"Language:         {state.get('language')} ({language_name(state.get('language', 'en'))})")
    print(f"Sub-queries:      {state.get('sub_queries')}")
    print(f"Routes:           {state.get('query_routes')}")
    print(f"Grades:           {state.get('grades')} (avg {state.get('avg_grade')})")
    print(f"Rewrites:         {state.get('iteration_count', 0)}")
    print(f"Critic retries:   {state.get('critic_iterations', 0)}")
    nv = state.get("numeric_verification") or {}
    print(f"Numeric verify:   score={nv.get('score')}  unverified={len(nv.get('unverified', []))}")
    print(f"Refused:          {state.get('refused', False)}")
    print(f"Low confidence:   {state.get('low_confidence', False)}")
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
