"""Live-data lanes: yfinance market data, Tavily web search, EDGAR full-text search.

Split out of `finagent.graph.agent` (methods are unchanged); mixed into
`AgenticRAGv4` ahead of `AgenticRAGv3` in the MRO.
"""

from __future__ import annotations

import re

from finagent.graph.state import AgentState, MarketIntent, EdgarQuery
from finagent.graph.full import _MARKET_MARKERS
from finagent.tools.market import call_tool as call_market_tool

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
".NASDAQ". Common aliases (AAPL,
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


class ExternalNodes:
    """Live-data lanes: yfinance market data, Tavily web search, EDGAR full-text search."""

    # US exchange suffixes the LLM sometimes wrongly appends; any OTHER dotted
    # suffix (.NS, .L, .TO, .HK, …) is a deliberate foreign listing we keep.
    _US_SUFFIX_RE = re.compile(r"\.(NASDAQ|NYSE|NYS|NMS|NASD|NAS|OQ|N|O|A|P|Z|BATS|ARCA)$", re.I)

    def _get_market_planner_llm(self):
        """Same LLM as the router/planner tier — structured output, fast."""
        if "market_planner" not in self._llms:
            from finagent.llm import build_llm

            self._llms["market_planner"] = build_llm(
                self.provider, self.planner_model, self.api_key, temperature=0.0,
            )
        return self._llms["market_planner"]

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
            # An in-corpus company with chunks in hand is answered from its
            # filing — never escalate to the web (generic IR/marketing pages
            # bury the real evidence and tank faithfulness, and the Tavily call
            # is wasted cost). Escalate only when retrieval is genuinely empty,
            # or the chunks are off-entity noise (company NOT in the corpus).
            in_corpus = state.get("company_in_corpus")
            retrieval_was_poor = (not chunks) or (
                avg_grade is not None and avg_grade < 2.0 and not in_corpus
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

