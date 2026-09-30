"""Step 6: live data. Market prices (yfinance), web search (Tavily) and EDGAR
full-text search run side by side after the numeric steps."""

from __future__ import annotations

import re
from datetime import date, timedelta

from langchain_core.messages import HumanMessage, SystemMessage

from finagent.agent.prompts import (EDGAR_PROMPT, EDGAR_SYSTEM, MARKET_PROMPT,
                                    MARKET_SYSTEM, history_block)
from finagent.agent.state import AgentState, EdgarQuery, MarketIntent
from finagent.tools.market import call_tool as call_market_tool

# Words that mean "this is about the stock", used when the planner mislabels a
# chart or price question.
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

# Words that mean "this needs the web" even when no part was routed `external`.
# Kept news-specific: a bare "current" also appears in "current ratio".
_WEB_NEWS_MARKERS = (
    "news", "latest", "headline", "press release", "announcement",
    "outlook", "forecast", "guidance", "analyst", "expected to perform",
    "upcoming month", "coming month", "next quarter", "this week", "today's",
    "recently", "happening", "sentiment", "what's new",
    "acquisition", "acquisitions", "acquire", "acquired", "merger", "merged",
    "takeover", "divestiture", "divest", "spin-off", "spinoff", "joint venture",
    "partnership", "deal", "buyout",
    "8-k", "8k",                    # event filings are not in the index
)

# US exchange suffixes a model sometimes appends (".NASDAQ"). Any other suffix
# (.NS, .L, .TO) is a real foreign listing and is kept.
_US_SUFFIX_RE = re.compile(r"\.(NASDAQ|NYSE|NYS|NMS|NASD|NAS|OQ|N|O|A|P|Z|BATS|ARCA)$", re.I)


# --------------------------------------------------------------------------- #
# market_data
# --------------------------------------------------------------------------- #

def _resolve_ticker(agent, company: str, symbol: str) -> str:
    """The real ticker. The company NAME is resolved against SEC data, which is
    more reliable than the model's symbol guess."""
    symbol = (symbol or "").strip()
    if "." in symbol and not _US_SUFFIX_RE.search(symbol):
        return symbol.upper()                       # a foreign listing: keep as is
    resolver = agent.xbrl.resolver
    if (company or "").strip():
        try:
            if (ticker := resolver.resolve(company.strip()).get("ticker")):
                return ticker
        except Exception:
            pass
    raw = _US_SUFFIX_RE.sub("", symbol).upper()
    if raw:
        try:
            r = resolver.resolve(raw)
            if r.get("match") == "ticker" and r.get("ticker"):   # exact match only
                return r["ticker"]
        except Exception:
            pass
    return raw


def market_data(agent, state: AgentState) -> dict:
    """Pick one yfinance call for a stock question and run it. `get_history`
    also produces the chart the frontend draws."""
    question = state["question"]
    sub_queries = state.get("sub_queries") or []
    text = (question + " " + " ".join(sub_queries)).lower()
    if "market" not in (state.get("query_routes") or []) and not any(
            m in text for m in _MARKET_MARKERS):
        return {"market_data": [], "charts": []}

    try:
        intent: MarketIntent = agent.llm("planner").with_structured_output(MarketIntent).invoke([
            SystemMessage(content=MARKET_SYSTEM),
            HumanMessage(content=history_block(state.get("chat_history"))
                         + MARKET_PROMPT.format(question=question)),
        ])
    except Exception as e:
        agent.llm_failed(state, "Market-data planning", e)
        return {"market_data": [], "charts": []}

    if intent.tool == "compare":
        intent.symbols = [s for s in (_resolve_ticker(agent, "", s)
                                      for s in intent.symbols or []) if s]
    else:
        intent.symbol = _resolve_ticker(agent, intent.company, intent.symbol)
    if intent.tool == "none" or not (intent.symbol or intent.symbols):
        return {"market_data": [], "charts": []}

    calls = [intent]
    # A volume or performance question needs the price history even if the
    # model picked news or a quote.
    sym = intent.symbol or (intent.symbols[0] if intent.symbols else "")
    wants_history = any(m in text for m in (
        "volume", "performance", "how has", "how is it doing", "trend", "perform"))
    if wants_history and sym and intent.tool not in ("get_history", "compare"):
        calls.append(MarketIntent(tool="get_history", symbol=sym, period="1y", interval="1d"))

    results: list[dict] = []
    charts: list[dict] = []
    for c in calls:
        kwargs: dict = {"symbols": c.symbols} if c.tool == "compare" else {"symbol": c.symbol}
        if c.tool == "get_history":
            kwargs.update(period=c.period, interval=c.interval)
        if c.tool == "get_news":
            kwargs["limit"] = 5
        res = call_market_tool(c.tool, **kwargs)
        results.append({"tool": c.tool, "args": kwargs, "ok": res.get("ok", False),
                        "data": res.get("data"), "error": res.get("error"),
                        "sub_query": question})
        if c.tool == "get_history" and res.get("ok") and res["data"].get("chart"):
            charts.append(res["data"]["chart"])
    return {"market_data": results, "charts": charts}


# --------------------------------------------------------------------------- #
# web_search
# --------------------------------------------------------------------------- #

def web_search(agent, state: AgentState) -> dict:
    """Search the web when a part was routed `external`, when the question asks
    for news, when the critic sent us here, or when there is no filing evidence."""
    question = state["question"]
    sub_queries = state.get("sub_queries") or [question]
    routes = state.get("query_routes") or ["narrative"] * len(sub_queries)
    queries = [s for s, r in zip(sub_queries, routes) if r == "external"]
    for s in [*sub_queries, question]:
        if s not in queries and any(m in (s or "").lower() for m in _WEB_NEWS_MARKERS):
            queries.append(s)
    if state.get("web_fallback_pending") and question not in queries:
        queries.append(question)

    # Last resort: the filings were tried and returned nothing at all. Any chunk
    # we do have is the right company's filing (fetch_filing ran first), and
    # adding web pages on top of it buries real evidence, so this fires only
    # on zero chunks.
    numbers_found = bool(state.get("xbrl_facts") or state.get("calc_results"))
    filings_tried = ("narrative" in routes) or ("numeric" in routes and not numbers_found)
    if not queries and filings_tried and not state.get("retrieved_chunks"):
        agent.log(state, "no filing evidence; searching the web for the question")
        queries = [question]
    if not queries:
        return {"web_results": []}

    hits: list[dict] = []
    for q in queries:
        try:
            hits.extend({"sub_query": q, **h} for h in agent.web.search(q))
        except Exception as e:
            agent.log(state, f"web search failed for {q!r}: {e}")

    # Keep one search's worth in total, taking turns across queries (trusted
    # sites first), so many queries cannot flood the writer's prompt.
    if len(hits) > agent.web_top_k:
        by_query: dict[str, list[dict]] = {}
        for h in hits:
            by_query.setdefault(h["sub_query"], []).append(h)
        for lst in by_query.values():
            lst.sort(key=lambda h: h.get("tier") != "trusted")
        hits = []
        while len(hits) < agent.web_top_k and any(by_query.values()):
            for lst in by_query.values():
                if lst and len(hits) < agent.web_top_k:
                    hits.append(lst.pop(0))
    return {"web_results": hits}


# --------------------------------------------------------------------------- #
# edgar_search
# --------------------------------------------------------------------------- #

def edgar_search(agent, state: AgentState) -> dict:
    """For "which companies disclosed X" parts: full-text search across every
    company's filings on EDGAR."""
    sub_queries = state.get("sub_queries") or [state["question"]]
    routes = state.get("query_routes") or ["narrative"] * len(sub_queries)
    subs = [s for s, r in zip(sub_queries, routes) if r == "cross_document"]
    if not subs:
        return {"edgar_results": []}

    today = date.today()
    start = (today - timedelta(days=3 * 365)).isoformat()   # recent filers, not decade-old hits
    results: list[dict] = []
    for sub_q in subs:
        try:
            eq: EdgarQuery = agent.llm("extractor").with_structured_output(EdgarQuery).invoke([
                SystemMessage(content=EDGAR_SYSTEM),
                HumanMessage(content=EDGAR_PROMPT.format(sub_query=sub_q)),
            ])
            phrase = (eq.phrase or "").strip()
        except Exception as e:
            agent.llm_failed(state, "EDGAR search", e)
            continue
        if not phrase:
            continue
        # Exact phrase first; if it matches nothing, all the words unquoted.
        quoted = f'"{phrase}"' if " " in phrase and '"' not in phrase else phrase
        res = None
        for q in ([quoted, phrase] if quoted != phrase else [phrase]):
            try:
                r = agent.edgar.run(q, forms=eq.forms or "10-K", n=10,
                                    startdt=start, enddt=today.isoformat())
            except Exception as e:
                agent.log(state, f"edgar search failed for {q!r}: {e}")
                continue
            if r.get("ok") and r.get("companies"):
                res = r
                break
        if res is None:
            agent.log(state, f"edgar: no hits for {phrase!r}")
            continue
        res["sub_query"] = sub_q
        results.append(res)
    return {"edgar_results": results}
