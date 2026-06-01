"""
final_rag.py  ·  src/graph/final_rag.py

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

from src.graph.full_rag import (
    AgenticRAGv3,
    SYNTH_V3_SYSTEM,
    append_comparison_row,  # noqa: F401  re-export
)
from src.graph.market_tools import call_tool as call_market_tool
from src.graph.state import AgentState, MarketIntent, NumericVerification
from src.graph.translate import detect_language, language_name, translate_text
from src.graph.web_search import WebSearcher


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

Pick the minimum set of calls that fully answers the question. If the user
asks for a chart, history, "show me", "1-year", "5-year", "ytd", etc. → use
get_history. For "what is X trading at" → get_quote. For "compare X and Y" →
compare. Use multiple calls when one isn't enough (e.g. quote + history).
Return an empty list if the question genuinely isn't about market data.
"""

MARKET_PLANNER_PROMPT = """\
Question: {question}

Sub-queries routed to `market`:
{market_subs}

Return a list of MarketToolCall objects.
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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.news_collection = news_collection
        self.web_top_k = web_top_k
        # Translation is sensitive to model quality (especially Indian languages
        # with their digit grouping and proper-noun handling). Default to the
        # strong tier; override via translator_model for cheap-tier runs.
        self.translator_model = translator_model or self.synth_model
        self.verifier_model = verifier_model or self.critic_model
        self.min_verify_score = min_verify_score
        self._web: Optional[WebSearcher] = None

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

    def _get_translator_llm(self):
        if "translator" not in self._llms:
            from src.llm import build_llm

            self._llms["translator"] = build_llm(
                self.provider, self.translator_model, self.api_key, temperature=0.0
            )
        return self._llms["translator"]

    def _get_verifier_llm(self):
        if "verifier" not in self._llms:
            from src.llm import build_llm

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
            from src.llm import build_llm

            self._llms["market_planner"] = build_llm(
                self.provider, self.planner_model, self.api_key, temperature=0.0,
            )
        return self._llms["market_planner"]

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

        sub_queries = state.get("sub_queries") or [state["question"]]
        routes = state.get("query_routes") or ["narrative"] * len(sub_queries)
        market_subs = [s for s, r in zip(sub_queries, routes) if r == "market"]
        if not market_subs:
            return {"market_data": [], "charts": []}

        market_q = " | ".join(market_subs) if len(market_subs) > 1 else market_subs[0]
        planner = self._get_market_planner_llm().with_structured_output(MarketIntent)
        try:
            intent: MarketIntent = planner.invoke([
                SystemMessage(content=MARKET_PLANNER_SYSTEM),
                HumanMessage(content=MARKET_PLANNER_PROMPT.format(
                    question=state["question"],
                    market_subs="\n".join(f"- {s}" for s in market_subs),
                )),
            ])
            calls = intent.calls[:6]
        except Exception as e:
            self._log(state, f"market planner failed ({e}); skipping market lane")
            return {"market_data": [], "charts": []}

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

        # Escalation: no explicit external sub-queries, but text retrieval
        # didn't find usable evidence → fall back to the original question.
        if not external_subs:
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

    def critic_node(self, state: AgentState) -> dict:
        """v3 critic with web-search hits AND market-data tool results added to
        the evidence pool. Without this, a claim grounded in a Tavily hit or
        a yfinance quote looks "unsupported" to the critic (which only sees
        text + table chunks by default), triggers a retry, and ends up refused
        even though the verifier accepts it.
        """
        pseudo_chunks: list[dict] = []

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

        evidence_items: list[str] = []
        idx = 1

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
                evidence_items.append(
                    f"[{idx}] LIVE MARKET (yfinance · get_history) — "
                    f"{s.get('symbol','?')} {s.get('period','?')} {s.get('interval','?')}\n"
                    f"Range {s.get('start','?')} → {s.get('end','?')}; "
                    f"first_close={s.get('first_close')}, last_close={s.get('last_close')}, "
                    f"high={s.get('high')}, low={s.get('low')}, "
                    f"pct_change={s.get('pct_change')}%."
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

        evidence_block = (
            "\n\n".join(evidence_items) if evidence_items else "(no evidence retrieved)"
        )
        sub_queries = "\n".join(f"- {q}" for q in state.get("sub_queries", []))

        prompt = f"""Question: {state['question']}

Sub-queries researched:
{sub_queries}

Numbered evidence (cite with `[N]`):
{evidence_block}

---
Write your answer now in well-structured markdown with [N] citations after
every factual claim. If the evidence above does contain usable material —
including web / news items — USE IT; don't fall back to "no information"
unless every single item is irrelevant."""

        llm = self._get_llm("synth")
        response = llm.invoke([
            SystemMessage(content=SYNTH_V3_SYSTEM),
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
        """Check every numeric claim in the draft against the evidence."""
        answer = state.get("draft_answer", "")
        if not self._has_numbers(answer):
            return {"numeric_verification": {"claims": [], "unverified": [], "score": 1.0}}

        evidence = self._build_evidence_block(state)
        llm = self._get_verifier_llm().with_structured_output(NumericVerification)

        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            report: NumericVerification = llm.invoke([
                SystemMessage(content=NUM_VERIFY_SYSTEM),
                HumanMessage(content=NUM_VERIFY_PROMPT.format(answer=answer, evidence=evidence)),
            ])
            claims = [c.model_dump() for c in report.claims]
        except Exception as e:
            self._log(state, f"verifier failed ({e})")
            return {"numeric_verification": {"claims": [], "unverified": [], "score": None}}

        if not claims:
            return {"numeric_verification": {"claims": [], "unverified": [], "score": 1.0}}

        unverified = [c for c in claims if not c.get("matched")]
        score = (len(claims) - len(unverified)) / len(claims)
        # Log unverified claims so they show up in the run's `errors` list.
        for u in unverified:
            self._log(state, f"unverified figure: {u.get('number')} — {u.get('claim')[:120]}")
        return {
            "numeric_verification": {
                "claims": claims,
                "unverified": unverified,
                "score": round(score, 3),
            }
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

    def _verify_router(self, state: AgentState) -> str:
        """After numeric verification: refuse / retry retrieval / continue to translate-out.

        The **verifier** is the source of truth for refusal: it does a precise
        per-number match against ALL evidence (text + tables + web). The
        critic is a cheaper retry trigger; if it disagrees with a passing
        verifier (e.g. it didn't see the web hits) we trust the verifier.
        """
        nv = state.get("numeric_verification") or {}
        score = nv.get("score")
        claims = nv.get("claims") if isinstance(nv, dict) else None

        # If critic asked for a retry and we still have budget, follow it.
        if state.get("needs_retry"):
            return "retrieve"

        out_of_retries = state.get("critic_iterations", 0) >= self.max_critic_retries
        verify_failed = (score is not None and score < self.min_verify_score)

        # We only refuse when (a) we've used up retries, (b) the verifier
        # actually saw numeric claims AND rejected enough of them. A passing
        # verifier overrides a complaining critic.
        if out_of_retries and verify_failed and claims:
            return "refuse"
        return "translate_out"

    # ------------------------------------------------------------------ #
    # Graph
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(AgentState)
        g.add_node("detect_lang", self.detect_language_node)
        g.add_node("translate_in", self.translate_in_node)
        g.add_node("planner", self.planner_node)
        g.add_node("router", self.router_node)
        g.add_node("retrieve", self.hybrid_retrieve_node)
        g.add_node("grader", self.grader_node)
        g.add_node("rewrite", self.rewrite_node)
        g.add_node("table_agent", self.table_agent_node)
        g.add_node("market_data", self.market_data_node)
        g.add_node("web_search", self.web_search_node)
        g.add_node("synthesize", self.synthesize_node)
        g.add_node("critic", self.critic_node)
        g.add_node("verify_numbers", self.verify_numbers_node)
        g.add_node("refuse", self.refuse_node)
        g.add_node("translate_out", self.translate_out_node)

        g.add_edge(START, "detect_lang")
        g.add_edge("detect_lang", "translate_in")
        g.add_edge("translate_in", "planner")
        g.add_edge("planner", "router")
        g.add_edge("router", "retrieve")
        g.add_edge("retrieve", "grader")
        g.add_conditional_edges(
            "grader", self._grade_router,
            {"rewrite": "rewrite", "table_agent": "table_agent"},
        )
        g.add_edge("rewrite", "retrieve")
        g.add_edge("table_agent", "market_data")
        g.add_edge("market_data", "web_search")
        g.add_edge("web_search", "synthesize")
        g.add_edge("synthesize", "critic")
        g.add_edge("critic", "verify_numbers")
        g.add_conditional_edges(
            "verify_numbers", self._verify_router,
            {"retrieve": "retrieve", "refuse": "refuse", "translate_out": "translate_out"},
        )
        g.add_edge("refuse", "translate_out")
        g.add_edge("translate_out", END)
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

    @staticmethod
    def _build_evidence_block(state: AgentState) -> str:
        parts: list[str] = []
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
