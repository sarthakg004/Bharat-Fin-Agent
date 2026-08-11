"""Evidence assembly, analyst-voice synthesis, the claim-checking critic, and
the refusal it can trigger.

Split out of `finagent.graph.agent`; mixed into `AgenticRAGv4` ahead of
`AgenticRAGv3` in the MRO.
"""

from __future__ import annotations

import re

from finagent.graph.state import AgentState
from finagent.graph.full import SYNTH_V3_SYSTEM
from finagent.prompts.critic import REFUSAL_TEMPLATE

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

Do NOT invent provenance
------------------------
State ONLY what the numbered evidence states. Do NOT add provenance metadata
that is not present in the evidence item you are citing — specifically:
- Never write XBRL tags or us-gaap concept names (e.g. "us-gaap:InventoryNet")
  unless that exact tag appears in the evidence.
- Never add filing identifiers, form types, accession numbers, or filing dates
  ("as filed in the FY2019 10-K", "per the 2018-12-31 10-K") unless the evidence
  item literally contains them.
- Never add methodology or sourcing notes ("sourced from the XBRL filing",
  "as reported in the cash-flow statement", "balance-sheet line item") that the
  evidence does not itself assert.
The figure plus its `[N]` citation is the complete, faithful answer — the [N]
already points the reader to the exact source. Tacking on unverifiable "where
this came from" phrasing reads authoritative but is ungrounded, and a fact-checker
scores it as unsupported even when the number is correct. When in doubt, say
less: cite the number and stop.

Facts vs inference
------------------
Never present your own inference as the filing's claim. If management does not
explicitly attribute a change to a driver, do NOT write that the filing
"implies" or "suggests" it — state what the figures show ("S&M grew 4% [2]
while revenue grew 12% [1], so S&M leveraged as a % of revenue") and note
plainly when the filing offers no attribution. A computed comparison of cited
figures is fine; an invented causal story is not.

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
When the question names NO fiscal period ("latest", "year-over-year", or
nothing at all), answer from the MOST RECENT fiscal period in the evidence and
name that period explicitly — never centre the answer on an older year when a
newer one is available in the evidence.

A QUARTER IS NOT A FISCAL YEAR. When the question asks about a fiscal YEAR
("in the latest fiscal year", "FY2025", "full-year") and the evidence carries
only an interim figure (a quarter, a half, a trailing-twelve-month or a
guidance number), you may NOT present that figure as the answer. Report it as
what it is — the period goes on the figure itself, "$1.7 billion (Q2 FY2026)" —
and say in the same breath that the full-year figure is not in the evidence.
Never put a quarterly number in a column or sentence the question framed as
annual, and never sum or annualise quarters yourself.

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




class SynthesisNodes:
    """Evidence assembly, analyst-voice synthesis, and the claim-checking critic."""

    # Per-source baseline confidence for a normalised evidence item. Ordered by
    # how authoritative the source is for a financial claim: exact filed XBRL is
    # ground truth; deterministic math over it is nearly as good; filing text and
    # tables are strong; live market data is fresh-but-point-in-time; EDGAR FTS
    # locates filers; web is the weakest.
    _EVIDENCE_BASE_CONF = {
        "xbrl": 0.98, "calc": 0.95, "table": 0.85, "filing": 0.75,
        "market": 0.90, "edgar": 0.70, "web": 0.55,
    }

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

        # Filing text excerpts — the reranker already ordered them, so they all
        # carry the same source-level confidence.
        for c in state.get("retrieved_chunks", []) or []:
            add("filing", c.get("text", "")[:300],
                source=c.get("source", ""),
                citation=c.get("source", ""),
                sub_query=c.get("sub_query", ""))

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
        is available to the critic, the audit
        trail, and cross-source validation — without disturbing the
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
        even though the evidence supports it.
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

    def _web_fallback_signal(self, state: AgentState) -> dict:
        """One-shot escalation signal: the draft admits the gathered evidence
        can't answer, and the web lane hasn't run yet — so route to web search
        and re-synthesize instead of shipping an "I couldn't find it" answer
        while a whole tool lane sits unused. `used` latches so a question the
        web genuinely can't answer either doesn't loop.

        This spends the SAME recovery budget as a re-draft or a re-retrieve. It
        used to be free, which is how a question could run three synthesize
        passes while `_retrieval_status` reported one. The counter is derived
        from `state`, not from the dict this update is merged into, so it lands
        on the same value whichever recovery the router ends up choosing.
        """
        draft = state.get("draft_answer", "") or ""
        if (not state.get("web_results")
                and not state.get("web_fallback_used")
                and state.get("critic_iterations", 0) < self.max_critic_retries
                and self._INSUFFICIENT_RE.search(draft)):
            self._log(state, "draft admits the evidence can't answer; "
                             "escalating to web search")
            return {"web_fallback_pending": True, "web_fallback_used": True,
                    "critic_iterations": state.get("critic_iterations", 0) + 1}
        return {"web_fallback_pending": False}

    def refuse_node(self, state: AgentState) -> dict:
        """Replace the final answer with an explicit refusal.

        Reached from `_critic_router` when the critic could support less than
        `REFUSE_BELOW_SUPPORT` of the draft's claims and every retry is spent —
        shipping a draft that is mostly unsupported is worse than declining.
        """
        claims = state.get("critic_feedback") or []
        detail = ""
        if claims:
            detail = " (unsupported: " + "; ".join(c[:80] for c in claims[:2]) + ")"
        web_clause = "" if state.get("web_results") else " or recent web sources"
        return {"final_answer": REFUSAL_TEMPLATE.format(
                    web_clause=web_clause, detail=detail),
                "refused": True, "needs_retry": False, "status": "refused"}

    def synthesize_node(self, state: AgentState) -> dict:
        """v3 synth + web results, presented as one unified numbered evidence list.

        Numbering order matches what the frontend sees (text → web → table),
        so a `[3]` in the answer points at the third card in the sidebar.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        chunks = state.get("retrieved_chunks", []) or []
        web_res = state.get("web_results", []) or []
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

        # 3. EDGAR full-text cross-document results — the set of companies whose
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

        # Extractive numeric mode (#4): when EVERY sub-query is a numeric lookup
        # (a single figure or ratio — no narrative component), the answer should
        # be the figure itself, not a paragraph wrapped around it. A terse,
        # extractive answer can't drift into unsupported provenance claims, which
        # is exactly what depressed faithfulness on the numeric set.
        routes = state.get("query_routes") or []
        numeric_only = bool(routes) and all(r == "numeric" for r in routes)
        extractive_block = ""
        if numeric_only:
            extractive_block = (
                "\nThis is a NUMERIC question. Answer EXTRACTIVELY:\n"
                "- Lead with the figure(s), each carrying its unit and period and a "
                "single `[N]` citation — e.g. \"**$5,409 million** (FY2019) [1].\"\n"
                "- Prefer XBRL FACT / DERIVED METRIC values verbatim when present.\n"
                "- At most one short line stating the basis IF it is visible in the "
                "evidence (e.g. the two operands of a ratio, each cited). Add nothing "
                "the evidence does not state — no XBRL tags, no filing dates, no "
                "methodology notes.\n"
                "- No overview paragraph, no restating the question, no filler.\n"
            )

        prompt = f"""{history_block}Question: {state['question']}

Sub-queries researched:
{sub_queries}

Numbered evidence (cite with `[N]`):
{evidence_block}
{feedback_block}{extractive_block}
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
        from finagent.llm import text_of
        answer = text_of(response)
        # Citation extraction: only `[N]` / `[N, M]` markers, no verbose tags.
        citations = sorted(set(re.findall(r"\[\d+(?:\s*,\s*\d+)*\]", answer)))
        return {
            "draft_answer": answer,
            "final_answer": answer,
            "citations": citations,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    def _critic_system(self) -> str:
        """Phase 9: use the analyst-voice critic (period/unit-aware numeric
        rigor) when enabled, else the generic hallucination critic."""
        return CRITIC_ANALYST_SYSTEM if self.analyst_voice else super()._critic_system()

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
    def _format_calc_result(r: dict) -> str:
        """One readable, auditable line (or block) for a derived-metric result."""
        ticker = r.get("ticker", "")
        metric = str(r.get("metric", "")).replace("_", " ")
        if r.get("series"):                       # trend
            pts = ", ".join(
                f"{s.get('period_label') or 'FY' + str(s.get('fy', s.get('period')))}"
                f"={s.get('value_str')}"
                for s in r["series"] if s.get("ok")
            )
            tail = f" — {r['summary']}" if r.get("summary") else ""
            return f"{ticker} {metric} trend: {pts}{tail}"
        # scalar (ratio / margin / growth / cagr). State the formula and the
        # input figures alongside the result: the RAGAS faithfulness judge (and
        # any reader) can't do the arithmetic from raw inputs alone, so a bare
        # "ratio = 24.26" chunk scores as unsupported even when it's exact.
        lines = [f"{ticker} {metric} = {r.get('value_str', '')}"]
        if r.get("formula"):
            lines.append(f"Derivation: {r['formula']}")
        inputs = [i for i in (r.get("inputs") or []) if isinstance(i, dict)]
        if inputs:
            lines.append("Inputs: " + "; ".join(
                f"{i.get('concept', '?')} ({i.get('period', i.get('fy', '?'))})"
                f" = {i.get('value_str', i.get('value', '?'))}"
                for i in inputs))
        if r.get("source"):
            lines.append(str(r["source"]))
        return "\n".join(lines)


