"""Steps 7-9: write the answer, fact-check it, and recover once or refuse."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from finagent.agent.prompts import (CRITIC_PROMPT, CRITIC_SYSTEM, REFUSAL_TEMPLATE,
                                    WRITER_EXTRACTIVE, WRITER_FEEDBACK, WRITER_PROMPT,
                                    WRITER_SYSTEM, history_block)
from finagent.agent.state import AgentState, CriticReport
from finagent.llm import text_of

# The draft itself saying the evidence cannot answer ("not specified in the
# provided sources"). A fallback for when the critic misses it or fails.
_INSUFFICIENT_RE = re.compile(
    r"not (?:explicitly )?(?:specified|stated|provided|available|disclosed"
    r"|described|outlined|mentioned)"
    r"|no (?:specific )?(?:information|data|details?|mention|evidence|figures?)"
    r"|not enough information"
    r"|cannot be (?:determined|calculated|computed|answered)"
    r"|unable to (?:determine|find|locate)"
    r"|do(?:es)? not (?:contain|state|specify|describe|disclose|provide"
    r"|discuss|mention|cover)", re.I)


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #

def format_calc_result(r: dict) -> str:
    """A calculator result as readable text: the value, the formula, the inputs."""
    ticker = r.get("ticker", "")
    metric = str(r.get("metric", "")).replace("_", " ")
    if r.get("series"):                                     # a multi-period trend
        points = ", ".join(
            f"{s.get('period_label') or 'FY' + str(s.get('fy', s.get('period')))}={s.get('value_str')}"
            for s in r["series"] if s.get("ok"))
        return f"{ticker} {metric} trend: {points}" + (f" — {r['summary']}" if r.get("summary") else "")
    lines = [f"{ticker} {metric} = {r.get('value_str', '')}"]
    if r.get("formula"):
        lines.append(f"Derivation: {r['formula']}")
    inputs = [i for i in (r.get("inputs") or []) if isinstance(i, dict)]
    if inputs:
        lines.append("Inputs: " + "; ".join(
            f"{i.get('concept', '?')} ({i.get('period', i.get('fy', '?'))})"
            f" = {i.get('value_str', i.get('value', '?'))}" for i in inputs))
    if r.get("source"):
        lines.append(str(r["source"]))
    return "\n".join(lines)


def _short(value_str: str) -> str:
    """"$10,069,000,000 ($10,069 million; $10.07 billion)" -> "$10,069 million"."""
    m = re.search(r"\(([^;)]+)", value_str or "")
    return m.group(1).strip() if m else (value_str or "?")


def _readable_formula(formula: str) -> str:
    """"cost_of_revenue / avg(inventory)" -> "cost of revenue ÷ average inventory"."""
    f = re.sub(r"avg\(([^)]*)\)", r"average \1", formula or "")
    return f.replace("_", " ").replace(" / ", " ÷ ").replace(" * ", " × ")


def format_calc_card(r: dict) -> str:
    """A calculator result for a person to read (the source card): the value,
    the formula in words, and the inputs as a table. The models read
    `format_calc_result` instead."""
    ticker = r.get("ticker", "")
    metric = str(r.get("metric", "")).replace("_", " ")
    if r.get("series"):
        rows = [f"| {s.get('period_label') or 'FY' + str(s.get('fy', s.get('period')))} "
                f"| {_short(s.get('value_str', ''))} |" for s in r["series"] if s.get("ok")]
        out = [f"**{ticker} {metric} trend**", "", "| Period | Value |", "|---|---|", *rows]
        return "\n".join(out + ([f"\n{r['summary']}"] if r.get("summary") else []))
    out = [f"**{ticker} {metric}: {r.get('value_str', '')}**"]
    if r.get("formula"):
        out += ["", f"Formula: {_readable_formula(r['formula'])}"]
    inputs = [i for i in (r.get("inputs") or []) if isinstance(i, dict)]
    if inputs:
        out += ["", "| Input | Period | Value |", "|---|---|---|"]
        out += [f"| {str(i.get('concept', '?')).replace('_', ' ')} "
                f"| {i.get('period', i.get('fy', '?'))} "
                f"| {_short(str(i.get('value_str', i.get('value', '?'))))} |" for i in inputs]
    return "\n".join(out)


def format_edgar_result(r: dict) -> str:
    lines = [f"  - {c.get('company', '?')}{(' (' + c['ticker'] + ')') if c.get('ticker') else ''}"
             f" — {c.get('form', '')} {c.get('date', '')}  {c.get('url', '')}"
             for c in r.get("companies", [])]
    total = r.get("total")
    head = (f"EDGAR full-text search {r.get('query', '')} ({total:,} matching filings; "
            f"showing {len(lines)} companies):" if isinstance(total, int)
            else f"EDGAR full-text search {r.get('query', '')}:")
    return head + "\n" + "\n".join(lines)


def _format_market(tool: str, data: dict) -> str:
    if tool == "get_history":
        s = data.get("summary", {})
        volume = ""
        if s.get("avg_volume") is not None:
            volume = (f"\nVolume: last={s.get('last_volume'):,}, avg={s.get('avg_volume'):,}, "
                      f"recent_avg={s.get('recent_avg_volume'):,} vs "
                      f"prior_avg={s.get('prior_avg_volume'):,} ({s.get('volume_change_pct')}% change"
                      f"{', SURGE' if s.get('volume_surge') else ''}).")
        return (f"{s.get('symbol', '?')} {s.get('period', '?')} {s.get('interval', '?')}\n"
                f"Range {s.get('start', '?')} → {s.get('end', '?')}; "
                f"first_close={s.get('first_close')}, last_close={s.get('last_close')}, "
                f"high={s.get('high')}, low={s.get('low')}, pct_change={s.get('pct_change')}%.{volume}")
    if tool == "compare":
        return "\n".join(f"  - {r.get('symbol')}: last={r.get('lastPrice')} "
                         f"prevClose={r.get('previousClose')} yearChange={r.get('yearChange')}"
                         for r in data.get("rows") or [])
    if tool == "get_news":
        return data.get("symbol", "?") + "\n" + "\n".join(
            f"  - {a.get('title', '')} ({a.get('publisher', '')})" for a in data.get("articles") or [])
    return ", ".join(f"{k}={v}" for k, v in data.items() if v not in (None, ""))[:1200]


def build_evidence(state: AgentState) -> list[dict]:
    """Everything gathered, as ONE numbered list.

    The writer cites item N as `[N]`, the critic checks against the same list,
    and the API returns it as the citation cards, so a `[3]` in the answer
    always points at the third card. Most authoritative sources come first.

    Each item: `prompt` (what the models read) plus the card fields.
    """
    items: list[dict] = []

    def add(kind, prompt, text, *, company="", ticker="", year="", source_url="",
            citation="", sub_query=""):
        items.append({"kind": kind, "prompt": prompt, "text": text, "company": company,
                      "ticker": ticker, "year": str(year), "source_url": source_url,
                      "citation": citation, "sub_query": sub_query})

    for f in state.get("xbrl_facts") or []:                 # exact filed figures
        period = f.get("period_label", "FY" + str(f.get("fy", "?")))
        entity = f.get("entity", f.get("ticker", ""))
        add("xbrl",
            f"XBRL FACT (authoritative — exact figure as filed) — {entity} "
            f"{f.get('concept', '')} {period}: {f.get('value_str', '')}\n"
            f"Source: {f.get('source', '')} (us-gaap:{f.get('tag', '')}).",
            f"**{entity} {f.get('concept', '')} ({period}): {_short(f.get('value_str', ''))}**\n\n"
            f"Exact figure as filed: {f.get('value_str', '')}, us-gaap:{f.get('tag', '')}, "
            f"{f.get('form', '')} for the period ending {f.get('end', '')}.",
            company=entity, ticker=f.get("ticker", ""), year=f.get("fy", "?"),
            citation=f.get("source", ""), sub_query=f.get("sub_query", ""))

    for r in state.get("calc_results") or []:               # math over those figures
        text = format_calc_result(r)
        add("calc", f"DERIVED METRIC (computed from exact XBRL inputs) — {text}", format_calc_card(r),
            company=r.get("ticker", "?"), ticker=r.get("ticker", ""),
            year=r.get("fy", r.get("end_period", "?")),
            citation=f"(Computed: {str(r.get('metric', '')).replace('_', ' ')} from XBRL)",
            sub_query=r.get("sub_query", ""))

    for c in state.get("retrieved_chunks") or []:           # filing passages
        add("text", f"FILING EXCERPT — {c.get('source', '')}\n{c.get('text', '')}",
            c.get("text", ""), company=c.get("company", "?"), ticker=c.get("ticker", ""),
            year=c.get("year", "?"), source_url=c.get("source_url", ""),
            citation=c.get("source", ""), sub_query=c.get("sub_query", ""))

    for m in state.get("market_data") or []:                # live market data
        if not m.get("ok"):
            continue
        data, tool = m.get("data") or {}, m.get("tool", "")
        sym = data.get("symbol") or (data.get("summary") or {}).get("symbol") or "—"
        add("market", f"LIVE MARKET (yfinance · {tool})\n{_format_market(tool, data)}",
            str({k: v for k, v in data.items() if k != "chart"})[:1500],
            company=sym, ticker=sym, year="—",
            citation=f"<Market: yfinance.{tool} {sym}>", sub_query=m.get("sub_query", ""))

    web = sorted(state.get("web_results") or [],            # web, newest first
                 key=lambda h: h.get("published_date") or "", reverse=True)
    for h in web:
        tier = "TRUSTED PRESS" if h.get("tier") == "trusted" else "WEB"
        pub = f" (published {h['published_date']})" if h.get("published_date") else ""
        content = (h.get("content") or "")[:1500]
        add("web", f"{tier}{pub} — {h.get('title', '')[:120]} ({h.get('url', '')})\n{content}",
            content, company=h.get("title", "")[:80] or "web",
            year=(h.get("published_date") or "")[:4] or "?", source_url=h.get("url", ""),
            citation=f"<News: {h.get('title', '')[:80]}>", sub_query=h.get("sub_query", ""))

    for r in state.get("edgar_results") or []:              # cross-company search
        text = format_edgar_result(r)
        first = (r.get("companies") or [{}])[0]
        add("edgar", f"EDGAR CROSS-DOCUMENT SEARCH — {text}", text, company="EDGAR search",
            year="—", source_url=first.get("url", ""),
            citation=f"<EDGAR: {r.get('query', '')}>", sub_query=r.get("sub_query", ""))
    return items


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #

def gather(agent, state: AgentState) -> dict:
    """Where the parallel lanes meet. If the question skipped filing search and
    every tool lane came back empty, send it through filing search once."""
    routes = state.get("query_routes") or []
    empty = not build_evidence(state)
    skipped_filings = bool(routes) and "narrative" not in routes
    if empty and skipped_filings and not state.get("corpus_fallback_used"):
        agent.log(state, "tool lanes found nothing; trying the filings")
        return {"corpus_fallback_pending": True, "corpus_fallback_used": True}
    return {"corpus_fallback_pending": False}


def synthesize(agent, state: AgentState) -> dict:
    """Write the answer from the numbered evidence."""
    evidence = build_evidence(state)
    block = "\n\n".join(f"[{i}] {e['prompt']}" for i, e in enumerate(evidence, 1))
    routes = state.get("query_routes") or []
    claims = state.get("unsupported_claims") or []
    prompt = WRITER_PROMPT.format(
        history=history_block(state.get("chat_history"), 6, 600),
        question=state["question"],
        sub_queries="\n".join(f"- {q}" for q in state.get("sub_queries", [])),
        evidence=block or "(no evidence retrieved)",
        feedback=WRITER_FEEDBACK.format(claims="\n".join(f"  - {c}" for c in claims[:5]))
        if claims else "",
        # A purely numeric question gets the figure, not an essay around it.
        extractive=WRITER_EXTRACTIVE if routes and all(r == "numeric" for r in routes) else "",
    )
    answer = text_of(agent.llm("writer").invoke(
        [SystemMessage(content=WRITER_SYSTEM), HumanMessage(content=prompt)]))
    return {"answer": answer, "evidence": evidence,
            "citations": sorted(set(re.findall(r"\[\d+(?:\s*,\s*\d+)*\]", answer)))}


def critic(agent, state: AgentState) -> dict:
    """Check every claim in the draft against the evidence the writer was given,
    and choose at most one recovery:

        filing search  the draft admits a gap and the question skipped filing search
        web search     the draft admits a gap, filings already searched, web unused
        retrieve     the evidence lacks the fact ("gather")
        redraft      the evidence is fine, the draft overstated it
    """
    out: dict = {"needs_retry": False, "web_fallback_pending": False,
                 "corpus_fallback_pending": False, "retry_queries": []}
    answer = state.get("answer", "")
    context = "\n\n".join(f"[{i}] {e['prompt']}"
                          for i, e in enumerate(state.get("evidence") or [], 1)) or "No evidence."
    try:
        report: CriticReport = agent.llm("critic").with_structured_output(CriticReport).invoke([
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(content=CRITIC_PROMPT.format(context=context, answer=answer)),
        ])
        verdicts = report.verdicts or []
    except Exception as e:
        # The draft still ships; the user is told it was not checked.
        agent.llm_failed(state, "Fact-check", e, always_notice=True)
        return {**out, "support_score": None, "unsupported_claims": []}

    unsupported = [v.claim for v in verdicts if not v.supported]
    if verdicts:
        out.update(support_score=round(1 - len(unsupported) / len(verdicts), 3),
                   unsupported_claims=unsupported, remedy=report.remedy)
    else:
        out.update(support_score=None, unsupported_claims=[])

    recoveries = state.get("recoveries", 0)
    if recoveries >= agent.MAX_RECOVERIES:
        return out
    routes = state.get("query_routes") or []
    admits_gap = report.draft_says_evidence_missing or bool(_INSUFFICIENT_RE.search(answer))
    if (admits_gap and routes and "narrative" not in routes
            and not state.get("corpus_fallback_used")):
        agent.log(state, "draft admits a missing figure; searching the filings")
        out.update(corpus_fallback_pending=True, corpus_fallback_used=True,
                   recoveries=recoveries + 1)
    elif (admits_gap and not state.get("web_results")
            and not state.get("web_fallback_used")):
        agent.log(state, "draft admits the evidence can't answer; trying web search")
        out.update(web_fallback_pending=True, web_fallback_used=True, recoveries=recoveries + 1)
    elif unsupported:
        out.update(needs_retry=True, recoveries=recoveries + 1)
        if report.remedy == "gather":
            # Search for the claims that failed, not the original query again.
            out["retry_queries"] = ["supporting evidence for: " + c for c in unsupported[-3:]]
    return out


def refuse(agent, state: AgentState) -> dict:
    """Replace a mostly unsupported answer with an explicit refusal."""
    claims = state.get("unsupported_claims") or []
    detail = " (unsupported: " + "; ".join(c[:80] for c in claims[:2]) + ")" if claims else ""
    web_clause = "" if state.get("web_results") else " or recent web sources"
    return {"answer": REFUSAL_TEMPLATE.format(web_clause=web_clause, detail=detail),
            "refused": True, "needs_retry": False}
