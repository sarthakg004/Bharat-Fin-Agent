"""Steps 4-5: numbers. Exact figures from SEC XBRL, then ratios computed in Python.

The model only decides WHAT to look up (company, line item, period, formula).
The figures come from the SEC and the arithmetic is done in code.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from finagent.agent.prompts import (CALC_EXTRACT_PROMPT, CALC_EXTRACT_SYSTEM, FORMULA_PROMPT,
                                    FORMULA_SYSTEM, XBRL_EXTRACT_PROMPT, XBRL_EXTRACT_SYSTEM,
                                    XBRL_TAG_SYSTEM, history_block)
from finagent.agent.state import (AgentState, CalcQuery, CalcQueryBatch, FormulaSpec,
                                  XBRLQuery, XBRLQueryBatch)
from finagent.llm import text_of

# The question gives its own formula ("... is defined as: ...").
_DEFINES_RE = re.compile(r"\b(?:defined|calculated|computed|measured)\s+as\b", re.I)
# Multi-period metrics; the formula planner only handles single-period ones.
_MULTIPERIOD_RE = re.compile(r"\b(growth|cagr|trend|compound annual)\b", re.I)
# The sub-query compares two periods ("year-over-year", "vs prior year", "improve").
_COMPARE_RE = re.compile(
    r"\byoy\b|year[\s-]over[\s-]year|\bvs\.?\b|versus|\bprior\b|previous"
    r"|improv|declin|change|compared|expand|contract", re.I)
_QUARTERLY_RE = re.compile(r"\bq[1-4]\b|quarter|10[\s-]?q\b", re.I)


def _quarter(q) -> Optional[str]:
    """The named fiscal quarter ('Q2'), or None."""
    fq = str(getattr(q, "fiscal_quarter", "") or "").strip().upper()
    return fq if fq in ("Q1", "Q2", "Q3", "Q4") else None


def _numeric_subs(state: AgentState) -> list[str]:
    sub_queries = state.get("sub_queries") or [state["question"]]
    routes = state.get("query_routes") or ["narrative"] * len(sub_queries)
    return [s for s, r in zip(sub_queries, routes) if r == "numeric"]


def _trust_the_named_company(agent, state: AgentState, sub_q: str, q) -> None:
    """Replace a ticker the model guessed with the company the sub-query names.

    The extractor sometimes recalls a ticker from memory and gets it wrong
    ("Amcor" -> AMR, which is Alpha Metallurgical Resources), and a figure for
    the wrong company would be presented as exact. When the sub-query opens
    with a registrant's exact name, that company wins.
    """
    if q is None or not getattr(q, "ticker", ""):
        return
    resolver = agent.xbrl.resolver
    words = re.findall(r"[\w&.-]+", re.sub(r"['’]s\b", "", sub_q))[:4]
    for n in range(len(words), 0, -1):
        named = resolver.resolve(" ".join(words[:n]))
        if named.get("match") == "name_exact":
            if resolver.resolve(q.ticker).get("cik") != named["cik"]:
                agent.log(state, f"company corrected: {q.ticker!r} -> {named['ticker']} "
                                 f"(named in {sub_q!r})")
                q.ticker = named["ticker"]
            return


def _extract(agent, state: AgentState, sub_queries: list[str], batch_schema,
             single_schema, system: str, single_prompt: str) -> list[tuple]:
    """One structured extraction per sub-query: [(sub_query, extraction or None)].

    All sub-queries go in one call. If that fails or returns the wrong number of
    items, fall back to one call per sub-query.
    """
    context = history_block(state.get("chat_history"), 4, 300)
    system = system.format(today=date.today().isoformat())
    llm = agent.llm("extractor")
    if len(sub_queries) > 1:
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sub_queries))
        try:
            out = llm.with_structured_output(batch_schema).invoke([
                SystemMessage(content=system),
                HumanMessage(content=f"{context}Numbered sub-queries:\n{numbered}\n\n"
                                     f"Return EXACTLY one extraction per numbered sub-query, "
                                     f"in the same order ({len(sub_queries)} entries)."),
            ])
            if len(out.queries or []) == len(sub_queries):
                for sub_q, q in zip(sub_queries, out.queries):
                    _trust_the_named_company(agent, state, sub_q, q)
                return list(zip(sub_queries, out.queries))
            agent.log(state, "batch extraction misaligned; extracting one by one")
        except Exception as e:
            agent.llm_failed(state, "Number lookup", e)

    pairs: list[tuple] = []
    for sub_q in sub_queries:
        try:
            q = llm.with_structured_output(single_schema).invoke([
                SystemMessage(content=system),
                HumanMessage(content=context + single_prompt.format(sub_query=sub_q)),
            ])
        except Exception as e:
            agent.llm_failed(state, "Number lookup", e)
            q = None
        _trust_the_named_company(agent, state, sub_q, q)
        pairs.append((sub_q, q))
    return pairs


# One sub-query may name several line items and years ("capex and D&A for
# FY2021 and FY2022"); every pair is looked up, up to this many.
MAX_LOOKUPS_PER_SUB_QUERY = 9


def xbrl(agent, state: AgentState) -> dict:
    """For each numeric sub-query asking for reported figures, fetch them from
    SEC XBRL. These are the highest-priority evidence the writer gets."""
    subs = _numeric_subs(state)
    if not subs:
        return {"xbrl_facts": []}
    facts: list[dict] = []
    for sub_q, q in _extract(agent, state, subs, XBRLQueryBatch, XBRLQuery,
                             XBRL_EXTRACT_SYSTEM, XBRL_EXTRACT_PROMPT):
        if q is None or not q.answerable or not (q.ticker and q.concept):
            continue
        fp = _quarter(q)
        concepts = list(dict.fromkeys([q.concept, *(q.other_concepts or [])]))
        periods = list(dict.fromkeys([q.period or None, *(q.other_periods or [])]))
        pairs = [(c, p) for c in concepts if c for p in periods][:MAX_LOOKUPS_PER_SUB_QUERY]
        for concept, period in pairs:
            try:
                res = agent.xbrl.run(ticker=q.ticker, concept=concept, period=period,
                                     quarterly=q.quarterly or bool(fp), fp=fp)
            except Exception as e:
                agent.log(state, f"xbrl lookup failed for {sub_q!r} ({concept}, {period}): {e}")
                continue
            res["sub_query"] = sub_q
            if res.get("ok"):
                facts.append(res)
            else:
                agent.log(state, f"xbrl miss for {sub_q!r} ({concept}, {period}): {res.get('error')}")
    return {"xbrl_facts": facts}


def pick_xbrl_tag(agent, concept: str, available_tags: list[str]) -> Optional[str]:
    """Ask the LLM for the best US-GAAP tag among those a company reports. Used
    by the XBRL client only when its curated concept map has no match."""
    try:
        resp = agent.llm("extractor").invoke([
            SystemMessage(content=XBRL_TAG_SYSTEM),
            HumanMessage(content=f"Concept: {concept}\n\nAvailable tags:\n"
                                 + "\n".join(available_tags[:200])),
        ])
        choice = (text_of(resp).strip().strip("`").split() or [""])[0]
        return None if choice.upper() == "NONE" else choice
    except Exception:
        return None


def calculator(agent, state: AgentState) -> dict:
    """For each numeric sub-query asking for a DERIVED metric (margin, ratio,
    growth, CAGR, trend), compute it from XBRL figures."""
    subs = _numeric_subs(state)
    if not subs:
        return {"calc_results": []}
    results: list[dict] = []
    for sub_q, q in _extract(agent, state, subs, CalcQueryBatch, CalcQuery,
                             CALC_EXTRACT_SYSTEM, CALC_EXTRACT_PROMPT):
        if q is None or not q.is_derived or not (q.ticker and q.metric):
            continue
        known = agent.calc.knows(q.metric)
        redefined = bool(_DEFINES_RE.search(sub_q))
        multiperiod = bool(_MULTIPERIOD_RE.search(q.metric) or _MULTIPERIOD_RE.search(sub_q))
        res = None
        # 1. A built-in formula, unless the question redefines the metric.
        if known and not redefined:
            res = _run_calc(agent, state, sub_q, q)
        # 2. An unknown or redefined metric: the LLM writes the formula's
        #    structure, the calculator fetches the figures and does the math.
        if (res is None or not res.get("ok")) and not multiperiod:
            spec = _plan_formula(agent, state, sub_q, q.metric)
            if spec is not None and spec.ok:
                try:
                    res = agent.calc.ratio_from_spec(
                        q.ticker, spec.model_dump(),
                        period=(_averaging_target_period(sub_q)
                                or (q.periods[0] if q.periods else None)),
                        metric_name=q.metric or "custom_metric")
                except Exception as e:
                    agent.log(state, f"planned formula failed for {sub_q!r}: {e}")
        # 3. Last resort: the built-in formula after all.
        if (res is None or not res.get("ok")) and known:
            res = _run_calc(agent, state, sub_q, q)
        if res is None:
            continue
        res["sub_query"] = sub_q
        if res.get("ok"):
            results.append(res)
        else:
            agent.log(state, f"calc miss for {sub_q!r}: {res.get('error')}")
    return {"calc_results": results}


def _averaging_target_period(sub_q: str) -> Optional[str]:
    """"FY2021 ratio using average X between FY2020 and FY2021" -> "FY2021".
    The earlier year only feeds the average; the latest one is the target."""
    if "average" not in sub_q.lower():
        return None
    years = sorted({int(y) for y in re.findall(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", sub_q)})
    return f"FY{years[-1]}" if len(years) >= 2 else None


def _run_calc(agent, state: AgentState, sub_q: str, q) -> Optional[dict]:
    """The built-in calculator formulas. None on an exception."""
    from finagent.tools.calculator import (AVG_DENOMINATOR_RATIOS, COMPOSITE_RATIOS,
                                           RATIOS, _canonical_metric)

    fp = _quarter(q)
    quarterly = bool(getattr(q, "quarterly", False) or fp or _QUARTERLY_RE.search(sub_q))
    try:
        metric = _canonical_metric(q.metric)
        # "latest vs prior" with no year named: find the newest filed period,
        # then compute both periods so the answer states the change.
        if (not q.periods and (metric in RATIOS or metric in COMPOSITE_RATIOS)
                and _COMPARE_RE.search(sub_q)):
            latest = agent.calc.ratio(q.ticker, metric, None, quarterly=quarterly)
            fy = latest.get("fy") if latest.get("ok") else None
            if fy:
                return agent.calc.trend(q.ticker, metric, [f"FY{fy - 1}", f"FY{fy}"],
                                        quarterly=quarterly,
                                        fp=latest.get("fp") if quarterly else None)
        target = _averaging_target_period(sub_q)
        if metric in AVG_DENOMINATOR_RATIOS and target:
            return agent.calc.ratio(q.ticker, metric, target)
        first = q.periods[0] if q.periods else None
        span = len(q.periods) >= 2
        return agent.calc.run(
            metric=q.metric, ticker=q.ticker, concept=q.concept, quarterly=quarterly,
            periods=q.periods, period=first,
            period_from=first if span else None, period_to=q.periods[-1] if span else None,
            start_period=first if span else None, end_period=q.periods[-1] if span else None,
            fp=fp)
    except Exception as e:
        agent.log(state, f"calc failed for {sub_q!r}: {e}")
        return None


def _plan_formula(agent, state: AgentState, sub_q: str, metric: str):
    """Ask the LLM to express `metric` as a formula over XBRL concepts."""
    try:
        return agent.llm("extractor").with_structured_output(FormulaSpec).invoke([
            SystemMessage(content=FORMULA_SYSTEM),
            HumanMessage(content=FORMULA_PROMPT.format(metric=metric, question=sub_q)),
        ])
    except Exception as e:
        agent.llm_failed(state, "Formula planning", e)
        return None
