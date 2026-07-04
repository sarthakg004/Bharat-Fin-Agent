"""Deterministic numeric lanes: XBRL facts, derived-metric calculator, table agent.

Split out of `finagent.graph.agent` (methods are unchanged); mixed into
`AgenticRAGv4` ahead of `AgenticRAGv3` in the MRO.
"""

from __future__ import annotations

import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from finagent.graph.state import (
    AgentState, XBRLQuery, XBRLQueryBatch, CalcQuery, CalcQueryBatch, FormulaSpec,
)
from finagent.tools.xbrl import CONCEPT_TAGS

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
growth/cagr only — the underlying concept. List EVERY fiscal year the sub-query
names: "FY2021 inventory turnover using average inventory between FY2020 and
FY2021" → periods ['FY2020','FY2021'] (the LAST period is the target year; the
earlier one only feeds the averaged input — never return just the earlier year).
Set is_derived=false for a single reported figure (revenue,
net income, total assets, …); those are handled by the XBRL facts tool, not here.
Use the conversation context to resolve a follow-up's company/periods.
"""

CALC_EXTRACT_PROMPT = """\
Numeric sub-query: {sub_query}

Return the derived-metric computation (is_derived, ticker, metric, concept, periods).
"""

# Canonical XBRL concepts the formula planner may reference (keys of
# xbrl.CONCEPT_TAGS). The planner must use ONLY these.
_PLANNER_CONCEPTS = ", ".join(sorted(CONCEPT_TAGS))

FORMULA_PLANNER_SYSTEM = f"""\
You turn a financial metric into a FORMULA over canonical accounting concepts.
You output STRUCTURE ONLY — never numbers. A separate deterministic step fetches
the exact figures from SEC XBRL and does the arithmetic, so your job is purely:
which concepts go in the numerator and denominator, and how.

Use ONLY these canonical concept names (map synonyms onto them — "sales"→revenue,
"COGS"→cost_of_revenue, "PP&E"→ppe_net, "shareholders' equity"→stockholders_equity,
"D&A"→depreciation_amortization, "CFO"→operating_cash_flow):
{_PLANNER_CONCEPTS}

Rules:
- If the QUESTION states its own definition ("X is defined as: A / B"), follow
  THAT definition exactly — it overrides the textbook formula.
- numerator_add / numerator_sub: concepts combined in the numerator.
- denominator_add / denominator_sub: the denominator. LEAVE THE DENOMINATOR
  EMPTY when the metric is a dollar amount, not a ratio (e.g. unadjusted EBITDA =
  operating_income + depreciation_amortization → numerator only).
- average_denominator: true when the denominator is a balance-sheet stock that is
  conventionally averaged over the prior and current year — turnover ratios
  (revenue / avg PP&E, COGS / avg inventory, revenue / avg total_assets) and
  returns (net_income / avg equity or assets). False for liquidity/leverage
  ratios measured at year-end (current ratio, quick ratio, debt-to-equity).
- is_percent: true for margins, returns, and "% of revenue" metrics.
- If the metric cannot be expressed from the listed concepts, set ok=false.
"""

FORMULA_PLANNER_PROMPT = """\
Metric: {metric}
Question: {question}

Express this metric as a formula over the canonical concepts.
"""

# The question states its OWN formula ("… is defined as: …", "calculated as").
# When it does, we plan from that definition rather than the hardcoded ratio, so
# a redefined metric (e.g. a custom quick-ratio variant) follows the question.
_DEFINES_RE = re.compile(r"\b(?:defined|calculated|computed|measured)\s+as\b", re.I)

# Multi-period metrics the hardcoded path owns (growth/cagr/trend); the
# single-period formula planner doesn't handle these.
_MULTIPERIOD_RE = re.compile(r"\b(growth|cagr|trend|compound annual)\b", re.I)


class NumericNodes:
    """Deterministic numeric lanes: XBRL facts, derived-metric calculator, table agent."""

    def _xbrl_pick_tag(self, concept: str, available_tags: list[str]):
        """LLM fallback for the XBRL client: pick the best US-GAAP tag for a
        concept from the tags a company actually reports. Used only when the
        curated concept→tag map misses (keeps cost near zero on the common path).
        """
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

    def _extract_batch(self, state: AgentState, sub_queries: list[str],
                       batch_schema, system: str, single_prompt: str,
                       single_schema, history_block: str) -> list[tuple]:
        """Run ONE structured-output extraction over all `sub_queries`,
        returning [(sub_query, extraction-or-None), ...] aligned by position.

        The batch call replaces N per-sub-query LLM calls. If it fails or the
        model returns a misaligned list, fall back to the per-sub-query loop so
        behaviour degrades to the legacy path, never to silent skips.
        """
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
            redefined = bool(_DEFINES_RE.search(sub_q))
            multiperiod = bool(_MULTIPERIOD_RE.search(q.metric)
                               or _MULTIPERIOD_RE.search(sub_q))
            res = None
            # Fast, audited path for a known metric the question doesn't redefine.
            if self.calc.knows(q.metric) and not redefined:
                res = self._run_calc(state, sub_q, q)
            # Dynamic formula planner: an unknown metric, OR one the question
            # redefines. The LLM returns a FORMULA over canonical XBRL concepts
            # (never numbers); ratio_from_spec fetches exact facts and computes
            # deterministically — so a planned metric is as faithful as a
            # hardcoded one. Single-period only (growth/cagr/trend stay hardcoded).
            if (res is None or not res.get("ok")) and not multiperiod:
                spec = self._plan_formula(state, sub_q, q.metric)
                if spec is not None and getattr(spec, "ok", False):
                    try:
                        res = self.calc.ratio_from_spec(
                            q.ticker, spec.model_dump(),
                            period=(self._averaging_target_period(sub_q)
                                    or (q.periods[0] if q.periods else None)),
                            metric_name=q.metric or "custom_metric")
                    except Exception as e:
                        self._log(state, f"dynamic formula failed for {sub_q!r}: {e}")
            # Last resort: a known metric the planner couldn't serve.
            if (res is None or not res.get("ok")) and self.calc.knows(q.metric):
                res = self._run_calc(state, sub_q, q)
            if res is None:
                continue
            res["sub_query"] = sub_q
            if res.get("ok"):
                results.append(res)
            else:
                self._log(state, f"calc miss for {sub_q!r}: {res.get('error')}")
        return {"calc_results": results}

    @staticmethod
    def _averaging_target_period(sub_q: str) -> Optional[str]:
        """'FY2021 ratio … average X between FY2020 and FY2021' → 'FY2021'.

        An averaged-input ratio names two years, but the target is always the
        LATEST — the earlier year only feeds the averaged input. Read it from
        the question itself so a wrong-period extraction (observed via a
        Langfuse trace: the LLM returning just FY2020, which computed the prior
        year's ratio and got refused as "no FY2021 evidence") can't reach the
        calculator. None when the phrasing doesn't apply.
        """
        if "average" not in sub_q.lower():
            return None
        years = sorted({int(y) for y in
                        re.findall(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", sub_q)})
        return f"FY{years[-1]}" if len(years) >= 2 else None

    def _run_calc(self, state: AgentState, sub_q: str, q) -> Optional[dict]:
        """The hardcoded deterministic calculator path (margins/ratios/growth/
        cagr/trend/days). Returns None on an exception so the caller can fall
        through to the dynamic planner."""
        from finagent.tools.calculator import AVG_DENOMINATOR_RATIOS, _canonical_metric

        try:
            metric = _canonical_metric(q.metric)
            target = self._averaging_target_period(sub_q)
            if metric in AVG_DENOMINATOR_RATIOS and target:
                return self.calc.ratio(q.ticker, metric, target)
            return self.calc.run(
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
            return None

    def _plan_formula(self, state: AgentState, sub_q: str, metric: str):
        """Ask the LLM to express `metric` as a FORMULA over canonical XBRL
        concepts (structure only, never numbers). Returns a FormulaSpec or None.
        Used when the calculator doesn't hardcode the metric, or the question
        supplies its own definition."""
        try:
            planner = self._get_router_llm().with_structured_output(FormulaSpec)
            spec = planner.invoke([
                SystemMessage(content=FORMULA_PLANNER_SYSTEM),
                HumanMessage(content=FORMULA_PLANNER_PROMPT.format(
                    metric=metric, question=sub_q)),
            ])
            if spec is not None and getattr(spec, "ok", False):
                self._log(state, f"planned formula for {metric!r}: "
                                 f"+{spec.numerator_add} -{spec.numerator_sub} "
                                 f"/ +{spec.denominator_add} -{spec.denominator_sub}"
                                 f"{' avg' if spec.average_denominator else ''}")
            return spec
        except Exception as e:
            self._log(state, f"formula planner failed for {metric!r}: {e}")
            return None

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

