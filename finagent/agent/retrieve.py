"""Steps 2-3: make sure the company's filing is indexed, then search the filings.

    fetch_filing   company not in the index? download its 10-K from EDGAR and ingest it
    retrieve       hybrid search on the rewritten query and on the raw question,
                   then keep the 8 best passages scored against the question
"""

from __future__ import annotations

import os
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from finagent.agent.prompts import GATE_PROMPT, GATE_SYSTEM, history_block
from finagent.agent.state import AgentState, CorpusGateQuery
from finagent.retrieval.filters import parse_quarter, parse_years
from finagent.runtime import current_context
from finagent.tools.sec_fetch import fiscal_year, is_filing_indexed

# Passages the raw question contributes next to the rewritten query. Measured:
# 1 slot -> 64 of 99 questions with evidence in the final 8, 2 -> 66, 5 -> 67,
# while the top-5 score falls. Two is the best trade.
QUESTION_SLOTS = 2
# For a narrative question the raw question gets the full budget: the rewritten
# query is keywords, and keywords are a poor match for prose. Measured: 71 -> 75 of 99.
NARRATIVE_QUESTION_SLOTS = 8

# A live fetch embeds 700-1,000 chunks per filing, and one Gemini key embeds
# 1,000 texts a day. ponytail: capped at 3 filings per question; raise it on a
# paid embedding key (it was 12 when the embedder was local).
MAX_FETCH_FILINGS = 3

# Cyrillic, Hebrew, Arabic, Devanagari, Japanese, CJK, Hangul.
_NON_LATIN_RE = re.compile(r"[Ѐ-ӿ֐-׿؀-ۿऀ-ॿ぀-ヿ㐀-鿿가-힯]")


def _mostly_non_english(text: str) -> bool:
    """True when over 30% of a chunk's letters are non-Latin script."""
    letters = sum(c.isalpha() for c in text or "")
    return letters >= 20 and len(_NON_LATIN_RE.findall(text)) / letters > 0.30


# --------------------------------------------------------------------------- #
# fetch_filing
# --------------------------------------------------------------------------- #

def fetch_filing(agent, state: AgentState) -> dict:
    """Fetch and ingest the company's 10-K when the index does not cover the question.

    Three outcomes: the company is indexed (nothing to do, unless the question
    asks about a year the index lacks), it is US-listed but missing (fetch), or
    it has no SEC id (leave it to web search).
    """
    if os.getenv("DISABLE_DYNAMIC_FETCH") == "1":        # the eval uses a fixed corpus
        return {"fetch_status": {}}
    question = state["question"]
    try:
        gq: CorpusGateQuery = agent.llm("extractor").with_structured_output(CorpusGateQuery).invoke([
            SystemMessage(content=GATE_SYSTEM),
            HumanMessage(content=history_block(state.get("chat_history"), 4, 300)
                         + GATE_PROMPT.format(question=question)),
        ])
        company = (gq.company or "").strip()
    except Exception as e:
        agent.llm_failed(state, "Company lookup", e)
        return {"fetch_status": {}}
    if not company:
        return {"fetch_status": {}}

    try:
        gate = agent.fetcher.gate(company)
    except Exception as e:
        agent.log(state, f"corpus gate failed for {company!r}: {e}")
        return {"fetch_status": {}}
    if gate["decision"] == "not_us_listed":
        return {"fetch_status": gate}

    # The gate checks the ticker; retrieval filters by matching the question
    # text against stored company names. If retrieval cannot match this company,
    # its search would run unfiltered over every other company, so fetch instead.
    if gate["decision"] == "already_indexed" and not _vocab_can_match(
            agent, company, gate.get("ticker"), gate.get("company")):
        agent.log(state, f"{company!r} is indexed but retrieval cannot match its name; fetching")
        gate = {**gate, "decision": "fetch"}

    # The fiscal years the question names ("FY2022", "FY22", "2022"): fetch the
    # 10-Ks that cover them. No year means the latest filing.
    years = sorted(set(parse_years(question)), reverse=True)[:MAX_FETCH_FILINGS]

    # A named quarter (Q1-Q3; Q4 is in the 10-K) wants that quarter's 10-Q.
    quarter = parse_quarter(question)
    if years and quarter in ("Q1", "Q2", "Q3"):
        done = _fetch_quarter(agent, state, gate, company, years[0], quarter)
        if done is not None:
            return done

    if gate["decision"] == "already_indexed":
        # "Indexed" is per company, so check the exact 10-K each year needs (no
        # year: the latest). The index's year label is the filing year, which
        # cannot tell a December company's FY2022 10-K (filed 2023) from FY2023.
        wanted = agent.fetcher.pick_filings(gate["cik"], n=max(1, len(years)),
                                            fiscal_years=years or None)
        urls = _indexed_values(agent, "source_url", gate.get("ticker") or company)
        missing = [f for f in wanted if not is_filing_indexed(f, urls)]
        if not missing:          # also when EDGAR lists none: search what is indexed
            return {"fetch_status": gate}
        if years:
            years = [fiscal_year(f["period"]) for f in missing]
        agent.log(state, f"the index lacks {gate.get('ticker')}'s 10-K for "
                         f"{', '.join(f'FY{y}' for y in years) or 'the latest year'}")

    agent.log(state, f"fetching the 10-K for {', '.join(f'FY{y}' for y in years) or 'the latest year'}"
                     f" for {gate['ticker']} from EDGAR")
    return _ingest(agent, state, gate, company, n=max(1, len(years)), fiscal_years=years or None)


def _fetch_quarter(agent, state: AgentState, gate: dict, company: str, fy: int,
                   quarter: str) -> Optional[dict]:
    """Fetch the 10-Q the company itself labelled fiscal year `fy`, `quarter`.
    None when no 10-Q carries that label (no XBRL data, or not filed yet): the
    caller then fetches the year's 10-K instead."""
    try:
        accn = agent.xbrl.filing_accession(gate["cik"], "10-Q", fy, quarter)
    except Exception as e:
        agent.log(state, f"could not look up the {quarter} FY{fy} 10-Q for {gate['ticker']}: {e}")
        return None
    if not accn:
        agent.log(state, f"no 10-Q labelled {quarter} FY{fy} for {gate['ticker']}; using the 10-K")
        return None
    urls = _indexed_values(agent, "source_url", gate.get("ticker") or company)
    if any(accn.replace("-", "") in u for u in urls):
        return {"fetch_status": {**gate, "decision": "already_indexed"}}
    agent.log(state, f"fetching the {quarter} FY{fy} 10-Q for {gate['ticker']} from EDGAR")
    return _ingest(agent, state, gate, company, filing_type="10-Q", accession=accn)


def _ingest(agent, state: AgentState, gate: dict, company: str, **kw) -> dict:
    """Download and index filings for the gated company; report how it went."""
    try:
        res = agent.fetcher.fetch_and_ingest(gate["ticker"], company=gate.get("company") or "", **kw)
    except Exception as e:
        agent.log(state, f"fetch failed for {gate['ticker']}: {e}")
        if type(e).__name__ == "EmbeddingQuotaExhausted":
            agent.notice(state, "The filing could not be indexed: today's embedding quota is used up.")
        return {"fetch_status": {**gate, "status": "error", "error": str(e)}}
    if res.get("ok"):
        agent.reset_retriever()          # its company list is now out of date
    return {"fetch_status": {**gate, "status": "fetched" if res.get("ok") else "error", **res}}


def _vocab_can_match(agent, *names: Optional[str]) -> bool:
    """Can retrieval's company filter resolve any of these names? Asks the real
    filter so the two cannot drift. Answers yes if the check itself fails."""
    try:
        return any(name and (agent.retriever.infer_filter(name) or {}).get("companies")
                   for name in names)
    except Exception:
        return True


def _indexed_values(agent, field: str, ticker: str) -> set[str]:
    from finagent.vectorstore import distinct_values

    try:
        return distinct_values(agent.collection, field, where_field="ticker", where_value=ticker)
    except Exception:
        return set()


# --------------------------------------------------------------------------- #
# retrieve
# --------------------------------------------------------------------------- #

def retrieve(agent, state: AgentState) -> dict:
    """Search the filings. If the embedding quota is spent, return nothing and
    let the tool lanes (XBRL, market, web) answer."""
    from finagent.vectorstore import EmbeddingQuotaExhausted

    try:
        return {"retrieved_chunks": _search(agent, state)}
    except EmbeddingQuotaExhausted as e:
        agent.log(state, f"embedding quota exhausted, skipping filing search ({e})")
        agent.notice(state, "Filing search was skipped: today's embedding quota is used up.")
        return {"retrieved_chunks": []}


def _search(agent, state: AgentState) -> list[dict]:
    status = state.get("fetch_status") or {}
    if status.get("decision") == "fetch" and status.get("status") != "fetched":
        # The company is not in the index and the fetch failed. Searching anyway
        # would return the nearest OTHER company's filing.
        agent.log(state, "company is not in the index; skipping filing search")
        return []

    sub_queries = state.get("sub_queries") or [state["question"]]
    routes = state.get("query_routes") or []
    # Only narrative and numeric parts are answered from filings.
    if routes and len(routes) == len(sub_queries):
        filing_subs = [s for s, r in zip(sub_queries, routes) if r in ("narrative", "numeric")]
    else:
        filing_subs = sub_queries

    # What to search on: the critic's retry queries, else the rewritten query
    # (which gets the whole 8-passage budget), else the sub-queries.
    retry = state.get("retry_queries") or []
    rq = (state.get("retrieval_query") or "").strip()
    if retry:
        queries = [(q, None) for q in retry]
    elif rq and filing_subs:
        queries = [(rq, agent.retrieve_cap)]
    else:
        queries = [(q, None) for q in filing_subs]
    # Also search on the question the user actually asked: a rewritten query is
    # a narrow lookup key and drops information. Worth 3 of 99 questions.
    if queries and state["question"] not in [q for q, _ in queries]:
        slots = NARRATIVE_QUESTION_SLOTS if "narrative" in routes else QUESTION_SLOTS
        queries.append((state["question"], slots))

    seen: set = set()
    chunks: list[dict] = []
    for query, top_k in queries:
        for text, meta in agent.retriever.search(query, top_k=top_k):
            # Deduplicate on the parent id, not the text: every chunk starts
            # with the same context header.
            pid = meta.get("parent_id")
            key = (meta.get("local_path", ""), meta.get("page", ""),
                   pid if pid is not None else text[:80])
            if key in seen or _mostly_non_english(text):
                continue
            seen.add(key)
            chunks.append({
                "text": text,
                "company": meta.get("company") or meta.get("ticker", "?"),
                "ticker": meta.get("ticker", ""),
                "year": meta.get("year", "?"),
                "source_url": meta.get("source_url", ""),
                "source": citation_tag(meta),
                "sub_query": query,
            })
    if retry:
        # A retry adds to the first search's passages instead of replacing them,
        # so the claims that were supported keep their evidence. The cap below
        # keeps each query's best passage, old and new, then the best by score.
        fresh = {c["text"] for c in chunks}
        chunks = [c for c in state.get("retrieved_chunks") or [] if c["text"] not in fresh] + chunks
    return _cap_pool(agent, state, chunks)


def citation_tag(meta: dict) -> str:
    """"[Apple 10-K 2024]". The year is the report's, not the figure's: a 2024
    report also carries 2023 figures."""
    form = str(meta.get("filing_type") or "10-K")
    form = {"10k": "10-K", "10q": "10-Q", "10k_annualreport": "10-K",
            "annual_report": "10-K"}.get(form.lower(), form)
    return f"[{meta.get('company') or meta.get('ticker', '?')} {form} {meta.get('year', '?')}]"


def _cap_pool(agent, state: AgentState, chunks: list[dict]) -> list[dict]:
    """Keep the `retrieve_cap` best passages, scored against the original question.

    Each query's best passage is kept first, so on a comparison question one
    company cannot crowd the other out. The rest is filled by score.
    """
    cap = agent.retrieve_cap
    if len(chunks) <= cap:
        return _fit_budget(agent, state, chunks)
    try:
        from finagent.retrieval.reranker import get_reranker

        scores = get_reranker(agent.reranker_model).predict(
            [(state["question"], c["text"]) for c in chunks])
    except Exception as e:
        agent.log(state, f"final rerank failed ({e}); keeping the first {cap}")
        return _fit_budget(agent, state, chunks[:cap])
    ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])
    kept: list[int] = []
    seen_queries: set = set()
    for i in ranked:
        if len(kept) >= cap:
            break
        if chunks[i]["sub_query"] not in seen_queries:
            seen_queries.add(chunks[i]["sub_query"])
            kept.append(i)
    for i in ranked:
        if len(kept) >= cap:
            break
        if i not in kept:
            kept.append(i)
    return _fit_budget(agent, state, [chunks[i] for i in sorted(kept, key=ranked.index)])


def _fit_budget(agent, state: AgentState, kept: list[dict]) -> list[dict]:
    """On the free Groq writer, trim the evidence so the request stays under
    Groq's 8,000-token limit. Drops whole passages from the bottom of the ranking."""
    budget = current_context().evidence_budget()
    if budget is None:
        return kept
    out, used = [], 0
    for c in kept:
        if used + len(c["text"]) <= budget:
            out.append(c)
            used += len(c["text"])
        elif not out:                       # the best passage alone is too long
            out.append({**c, "text": c["text"][:budget]})
            used = budget
    if len(out) < len(kept):
        agent.notice(state, f"Evidence was trimmed from {len(kept)} to {len(out)} passages "
                            f"to fit the free Groq model's request limit.")
    return out
