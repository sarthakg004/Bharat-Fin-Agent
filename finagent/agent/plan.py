"""Step 1: plan. Split the question into routed parts, then write the one
keyword query that searches the filings."""

from __future__ import annotations

import re
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

from finagent.agent.prompts import (PLAN_PROMPT, PLAN_SYSTEM, RETRIEVAL_QUERY_SHOTS,
                                    RETRIEVAL_QUERY_SYSTEM, history_block)
from finagent.agent.state import AgentState, QueryPlan
from finagent.llm import classify_error, text_of

# A real search query has at least company + year + a caption. Shorter output
# means the model answered the question instead ("Approximately 1.33.").
MIN_RETRIEVAL_QUERY_WORDS = 6

# The model sometimes refuses at length instead of writing a query. The prompt
# asks for keywords only, so first-person prose is never a valid query.
_NOT_A_QUERY = re.compile(
    r"\b(i am|i'm|i cannot|i can't|i am not|i'm not|unable to|not able to|"
    r"as an ai|sorry|apolog|there is no|cannot determine|insufficient "
    r"information|does not (?:provide|contain))\b", re.I)


def _first_line(text: str) -> str:
    """The first non-empty line, without the quotes and labels models add."""
    for line in (text or "").splitlines():
        line = line.strip().strip('"').strip("`").strip()
        for prefix in ("query:", "search query:", "rewrite:"):
            if line.lower().startswith(prefix):
                line = line[len(prefix):].strip()
        if line:
            return line
    return ""


def planner(agent, state: AgentState) -> dict:
    """One structured call returns the sub-queries and a lane for each.

    If the model's output cannot be parsed, the question itself becomes the
    single sub-query. If the provider is down, the error goes to the user.
    """
    question = state["question"]
    context = history_block(state.get("chat_history"))
    try:
        out: QueryPlan = agent.llm("planner").with_structured_output(QueryPlan).invoke([
            SystemMessage(content=PLAN_SYSTEM),
            HumanMessage(content=context + PLAN_PROMPT.format(
                question=question, today=date.today().isoformat())),
        ])
        pairs = [(q.query.strip(), q.route)
                 for q in (out.queries or []) if (q.query or "").strip()][:8]
    except Exception as e:
        if classify_error(e).kind != "other":
            raise
        agent.log(state, f"planner output unusable ({e}); searching on the question")
        pairs = []
    pairs = pairs or [(question, "narrative")]
    routes = [r for _, r in pairs]
    return {"sub_queries": [q for q, _ in pairs],
            "query_routes": routes,
            "retrieval_query": retrieval_query(agent, state, question, routes, context)}


def retrieval_query(agent, state: AgentState, question: str, routes: list[str],
                    context: str = "") -> str:
    """Write the one query that searches the filings.

    A separate call from the plan: the plan is written in prose, this is terse
    keywords. Skipped when the question has no narrative part. Returns "" on any
    failure, which makes retrieval search on the sub-queries instead.
    """
    if "narrative" not in routes:       # only narrative parts go through filing search
        return ""
    # The examples go inside one user message. Sent as separate chat turns,
    # Qwen returned an empty reply for most questions.
    examples = "\n\n".join(f"Question: {q}\nQuery: {a}" for q, a in RETRIEVAL_QUERY_SHOTS)
    msgs = [SystemMessage(content=RETRIEVAL_QUERY_SYSTEM),
            HumanMessage(content=f"Examples:\n\n{examples}\n\n{context}Question: {question}\nQuery:")]
    try:
        out = _first_line(text_of(agent.llm("planner").invoke(msgs)))
    except Exception as e:
        agent.llm_failed(state, "Search-query rewrite", e)
        return ""
    if len(out.split()) < MIN_RETRIEVAL_QUERY_WORDS or _NOT_A_QUERY.search(out):
        agent.log(state, f"discarded a non-query rewrite ({out[:60]!r})")
        return ""
    return out
