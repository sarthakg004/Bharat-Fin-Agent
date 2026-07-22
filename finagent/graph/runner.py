"""
runner.py  ·  finagent/graph/runner.py

Run the production agent and capture a *readable trace* of what each node did,
for the notebook's end-to-end walkthrough. Nothing here re-implements the agent —
it drives the compiled graph (`builder.build_graph`) and formats the result.

    run_with_trace(question)        — stream the full graph, capture each node.
    run_with_route_trace(question)  — just the fused plan+route call (cheap), show routing.
    run_with_full_trace(question)   — full run, returns decomposition + answer.
    format_trace(result)            — pretty multi-section string for printing.
"""

from __future__ import annotations


from finagent.graph.builder import build_agent, build_graph


def run_with_trace(
    question: str, provider: str = "groq", recursion_limit: int = 50
) -> dict:
    """Stream the full graph; return per-node deltas + final state.

    Returns ``{"question", "steps": [(node, delta_dict), ...], "final_answer",
    "state"}``.
    """
    graph = build_graph(provider=provider)
    initial = {
        "question": question, "iteration_count": 0, "errors": [],
        "table_results": [], "web_results": [],
    }
    steps, state = [], {}
    for mode, data in graph.stream(
        initial, stream_mode=["updates", "values"], config={"recursion_limit": recursion_limit}
    ):
        if mode == "values":
            state = data
        elif mode == "updates" and isinstance(data, dict):
            for node, delta in data.items():
                steps.append((node, delta or {}))
    return {
        "question": question,
        "steps": steps,
        "final_answer": state.get("final_answer", ""),
        "state": state,
    }


def run_with_route_trace(
    question: str, provider: str = "groq"
) -> dict:
    """Run only the fused plan+route step (ONE LLM call; the router is a
    no-op when the plan already carries routes) and return the routing decision."""
    agent = build_agent(provider=provider)
    state = {"question": question, "iteration_count": 0, "errors": []}
    state.update(agent.planner_node(state) or {})
    state.update(agent.router_node(state) or {})
    routes = state.get("query_routes", []) or []
    return {
        "question": question,
        "sub_queries": state.get("sub_queries", []),
        "routes": routes,
        "decision": ", ".join(sorted(set(routes))) if routes else "retrieve",
        "reasoning": f"{len(state.get('sub_queries', []))} sub-query(ies); "
                     f"routes={routes}",
    }


def run_with_full_trace(
    question: str, provider: str = "groq"
) -> dict:
    """Full run; return decomposition, routes, and the final cited answer."""
    res = run_with_trace(question, provider=provider)
    st = res["state"]
    res["sub_queries"] = st.get("sub_queries", [])
    res["routes"] = st.get("query_routes", [])
    res["citations"] = st.get("citations", [])
    return res


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #

def _summarize_delta(node: str, delta: dict) -> str:
    """One short, human-readable line per node's state change."""
    if node == "planner":
        return "sub-queries: " + "; ".join(delta.get("sub_queries", []) or [])
    if node == "router":
        return "routes: " + ", ".join(delta.get("query_routes", []) or [])
    if node in ("retrieve",):
        return f"retrieved {len(delta.get('retrieved_chunks', []) or [])} chunks"
    if node == "grader":
        return f"avg_grade={delta.get('avg_grade')}  kept {len(delta.get('retrieved_chunks', []) or [])} chunks"
    if node == "rewrite":
        rw = delta.get("sub_queries", []) or []
        return "rewritten: " + "; ".join(rw)
    if node in ("table_agent", "market_data", "web_search"):
        key = {"table_agent": "table_results", "market_data": "market_data",
               "web_search": "web_results"}[node]
        return f"{len(delta.get(key, []) or [])} result(s)"
    if node == "synthesize":
        return (delta.get("draft_answer", "") or "")[:200].replace("\n", " ") + "…"
    if node == "critic":
        return f"grading_score={delta.get('grading_score')}  needs_retry={delta.get('needs_retry')}"
    if node == "verify_numbers":
        nv = delta.get("numeric_verification") or {}
        return f"verification score={nv.get('score')}  unverified={len(nv.get('unverified', []) or [])}"
    if node == "refuse":
        return "refused — figures could not be grounded"
    return ", ".join(f"{k}" for k in delta) or "(no state change)"


def format_trace(result: dict) -> str:
    """Render a `run_with_trace` result as a sectioned, readable string."""
    lines = [f"Question: {result['question']}", "=" * 60]
    for node, delta in result["steps"]:
        title = node.replace("_", " ").title()
        bar = "─" * max(3, 56 - len(title))
        lines.append(f"\n── {title} {bar}")
        lines.append("  " + _summarize_delta(node, delta))
    lines.append("\n" + "=" * 60)
    lines.append("Final Answer:\n" + (result.get("final_answer") or "(none)"))
    return "\n".join(lines)
