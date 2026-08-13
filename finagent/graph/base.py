"""Base layer of the agent: the planner and critic nodes, plus `run()`.

Later layers override or wrap these (`corrective.py` → `full.py` → `agent.py`);
`agent.py`'s `AgenticRAGv4` is the class the API actually serves. Nothing here
is instantiated on its own — it is reached through `super()` from those layers.
"""

from __future__ import annotations

from typing import Optional

from dotenv import load_dotenv

from finagent.graph.state import AgentState, CriticReport, SubQueries
from finagent.runtime import DEFAULTS as RUNTIME_DEFAULTS
from finagent.runtime import RuntimeContext, create_llm, current_context
from finagent.vectorstore import DEFAULT_EMBED_MODEL
from finagent.config import settings
from finagent.prompts.critic import CRITIC_PROMPT, CRITIC_SYSTEM
from finagent.prompts.planner import PLANNER_PROMPT

load_dotenv()


class AgenticRAG:
    """Planner and critic over a Qdrant collection.

    Holds RESOURCES only and is immutable after construction, so one instance
    is safe to share across concurrent requests. Which provider, model and API
    key to use is per-request: pass a `RuntimeContext` to `run()`, or let the
    API layer put one in the LangGraph config.
    """

    # Re-exported from `finagent.runtime` because callers still read
    # `AgenticRAG.DEFAULTS` (see research/orchestrator.py).
    DEFAULTS = RUNTIME_DEFAULTS

    def __init__(
        self,
        collection_name: str = settings.us_collection,
        embedding_model: str = DEFAULT_EMBED_MODEL,
        collections: Optional[list[str]] = None,
    ):
        self.collection_name = collection_name
        # Filings collections to retrieve over. Defaults to the single
        # `collection_name`; pass several to pull from all of them and rerank.
        self.collections = collections or [collection_name]
        self.embedding_model = embedding_model
        self._graph = None

    def run(self, question: str, ctx: Optional[RuntimeContext] = None) -> AgentState:
        """Run the graph on one question and return the final state.

        `ctx` is injected here rather than held on the agent so one shared
        agent can serve concurrent requests with different providers.
        """
        initial: AgentState = {
            "question": question,
            "iteration_count": 0,
            "errors": [],
            "web_results": [],
        }
        return self.graph.invoke(
            initial,
            config={"configurable": {"runtime_context": ctx or RuntimeContext()}},
        )

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

    def planner_node(self, state: AgentState) -> dict:
        """Decompose the question into 1-8 sub-queries.

        Prepends a compact transcript when `chat_history` is present, so the
        planner can resolve follow-ups like "what about the previous year?".
        """
        question = state["question"]
        history = state.get("chat_history") or []
        history_block = ""
        if history:
            lines = [
                f"{('User' if t.get('role')=='user' else 'Assistant')}: "
                f"{(t.get('content') or '')[:400]}"
                for t in history[-6:]
            ]
            history_block = (
                "Recent conversation (most recent last):\n"
                + "\n".join(lines) + "\n\n"
            )

        from datetime import date

        llm = self._get_llm("planner").with_structured_output(SubQueries)
        try:
            out: SubQueries = llm.invoke(
                history_block + PLANNER_PROMPT.format(
                    question=question, today=date.today().isoformat())
            )
            queries = [q.strip() for q in out.queries if q.strip()][:8]
        except Exception as e:
            queries = []
            self._log(state, f"planner failed ({e}); falling back to original question")
        if not queries:
            queries = [question]
        return {"sub_queries": queries}

    def _critic_system(self) -> str:
        """The critic's system prompt; subclasses swap in the analyst voice."""
        return CRITIC_SYSTEM

    def critic_node(self, state: AgentState) -> dict:
        """Score each claim in the draft against the retrieved context."""
        from langchain_core.messages import HumanMessage, SystemMessage

        answer = state.get("draft_answer", "")
        chunks = state.get("retrieved_chunks", [])
        context = "\n\n".join(f"{c['source']}\n{c['text']}" for c in chunks) or "No context."

        llm = self._get_llm("critic").with_structured_output(CriticReport)
        prompt = CRITIC_PROMPT.format(context=context, answer=answer)
        try:
            report: CriticReport = llm.invoke(
                [SystemMessage(content=self._critic_system()), HumanMessage(content=prompt)]
            )
            verdicts = report.verdicts
        except Exception as e:
            self._log(state, f"critic failed ({e})")
            # `critic_feedback` must be CLEARED, not left alone. A previous
            # pass's claims surviving a failed critic would be re-injected into
            # the next re-draft prompt as if they were fresh findings.
            return {"grading_score": None, "needs_retry": False,
                    "critic_feedback": []}

        if not verdicts:
            return {"grading_score": None, "needs_retry": False,
                    "critic_feedback": []}

        supported = sum(1 for v in verdicts if v.supported)
        score = supported / len(verdicts) if verdicts else None
        unsupported = [v.claim for v in verdicts if not v.supported]

        errors = list(state.get("errors", []))
        for claim in unsupported:
            errors.append(f"unsupported claim: {claim}")
        return {
            "grading_score": round(score, 3),
            "needs_retry": bool(unsupported),
            "errors": errors,
            # The specific claims the critic could not support — fed to a focused
            # re-draft (active-critic recovery) so the synth fixes/drops them.
            "critic_feedback": list(unsupported),
            # …and the critic's own verdict on WHICH recovery would fix them.
            # It just read the draft against the evidence, so it is better
            # placed than a downstream heuristic to tell "the answer overstated
            # what we have" from "we never had this fact".
            "critic_remedy": report.remedy,
        }

    # ------------------------------------------------------------------ #
    # Graph + helpers
    # ------------------------------------------------------------------ #

    @property
    def graph(self):
        """Compiled LangGraph app (built once, then cached)."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _citation_tag(self, meta: dict) -> str:
        """Build a citation tag like '[Apple 10-K 2024, p. 102]'.

        The year is the *report's* year (which document the text came from), NOT
        the fiscal year of any figure inside it. Annual reports carry prior-year
        comparatives, so an FY23 number can legitimately appear in the 2024
        report — the tag points to the source document, and the answer text
        states which fiscal year a figure refers to.
        """
        name = meta.get("company") or meta.get("ticker", "?")
        year = meta.get("year", "?")
        return f"[{name} 10-K {year}, p. {meta.get('page', '?')}]"

    @staticmethod
    def _log(state: AgentState, msg: str) -> None:
        state.setdefault("errors", []).append(msg)

    def _get_llm(self, role: str):
        """The LLM for a role, built from the RUNNING REQUEST's context.

        Every LLM in the graph comes through here. Nothing is cached on the
        agent: the client is cheap next to the network call it wraps, and a
        cache keyed on a shared agent is what let one request's API key serve
        the next one's question.
        """
        return create_llm(current_context(), role)
