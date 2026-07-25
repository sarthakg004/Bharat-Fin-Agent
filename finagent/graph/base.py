"""
base.py  ·  finagent/graph/base.py

Week-2 agentic RAG built on LangGraph, with LangSmith tracing.

Pipeline (StateGraph):

    START → planner → retrieve → synthesize → critic → END

    planner     decomposes the question into 1-3 sub-queries (structured output)
    retrieve    similarity-search each sub-query, merge + dedupe
    synthesize  strong LLM writes the final answer with inline citations
    critic      checks each claim against the context (logs failures; no loop yet)

This is the same retrieve→generate idea as the naive baseline, but every step is
an observable LangGraph node and the planner/critic add multi-hop handling and a
hallucination check.

LangSmith
---------
Set these in .env (no code changes needed — LangChain auto-traces):

    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=ls__...
    LANGCHAIN_PROJECT=finagent-week2     # optional

Usage as a library
------------------
    from finagent.graph.base import AgenticRAG

    agent = AgenticRAG()  # collection defaults to settings.us_collection
    result = agent.run("Compare Microsoft and Apple revenue growth in FY23.")
    print(result["final_answer"])
    print(result["citations"])

CLI
---
    python -m finagent.graph.base \\
        --collection us_filings \\
        --question "Compare Microsoft and Apple revenue growth in FY23."

    # Batch over an eval set and append a row to results/comparison.csv:
    python -m finagent.graph.base \\
        --collection us_filings \\
        --dataset data/us/eval/financebench/data/financebench_open_source.jsonl \\
        --sample 100
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv
from tqdm import tqdm

from finagent.graph.state import AgentState, CriticReport, SubQueries
from finagent.runtime import DEFAULTS as RUNTIME_DEFAULTS
from finagent.runtime import RuntimeContext, create_llm, current_context
from finagent.vectorstore import DEFAULT_EMBED_MODEL
from finagent.config import settings

load_dotenv()


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

from finagent.prompts.critic import CRITIC_PROMPT, CRITIC_SYSTEM  # noqa: F401
from finagent.prompts.planner import PLANNER_PROMPT  # noqa: F401
from finagent.prompts.synthesizer import (  # noqa: F401
    SYNTHESIZER_PROMPT,
    SYNTHESIZER_SYSTEM,
)


class AgenticRAG:
    """LangGraph agentic RAG: planner → retrieve → synthesize → critic.

    Parameters
    ----------
    collection_name : str
        Qdrant collection (defaults to settings.us_collection).
    embedding_model : str
        MUST match the model used at ingestion.
    top_k : int
        Chunks retrieved per sub-query. Fixed at construction — a request
        cannot change how much evidence the agent gathers. (v4 supersedes this
        with `final_top_k`, which the ephemeral fetch/upload lane reuses so
        both lanes keep the same number of chunks per sub-query.)

    The agent holds RESOURCES only and is immutable after construction, so one
    instance is safe to share across concurrent requests. Which provider, model
    and API key to use is per-request: pass a `finagent.runtime.RuntimeContext`
    to `run()`, or let the API layer put one in the LangGraph config.
    """

    # Per-provider model defaults now live in `finagent.runtime` (the agent is
    # provider-agnostic); re-exported here because callers still read
    # `AgenticRAG.DEFAULTS`.
    DEFAULTS = RUNTIME_DEFAULTS

    def __init__(
        self,
        collection_name: str = settings.us_collection,
        embedding_model: str = DEFAULT_EMBED_MODEL,
        top_k: int = 5,
        collections: Optional[list[str]] = None,
    ):
        # Everything on the agent is a long-lived, shared RESOURCE and is read
        # -only after construction. Which provider/model/key to use is per
        # -request and lives in `finagent.runtime.RuntimeContext`, delivered
        # through the LangGraph config — see `_get_llm`.
        self.collection_name = collection_name
        # Filings collections to retrieve over. Defaults to the single
        # `collection_name`; pass several to let the agent pull from all of
        # them and rerank — the question decides what's relevant.
        self.collections = collections or [collection_name]
        self.embedding_model = embedding_model
        self.top_k = top_k

        # Lazy resources.
        self._retriever = None
        self._graph = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, question: str, ctx: Optional[RuntimeContext] = None) -> AgentState:
        """Run the full graph on one question. Returns the final state dict.

        `ctx` carries the per-request LLM configuration (provider/model/key).
        It is injected here rather than held on the agent so one shared agent
        can serve concurrent requests with different providers. None → defaults.
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

    def run_dataset(
        self,
        df,
        output_path: Union[str, Path] = "results/agentic_rag_outputs.json",
        question_col: str = "question",
        delay_between: float = 0.5,
    ) -> list[dict]:
        """Run the graph over a DataFrame, saving incrementally (resumable).

        Output records are shaped like the naive baseline's so the same
        RAGASEvaluator can score them (question, answer, retrieved_chunks, ...).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results: list[dict] = []
        answered: set[str] = set()
        if output_path.exists():
            results = json.load(open(output_path))
            answered = {r["question"] for r in results}
            print(f"Resuming: {len(answered)} already answered")

        rows = df.to_dict(orient="records")
        todo = [r for r in rows if str(r[question_col]) not in answered]
        print(f"Questions to process: {len(todo)}")

        for row in tqdm(todo, desc="agentic_rag"):
            question = str(row[question_col])
            try:
                state = self.run(question)
                rec = {
                    "question": question,
                    "answer": state.get("final_answer", ""),
                    "retrieved_chunks": [c["text"] for c in state.get("retrieved_chunks", [])],
                    "sub_queries": state.get("sub_queries", []),
                    "citations": state.get("citations", []),
                    "grading_score": state.get("grading_score"),
                    "needs_retry": state.get("needs_retry"),
                    "errors": state.get("errors", []),
                }
                for k, v in row.items():
                    rec.setdefault(k, v)
                results.append(rec)
            except Exception as e:
                print(f"\n  ! failed: {question[:60]}... — {e}")
                results.append(
                    {"question": question, "answer": "", "retrieved_chunks": [], "error": str(e)}
                )

            json.dump(results, open(output_path, "w"), indent=2, default=str)
            time.sleep(delay_between)

        print(f"\nSaved {len(results)} results → {output_path}")
        return results

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

    def planner_node(self, state: AgentState) -> dict:
        """Decompose the question into 1-3 sub-queries (structured output).

        If `chat_history` is present, prepend a compact transcript so the
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

    def retrieve_node(self, state: AgentState) -> dict:
        """Retrieve top-k chunks per sub-query, merge and dedupe."""
        retriever = self._get_retriever()
        seen: set[str] = set()
        chunks: list[dict] = []
        for sub_q in state.get("sub_queries") or [state["question"]]:
            for doc in retriever.similarity_search(sub_q, k=self.top_k):
                key = f"{doc.metadata.get('local_path','')}:{doc.metadata.get('page','')}:{doc.page_content[:80]}"
                if key in seen:
                    continue
                seen.add(key)
                m = doc.metadata
                chunks.append(
                    {
                        "text": doc.page_content,
                        "company": m.get("company") or m.get("ticker", "?"),
                        "year": m.get("year", "?"),
                        "page": m.get("page", "?"),
                        "source": self._citation_tag(m),
                        "sub_query": sub_q,
                    }
                )
        return {"retrieved_chunks": chunks}

    def synthesize_node(self, state: AgentState) -> dict:
        """Strong LLM writes the final answer with [N]-style inline citations."""
        from langchain_core.messages import HumanMessage, SystemMessage

        chunks = state.get("retrieved_chunks", [])
        # 1-based numbering aligns with the citation IDs the user sees in the
        # sidebar; the LLM is told to cite by these numbers.
        context = "\n\n".join(
            f"[{i + 1}] {c.get('source','')}\n{c['text']}"
            for i, c in enumerate(chunks)
        ) or "No context retrieved."
        sub_queries = "\n".join(f"- {q}" for q in state.get("sub_queries", []))

        llm = self._get_llm("synth")
        prompt = SYNTHESIZER_PROMPT.format(
            question=state["question"], sub_queries=sub_queries, context=context
        )
        response = llm.invoke(
            [SystemMessage(content=SYNTHESIZER_SYSTEM), HumanMessage(content=prompt)]
        )
        from finagent.llm import text_of
        answer = text_of(response)
        citations = sorted(set(re.findall(r"\[[^\]]+\]", answer)))
        return {
            "draft_answer": answer,
            "final_answer": answer,
            "citations": citations,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    def _critic_system(self) -> str:
        """The critic's system prompt. Subclasses override to swap in the
        analyst-voice critic (Phase 9) behind a flag."""
        return CRITIC_SYSTEM

    def critic_node(self, state: AgentState) -> dict:
        """Check each claim in the draft against the context. Logs failures.

        Day-6 scope: we score and flag but do NOT loop yet. The retry edge is
        wired in a later week.
        """
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
            return {"grading_score": None, "needs_retry": False}

        if not verdicts:
            return {"grading_score": None, "needs_retry": False}

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
        }

    # ------------------------------------------------------------------ #
    # Graph
    # ------------------------------------------------------------------ #

    @property
    def graph(self):
        """Compiled LangGraph app (built once, then cached)."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(AgentState)
        builder.add_node("planner", self.planner_node)
        builder.add_node("retrieve", self.retrieve_node)
        builder.add_node("synthesize", self.synthesize_node)
        builder.add_node("critic", self.critic_node)

        builder.add_edge(START, "planner")
        builder.add_edge("planner", "retrieve")
        builder.add_edge("retrieve", "synthesize")
        builder.add_edge("synthesize", "critic")
        builder.add_edge("critic", END)

        return builder.compile()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

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

    def _get_retriever(self):
        if self._retriever is None:
            from finagent.vectorstore import build_store

            self._retriever = build_store(
                self.collection_name, self.embedding_model
            )
        return self._retriever

    def _get_llm(self, role: str):
        """The LLM for a role, built from the RUNNING REQUEST's context.

        Every LLM in the graph comes through here. Nothing is cached on the
        agent: the client is cheap next to the network call it wraps, and a
        cache keyed on a shared agent is what let one request's API key serve
        the next one's question.
        """
        return create_llm(current_context(), role)


# --------------------------------------------------------------------------- #
# Comparison table helper (Day 7)
# --------------------------------------------------------------------------- #

def append_comparison_row(
    config_name: str,
    metrics: dict,
    csv_path: Union[str, Path] = "results/comparison.csv",
) -> None:
    """Append/update one configuration's mean metrics in results/comparison.csv.

    `metrics` keys are metric names (faithfulness, answer_relevancy, ...).
    Re-running with the same config_name overwrites that row.
    """
    import pandas as pd

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"configuration": config_name, **{k: round(float(v), 4) for k, v in metrics.items()}}

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["configuration"] != config_name]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} with row '{config_name}'")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the Week-2 agentic RAG graph.")
    p.add_argument("--collection", default=settings.us_collection)
    p.add_argument(
        "--provider",
        choices=["groq", "gemini", "openai", "anthropic"],
        default="groq",
    )
    p.add_argument("--planner-model", default=None)
    p.add_argument("--synth-model", default=None)
    p.add_argument("--critic-model", default=None)
    p.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--question", default=None, help="Single question to answer")
    p.add_argument("--dataset", default=None, help="JSONL/parquet eval set for a batch run")
    p.add_argument("--question-col", default="question")
    p.add_argument("--sample", type=int, default=None, help="Limit batch to first N rows")
    p.add_argument("--output", default="results/agentic_rag_outputs.json")
    return p


def _load_dataset(path: str, question_col: str):
    import pandas as pd

    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith((".jsonl", ".json")):
        return pd.read_json(path, lines=path.endswith(".jsonl"))
    raise ValueError(f"Unsupported dataset format: {path}")


def main():
    args = _build_cli().parse_args()

    # LLM choice is per-run, not a property of the agent — build the context
    # from the CLI flags and inject it at run().
    ctx = RuntimeContext(provider=args.provider, synth_model=args.synth_model,
                         top_k=args.top_k)

    agent = AgenticRAG(
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
    )

    if args.dataset:
        df = _load_dataset(args.dataset, args.question_col)
        if args.sample:
            df = df.head(args.sample)
        agent.run_dataset(df, output_path=args.output, question_col=args.question_col)
        return

    if not args.question:
        raise SystemExit("Provide --question or --dataset.")

    state = agent.run(args.question, ctx)
    print("\n" + "=" * 60)
    print(f"Question:    {state['question']}")
    print(f"Sub-queries: {state.get('sub_queries')}")
    print(f"\nAnswer:\n{state.get('final_answer')}")
    print(f"\nCitations:   {state.get('citations')}")
    print(f"Grade:       {state.get('grading_score')}  needs_retry={state.get('needs_retry')}")
    if state.get("errors"):
        print(f"Errors:      {state['errors']}")


if __name__ == "__main__":
    main()
