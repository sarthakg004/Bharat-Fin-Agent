"""
corrective.py  ·  finagent/graph/corrective.py

Corrective RAG built on top of the v1 agentic graph. Two upgrades:

1. **Hybrid retrieval + cross-encoder rerank**
       BM25 top-k  ⋃  dense top-k  ──►  rerank  ──►  top-N

2. **Self-correcting control flow**

       START → planner → retrieve → grader ──── conditional ────┐
                          ▲                                     │
                          │                                     ▼
                          └── rewrite ◄── (avg_grade below threshold,
                                            iteration_count < cap)
                                                                │
                                                                ▼
                                        synthesize → critic ─── conditional
                                                       │              │
                                                       └── retrieve ◄─┘
                                                  (needs_retry,
                                                   critic_iterations < cap)
                                                       │
                                                       ▼
                                                      END

The grader (a fast LLM, structured output) scores each retrieved chunk 1-5;
if the mean is below threshold the rewriter reformulates the question and we
re-retrieve. After synthesis, the critic checks every claim against the
context; if it fails, we go back to retrieve with the failing claims as
focused sub-queries. Both loops are capped to avoid spinning forever.

Usage as a library
------------------
    from finagent.graph.corrective import AgenticRAGv2

    agent = AgenticRAGv2(collection_name="india_filings", market="india")
    state = agent.run("Compare Reliance and Tata Motors revenue in FY23.")
    print(state["final_answer"])
    print("avg_grade:", state["avg_grade"], "low_conf:", state.get("low_confidence"))

CLI
---
    python -m finagent.graph.corrective \\
        --collection india_filings --market india \\
        --question "Compare Reliance and Tata Motors revenue in FY23."
"""

from __future__ import annotations

import argparse
from typing import Optional, Union

from finagent.graph.base import AgenticRAG, append_comparison_row
from finagent.graph.state import (
    AgentState,
    ChunkScore,
    CriticReport,
    GraderReport,
    RewrittenQuery,
    SubQueries,
)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

GRADER_PROMPT = """\
Question: {question}

For each excerpt below, give a relevance score:
  1 = irrelevant — wrong company / wrong topic
  2 = tangentially related — same broad topic but different ENTITY than the
      question asks about (e.g. question is about ServiceNow, excerpt is a
      Lockheed Martin 10-K discussing "U.S. DoD revenue")
  3 = related but does not directly answer
  4 = partially answers
  5 = directly answers the question

Be strict about ENTITY match. If the question asks about Company X but the
excerpt is a filing from Company Y, score it 1 or 2 even when both companies
operate in the same industry or discuss the same topic (defense contracts,
cloud revenue, etc.). Score 4 or 5 only when the excerpt is from / about the
same entity AND addresses the specific question.

Return one score per excerpt, in the same order.

Excerpts:
{excerpts}
"""

REWRITE_PROMPT = """\
The question below did not retrieve enough relevant context. Reformulate it
to improve retrieval. Add domain-specific synonyms or financial terminology
where appropriate (e.g. expand "earnings" to "net profit attributable to
shareholders"). Keep the rewrite self-contained — no pronouns referring back
to the original.

Original question: {question}

Previous rewrites that did not work well:
{history}

Return only the rewritten question.
"""


# --------------------------------------------------------------------------- #
# Hybrid retriever + reranker — MOVED to finagent.retrieval during the layout
# restructure. Re-exported here so existing import paths keep working, e.g.:
#     from finagent.graph.corrective import HybridRetriever, _get_shared_reranker
# Remove this shim once every consumer imports from finagent.retrieval directly.
# --------------------------------------------------------------------------- #
from finagent.retrieval.hybrid import HybridRetriever  # noqa: E402,F401
from finagent.retrieval.reranker import (  # noqa: E402,F401
    CrossEncoderReranker,
    _get_shared_reranker,
    _SHARED_RERANKERS,
)


# --------------------------------------------------------------------------- #
# AgenticRAGv2
# --------------------------------------------------------------------------- #

class AgenticRAGv2(AgenticRAG):
    """Corrective-RAG agent: hybrid retrieval + grader + rewriter + critic loop.

    Subclasses AgenticRAG and only overrides what changes:
      * `_get_hybrids()` builds the hybrid retrievers (one per collection)
      * `hybrid_retrieve_node` replaces `retrieve_node`
      * `grader_node` and `rewrite_node` are new
      * `critic_node` is extended to schedule a retrieve-loop on failure
      * `synthesize_node` adds the `low_confidence` flag
      * `_build_graph` wires the conditional edges
    """

    def __init__(
        self,
        *args,
        reranker_model: str = HybridRetriever.DEFAULT_RERANKER,
        bm25_top_k: int = 10,
        dense_top_k: int = 10,
        final_top_k: int = 5,
        grader_model: Optional[str] = None,
        grade_threshold: float = 3.0,
        min_keep_grade: int = 4,
        very_poor_grade: float = 2.5,
        max_rewrites: int = 3,
        max_critic_retries: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reranker_model = reranker_model
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.final_top_k = final_top_k
        # Grader is fast & cheap — default to the planner-tier model.
        self.grader_model = grader_model or self.planner_model
        self.grade_threshold = grade_threshold
        # Chunks whose per-chunk grade falls below `min_keep_grade` are dropped
        # before synthesis. Default 4 means we keep only "partially answers" (4)
        # or "directly answers" (5) — this is what prevents off-entity noise
        # from polluting the synth (e.g. LMT/BA chunks on a ServiceNow query).
        self.min_keep_grade = min_keep_grade
        # If the AVERAGE chunk grade is below this AND the post-filter list is
        # empty, rewriting won't help (the corpus genuinely lacks the entity)
        # — short-circuit straight to the next stage.
        self.very_poor_grade = very_poor_grade
        self.max_rewrites = max_rewrites
        self.max_critic_retries = max_critic_retries
        self._hybrids: Optional[list[HybridRetriever]] = None

    # ------------------------------------------------------------------ #
    # Resources
    # ------------------------------------------------------------------ #

    def _get_hybrids(self) -> list[HybridRetriever]:
        """One hybrid retriever per filings collection (BM25 + dense + rerank).
        Retrieving over several collections and letting the grader/reranker sort
        it out is how the agent serves both US + India without a market toggle."""
        if self._hybrids is None:
            from finagent.vectorstore import build_store

            self._hybrids = [
                HybridRetriever(
                    build_store(c, self.embedding_model, self.chroma_dir),
                    reranker_model=self.reranker_model,
                    bm25_top_k=self.bm25_top_k,
                    dense_top_k=self.dense_top_k,
                    final_top_k=self.final_top_k,
                )
                for c in self.collections
            ]
        return self._hybrids

    def _get_grader_llm(self):
        """A fourth LLM role — reuses the role cache from the base class."""
        if "grader" not in self._llms:
            from finagent.llm import build_llm

            self._llms["grader"] = build_llm(
                self.provider, self.grader_model, self.api_key, temperature=0.0
            )
        return self._llms["grader"]

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

    def hybrid_retrieve_node(self, state: AgentState) -> dict:
        """BM25 + dense + cross-encoder rerank, per sub-query, across every
        configured collection, merged/deduped. The grader then drops chunks
        that aren't relevant to the question (e.g. the other market)."""
        hybrids = self._get_hybrids()
        seen: set = set()
        chunks: list[dict] = []
        for sub_q in state.get("sub_queries") or [state["question"]]:
            for hyb in hybrids:
                for text, meta in hyb.search(sub_q):
                    key = (meta.get("local_path", ""), meta.get("page", ""), text[:80])
                    if key in seen:
                        continue
                    seen.add(key)
                    chunks.append({
                        "text": text,
                        "company": meta.get("company") or meta.get("ticker", "?"),
                        "year": meta.get("year", "?"),
                        "page": meta.get("page", "?"),
                        "source": self._citation_tag(meta),
                        "sub_query": sub_q,
                    })
        return {"retrieved_chunks": chunks}

    def grader_node(self, state: AgentState) -> dict:
        """Score each chunk's relevance (1-5) AND drop the low-graded ones.

        Off-entity noise is the main failure mode for questions about
        companies not in the corpus (the retriever happily returns
        topically-similar chunks from a different company). The filter step
        below removes them before the synthesizer ever sees them.

        The `avg_grade` is computed on the FULL set so the rewrite-loop
        router still gets a faithful "how good was retrieval overall?" signal.
        """
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return {"grades": [], "avg_grade": 0.0, "retrieved_chunks": []}

        # Truncate excerpts to keep the grader prompt small.
        excerpts = "\n\n".join(
            f"[{i + 1}] {c['source']}\n{c['text'][:500]}"
            for i, c in enumerate(chunks)
        )
        llm = self._get_grader_llm().with_structured_output(GraderReport)
        try:
            report: GraderReport = llm.invoke(
                GRADER_PROMPT.format(question=state["question"], excerpts=excerpts)
            )
            scores = [s.score for s in report.scores][: len(chunks)]
            while len(scores) < len(chunks):
                scores.append(3)
        except Exception as e:
            self._log(state, f"grader failed ({e}); assuming neutral score 3")
            scores = [3] * len(chunks)

        avg = round(sum(scores) / len(scores), 2) if scores else 0.0

        kept_chunks: list[dict] = []
        kept_grades: list[int] = []
        dropped = 0
        for chunk, score in zip(chunks, scores):
            if score >= self.min_keep_grade:
                kept_chunks.append(chunk)
                kept_grades.append(score)
            else:
                dropped += 1
        if dropped:
            self._log(
                state,
                f"grader dropped {dropped}/{len(chunks)} chunks below "
                f"grade {self.min_keep_grade}",
            )
        return {
            "grades": kept_grades,
            "avg_grade": avg,
            "retrieved_chunks": kept_chunks,
        }

    def rewrite_node(self, state: AgentState) -> dict:
        """Reformulate the question to improve retrieval. Caps via iteration_count."""
        iteration = state.get("iteration_count", 0)
        history = list(state.get("rewrite_history", []))
        history_str = "\n".join(f"- {q}" for q in history) or "(none)"
        llm = self._get_grader_llm().with_structured_output(RewrittenQuery)
        try:
            out: RewrittenQuery = llm.invoke(
                REWRITE_PROMPT.format(question=state["question"], history=history_str)
            )
            rewritten = out.query.strip()
        except Exception as e:
            self._log(state, f"rewriter failed ({e}); keeping original question")
            rewritten = state["question"]

        # The next retrieve pass uses the rewrite as the (only) sub-query.
        return {
            "sub_queries": [rewritten],
            "rewrite_history": history + [rewritten],
            "iteration_count": iteration + 1,
        }

    def synthesize_node(self, state: AgentState) -> dict:
        """Same as the parent, plus a low-confidence flag when retrieval stayed weak."""
        out = super().synthesize_node(state)
        low_conf = (
            state.get("avg_grade", 0.0) < self.grade_threshold
            and state.get("iteration_count", 0) >= self.max_rewrites
        )
        out["low_confidence"] = bool(low_conf)
        return out

    def critic_node(self, state: AgentState) -> dict:
        """Extends the parent's critic with a retrieval-retry hint.

        When the parent flags unsupported claims AND we still have retries
        left, point the next retrieve pass at those claims as focused
        sub-queries. Otherwise let the flow end.
        """
        out = super().critic_node(state)
        crit_iter = state.get("critic_iterations", 0)

        if not out.get("needs_retry") or crit_iter >= self.max_critic_retries:
            # Either converged, or out of retries — let the router END.
            out["needs_retry"] = False
            return out

        # Build hint sub-queries from the unsupported-claim errors the parent
        # appended this turn. (Format set by parent: "unsupported claim: <c>".)
        new_errors = out.get("errors", [])
        hints = [
            "supporting evidence for: " + e[len("unsupported claim: "):]
            for e in new_errors[-3:]
            if e.startswith("unsupported claim: ")
        ] or [state["question"]]

        out["sub_queries"] = hints
        out["critic_iterations"] = crit_iter + 1
        return out

    # ------------------------------------------------------------------ #
    # Graph
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(AgentState)
        g.add_node("planner", self.planner_node)
        g.add_node("retrieve", self.hybrid_retrieve_node)
        g.add_node("grader", self.grader_node)
        g.add_node("rewrite", self.rewrite_node)
        g.add_node("synthesize", self.synthesize_node)
        g.add_node("critic", self.critic_node)

        g.add_edge(START, "planner")
        g.add_edge("planner", "retrieve")
        g.add_edge("retrieve", "grader")
        g.add_conditional_edges(
            "grader", self._grade_router,
            {"rewrite": "rewrite", "synthesize": "synthesize"},
        )
        g.add_edge("rewrite", "retrieve")
        g.add_edge("synthesize", "critic")
        g.add_conditional_edges(
            "critic", self._critic_router,
            {"retrieve": "retrieve", "end": END},
        )
        return g.compile()

    # ------------------------------------------------------------------ #
    # Routers
    # ------------------------------------------------------------------ #

    def _grade_router(self, state: AgentState) -> str:
        avg = state.get("avg_grade", 0.0)
        kept = state.get("retrieved_chunks") or []
        # All chunks filtered out AND average grade is very low → rewriting
        # over the same corpus won't surface anything (the entity isn't
        # there). Skip straight to the next stage (web/table/synth).
        if not kept and avg < self.very_poor_grade:
            return "synthesize"
        if avg >= self.grade_threshold:
            return "synthesize"
        if state.get("iteration_count", 0) >= self.max_rewrites:
            return "synthesize"          # cap hit; synth will set low_confidence
        return "rewrite"

    def _critic_router(self, state: AgentState) -> str:
        return "retrieve" if state.get("needs_retry") else "end"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the Corrective-RAG (v2) graph.")
    p.add_argument("--collection", default="us_filings")
    p.add_argument("--chroma-dir", default="data/chroma")
    p.add_argument("--market", choices=["india", "us"], default="us")
    p.add_argument("--provider", choices=["groq", "gemini", "openai", "anthropic"],
                   default="groq")
    p.add_argument("--planner-model", default=None)
    p.add_argument("--synth-model", default=None)
    p.add_argument("--critic-model", default=None)
    p.add_argument("--grader-model", default=None)
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--reranker-model", default=HybridRetriever.DEFAULT_RERANKER)
    p.add_argument("--bm25-top-k", type=int, default=10)
    p.add_argument("--dense-top-k", type=int, default=10)
    p.add_argument("--final-top-k", type=int, default=5)
    p.add_argument("--grade-threshold", type=float, default=3.0)
    p.add_argument("--max-rewrites", type=int, default=3)
    p.add_argument("--max-critic-retries", type=int, default=2)
    p.add_argument("--question", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--question-col", default="question")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--output", default="results/corrective_rag_outputs.json")
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

    agent = AgenticRAGv2(
        collection_name=args.collection,
        chroma_dir=args.chroma_dir,
        market=args.market,
        embedding_model=args.embedding_model,
        provider=args.provider,
        planner_model=args.planner_model,
        synth_model=args.synth_model,
        critic_model=args.critic_model,
        grader_model=args.grader_model,
        reranker_model=args.reranker_model,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        final_top_k=args.final_top_k,
        grade_threshold=args.grade_threshold,
        max_rewrites=args.max_rewrites,
        max_critic_retries=args.max_critic_retries,
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
    print(f"Question:        {state['question']}")
    print(f"Sub-queries:     {state.get('sub_queries')}")
    print(f"Grades:          {state.get('grades')}  (avg {state.get('avg_grade')})")
    print(f"Rewrites used:   {state.get('iteration_count', 0)}")
    print(f"Critic retries:  {state.get('critic_iterations', 0)}")
    print(f"\nAnswer:\n{state.get('final_answer')}")
    print(f"\nCitations:       {state.get('citations')}")
    print(f"Critic grade:    {state.get('grading_score')}  needs_retry={state.get('needs_retry')}")
    print(f"Low confidence:  {state.get('low_confidence')}")
    if state.get("errors"):
        print(f"Errors:          {state['errors']}")


if __name__ == "__main__":
    main()
