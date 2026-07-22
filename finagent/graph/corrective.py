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

    agent = AgenticRAGv2(collection_name="us_filings")
    state = agent.run("Compare Microsoft and Apple revenue in FY23.")
    print(state["final_answer"])
    print("avg_grade:", state["avg_grade"], "low_conf:", state.get("low_confidence"))

CLI
---
    python -m finagent.graph.corrective \\
        --collection us_filings \\
        --question "Compare Microsoft and Apple revenue in FY23."
"""

from __future__ import annotations

import argparse
import re
from typing import Optional, Union

from finagent.graph.base import AgenticRAG, append_comparison_row
from finagent.runtime import RuntimeContext
from finagent.graph.state import (
    AgentState,
    ChunkScore,
    CriticReport,
    GraderReport,
    RewrittenQuery,
    SubQueries,
)
from finagent.vectorstore import DEFAULT_EMBED_MODEL


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


# Codepoint ranges for scripts a US English filing should not be written in
# (Cyrillic, Hebrew, Arabic, Devanagari, Hiragana/Katakana, CJK, Hangul).
_NON_LATIN_RE = re.compile(
    r"[Ѐ-ӿ֐-׿؀-ۿऀ-ॿ"
    r"぀-ヿ㐀-鿿가-힯]"
)


def _mostly_non_english(text: str) -> bool:
    """True if a chunk is predominantly non-Latin script — i.e. foreign-language
    noise that has no place in an English US-filings answer. Deterministic and
    cheap (no langdetect call); ignores stray foreign words in otherwise-English
    text by requiring non-Latin to exceed 30% of the alphabetic characters."""
    if not text:
        return False
    letters = sum(c.isalpha() for c in text)
    if letters < 20:                     # too short to judge confidently
        return False
    return len(_NON_LATIN_RE.findall(text)) / letters > 0.30


from finagent.retrieval.hybrid import HybridRetriever  # noqa: E402


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
        pool_top_k: int = 48,
        final_top_k: int = 5,
        # After merging every sub-query's hits into one pool, a final
        # cross-encoder rerank against the ORIGINAL question keeps only the
        # `retrieve_cap` best passages. Multi-hop questions decompose into
        # several sub-queries, each contributing up to `final_top_k` chunks, so
        # without this cap a single question could carry 15-25 passages into the
        # synthesizer — measured to LOWER faithfulness/precision (more material
        # to over-claim against, more noise). Capping to a tight, globally-best
        # set is the single biggest faithfulness lever in the eval. None = no cap.
        retrieve_cap: Optional[int] = 8,
        grade_threshold: float = 3.0,
        min_keep_grade: int = 4,
        very_poor_grade: float = 2.5,
        max_rewrites: int = 3,
        max_critic_retries: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reranker_model = reranker_model
        self.pool_top_k = pool_top_k
        self.final_top_k = final_top_k
        self.retrieve_cap = retrieve_cap
        # Grader is fast & cheap — default to the planner-tier model.
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
        """One hybrid retriever per filings collection (fused lexical+dense, reranked).
        Retrieving over several collections and letting the grader/reranker sort
        it out keeps every collection searchable without a toggle."""
        if self._hybrids is None:
            from finagent.vectorstore import build_store

            self._hybrids = [
                HybridRetriever(
                    build_store(c, self.embedding_model),
                    reranker_model=self.reranker_model,
                    pool_top_k=self.pool_top_k,
                    final_top_k=self.final_top_k,
                )
                for c in self.collections
            ]
        return self._hybrids

    def _get_grader_llm(self):
        """A fourth LLM role — reuses the role cache from the base class."""
        return self._get_llm("grader")

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
        dropped_lang = 0
        # Per-sub-query routing: filing retrieval is for `narrative` sub-queries
        # (and `numeric`, since the 10-K prose is a useful fallback when XBRL
        # lacks a metric). A `market` / `external` / `cross_document` sub-query is
        # answered by its own tool (yfinance / web / EDGAR), so retrieving the
        # filings for it only injects off-topic chunks — skip those. When routes
        # are unknown (older graph, or a single bare question) retrieve for all.
        sub_queries = state.get("sub_queries") or [state["question"]]
        routes = state.get("query_routes") or []
        if routes and len(routes) == len(sub_queries):
            retrieve_subs = [s for s, r in zip(sub_queries, routes)
                             if r in ("narrative", "numeric")]
        else:
            retrieve_subs = sub_queries
        for sub_q in retrieve_subs:
            for hyb in hybrids:
                for text, meta in hyb.search(sub_q):
                    key = (meta.get("local_path", ""), meta.get("page", ""), text[:80])
                    if key in seen:
                        continue
                    seen.add(key)
                    # The corpus is English US filings — drop any chunk that's
                    # predominantly non-Latin script (stray foreign-language
                    # exhibits, OCR noise) so it never reaches the synthesizer.
                    if _mostly_non_english(text):
                        dropped_lang += 1
                        continue
                    chunks.append({
                        "text": text,
                        "company": meta.get("company") or meta.get("ticker", "?"),
                        "year": meta.get("year", "?"),
                        "page": meta.get("page", "?"),
                        "source": self._citation_tag(meta),
                        "sub_query": sub_q,
                    })
        if dropped_lang:
            self._log(state, f"dropped {dropped_lang} non-English chunk(s) from retrieval")
        chunks = self._cap_pool(state, chunks)
        # Is the question's company actually in the corpus? The retriever's
        # metadata-vocab filter (no LLM) tells us: a company match means the
        # authoritative filing is indexed, so a low grade later means "hard
        # question", not "wrong company" — keep the filing instead of escalating
        # to generic web pages that only dilute faithfulness.
        in_corpus = False
        for hyb in hybrids:
            if any((hyb.infer_filter(sq) or {}).get("companies") for sq in retrieve_subs):
                in_corpus = True
                break
        return {"retrieved_chunks": chunks, "company_in_corpus": in_corpus}

    def _cap_pool(self, state: AgentState, chunks: list[dict]) -> list[dict]:
        """Global second-stage rerank: score the merged sub-query pool against
        the ORIGINAL question and keep the `retrieve_cap` best passages.

        The per-sub-query reranker inside each `HybridRetriever` only sees one
        sub-query at a time, so a 6-sub-query comparison question arrives here
        with up to 30 chunks. This final cross-encoder pass re-scores them all
        against the user's actual question and trims to a tight set — fewer,
        higher-precision passages measurably raised faithfulness in the eval.
        """
        cap = self.retrieve_cap
        if not cap or len(chunks) <= cap:
            return chunks
        try:
            from finagent.retrieval.reranker import _get_shared_reranker
            reranker = _get_shared_reranker(self.reranker_model)
            scores = reranker.predict([(state["question"], c["text"]) for c in chunks])
            ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])
            # Per-sub-query floor: a comparison question's sub-queries each cover
            # a different (entity × year); ranking the merged pool purely against
            # the ORIGINAL question lets one entity's chunks crowd out the
            # other's entirely. Reserve each sub-query's single best chunk first,
            # then fill the remaining slots by global score.
            kept_idx: list[int] = []
            seen_subs: set = set()
            for i in ranked:
                sub = chunks[i].get("sub_query", "")
                if sub not in seen_subs:
                    seen_subs.add(sub)
                    kept_idx.append(i)
                if len(kept_idx) >= cap:
                    break
            for i in ranked:
                if len(kept_idx) >= cap:
                    break
                if i not in kept_idx:
                    kept_idx.append(i)
            kept = [chunks[i] for i in sorted(kept_idx, key=ranked.index)]
            self._log(state, f"capped retrieval pool {len(chunks)}→{len(kept)} "
                             f"by cross-encoder rerank vs. the question")
            return kept
        except Exception as e:
            # Reranking is a precision optimisation — never let it drop evidence
            # on a model/load error. Fall back to a simple head-truncation.
            self._log(state, f"pool cap rerank failed ({e}); truncating to {cap}")
            return chunks[:cap]

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
        graded_ok = True
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
            graded_ok = False

        avg = round(sum(scores) / len(scores), 2) if scores else 0.0

        # A fallback score carries no per-chunk signal — filtering on it would
        # throw away the ENTIRE evidence set whenever the grader LLM hiccups
        # (that exact failure once discarded a freshly fetched 8-K wholesale).
        if not graded_ok:
            return {"grades": scores, "avg_grade": avg, "retrieved_chunks": chunks}

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
        # Never starve the synthesizer: if EVERY chunk fell below the bar but
        # the pool is at least borderline-relevant (avg ≥ very_poor_grade),
        # keep the best few — the critic and numeric verifier still guard
        # whatever gets written from them. When the average is VERY poor the
        # pool is off-entity noise (a JPMorgan filing for an ICICI question):
        # keeping it would show the user wrong-company sources, so drop it all
        # — empty retrieval is exactly what triggers the web escalation.
        if not kept_chunks and chunks:
            # Keep the best few when the pool is borderline-relevant OR the
            # company is in-corpus (the filing IS the authoritative source, even
            # if no single chunk literally restates a hard narrative question —
            # web pages would only hurt; the verifier still guards the figures).
            if avg >= self.very_poor_grade or state.get("company_in_corpus"):
                order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:5]
                kept_chunks = [chunks[i] for i in sorted(order)]
                kept_grades = [scores[i] for i in sorted(order)]
                self._log(state, f"all chunks below grade {self.min_keep_grade}; "
                                 f"keeping top {len(kept_chunks)} by score")
            else:
                self._log(state, f"all chunks graded off-entity/irrelevant "
                                 f"(avg {avg}); dropping them so the web "
                                 f"escalation can take over")
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
        # Routes are cleared so the router re-classifies the rewritten query
        # (in graphs that route after a rewrite) instead of zipping it against
        # the stale plan's routes.
        return {
            "sub_queries": [rewritten],
            "query_routes": [],
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
    p.add_argument("--provider", choices=["groq", "gemini", "openai", "anthropic"],
                   default="groq")
    p.add_argument("--planner-model", default=None)
    p.add_argument("--synth-model", default=None)
    p.add_argument("--critic-model", default=None)
    p.add_argument("--grader-model", default=None)
    p.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--reranker-model", default=HybridRetriever.DEFAULT_RERANKER)
    p.add_argument("--pool-top-k", type=int, default=48)
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

    # LLM choice is per-run, not a property of the agent — build the context
    # from the CLI flags and inject it at run().
    ctx = RuntimeContext(provider=args.provider, synth_model=args.synth_model,
                         top_k=args.final_top_k)

    agent = AgenticRAGv2(
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        pool_top_k=args.pool_top_k,
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

    state = agent.run(args.question, ctx)
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
