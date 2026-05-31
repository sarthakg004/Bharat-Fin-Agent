"""
state.py  ·  src/graph/state.py

Shared state for the agentic RAG LangGraph, plus the Pydantic schemas used
for structured LLM outputs (planner sub-queries, critic verdict).

`AgentState` is a TypedDict — the canonical LangGraph state schema. Every node
reads it and returns a partial dict that LangGraph merges back in. Each field
is written by exactly one node, so no custom reducers are needed.

Fields that aren't used yet (table_results, web_results) are placeholders for
later weeks (the table agent and web-search agent). They default to empty lists
so downstream nodes can treat them uniformly.
"""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
    """Shared state passed between every node in the graph."""

    question: str                      # the user's original question
    sub_queries: list[str]             # planner's decomposition (1-3 queries)
    retrieved_chunks: list[dict]       # [{text, company, year, page, source, sub_query}]
    table_results: list[Any]           # placeholder — future table agent
    web_results: list[Any]             # placeholder — future web-search agent
    draft_answer: str                  # synthesizer output (pre-critique)
    final_answer: str                  # answer returned to the caller
    citations: list[str]               # citation tags used, e.g. "[TCS AR FY23, p. 102]"
    grading_score: float               # critic: fraction of claims supported (0-1)
    iteration_count: int               # how many synthesize attempts so far
    errors: list[str]                  # non-fatal problems logged along the way
    needs_retry: bool                  # critic flag; drives the critic-retry loop

    # --- Corrective-RAG additions (graded retrieval + rewrite loop) -------- #
    grades: list[int]                  # grader: per-chunk relevance score (1-5)
    avg_grade: float                   # grader: mean of grades
    rewrite_history: list[str]         # past rewritten sub-queries
    critic_iterations: int             # critic-retry counter (cap = max_critic_retries)
    low_confidence: bool               # synth flag when grader stayed below threshold

    # --- Table-agent additions (v3) ---------------------------------------- #
    # `table_results` is already declared above; the v3 graph fills it with
    # one entry per numeric sub-query.
    query_routes: list[str]            # one of "narrative" | "numeric" | "external" per sub-query


# --------------------------------------------------------------------------- #
# Structured-output schemas (used with llm.with_structured_output(...))
# --------------------------------------------------------------------------- #

class SubQueries(BaseModel):
    """Planner output: the question decomposed into focused sub-queries."""

    queries: list[str] = Field(
        description=(
            "1 to 3 self-contained sub-queries. Use a single query for simple "
            "questions; split multi-hop or comparison questions (e.g. comparing "
            "two companies) into one query per entity/fact."
        )
    )


class ClaimVerdict(BaseModel):
    """One factual claim from the draft answer and whether the context supports it."""

    claim: str = Field(description="A single factual claim extracted from the answer.")
    supported: bool = Field(description="True if the retrieved context supports the claim.")
    reason: str = Field(description="Brief justification for the verdict.")


class CriticReport(BaseModel):
    """Critic output: per-claim support verdicts for the draft answer."""

    verdicts: list[ClaimVerdict] = Field(
        description="One entry per factual claim found in the answer."
    )


# --------------------------------------------------------------------------- #
# Corrective-RAG schemas
# --------------------------------------------------------------------------- #

class ChunkScore(BaseModel):
    """Per-excerpt relevance score from the grader."""

    score: int = Field(
        ge=1, le=5,
        description="1 = irrelevant, 3 = related, 5 = directly answers the question",
    )
    reason: str = Field(description="Brief justification for the score.")


class GraderReport(BaseModel):
    """Grader output: one score per excerpt, in the same order as presented."""

    scores: list[ChunkScore] = Field(
        description="Per-excerpt scores, in input order."
    )


class RewrittenQuery(BaseModel):
    """Rewriter output: a single reformulated, retrieval-friendly question."""

    query: str = Field(
        description=(
            "The reformulated question. Self-contained, ideally with "
            "domain-specific synonyms (e.g. 'net profit attributable to "
            "shareholders' instead of 'earnings')."
        )
    )


# --------------------------------------------------------------------------- #
# Router schemas (table-augmented RAG)
# --------------------------------------------------------------------------- #

from typing import Literal


class QueryRoute(BaseModel):
    """Per-sub-query routing verdict."""

    sub_query: str = Field(description="The sub-query being classified, copied verbatim.")
    route: Literal["narrative", "numeric", "external"] = Field(
        description=(
            "narrative = text retrieval over filings (default for prose-y questions); "
            "numeric = table agent (specific figures, ratios, growth %, "
            "currency amounts, multi-year comparisons); "
            "external = web search (current events, prices) — not yet implemented."
        )
    )
    reason: str = Field(description="One-line justification.")


class RouterReport(BaseModel):
    """Router output: one verdict per sub-query, in the same order."""

    routes: list[QueryRoute] = Field(description="One per sub-query, in input order.")


class TableTitle(BaseModel):
    """LLM-extracted title for a single table."""

    title: str = Field(description="Concise table title (≤ 120 chars).")


class PandasCode(BaseModel):
    """Pandas code emitted by the table agent to compute an answer."""

    code: str = Field(description="Python code; assigns the final answer to `result`.")
    explanation: str = Field(description="One sentence explaining what the code does.")
