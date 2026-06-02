"""
base.py  ·  finagent/graph/base.py

Week-2 agentic RAG built on LangGraph, with LangSmith tracing.

Pipeline (StateGraph):

    START → planner → retrieve → synthesize → critic → END

    planner     decomposes the question into 1-3 sub-queries (structured output)
    retrieve    similarity-search each sub-query in Chroma, merge + dedupe
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

    agent = AgenticRAG(collection_name="india_filings", market="india")
    result = agent.run("Compare TCS and Infosys revenue growth in FY23.")
    print(result["final_answer"])
    print(result["citations"])

CLI
---
    python -m finagent.graph.base \\
        --collection india_filings --market india \\
        --question "Compare TCS and Infosys revenue growth in FY23."

    # Batch over an eval set and append a row to results/comparison.csv:
    python -m finagent.graph.base \\
        --collection us_filings --market us \\
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
from finagent.llm import build_llm, resolve_api_key

load_dotenv()


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

PLANNER_PROMPT = """\
You are a query planner for a financial-filings question-answering system.
Decompose the user's question into 1-3 focused, self-contained sub-queries.

Rules:
- Simple, single-fact questions → return ONE sub-query (often the original).
- Comparison or multi-hop questions (e.g. "compare X and Y", "growth from A to B")
  → split into one sub-query per entity or fact so each can be retrieved separately.
- Each sub-query must stand on its own (no pronouns referring to the question).

Question: {question}
"""

SYNTHESIZER_SYSTEM = """\
You are a meticulous financial analyst. Write a clear, accurate answer using
ONLY the numbered excerpts supplied below.

Citations
---------
Cite by **number** only. After every factual claim, append the index of the
excerpt(s) that support it in square brackets, e.g.
"Apple's net sales were $394.3 billion [1]."
Multiple sources for one claim: `[1,3]`. NEVER write out the source title,
the URL, or the full tag — the user sees those in a sidebar already.

Formatting (markdown)
---------------------
- Open with a short overview paragraph that directly answers the question.
- Use **bold** for the key figures and entity names.
- Use bullet lists when summarising 3+ points.
- Use a GitHub-flavoured markdown table when comparing two or more items
  across the same metrics (e.g. revenue / margin / growth for two companies).
- Use `## sub-headings` only when the answer has 2+ logical sections.
- Keep paragraphs short (2-4 sentences); leave a blank line between them.

Aim for a thorough, well-structured answer — long enough to fully address the
question but with no filler.

If the excerpts do NOT contain enough information to answer:
- Say so in one short sentence.
- Do NOT include any citation numbers in that sentence.
- Do NOT invent figures, companies, or sources.
"""

SYNTHESIZER_PROMPT = """\
Question: {question}

Sub-queries researched:
{sub_queries}

Numbered excerpts (cite with `[N]`):
{context}

---
Write the answer now in well-structured markdown, with a [N] citation
after every factual claim.
"""

CRITIC_SYSTEM = """\
You are a hallucination critic. Given an answer and the source excerpts it was
based on, extract each distinct factual claim from the answer and decide whether
the excerpts SUPPORT it. Judge only against the excerpts, not your own knowledge.
"""

CRITIC_PROMPT = """\
Source excerpts:
{context}

---
Answer to check:
{answer}

---
Extract the factual claims and mark each supported / not supported.
"""


class AgenticRAG:
    """LangGraph agentic RAG: planner → retrieve → synthesize → critic.

    Parameters
    ----------
    collection_name : str
        Chroma collection ("us_filings" or "india_filings").
    chroma_dir : str
        Persistent Chroma directory (must match ingestion).
    market : str
        "india" or "us" — only affects the citation wording ("AR" vs "10-K").
    embedding_model : str
        MUST match the model used at ingestion.
    provider : str
        "groq" (default), "gemini", "openai", or "anthropic" for all three LLM
        roles.
    planner_model / synth_model / critic_model : str
        Per-role models — mix freely. Synthesis/critic use the strongest model
        by default; the planner uses a fast one. If None, per-provider defaults
        apply. Pass any model your key has access to.
    top_k : int
        Chunks retrieved per sub-query.
    api_key : str
        Falls back to the provider's env var (GROQ_API_KEY / GEMINI_API_KEY /
        OPENAI_API_KEY / ANTHROPIC_API_KEY).
    """

    # Groq tier picks:
    #   planner / grader / router (fast, structured output)  → llama-3.1-8b-instant
    #   synth + critic  (long-form writing, reasoning)        → openai/gpt-oss-120b
    # 120B has its own TPD bucket on Groq so it relieves the 70B-versatile
    # quota that was getting hammered, and produces noticeably better
    # multi-section / markdown / citation output.
    DEFAULTS = {
        "groq": {
            "planner": "llama-3.1-8b-instant",
            "synth":   "openai/gpt-oss-120b",
            "critic":  "openai/gpt-oss-120b",
        },
        "gemini": {
            "planner": "gemini-2.5-flash",
            "synth": "gemini-2.5-flash",
            "critic": "gemini-2.5-flash",
        },
        "openai": {
            "planner": "gpt-4o-mini",
            "synth": "gpt-4o",
            "critic": "gpt-4o",
        },
        "anthropic": {
            "planner": "claude-haiku-4-5",
            "synth": "claude-sonnet-4-6",
            "critic": "claude-sonnet-4-6",
        },
    }

    def __init__(
        self,
        collection_name: str = "us_filings",
        chroma_dir: Union[str, Path] = "data/chroma",
        market: str = "us",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        provider: str = "groq",
        planner_model: Optional[str] = None,
        synth_model: Optional[str] = None,
        critic_model: Optional[str] = None,
        top_k: int = 5,
        api_key: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.chroma_dir = str(chroma_dir)
        self.market = market
        self.embedding_model = embedding_model
        self.top_k = top_k

        self.provider = provider.lower()
        if self.provider not in self.DEFAULTS:
            raise ValueError(
                f"Unknown provider {provider!r}. Choose one of {list(self.DEFAULTS)}."
            )
        d = self.DEFAULTS[self.provider]
        self.planner_model = planner_model or d["planner"]
        self.synth_model = synth_model or d["synth"]
        self.critic_model = critic_model or d["critic"]
        # api_key=None lets build_llm pick up every {ENV}, {ENV}2, ... from
        # .env and rotate on rate-limit errors. We still validate up front.
        resolve_api_key(self.provider, api_key)
        self.api_key = api_key

        # Lazy resources.
        self._retriever = None
        self._llms: dict[str, object] = {}
        self._graph = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, question: str) -> AgentState:
        """Run the full graph on one question. Returns the final state dict."""
        initial: AgentState = {
            "question": question,
            "iteration_count": 0,
            "errors": [],
            "table_results": [],
            "web_results": [],
        }
        return self.graph.invoke(initial)

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

        llm = self._get_llm("planner").with_structured_output(SubQueries)
        try:
            out: SubQueries = llm.invoke(
                history_block + PLANNER_PROMPT.format(question=question)
            )
            queries = [q.strip() for q in out.queries if q.strip()][:3]
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
        answer = response.content
        citations = sorted(set(re.findall(r"\[[^\]]+\]", answer)))
        return {
            "draft_answer": answer,
            "final_answer": answer,
            "citations": citations,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

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
                [SystemMessage(content=CRITIC_SYSTEM), HumanMessage(content=prompt)]
            )
            verdicts = report.verdicts
        except Exception as e:
            self._log(state, f"critic failed ({e})")
            return {"grading_score": None, "needs_retry": False}

        if not verdicts:
            return {"grading_score": None, "needs_retry": False}

        supported = sum(1 for v in verdicts if v.supported)
        score = supported / len(verdicts)
        unsupported = [v.claim for v in verdicts if not v.supported]

        errors = list(state.get("errors", []))
        for claim in unsupported:
            errors.append(f"unsupported claim: {claim}")
        return {
            "grading_score": round(score, 3),
            "needs_retry": bool(unsupported),
            "errors": errors,
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
        """Build a citation tag like '[Reliance AR 2024, p. 102]'.

        The year is the *report's* year (which document the text came from), NOT
        the fiscal year of any figure inside it. Annual reports carry prior-year
        comparatives, so an FY23 number can legitimately appear in the 2024
        report — the tag points to the source document, and the answer text
        states which fiscal year a figure refers to.
        """
        name = meta.get("company") or meta.get("ticker", "?")
        year = meta.get("year", "?")
        doc_kind = "10-K" if self.market == "us" else "AR"
        return f"[{name} {doc_kind} {year}, p. {meta.get('page', '?')}]"

    @staticmethod
    def _log(state: AgentState, msg: str) -> None:
        state.setdefault("errors", []).append(msg)

    def _get_retriever(self):
        if self._retriever is None:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings

            from finagent.chroma_client import chroma_kwargs_for_langchain
            from finagent.device import get_device

            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": get_device()},
                encode_kwargs={"normalize_embeddings": True},
            )
            self._retriever = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                **chroma_kwargs_for_langchain(self.chroma_dir),
            )
        return self._retriever

    def _get_llm(self, role: str):
        """Build (and cache) the LLM for a role: 'planner' | 'synth' | 'critic'."""
        if role not in self._llms:
            model = {
                "planner": self.planner_model,
                "synth": self.synth_model,
                "critic": self.critic_model,
            }[role]
            self._llms[role] = build_llm(
                self.provider, model, self.api_key, temperature=0.0
            )
        return self._llms[role]


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
    p.add_argument("--collection", default="us_filings")
    p.add_argument("--chroma-dir", default="data/chroma")
    p.add_argument("--market", choices=["india", "us"], default="us")
    p.add_argument(
        "--provider",
        choices=["groq", "gemini", "openai", "anthropic"],
        default="groq",
    )
    p.add_argument("--planner-model", default=None)
    p.add_argument("--synth-model", default=None)
    p.add_argument("--critic-model", default=None)
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
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

    agent = AgenticRAG(
        collection_name=args.collection,
        chroma_dir=args.chroma_dir,
        market=args.market,
        embedding_model=args.embedding_model,
        provider=args.provider,
        planner_model=args.planner_model,
        synth_model=args.synth_model,
        critic_model=args.critic_model,
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

    state = agent.run(args.question)
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
