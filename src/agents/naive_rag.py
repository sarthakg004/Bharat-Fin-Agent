"""
naive_rag.py  ·  src/agents/naive_rag.py

The simplest possible RAG pipeline.

Flow:
    question
        ↓ embed (same BGE model used at ingestion)
        ↓ top-k similarity search in Chroma
        ↓ stuff all chunks into a single prompt
        ↓ Groq LLM generates the answer
        ↓ return structured result dict

This is the Week 1 baseline. Every improvement in later weeks
(hybrid retrieval, table agent, hallucination critic) is measured
against the numbers this produces.

Usage:
    from src.agents.naive_rag import NaiveRAG

    rag = NaiveRAG(
        collection_name="us_filings",
        chroma_dir="data/chroma",
        llm_provider="groq",          # or "gemini"
    )

    result = rag.answer("What was Apple's net revenue in FY2023?")
    print(result["answer"])
    print(f"Latency: {result['latency']:.2f}s")

    # Batch run over a DataFrame and save JSON
    rag.run_dataset(df, output_path="results/naive_rag_outputs.json")

CLI:
    python -m src.agents.naive_rag \\
        --collection us_filings \\
        --chroma-dir data/chroma \\
        --provider groq \\
        --question "What was Apple's R&D expense in FY2023?"
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv
from tqdm import tqdm

from src.llm import build_llm, resolve_api_key

load_dotenv()  # picks up provider API keys from a .env file if present

# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """\
You are a financial analyst assistant with access to numbered excerpts from
SEC 10-K filings and Indian annual reports. Answer the user's question using
ONLY the supplied excerpts.

Citations: cite by number only — after every factual claim, append the index
of the excerpt(s) that support it in square brackets, e.g. \"net sales were \
$394.3 billion [1]\". Multiple sources for one claim: `[1,3]`. Do NOT write \
out source titles, URLs, or full tags in the prose.

Formatting: write the answer in clean markdown. Use **bold** for key figures \
and entity names, bullet lists for 3+ points, a GitHub-flavoured markdown \
table when comparing two or more items on the same metrics, and `## \
sub-headings` only when the answer has 2+ logical sections. Keep paragraphs \
short and leave a blank line between them.

Aim for a thorough, well-structured answer; no filler. If the excerpts don't \
contain enough information, say so in one short sentence WITHOUT any citation \
numbers — and do not invent figures.
"""

RAG_PROMPT = """\
Numbered excerpts (cite with `[N]`):
{context}

---
Question: {question}

Answer in well-structured markdown with [N] citations after every factual claim.
"""


class NaiveRAG:
    """
    Minimal RAG pipeline: embed → retrieve → stuff-into-prompt → generate.

    Parameters
    ----------
    collection_name : str
        Chroma collection to query. Must match the collection used during
        ingestion (e.g. "us_filings" or "india_filings").
    chroma_dir : str
        Persistent Chroma directory (same value as used in ingest.py).
    embedding_model : str
        MUST match the model used at ingestion. Defaults to bge-small.
    llm_provider : str
        "groq" (default), "gemini", "openai", or "anthropic". Gemini's free
        tier is ~5 requests/day — fine for smoke tests, not full runs.
    generator_model : str
        Model for answer generation. If None, a sensible default is chosen
        per provider. Pass any model your key has access to.
    top_k : int
        Number of chunks to retrieve per question.
    api_key : str
        If None, reads from the provider's env var (GROQ_API_KEY, GEMINI_API_KEY,
        OPENAI_API_KEY, or ANTHROPIC_API_KEY).
    """

    # Default generator model per provider (override via generator_model=).
    # Groq: openai/gpt-oss-120b — 120B open-weights model with better long-form
    # writing + reasoning than llama-3.1-8b-instant for the same Groq endpoint.
    DEFAULT_MODELS = {
        "groq": "openai/gpt-oss-120b",
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5",
    }

    def __init__(
        self,
        collection_name: str = "us_filings",
        chroma_dir: Union[str, Path] = "data/chroma",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        llm_provider: str = "groq",
        generator_model: Optional[str] = None,
        top_k: int = 5,
        api_key: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.chroma_dir = str(chroma_dir)
        self.embedding_model = embedding_model
        self.top_k = top_k

        self.llm_provider = llm_provider.lower()
        if self.llm_provider not in self.DEFAULT_MODELS:
            raise ValueError(
                f"Unknown llm_provider {llm_provider!r}. "
                f"Choose one of {list(self.DEFAULT_MODELS)}."
            )
        self.generator_model = generator_model or self.DEFAULT_MODELS[self.llm_provider]
        # When api_key is None, build_llm collects every {ENV}, {ENV}2, ... from
        # the environment and rotates across them on rate-limit errors.
        # Validate that at least one key exists; raise a clean error if not.
        resolve_api_key(self.llm_provider, api_key)
        self.api_key = api_key

        # Initialised lazily so the class is fast to import.
        self._retriever = None
        self._llm = None

    # ------------------------------------------------------------------ #
    # Public methods
    # ------------------------------------------------------------------ #

    def answer(self, question: str) -> dict:
        """Answer a single question. Returns a structured result dict.

        Return schema
        -------------
        {
            "question"         : str   — original question
            "retrieved_chunks" : list  — top-k chunk texts (for RAGAS contexts)
            "chunk_metadata"   : list  — Chroma metadata for each chunk
            "prompt"           : str   — full prompt sent to the LLM
            "answer"           : str   — generated answer
            "latency"          : float — wall-clock seconds
            "input_tokens"     : int   — approximate input token count
            "output_tokens"    : int   — approximate output token count
            "cost"             : float — always 0.0 for Groq free tier
            "model"            : str   — model used
        }
        """
        t0 = time.time()

        # 1. Retrieve
        docs = self._get_retriever().similarity_search(question, k=self.top_k)

        # 2. Build context string — 1-based numbering so the model can cite
        # by these indices and the frontend can scroll to the matching card.
        context_parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            source = (
                f"{meta.get('company') or meta.get('ticker', '?')}, "
                f"FY{meta.get('year', '?')}, page {meta.get('page', '?')}"
            )
            context_parts.append(f"[{i}] ({source})\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        # 3. Build prompt
        prompt = RAG_PROMPT.format(context=context, question=question)

        # 4. Generate
        llm = self._get_llm()
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        answer_text = response.content

        # 5. Token counts (best-effort).
        input_tokens, output_tokens = self._extract_usage(response, prompt, answer_text)

        return {
            "question": question,
            "retrieved_chunks": [d.page_content for d in docs],
            "chunk_metadata": [d.metadata for d in docs],
            "prompt": prompt,
            "answer": answer_text,
            "latency": round(time.time() - t0, 3),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": 0.0,   # Groq is free
            "model": self.generator_model,
        }

    def run_dataset(
        self,
        df,
        output_path: Union[str, Path] = "results/naive_rag_outputs.json",
        question_col: str = "question",
        delay_between: float = 0.5,
    ) -> list[dict]:
        """Run the pipeline over every question in a DataFrame.

        Saves incremental results to JSON after every question so a
        crash doesn't lose progress. Already-answered questions are
        skipped if the output file exists.

        Args:
            df: DataFrame with at least a `question_col` column.
                Any additional columns (answer, company, etc.) are
                merged into the output for traceability.
            output_path: Where to save/resume the JSON output list.
            question_col: Column name for questions.
            delay_between: Seconds to wait between questions.
                Groq free tier is rate-limited; 0.5s is polite.

        Returns:
            List of result dicts (same structure as `answer()`).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing results for resume capability.
        results: list[dict] = []
        answered_questions: set[str] = set()
        if output_path.exists():
            with open(output_path) as f:
                results = json.load(f)
            answered_questions = {r["question"] for r in results}
            print(f"Resuming: {len(answered_questions)} questions already answered")

        # Convert DataFrame to list of row dicts for iteration.
        rows = df.to_dict(orient="records")
        todo = [r for r in rows if r[question_col] not in answered_questions]
        print(f"Questions to process: {len(todo)}")

        for row in tqdm(todo, desc="naive_rag"):
            question = str(row[question_col])
            try:
                result = self.answer(question)
                # Merge in any extra columns from the eval set (company,
                # year, ground_truth, etc.) so they're in the same record.
                for k, v in row.items():
                    if k not in result:
                        result[k] = v
                results.append(result)
            except Exception as e:
                print(f"\n  ! failed: {question[:60]}... — {e}")
                results.append({
                    "question": question,
                    "answer": "",
                    "retrieved_chunks": [],
                    "error": str(e),
                    "latency": 0.0,
                    "cost": 0.0,
                })

            # Save after every question (crash-safe).
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            time.sleep(delay_between)

        print(f"\nSaved {len(results)} results → {output_path}")
        self._print_summary(results)
        return results

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_retriever(self):
        """Lazily initialise the Chroma retriever."""
        if self._retriever is None:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                encode_kwargs={"normalize_embeddings": True},
            )
            self._retriever = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=self.chroma_dir,
            )
            # Quick sanity check
            n = self._retriever._collection.count()
            print(f"Chroma collection '{self.collection_name}' loaded — {n} chunks")
        return self._retriever

    def _get_llm(self):
        """Lazily initialise the generator LLM for the chosen provider."""
        if self._llm is None:
            self._llm = build_llm(
                self.llm_provider, self.generator_model, self.api_key,
                temperature=0.0,  # deterministic for reproducibility
            )
        return self._llm

    def _extract_usage(self, response, prompt: str, answer_text: str) -> tuple[int, int]:
        """Best-effort (input, output) token counts across providers.

        Both ChatGroq and ChatGoogleGenerativeAI populate the standard
        `usage_metadata` field; we fall back to provider-specific metadata
        and finally a char-based estimate.
        """
        usage = getattr(response, "usage_metadata", None)
        if usage:
            return usage.get("input_tokens", 0), usage.get("output_tokens", 0)

        meta = getattr(response, "response_metadata", {}) or {}
        token_usage = meta.get("token_usage") or meta.get("usage") or {}
        return (
            token_usage.get("prompt_tokens", self._estimate_tokens(prompt)),
            token_usage.get("completion_tokens", self._estimate_tokens(answer_text)),
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count: ~4 chars per token is a reasonable approximation."""
        return len(text) // 4

    @staticmethod
    def _print_summary(results: list[dict]) -> None:
        ok = [r for r in results if r.get("answer") and not r.get("error")]
        failed = [r for r in results if r.get("error")]
        total_latency = sum(r.get("latency", 0) for r in ok)
        avg_latency = total_latency / len(ok) if ok else 0
        print(f"\nAnswered: {len(ok)} | Failed: {len(failed)}")
        print(f"Avg latency: {avg_latency:.2f}s | "
              f"Total time: {total_latency:.0f}s")


# --------------------------------------------------------------------------- #
# CLI entry point (for quick one-off questions)
# --------------------------------------------------------------------------- #

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run a single question through NaiveRAG")
    parser.add_argument("--collection", default="us_filings")
    parser.add_argument("--chroma-dir", default="data/chroma")
    parser.add_argument(
        "--provider",
        choices=["groq", "gemini", "openai", "anthropic"],
        default="groq",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Generator model; defaults per provider if omitted",
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    rag = NaiveRAG(
        collection_name=args.collection,
        chroma_dir=args.chroma_dir,
        embedding_model=args.embedding_model,
        llm_provider=args.provider,
        generator_model=args.model,
        top_k=args.top_k,
    )

    result = rag.answer(args.question)
    print("\n" + "=" * 60)
    print(f"Question: {result['question']}")
    print(f"\nAnswer:   {result['answer']}")
    print(f"\nLatency:  {result['latency']}s | "
          f"Tokens: {result['input_tokens']}in / {result['output_tokens']}out")
    print(f"\nTop chunks used:")
    for i, (chunk, meta) in enumerate(
        zip(result["retrieved_chunks"], result["chunk_metadata"]), 1
    ):
        co = meta.get("company") or meta.get("ticker", "?")
        print(f"  {i}. [{co} FY{meta.get('year','?')} p{meta.get('page','?')}] "
              f"{chunk[:100]}...")


if __name__ == "__main__":
    main()