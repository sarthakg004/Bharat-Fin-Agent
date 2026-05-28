"""
ragas.py  ·  src/evaluation/ragas.py

Week 1 evaluation harness. Takes the outputs from naive_rag.py and
measures quality using four RAGAS metrics:

    Faithfulness          — are all claims in the answer grounded in
                            the retrieved context?
    Response Relevancy    — does the answer actually address the question?
    Context Precision     — are the retrieved chunks relevant (is signal
                            ranked above noise)?
    Context Recall        — do the retrieved chunks cover the gold answer?

The judge LLM is intentionally DIFFERENT from the generator LLM to
avoid self-evaluation bias. Generator uses llama-3.1-8b-instant;
judge uses llama-3.3-70b-versatile (the strongest free Groq model).

Usage:
    from src.evaluation.ragas import RAGASEvaluator

    ev = RAGASEvaluator(judge_provider="groq")   # or "gemini"
    df_results = ev.evaluate(
        outputs_path="results/naive_rag_outputs.json",
        output_csv="results/week1_naive_baseline.csv",
    )
    print(df_results[["faithfulness","answer_relevancy",
                       "context_precision","context_recall"]].mean())

CLI:
    python -m src.evaluation.ragas \\
        --outputs results/naive_rag_outputs.json \\
        --output-csv results/week1_naive_baseline.csv \\
        --judge-provider groq \\
        --sample 50
"""

from __future__ import annotations

import sys
import types

# --------------------------------------------------------------------------- #
# Compatibility shim — MUST run before `import ragas`.
# ragas 0.4.3 still does `from langchain_community.chat_models.vertexai import
# ChatVertexAI`, but langchain-community 0.4.x (the LangChain v1 stack) removed
# that module. We never use Vertex AI, so we register a stub so the import
# chain resolves. Drop this once ragas stops importing the dead path.
# --------------------------------------------------------------------------- #
_VERTEXAI_MOD = "langchain_community.chat_models.vertexai"
if _VERTEXAI_MOD not in sys.modules:
    _shim = types.ModuleType(_VERTEXAI_MOD)
    _shim.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[_VERTEXAI_MOD] = _shim

import json
import os
import time
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# How long to wait between RAGAS metric calls (Groq free tier: ~30 req/min).
# With 4 metrics × N questions, budget ~2s per metric call to stay safe.
GROQ_RATE_LIMIT_DELAY = 2.0


class RAGASEvaluator:
    """Evaluate RAG outputs with four RAGAS metrics using Groq as judge.

    Parameters
    ----------
    judge_provider : str
        "groq" (default) or "gemini". The judge makes several calls per
        question per metric, so Gemini's ~5-requests/day free tier is only
        viable for a 1-2 question smoke test — use Groq for real runs.
    judge_model : str
        Judge LLM model. Should be DIFFERENT and STRONGER than the generator
        to avoid self-evaluation bias. If None, defaults per provider
        (Groq: llama-3.3-70b-versatile; Gemini: gemini-2.5-flash).
    embedding_model : str
        Used by the ResponseRelevancy metric. Match your ingestion model.
    api_key : str
        Falls back to the provider's env var (GROQ_API_KEY / GEMINI_API_KEY).
    """

    DEFAULT_MODELS = {
        "groq": "llama-3.3-70b-versatile",
        "gemini": "gemini-2.5-flash",
    }
    API_KEY_ENV = {
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }

    def __init__(
        self,
        judge_provider: str = "groq",
        judge_model: Optional[str] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        api_key: Optional[str] = None,
    ):
        self.judge_provider = judge_provider.lower()
        if self.judge_provider not in self.DEFAULT_MODELS:
            raise ValueError(
                f"Unknown judge_provider {judge_provider!r}. "
                f"Choose one of {list(self.DEFAULT_MODELS)}."
            )
        self.judge_model = judge_model or self.DEFAULT_MODELS[self.judge_provider]
        self.embedding_model = embedding_model

        env_var = self.API_KEY_ENV[self.judge_provider]
        self.api_key = api_key or os.getenv(env_var)
        if not self.api_key:
            raise ValueError(f"{env_var} not found. Set it in your .env file.")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        outputs_path: Union[str, Path],
        output_csv: Union[str, Path] = "results/week1_naive_baseline.csv",
        ground_truth_col: str = "answer",
        sample: Optional[int] = None,
        batch_size: int = 10,
    ) -> pd.DataFrame:
        """Run RAGAS over all outputs and save results CSV.

        Args:
            outputs_path: JSON file produced by NaiveRAG.run_dataset().
            output_csv: Where to write per-question metric scores + summary.
            ground_truth_col: Key in outputs JSON that holds the gold answer.
                For FinanceBench this is "answer".
            sample: If set, evaluate only the first N rows. Useful for
                smoke-testing before running the full 150 questions.
            batch_size: Process this many questions per RAGAS call.
                Smaller = more API calls but safer against Groq timeouts.

        Returns:
            DataFrame with one row per question and columns:
            question, answer, faithfulness, answer_relevancy,
            context_precision, context_recall, error.
        """
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        # ---- 1. Load outputs ------------------------------------------------
        outputs = self._load_outputs(outputs_path)
        # Drop failures (empty answer)
        outputs = [o for o in outputs if o.get("answer") and not o.get("error")]
        if sample:
            outputs = outputs[:sample]
        print(f"Evaluating {len(outputs)} outputs with RAGAS "
              f"(judge: {self.judge_model})")

        # ---- 2. Configure RAGAS LLM + embeddings ----------------------------
        ragas_llm, ragas_embeddings = self._build_ragas_clients()

        # ---- 3. Evaluate in batches -----------------------------------------
        all_scores: list[dict] = []
        batches = [
            outputs[i: i + batch_size]
            for i in range(0, len(outputs), batch_size)
        ]

        for batch_idx, batch in enumerate(
            tqdm(batches, desc="RAGAS batches"), start=1
        ):
            try:
                scores = self._evaluate_batch(
                    batch,
                    ragas_llm,
                    ragas_embeddings,
                    ground_truth_col,
                )
                all_scores.extend(scores)
            except Exception as e:
                print(f"\n  ! Batch {batch_idx} failed: {e}")
                # Mark each row in the batch as failed rather than losing them.
                for row in batch:
                    all_scores.append({
                        "question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                        "faithfulness": None,
                        "answer_relevancy": None,
                        "context_precision": None,
                        "context_recall": None,
                        "error": str(e),
                    })

            # Polite delay between batches for Groq rate limit.
            if batch_idx < len(batches):
                time.sleep(GROQ_RATE_LIMIT_DELAY)

        # ---- 4. Build results DataFrame -------------------------------------
        df = pd.DataFrame(all_scores)

        # Append a summary row at the bottom.
        numeric_cols = [
            "faithfulness", "answer_relevancy",
            "context_precision", "context_recall",
        ]
        summary = {col: df[col].mean() for col in numeric_cols}
        summary["question"] = "*** MEAN ***"
        summary["answer"] = ""
        df = pd.concat(
            [df, pd.DataFrame([summary])], ignore_index=True
        )

        df.to_csv(output_csv, index=False)
        print(f"\nResults written → {output_csv}")
        self._print_summary(df, numeric_cols)
        return df

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_ragas_clients(self):
        """Build RAGAS-compatible LLM and embedding wrappers."""
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        if self.judge_provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=self.judge_model,
                google_api_key=self.api_key,
                temperature=0.0,
                max_retries=3,
            )
        else:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model=self.judge_model,
                api_key=self.api_key,
                temperature=0.0,
                max_retries=3,
            )
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)

    def _evaluate_batch(
        self,
        batch: list[dict],
        ragas_llm,
        ragas_embeddings,
        ground_truth_col: str,
    ) -> list[dict]:
        """Run RAGAS on one batch and return per-row score dicts."""
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )

        samples = []
        for row in batch:
            ground_truth = str(row.get(ground_truth_col, ""))
            contexts = row.get("retrieved_chunks", [])
            if not isinstance(contexts, list):
                contexts = [str(contexts)]
            # RAGAS needs non-empty contexts.
            if not contexts:
                contexts = ["No context retrieved."]

            samples.append(
                SingleTurnSample(
                    user_input=row["question"],
                    retrieved_contexts=contexts,
                    response=row["answer"],
                    reference=ground_truth,
                )
            )

        dataset = EvaluationDataset(samples=samples)

        result = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(),
                ResponseRelevancy(),
                LLMContextPrecisionWithReference(),
                LLMContextRecall(),
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False,   # keep going if one metric fails
        )

        # result.to_pandas() gives one row per sample with metric columns.
        scores_df = result.to_pandas()

        # Map RAGAS column names to our convention.
        col_map = {
            "faithfulness": "faithfulness",
            "answer_relevancy": "answer_relevancy",
            "context_precision": "context_precision",
            "context_recall": "context_recall",
            # RAGAS 0.2.x uses these names:
            "response_relevancy": "answer_relevancy",
            "llm_context_precision_with_reference": "context_precision",
            "llm_context_recall": "context_recall",
        }

        scores = []
        for i, row in enumerate(batch):
            score_row = {
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "company": row.get("company") or row.get("company_name") or row.get("ticker", ""),
                "year": row.get("year", ""),
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None,
                "error": None,
            }
            if i < len(scores_df):
                for ragas_col, our_col in col_map.items():
                    if ragas_col in scores_df.columns:
                        score_row[our_col] = scores_df.iloc[i].get(ragas_col)
            scores.append(score_row)

        return scores

    @staticmethod
    def _load_outputs(path) -> list[dict]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Outputs file not found: {path}\n"
                "Run naive_rag.py first to generate it."
            )
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _print_summary(df: pd.DataFrame, numeric_cols: list[str]) -> None:
        # Drop the summary row for per-question stats
        data = df[df["question"] != "*** MEAN ***"]
        print()
        print("=" * 55)
        print("RAGAS Evaluation — Week 1 Naive Baseline")
        print("=" * 55)
        print(f"{'Metric':<28} {'Mean':>8}  {'Std':>8}")
        print("-" * 55)
        for col in numeric_cols:
            vals = pd.to_numeric(data[col], errors="coerce").dropna()
            mean = vals.mean() if len(vals) else float("nan")
            std = vals.std() if len(vals) else float("nan")
            print(f"  {col:<26} {mean:>8.3f}  {std:>8.3f}")
        print("=" * 55)
        print()
        print("Copy this into your README as the Week 1 baseline table:")
        print()
        print("| Configuration | Faithfulness | Answer Relevancy | "
              "Context Precision | Context Recall |")
        print("|---|---|---|---|---|")
        row_vals = []
        for col in numeric_cols:
            vals = pd.to_numeric(data[col], errors="coerce").dropna()
            row_vals.append(f"{vals.mean():.2f}" if len(vals) else "N/A")
        print(f"| Naive RAG | {' | '.join(row_vals)} |")


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument(
        "--outputs",
        default="results/naive_rag_outputs.json",
        help="Path to JSON produced by naive_rag.py",
    )
    parser.add_argument(
        "--output-csv",
        default="results/week1_naive_baseline.csv",
        help="Where to save the per-question scores CSV",
    )
    parser.add_argument(
        "--judge-provider",
        choices=["groq", "gemini"],
        default="groq",
        help="LLM provider for the RAGAS judge (gemini free tier is tiny)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model; defaults per provider if omitted",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-en-v1.5",
        help="Same embedding model used at ingestion",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Evaluate only the first N rows (useful for smoke test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Questions per RAGAS batch (default 10)",
    )
    args = parser.parse_args()

    ev = RAGASEvaluator(
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
    )
    ev.evaluate(
        outputs_path=args.outputs,
        output_csv=args.output_csv,
        sample=args.sample,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()