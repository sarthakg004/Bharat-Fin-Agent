"""
ragas.py  ·  finagent/evaluation/ragas.py

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
    from finagent.evaluation.ragas import RAGASEvaluator

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
import time
import warnings
from pathlib import Path
from typing import Optional, Union

# ragas 0.4.3 still works with the Langchain* wrappers but emits a deprecation
# notice steering toward its instructor/litellm `llm_factory`. We intentionally
# keep the langchain wrappers so the judge can be any of our four providers via
# one code path, so silence that specific (cosmetic) notice.
warnings.filterwarnings("ignore", message=r".*LangchainLLMWrapper is deprecated.*")
warnings.filterwarnings("ignore", message=r".*LangchainEmbeddingsWrapper is deprecated.*")

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from finagent.llm import build_llm, resolve_api_key

load_dotenv()

# How long to wait between RAGAS metric calls (Groq free tier: ~30 req/min).
# With 4 metrics × N questions, budget ~2s per metric call to stay safe.
GROQ_RATE_LIMIT_DELAY = 2.0


class RAGASEvaluator:
    """Evaluate RAG outputs with four RAGAS metrics using Groq as judge.

    Parameters
    ----------
    judge_provider : str
        "groq" (default), "gemini", "openai", or "anthropic". The judge makes
        several calls per question per metric, so Gemini's ~5-requests/day free
        tier is only viable for a 1-2 question smoke test — use Groq/OpenAI/
        Anthropic for real runs.
    judge_model : str
        Judge LLM model. Should be DIFFERENT and STRONGER than the generator
        to avoid self-evaluation bias. If None, defaults per provider. Pass any
        model your key has access to (this is how you pick a different judge).
    embedding_model : str
        Used by the ResponseRelevancy metric. Match your ingestion model.
    api_key : str
        Falls back to the provider's env var (GROQ_API_KEY / GEMINI_API_KEY /
        OPENAI_API_KEY / ANTHROPIC_API_KEY).
    """

    DEFAULT_MODELS = {
        "groq": "llama-3.3-70b-versatile",
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-6",
    }

    def __init__(
        self,
        judge_provider: str = "groq",
        judge_model: Optional[str] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        api_key: Optional[str] = None,
        timeout: int = 300,
        max_workers: int = 4,
    ):
        self.judge_provider = judge_provider.lower()
        if self.judge_provider not in self.DEFAULT_MODELS:
            raise ValueError(
                f"Unknown judge_provider {judge_provider!r}. "
                f"Choose one of {list(self.DEFAULT_MODELS)}."
            )
        self.judge_model = judge_model or self.DEFAULT_MODELS[self.judge_provider]
        self.embedding_model = embedding_model
        # Keep api_key as None when not user-supplied so build_llm can collect
        # every {ENV}, {ENV}2, ... key from .env and rotate across them.
        resolve_api_key(self.judge_provider, api_key)
        self.api_key = api_key
        # Free-tier judges (Groq/Gemini) rate-limit; the RAGAS default of 16
        # parallel workers triggers TimeoutErrors. Fewer workers + a longer
        # per-job timeout keeps the run stable.
        self.timeout = timeout
        self.max_workers = max_workers

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

        Resumable: if `output_csv` already exists, questions that already have
        at least one non-null RAGAS score are skipped and their rows are kept
        as-is. Scores are flushed to CSV after every batch so partial progress
        is never lost across crashes or quota exhaustions.

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

        numeric_cols = [
            "faithfulness", "answer_relevancy",
            "context_precision", "context_recall",
        ]

        # ---- 1. Load outputs ------------------------------------------------
        outputs = self._load_outputs(outputs_path)
        outputs = [o for o in outputs if o.get("answer") and not o.get("error")]
        if sample:
            outputs = outputs[:sample]

        # ---- 1b. Resume: load already-scored rows ---------------------------
        # A row is "complete" only when ALL four metrics are present.  Rows
        # with partial scores (some metrics timed out) are re-evaluated so
        # the entire metric set is filled in — their old scores are discarded
        # and replaced with fresh ones.
        already_scored: dict[str, dict] = {}
        if output_csv.exists():
            try:
                existing = pd.read_csv(output_csv)
                complete_mask = (
                    (existing["question"] != "*** MEAN ***")
                    & existing[numeric_cols].notna().all(axis=1)
                )
                for _, row in existing[complete_mask].iterrows():
                    already_scored[str(row["question"])] = row.to_dict()
                total_existing = (existing["question"] != "*** MEAN ***").sum()
                partial = total_existing - len(already_scored)
                if already_scored or partial:
                    print(f"Resume: {len(already_scored)} fully scored, "
                          f"{partial} partial (will re-evaluate) — skipping "
                          f"{len(already_scored)}.")
            except Exception as e:
                print(f"  ! Could not read existing CSV ({e}); starting fresh.")

        pending = [o for o in outputs if o.get("question", "") not in already_scored]
        print(f"Evaluating {len(pending)}/{len(outputs)} outputs with RAGAS "
              f"(judge: {self.judge_model})")

        # ---- 2. Configure RAGAS LLM + embeddings ----------------------------
        ragas_llm, ragas_embeddings = self._build_ragas_clients()

        # ---- 3. Evaluate one question at a time, check exhaustion after each --
        # Evaluating question-by-question (not in batches of N) lets us check
        # the key-pool exhaustion state after every 4 LLM calls (one per metric).
        # RAGAS uses raise_exceptions=False so AllKeysExhaustedError is silently
        # converted to NaN — we detect it via the _EXHAUST_COUNT counter that
        # _exhausted() increments every time a full rotation cycle fails.
        from finagent.llm import AllKeysExhaustedError, _EXHAUST_COUNT

        new_scores: list[dict] = []
        flush_every = max(1, batch_size)   # flush CSV this often (questions)

        for q_idx, row in enumerate(tqdm(pending, desc="RAGAS questions"), start=1):
            exhaust_before = _EXHAUST_COUNT.get(self.judge_provider, 0)

            score = self._evaluate_one(row, ragas_llm, ragas_embeddings, ground_truth_col)
            new_scores.append(score)

            exhaust_after = _EXHAUST_COUNT.get(self.judge_provider, 0)
            new_exhaustions = exhaust_after - exhaust_before

            # If the key pool exhausted even ONCE while scoring this question
            # every key in the pool is rate-limited (daily or per-minute quota).
            # Stop immediately — there is nothing to gain by continuing.
            if new_exhaustions >= 1:
                self._flush_csv(output_csv, already_scored, new_scores, numeric_cols)
                scored_so_far = len(already_scored) + len(new_scores)
                remaining = len(pending) - q_idx
                print(f"\n⏸  LIMIT EXHAUSTED — all {self.judge_provider} keys are rate-limited.")
                print(f"   {scored_so_far}/{len(outputs)} rows scored so far.")
                print(f"   Progress saved to {output_csv}")
                print(f"   Re-run the same command tomorrow to score the remaining {remaining} rows.")
                raise AllKeysExhaustedError(
                    f"All {self.judge_provider} keys exhausted after question {q_idx}"
                )

            # Flush CSV periodically so progress is never lost.
            if q_idx % flush_every == 0 or q_idx == len(pending):
                self._flush_csv(output_csv, already_scored, new_scores, numeric_cols)

            time.sleep(GROQ_RATE_LIMIT_DELAY)

        # ---- 4. Build final DataFrame ----------------------------------------
        df = self._flush_csv(
            output_csv, already_scored, new_scores, numeric_cols, return_df=True,
        )
        print(f"\nResults written → {output_csv}")
        self._print_summary(df, numeric_cols)
        return df

    def _flush_csv(
        self,
        output_csv: Path,
        already_scored: dict,
        new_scores: list[dict],
        numeric_cols: list[str],
        return_df: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Merge already-scored + new_scores, append MEAN row, write CSV."""
        all_rows = list(already_scored.values()) + new_scores
        df = pd.DataFrame(all_rows)
        if df.empty:
            return df if return_df else None
        summary = {col: pd.to_numeric(df[col], errors="coerce").mean()
                   for col in numeric_cols if col in df.columns}
        summary["question"] = "*** MEAN ***"
        summary["answer"] = ""
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        df.to_csv(output_csv, index=False)
        return df if return_df else None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_ragas_clients(self):
        """Build RAGAS-compatible LLM and embedding wrappers."""
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        # Rotation IS enabled here. `RotatingChatModel` IS a `BaseChatModel`,
        # so ragas' `LangchainLLMWrapper` accepts it, and its `_generate` /
        # `_agenerate` rotate across every {ENV}, {ENV}2, ... key on rate-limit
        # errors. If your keys are from separate orgs each has its own TPD
        # bucket, which is what unblocks high-volume judge runs.
        llm = build_llm(self.judge_provider, self.judge_model, self.api_key,
                        temperature=0.0)
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)

    def _evaluate_one(
        self,
        row: dict,
        ragas_llm,
        ragas_embeddings,
        ground_truth_col: str,
    ) -> dict:
        """Score a single question across all 4 RAGAS metrics.

        Each metric is evaluated independently so a timeout on one metric
        doesn't wipe the others.  AllKeysExhaustedError is NOT caught here —
        it propagates to the caller which can flush state and stop cleanly.
        """
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
        from ragas.run_config import RunConfig

        ground_truth = str(row.get(ground_truth_col, ""))
        contexts = row.get("retrieved_chunks", [])
        if not isinstance(contexts, list):
            contexts = [str(contexts)]
        if not contexts:
            contexts = ["No context retrieved."]

        sample = SingleTurnSample(
            user_input=row["question"],
            retrieved_contexts=contexts,
            response=row["answer"],
            reference=ground_truth,
        )
        dataset = EvaluationDataset(samples=[sample])
        # timeout per sub-job (one LLM call); low enough that a slow-but-not-
        # exhausted key fails fast.  AllKeysExhaustedError propagates through
        # raise_exceptions=True before the timeout can matter.
        run_cfg = RunConfig(timeout=min(self.timeout, 60), max_workers=1)

        col_map = {
            "faithfulness": "faithfulness",
            "response_relevancy": "answer_relevancy",
            "answer_relevancy": "answer_relevancy",
            "llm_context_precision_with_reference": "context_precision",
            "context_precision": "context_precision",
            "llm_context_recall": "context_recall",
            "context_recall": "context_recall",
        }

        # Evaluate each metric separately so a timeout on one doesn't null
        # the others.  AllKeysExhaustedError propagates (not caught here).
        metric_defs = [
            Faithfulness(),
            ResponseRelevancy(strictness=1),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ]

        score_row: dict = {
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

        for metric in metric_defs:
            try:
                result = evaluate(
                    dataset=dataset,
                    metrics=[metric],
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    run_config=run_cfg,
                    raise_exceptions=True,   # propagates AllKeysExhaustedError immediately
                )
                sdf = result.to_pandas()
                if not sdf.empty:
                    for ragas_col, our_col in col_map.items():
                        if ragas_col in sdf.columns:
                            val = sdf.iloc[0].get(ragas_col)
                            if val is not None and str(val) not in ("nan", "None"):
                                score_row[our_col] = float(val)
            except Exception as e:
                from finagent.llm import AllKeysExhaustedError
                if isinstance(e, AllKeysExhaustedError):
                    raise   # propagate → outer loop stops the run and saves progress
                # TimeoutError or other transient failure — leave this metric as None
                if score_row["error"] is None:
                    score_row["error"] = str(e)

        return score_row

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
        choices=["groq", "gemini", "openai", "anthropic"],
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