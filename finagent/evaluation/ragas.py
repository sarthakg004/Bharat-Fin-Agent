"""Answer-quality harness. Takes a run's outputs JSON and measures it with six
RAGAS metrics:

    Faithfulness          is EVERY claim extracted from the answer entailed by
                          the retrieved context?
    Response Groundedness is the answer AS A WHOLE supported by the context
                          (0/1/2 by two judges, averaged)? Coarser than
                          faithfulness and far cheaper, and it does not punish an
                          answer for citation boilerplate.
    Response Relevancy    does the answer address the question?
    Context Precision     are the retrieved chunks relevant — is signal ranked
                          above noise?
    Context Recall        do the retrieved chunks cover the gold answer?
    Answer Correctness    is the answer RIGHT? The only metric here that reads
                          gold as the target rather than as a coverage checklist.
                          Every other metric can be perfect on a confidently
                          wrong answer: one fabricated from a chunk scores 1.0
                          faithfulness if it is faithful to that chunk.

The judge LLM is deliberately DIFFERENT from the generator to avoid
self-evaluation bias. The agent generates on Gemini; the judge runs on Groq
(`judge_provider="groq"`, the default) unless told otherwise.
"""

from __future__ import annotations

import sys
import types

import os

# Opt out of RAGAS telemetry. It phones home several times per metric, and where
# DNS for that host doesn't resolve (offline, split-horizon DNS, blocked egress)
# each call blocks through urllib3's retry ladder: a single-question five-metric
# score spent ~115 s of its 125 s here. Must be set before `import ragas`.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

# ragas 0.4.3 still imports ChatVertexAI from a module langchain-community 0.4.x
# removed. Vertex is never used, so a stub is enough to resolve the import chain.
# Must also run before `import ragas`. Drop when ragas stops importing it.
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
# With 6 metrics × N questions, budget ~2s per metric call to stay safe.
GROQ_RATE_LIMIT_DELAY = 2.0

# The scored columns, in report order. `groundedness` is RAGAS'
# `ResponseGroundedness` (NVIDIA's metric): two judges rate the WHOLE answer
# against the context 0/1/2, averaged and normalised. Faithfulness asks the
# same question per extracted claim, which on this corpus penalises the
# provenance boilerplate the synthesizer adds (see answer_match.py) — so the
# pair reads as "is it grounded overall" vs "is every claim grounded".
#
# `answer_correctness` is RAGAS' weighted blend of a factual F1 (claims in the
# answer vs claims in the gold answer, decomposed by the judge) and embedding
# cosine similarity, default 0.75/0.25. It is the most expensive metric in the
# set — the factual half is a second claim-decomposition — which is why it sits
# LAST: `_score_row` skips any metric already carried from a prior partial run,
# so a resumed pass does not re-buy it.
METRIC_COLUMNS = ("faithfulness", "groundedness", "answer_relevancy",
                  "context_precision", "context_recall", "answer_correctness")

# Context caps handed to the judge, per chunk and in total. Chunks are
# parent-document passages (up to 4000 chars) and a question can retrieve 25 of
# them, so the raw evidence runs to ~50k chars ≈ 12k tokens. Groq's free tier
# caps a SINGLE request at the model's TPM (8k for qwen, 12k for llama-70b) and
# rejects anything larger with a 413 — which RAGAS surfaces as a missing score,
# not an error. These caps keep every judge prompt inside that wall. They are
# applied identically to every row, so runs stay comparable; raise them on a
# paid judge with a real context budget.
CONTEXT_CHAR_CAP = 2000        # per chunk
CONTEXT_TOTAL_CHAR_CAP = 24000  # all chunks in one prompt (~6k tokens)

# How many questions in a row may fail to score at all — each after waiting out
# the pool's cooldown — before the run gives up. One is normal on a free tier
# (an expensive question outruns a key's per-minute budget); several in a row
# means the daily budget really is spent and waiting will not help.
MAX_CONSECUTIVE_EXHAUSTIONS = 4

# Groq's per-minute token bucket refills in 60s; +10s so we land after it.
TOKEN_BUCKET_REFILL_S = 70


def _score_of(row: Optional[dict], column: str) -> Optional[float]:
    """A previously recorded metric value, or None if it was never scored.

    Values come back off a CSV, so a missing score is `nan` rather than None —
    and `nan` is not falsy, so a plain truth test would carry it forward as a
    real score and the metric would never be retried.
    """
    if not row:
        return None
    val = row.get(column)
    if val is None:
        return None
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None
    return None if val != val else val          # nan -> never scored


def _cap_contexts(contexts: list) -> list[str]:
    """Trim the retrieved contexts to what one judge request can carry.

    Chunks keep their retrieval ORDER (the reranker's), so what gets dropped is
    the tail the agent ranked least relevant — the same thing a smaller `top_k`
    would have dropped.
    """
    out: list[str] = []
    budget = CONTEXT_TOTAL_CHAR_CAP
    for c in contexts:
        text = str(c).strip()[:CONTEXT_CHAR_CAP]
        if not text:
            continue
        if len(text) > budget:
            text = text[:budget]
        if not text:
            break
        out.append(text)
        budget -= len(text)
    return out


class RAGASEvaluator:
    """Evaluate RAG outputs with the metrics in `METRIC_COLUMNS`.

    `judge_model` should be DIFFERENT and stronger than the generator, to avoid
    self-evaluation bias; None picks a per-provider default. The judge makes
    several calls per question per metric, so Gemini's free tier only stretches
    to a 1-2 question smoke test — use Groq/OpenAI/Anthropic for real runs.
    `embedding_model` serves ResponseRelevancy and should match ingestion.
    `api_key` falls back to the provider's env var.
    """

    # Judge defaults. On Groq the judge is `llama-3.3-70b-versatile`: it is the
    # largest non-reasoning model on the free tier and the only one with a
    # 12k-TPM per-request budget (every other option caps at 8k, which a
    # faithfulness prompt over real filing chunks exceeds outright). Qwen 3.x
    # were the previous default and are a false economy here — reasoning tokens
    # eat the completion budget, so structured metrics come back empty.
    # Everything except groq comes from `runtime.DEFAULTS`, the table the rest
    # of the app already uses, because a SECOND copy of the model names is how
    # this broke: the Gemini migration updated runtime.py and left "gemini" here
    # pointing at `gemini-2.5-flash`, which Google has since retired. Every
    # judge call 404'd, and RAGAS answers a failed call by "skipping a sample by
    # assigning it nan" — so the run kept going and would have produced a fully
    # empty scorecard rather than an error.
    #
    # The judge takes the CRITIC slot, not synth. It is the highest-volume role
    # in the eval (six metrics x every row, several calls each) and the free
    # tier meters requests per day PER MODEL, so putting it on the same model as
    # the answers would drain one bucket twice as fast instead of using two.
    from finagent.runtime import DEFAULTS as _RUNTIME_DEFAULTS

    DEFAULT_MODELS = {
        "groq": "llama-3.3-70b-versatile",
        **{p: m["critic"] for p, m in _RUNTIME_DEFAULTS.items() if p != "groq"},
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
        # "gold", not "answer". `answer_correctness` scores the response AGAINST
        # this column, so defaulting it to the response's own column compares the
        # answer to itself and returns a silent, confident ~1.0 — the newest and
        # most expensive metric reporting a perfect score for free. Only
        # `runner.py` passed the override; this module's own CLI did not.
        ground_truth_col: str = "gold",
        sample: Optional[int] = None,
        batch_size: int = 10,
    ) -> pd.DataFrame:
        """Run RAGAS over all outputs and save the results CSV.

        Resumable: if `output_csv` exists, questions that already carry at least
        one non-null score are skipped and their rows kept as-is. Scores flush
        after every batch, so partial progress survives a crash or a quota
        exhaustion.

        `ground_truth_col` names the key holding the gold answer. A smaller
        `batch_size` means more API calls but is safer against judge timeouts.
        """
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        numeric_cols = list(METRIC_COLUMNS)

        # ---- 1. Load outputs ------------------------------------------------
        outputs = self._load_outputs(outputs_path)
        outputs = [o for o in outputs if o.get("answer") and not o.get("error")]
        if sample:
            outputs = outputs[:sample]

        # ---- 1b. Resume: load already-scored rows ---------------------------
        # A row is complete only when EVERY metric is present. Incomplete rows
        # are re-queued, but their existing scores are carried into the retry
        # so only the MISSING metrics are re-run.
        #
        # This used to discard partial rows and re-score them from scratch, and
        # on a rate-limited judge that never converged: the cheap context-free
        # metric would land, the four context-heavy ones would be refused, and
        # the next attempt threw away the one that succeeded and repeated the
        # whole thing. Keeping partials makes every attempt monotonic.
        already_scored: dict[str, dict] = {}
        prior: dict[str, dict] = {}
        if output_csv.exists():
            try:
                existing = pd.read_csv(output_csv)
                real = existing[existing["question"] != "*** MEAN ***"]
                complete = real[numeric_cols].notna().all(axis=1)
                for (_, row), done in zip(real.iterrows(), complete):
                    key = str(row["question"])
                    prior[key] = row.to_dict()
                    if done:
                        already_scored[key] = row.to_dict()
                partial = len(real) - len(already_scored)
                if already_scored or partial:
                    print(f"Resume: {len(already_scored)} fully scored, "
                          f"{partial} partial (missing metrics only will be "
                          f"re-run) — skipping {len(already_scored)}.")
            except Exception as e:
                print(f"  ! Could not read existing CSV ({e}); starting fresh.")

        pending = [o for o in outputs if o.get("question", "") not in already_scored]
        print(f"Evaluating {len(pending)}/{len(outputs)} outputs with RAGAS "
              f"(judge: {self.judge_model})")

        # ---- 2. Configure RAGAS LLM + embeddings ----------------------------
        self._shorten_exhaustion_latch()
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
        consecutive_exhaustions = 0

        for q_idx, row in enumerate(tqdm(pending, desc="RAGAS questions"), start=1):
            exhaust_before = _EXHAUST_COUNT.get(self.judge_provider, 0)

            score = self._evaluate_one(
                row, ragas_llm, ragas_embeddings, ground_truth_col,
                prior=prior.get(row.get("question", "")),
            )
            new_scores.append(score)

            exhaust_after = _EXHAUST_COUNT.get(self.judge_provider, 0)
            new_exhaustions = exhaust_after - exhaust_before

            # An exhaustion while scoring ONE question does not mean the day's
            # quota is gone. Five metrics over real filing chunks cost ~40k judge
            # tokens, and a free key holds 12k per MINUTE — so a single expensive
            # question can spend every key's current minute all by itself, while
            # each key still has ~99% of its daily budget. Treating that as fatal
            # deadlocked the run: the first pending question tripped it, the row
            # was discarded as partial, and every retry re-hit the same question
            # and stopped again — three attempts in a row scored nothing.
            #
            # So: wait out the cooldown the rotator already latched, and carry
            # on. Give up only when several questions IN A ROW cannot be scored
            # even after waiting, which is what a genuinely spent pool looks like.
            if new_exhaustions >= 1:
                self._flush_csv(output_csv, already_scored, new_scores, numeric_cols,
                                prior=prior)
                scored = sum(1 for c in METRIC_COLUMNS if score.get(c) is not None)
                consecutive_exhaustions = 0 if scored else consecutive_exhaustions + 1
                if consecutive_exhaustions >= MAX_CONSECUTIVE_EXHAUSTIONS:
                    scored_so_far = len(already_scored) + len(new_scores)
                    remaining = len(pending) - q_idx
                    print(f"\n⏸  LIMIT EXHAUSTED — all {self.judge_provider} keys are "
                          f"rate-limited across {consecutive_exhaustions} questions in a row.")
                    print(f"   {scored_so_far}/{len(outputs)} rows scored so far.")
                    print(f"   Progress saved to {output_csv}")
                    print(f"   Re-run the same command to score the remaining {remaining} rows.")
                    raise AllKeysExhaustedError(
                        f"All {self.judge_provider} keys exhausted after question {q_idx}"
                    )
                self._wait_out_cooldown(q_idx, scored)
                continue
            consecutive_exhaustions = 0

            # Flush CSV periodically so progress is never lost.
            if q_idx % flush_every == 0 or q_idx == len(pending):
                self._flush_csv(output_csv, already_scored, new_scores, numeric_cols,
                                prior=prior)

            time.sleep(GROQ_RATE_LIMIT_DELAY)

        # ---- 4. Build final DataFrame ----------------------------------------
        df = self._flush_csv(
            output_csv, already_scored, new_scores, numeric_cols,
            prior=prior, return_df=True,
        )
        print(f"\nResults written → {output_csv}")
        self._print_summary(df, numeric_cols)
        return df

    @staticmethod
    def _shorten_exhaustion_latch() -> None:
        """Stop the pool's fail-fast latch from turning a rate limit into a NaN.

        When a full key rotation fails, `RotatingChatModel` latches the provider
        as exhausted and then raises WITHOUT touching the network until the latch
        expires — right for a user waiting on one request, wrong here. RAGAS
        wraps every metric call in tenacity with exponential backoff, so it would
        happily ride out a rate limit and succeed; but under a 300 s latch all
        ten of its retries fail instantly inside the window and the metric comes
        back unscored. That is why the context-heavy metrics never landed while
        the one context-free metric always did.

        A short latch lets the retries actually reach the network, by which time
        the per-minute buckets have refilled. Batch scorer only — the serve path
        keeps the long latch that fails fast for a waiting user.
        """
        from finagent import llm

        llm._EXHAUST_COOLDOWN_S = 5.0
        llm._EXHAUST_COOLDOWN_MAX_S = 5.0

    def _wait_out_cooldown(self, q_idx: int, scored: int) -> None:
        """Sleep one token-bucket refill, then clear the rotator's latch.

        On a full rotation failure the pool latches `_EXHAUSTED_UNTIL` and fails
        fast without touching the network until it expires — so a scorer that
        doesn't sleep just burns the rest of the queue scoring nothing. But the
        latch is sized for a *user* waiting on one request (fail fast, don't
        hang); what actually ran out here is a per-MINUTE token bucket, which
        refills in 60 s. Honouring the full 300 s latch cost ~170 s per question
        against a ~70 s recovery. A batch scorer wants the shorter, true wait,
        so sleep the refill and drop the latch. If the pool really is spent, the
        next questions fail again and the consecutive-exhaustion guard stops the
        run within a few minutes.
        """
        from finagent.llm import _EXHAUSTED_UNTIL

        print(f"\n   pool rate-limited on question {q_idx} "
              f"({scored}/{len(METRIC_COLUMNS)} metrics scored); waiting "
              f"{TOKEN_BUCKET_REFILL_S}s for the per-minute buckets to refill, "
              f"then continuing.", flush=True)
        time.sleep(TOKEN_BUCKET_REFILL_S)
        _EXHAUSTED_UNTIL[self.judge_provider] = 0.0

    def _flush_csv(
        self,
        output_csv: Path,
        already_scored: dict,
        new_scores: list[dict],
        numeric_cols: list[str],
        prior: Optional[dict] = None,
        return_df: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Merge every row we know about, append MEAN row, write CSV.

        `prior` must be included or a flush ERASES the partial rows this attempt
        has not re-reached yet: they are not in `already_scored` (which holds
        complete rows only) and not yet in `new_scores`. They survive in memory,
        so a run that finishes writes them back — but a run that dies mid-attempt
        leaves a CSV missing them, and the next attempt re-scores them from
        scratch. That is the carry-forward guarantee failing in exactly the case
        it exists for.
        """
        merged: dict[str, dict] = dict(prior or {})
        merged.update(already_scored)
        for row in new_scores:
            merged[str(row.get("question", ""))] = row
        all_rows = list(merged.values())
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
        # Reasoning OFF for the judge. Qwen 3.x on Groq spends the completion
        # budget thinking and returns `finish_reason=length` with EMPTY content
        # on RAGAS' structured-output metrics — which RAGAS surfaces as "increase
        # max_tokens" and the harness stored as a NaN. That is what scored v4's
        # faithfulness on 7 of 150 rows. Measured on one faithfulness call:
        # reasoning on = 4.2 s, truncated, no score; off = 0.3 s, valid JSON.
        extra = ({"reasoning_effort": "none"}
                 if self.judge_provider == "groq"
                 and self.judge_model.startswith("qwen/") else {})
        llm = build_llm(self.judge_provider, self.judge_model, self.api_key,
                        temperature=0.0, **extra)
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)

    @staticmethod
    def _install_asyncio_cleanup_filter() -> None:
        """Suppress the harmless 'Event loop is closed' noise from httpx/anyio.

        RAGAS creates a new asyncio event loop per evaluate() call.  When that
        loop closes, open httpx connections try to clean up asynchronously and
        land in already-closed Tasks.  asyncio prints 'Task exception was never
        retrieved' for each one.  This is a known upstream issue (httpx +
        anyio + RAGAS teardown ordering) — the evaluation results are correct.
        We swallow only that specific exception class so genuine async errors
        still surface.
        """
        import asyncio

        def _handler(loop, context):
            exc = context.get("exception")
            if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
                return  # swallow cleanup noise
            loop.default_exception_handler(context)

        try:
            loop = asyncio.get_event_loop()
            loop.set_exception_handler(_handler)
        except RuntimeError:
            pass  # no running loop yet — handler will be set when RAGAS creates one

    def _evaluate_one(
        self,
        row: dict,
        ragas_llm,
        ragas_embeddings,
        ground_truth_col: str,
        prior: Optional[dict] = None,
    ) -> dict:
        """Score a single question across all five RAGAS metrics — in ONE
        `evaluate()` call.

        `prior` is this question's row from a previous attempt: any metric that
        already has a value there is carried over and NOT re-run, so a
        rate-limited run converges instead of re-paying for scores it holds.

        One call, not one per metric. A single `evaluate()` costs ~30 s of fixed
        overhead (executor + event-loop setup and teardown) around HTTP calls
        that each take under a second; five of them per question was ~150 s of
        pure overhead. `raise_exceptions=False` keeps the per-metric isolation
        that the old loop was written for: a metric that fails comes back NaN
        and the other four still score. Key exhaustion is detected by the
        caller from `_EXHAUST_COUNT`, which is why it need not raise here.
        """
        self._install_asyncio_cleanup_filter()
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import (
            AnswerCorrectness,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseGroundedness,
            ResponseRelevancy,
        )
        from ragas.run_config import RunConfig

        ground_truth = str(row.get(ground_truth_col, ""))
        contexts = row.get("retrieved_chunks", [])
        if not isinstance(contexts, list):
            contexts = [str(contexts)]
        contexts = _cap_contexts(contexts)
        if not contexts:
            contexts = ["No context retrieved."]

        sample = SingleTurnSample(
            user_input=row["question"],
            retrieved_contexts=contexts,
            response=row["answer"],
            reference=ground_truth,
        )
        dataset = EvaluationDataset(samples=[sample])
        # `max_workers` matters most for context precision: it issues ONE LLM
        # call per retrieved chunk, so at 1 worker a 12-chunk question serialises
        # 12 round-trips and blows any sane timeout. The key pool absorbs the
        # concurrency. The timeout has to clear the slowest metric's full fan-out,
        # not one call. AllKeysExhaustedError propagates through
        # raise_exceptions=True before the timeout can matter.
        run_cfg = RunConfig(timeout=min(self.timeout, 240),
                            max_workers=self.max_workers)

        col_map = {
            "faithfulness": "faithfulness",
            "nv_response_groundedness": "groundedness",
            "response_relevancy": "answer_relevancy",
            "answer_relevancy": "answer_relevancy",
            "llm_context_precision_with_reference": "context_precision",
            "context_precision": "context_precision",
            "llm_context_recall": "context_recall",
            "context_recall": "context_recall",
            "answer_correctness": "answer_correctness",
        }

        factories = {
            "faithfulness": Faithfulness,
            "groundedness": ResponseGroundedness,
            "answer_relevancy": lambda: ResponseRelevancy(strictness=1),
            "context_precision": LLMContextPrecisionWithReference,
            "context_recall": LLMContextRecall,
            "answer_correctness": AnswerCorrectness,
        }
        carried = {c: _score_of(prior, c) for c in METRIC_COLUMNS}
        wanted = [c for c in METRIC_COLUMNS if carried[c] is None]
        # `answer_correctness` scores the answer AGAINST the gold answer, so
        # without one it is not a low score — it is no score. Letting it run
        # anyway returns ~0 and drags the reported mean down with rows that were
        # never measurable. The context metrics take `reference` too, but as a
        # coverage checklist where an empty one is merely uninformative.
        if not ground_truth.strip():
            wanted = [c for c in wanted if c != "answer_correctness"]
        metric_defs = [factories[c]() for c in wanted]

        score_row: dict = {
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "company": row.get("company") or row.get("company_name") or row.get("ticker", ""),
            "year": row.get("year", ""),
            **carried,
            "error": None,
        }
        if not metric_defs:
            return score_row

        try:
            result = evaluate(
                dataset=dataset,
                metrics=metric_defs,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                run_config=run_cfg,
                raise_exceptions=False,  # a failed metric → NaN, not a dead row
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
            score_row["error"] = str(e)

        missing = [c for c in wanted if score_row[c] is None]
        if missing and not score_row["error"]:
            score_row["error"] = f"no score from judge: {', '.join(missing)}"
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
        print("=" * 60)
        print(f"RAGAS Evaluation — {len(data)} questions")
        print("=" * 60)
        print(f"{'Metric':<28} {'Mean':>8}  {'Std':>8}  {'n':>5}")
        print("-" * 60)
        for col in numeric_cols:
            vals = pd.to_numeric(data[col], errors="coerce").dropna()
            mean = vals.mean() if len(vals) else float("nan")
            std = vals.std() if len(vals) else float("nan")
            # `n` < the question count means the judge failed on that metric
            # (timeout / over-sized prompt) and the mean silently skipped rows.
            print(f"  {col:<26} {mean:>8.3f}  {std:>8.3f}  {len(vals):>5}")
        print("=" * 60)


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