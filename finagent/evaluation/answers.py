"""Answer evaluation on FinanceBench: run the production agent, then score it.

    run     answer every question with `service.run_agent` (what a user gets)
    score   six RAGAS metrics + a judge-free numeric accuracy check + a report

RAGAS metrics, and what each one can and cannot tell you:

    faithfulness        is every claim in the answer supported by the evidence?
    groundedness        is the answer as a whole supported by the evidence?
    answer_relevancy    does the answer address the question?
    context_precision   are the retrieved passages relevant?
    context_recall      do the retrieved passages cover the gold answer?
    answer_correctness  does the answer match the gold answer?
    verdict             same answer as the gold, whatever the length? (1 or 0)

Only the last two compare against the gold answer. The first two stay high
for an honest "the evidence does not say" and for an answer faithful to the
wrong passage. And answer_correctness counts every extra true statement as a
mistake: gold answers are one line, so a correct but long answer scores about
0.4. `verdict` is our own yes/no criterion (RAGAS AspectCritic), not a
standard metric: it ignores length, and the report checks it against the
judge-free `numeric_accuracy` on the numeric questions.

Both steps are resumable: re-running continues where the last run stopped.

    python -m finagent.evaluation.answers run   --output results/v7/answers.json
    python -m finagent.evaluation.answers score --output results/v7/answers.json

The free-tier judge cannot score 127 answers in a day. With the Claude Code CLI
logged in, Claude can write and judge instead, with no API key:

    ... run   --output results/v7/answers.json --writer haiku
    ... score --output results/v7/answers.json --judge-provider claude-cli --judge-model claude-sonnet-5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import types
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

os.environ.setdefault("QDRANT_EVAL_URL", "http://localhost:6333")
os.environ.setdefault("FINANCEBENCH_COLLECTION", "sweep_p2500_c600_gemini-embedding-2_hdr_tbl-md")
# RAGAS phones home several times per metric and blocks when that host is unreachable.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

METRICS = ("faithfulness", "groundedness", "answer_relevancy",
           "context_precision", "context_recall", "answer_correctness", "verdict")
VERDICT_DEFINITION = (
    "Compare the response with the reference answer. Answer Yes if the response reaches "
    "the same answer: the same figure (rounding and units aside) or the same conclusion. "
    "Extra correct detail does not matter, however long. Answer No if the figure or "
    "conclusion differs, or if the response does not answer (a refusal or 'cannot be "
    "determined').")
# A different model from the writer (Haiku in the Claude runs) and stronger.
JUDGE = ("claude-cli", "claude-sonnet-5")
QUESTION_TIMEOUT_S = 300
# The judge reads the retrieved passages; these caps keep its prompt bounded.
CONTEXT_CHAR_CAP, CONTEXT_TOTAL_CHAR_CAP = 2000, 24000
REFUSAL_PREFIX = "I don't have enough information to answer this"
# The 99 questions whose evidence survives HTML parsing (the retrieval eval's set).
RECOVERABLE_IDS = Path("results/financebench_retrieval_queries.json")
# FinanceBench filings the eval index holds the wrong period of (the SEC match picked
# the next quarter or the prior fiscal year), or none at all. Their questions cannot
# be answered from the index, so the report also scores the set without them.
FILING_NOT_INDEXED = {
    "JOHNSON_JOHNSON_2022_10K",   # jnj-20220102 is FY2021
    "BESTBUY_2024Q2_10Q",         # bby-20241102 is Q3 FY2025
    "AMCOR_2023Q2_10Q",           # amcr-20231231 is Q2 FY2024
    "JPMORGAN_2021Q1_10Q",        # jpm-20210930 is Q3
    "JPMORGAN_2022Q2_10Q",        # not fetched
    "JPMORGAN_2023Q2_10Q",        # jpm-20230930 is Q3
    "MGMRESORTS_2023Q2_10Q",      # mgm-20230930 is Q3
    "Pfizer_2023Q2_10Q",          # pfe-20231001 is Q3
}


# --------------------------------------------------------------------------- #
# Claude through the local CLI
# --------------------------------------------------------------------------- #

class ClaudeCLI(BaseChatModel):
    """A chat model backed by `claude -p`, so an eval run can use the Claude
    Code login instead of an API key. One process per call, no tools."""

    model: str = "sonnet"
    # The CLI thinks by default. Off for the judge: 1,118 -> 150 output tokens
    # on one statement-classification prompt.
    thinking: bool = True

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    def _generate(self, messages, stop=None, run_manager=None, schema: Optional[dict] = None,
                  **_: Any) -> ChatResult:
        from finagent.llm import text_of

        system = "\n\n".join(text_of(m) for m in messages if isinstance(m, SystemMessage))
        prompt = "\n\n".join(text_of(m) for m in messages if not isinstance(m, SystemMessage))
        cmd = ["claude", "-p", "--model", self.model, "--output-format", "json", "--tools", "",
               "--system-prompt", system or "Follow the instructions in the message exactly.",
               "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands",
               "--setting-sources", ""]
        if schema:
            cmd += ["--json-schema", json.dumps(schema)]
        error = ""
        for attempt in range(4):
            # An empty directory: the CLI must not pick up this project's notes.
            done = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                  timeout=600, cwd=tempfile.gettempdir(),
                                  env=None if self.thinking else {**os.environ, "MAX_THINKING_TOKENS": "0"})
            try:
                out = json.loads(done.stdout)
            except ValueError:
                out = {"is_error": True, "result": done.stderr or done.stdout}
            if not out.get("is_error") and (out.get("result") or out.get("structured_output")):
                usage = out.get("usage") or {}
                tokens_in = ((usage.get("input_tokens") or 0)
                             + (usage.get("cache_read_input_tokens") or 0)
                             + (usage.get("cache_creation_input_tokens") or 0))
                tokens_out = usage.get("output_tokens") or 0
                return ChatResult(generations=[ChatGeneration(message=AIMessage(
                    content=out.get("result") or "",
                    additional_kwargs={"structured_output": out.get("structured_output")},
                    response_metadata={"stop_reason": out.get("stop_reason") or "end_turn"},
                    usage_metadata={"input_tokens": tokens_in, "output_tokens": tokens_out,
                                    "total_tokens": tokens_in + tokens_out}))])
            error = str(out.get("result"))[:300]
            time.sleep(15 * (attempt + 1))
        raise RuntimeError(f"claude CLI failed: {error}")

    def with_structured_output(self, schema, **_: Any):
        def parse(message: AIMessage):
            data = message.additional_kwargs.get("structured_output")
            return schema.model_validate(data if data is not None else json.loads(message.content))
        return self.bind(schema=schema.model_json_schema()) | RunnableLambda(parse)


# --------------------------------------------------------------------------- #
# run: answer the questions
# --------------------------------------------------------------------------- #

@contextmanager
def _time_limit(seconds: int):
    """Abort one question after `seconds` (a stuck network read once stalled a run)."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def fire(signum, frame):
        raise TimeoutError(f"question exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, fire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def run(output: Path, sample: Optional[int] = None, writer: Optional[str] = None) -> list[dict]:
    """Answer each question and save after every one. Rows with an error are retried.

    `writer` names a Claude model ("sonnet", "haiku") that writes and fact-checks
    through the local CLI. Planning, extraction and retrieval stay as in production.
    """
    from tqdm import tqdm

    from finagent.api.service import run_agent
    from finagent.config import settings
    from finagent.evaluation.retrieval import load_questions
    from finagent.llm import classify_error

    os.environ["DISABLE_DYNAMIC_FETCH"] = "1"       # the eval corpus is fixed
    if writer:
        from finagent.agent import agent as agent_module

        production = agent_module.create_llm
        agent_module.create_llm = lambda ctx, role: (
            ClaudeCLI(model=writer) if role in ("writer", "critic") else production(ctx, role))
    questions = load_questions()[:sample]
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if output.exists():
        done = {r["financebench_id"]: r for r in json.loads(output.read_text()) if not r.get("error")}

    rows: list[dict] = []
    for q in tqdm(questions, desc="answering"):
        if q["financebench_id"] in done:
            rows.append(done[q["financebench_id"]])
            continue
        row = {"financebench_id": q["financebench_id"], "question": q["question"],
               "gold": q.get("answer", ""), "qtype": q["qtype"], "company": q.get("company", ""),
               "answer": "", "retrieved_chunks": [], "error": None,
               "writer": writer or "production"}
        # A few questions never name their company. The benchmark is open-book
        # over a known filing, so name it for retrieval's company filter.
        question = q["question"]
        if q.get("company") and q["company"].split()[0].lower() not in question.lower():
            question = f"{question} (Company: {q['company']})"
        t0 = time.time()
        try:
            with _time_limit(QUESTION_TIMEOUT_S):
                res = run_agent(question, collection=settings.financebench_collection)
            meta = res["metadata"]
            row.update(answer=res["answer"],
                       retrieved_chunks=[c["text"] for c in res["chunks"]],
                       refused=meta["refused"], support_score=meta["support_score"],
                       latency_s=round(time.time() - t0, 2),
                       input_tokens=meta["input_tokens"], output_tokens=meta["output_tokens"])
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            if classify_error(e).kind == "quota":
                rows.append(row)
                output.write_text(json.dumps(rows, indent=2))
                print("\nDaily quota used up. Re-run the same command tomorrow to resume.")
                break
        rows.append(row)
        output.write_text(json.dumps(rows, indent=2))
    return rows


# --------------------------------------------------------------------------- #
# Numeric accuracy (no judge)
# --------------------------------------------------------------------------- #

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.findall(str(text).replace(",", "")):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def _is_year(x: float) -> bool:
    return x == int(x) and 1990 <= x <= 2035


def numeric_match(gold: str, answer: str, tol: float = 0.01) -> Optional[bool]:
    """Is a gold figure in the answer, within 1%? None when the gold answer has
    no figure. "0.40" also matches "40%".

    Years do not count as figures: "FY2024" in both the gold answer and a
    refusal used to score as a correct answer.
    """
    gold_nums = [g for g in _numbers(gold) if not _is_year(g)]
    if not gold_nums:
        return None
    answer_nums = [a for a in _numbers(answer) if not _is_year(a)]
    for g in gold_nums:
        for target in {g, g * 100.0, g / 100.0}:
            for a in answer_nums:
                if (abs(a) < 1e-9) if target == 0 else (abs(a - target) / abs(target) <= tol):
                    return True
    return False


def numeric_accuracy(rows: list[dict]) -> dict:
    """Share of numeric questions whose gold figure appears in the answer."""
    verdicts = [v for r in rows if r.get("qtype") == "numeric"
                and (v := numeric_match(r.get("gold", ""), r.get("answer", ""))) is not None]
    n = len(verdicts)
    return {"n": n, "correct": sum(verdicts), "accuracy": round(sum(verdicts) / n, 4) if n else None}


def verdict_agreement(rows: list[dict], scored: dict) -> dict:
    """How often the judge's verdict agrees with the judge-free numeric check."""
    pairs = [(v, scored[r["financebench_id"]]["verdict"] == 1) for r in rows
             if r.get("qtype") == "numeric" and (scored.get(r["financebench_id"]) or {}).get("verdict") is not None
             and (v := numeric_match(r.get("gold", ""), r.get("answer", ""))) is not None]
    return {"n": len(pairs), "agree": sum(a == b for a, b in pairs)}


# --------------------------------------------------------------------------- #
# score: RAGAS
# --------------------------------------------------------------------------- #

def _ragas_clients(judge_provider: str, judge_model: str):
    # ragas still imports a module langchain-community removed; a stub satisfies it.
    mod = "langchain_community.chat_models.vertexai"
    if mod not in sys.modules:
        shim = types.ModuleType(mod)
        shim.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[mod] = shim
    warnings.filterwarnings("ignore", message=r".*Langchain(LLM|Embeddings)Wrapper is deprecated.*")

    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from finagent.llm import build_llm

    # A small local embedder for the similarity parts of two metrics.
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                       encode_kwargs={"normalize_embeddings": True})
    judge = (ClaudeCLI(model=judge_model, thinking=False) if judge_provider == "claude-cli"
             else build_llm(judge_provider, judge_model))
    return LangchainLLMWrapper(judge), LangchainEmbeddingsWrapper(embeddings)


def _cap_contexts(contexts: list) -> list[str]:
    out, budget = [], CONTEXT_TOTAL_CHAR_CAP
    for c in contexts:
        text = str(c).strip()[:min(CONTEXT_CHAR_CAP, budget)]
        if text:
            out.append(text)
            budget -= len(text)
    return out or ["No context retrieved."]


def _score_row(row: dict, prior: dict, llm, embeddings) -> dict:
    """Score one answer. Metrics already scored in `prior` are kept, not re-bought."""
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import (AnswerCorrectness, AspectCritic, Faithfulness,
                               LLMContextPrecisionWithReference, LLMContextRecall,
                               ResponseGroundedness, ResponseRelevancy)
    from ragas.metrics.base import MetricType
    from ragas.run_config import RunConfig

    factories = {
        "faithfulness": Faithfulness, "groundedness": ResponseGroundedness,
        "answer_relevancy": lambda: ResponseRelevancy(strictness=1),
        "context_precision": LLMContextPrecisionWithReference,
        "context_recall": LLMContextRecall, "answer_correctness": AnswerCorrectness,
        # Question, answer and gold only: the passages would turn it into a support check.
        "verdict": lambda: AspectCritic(name="verdict", definition=VERDICT_DEFINITION,
                                        required_columns={MetricType.SINGLE_TURN: {
                                            "user_input", "response", "reference"}}),
    }
    ragas_names = {"nv_response_groundedness": "groundedness",
                   "response_relevancy": "answer_relevancy",
                   "llm_context_precision_with_reference": "context_precision",
                   "llm_context_recall": "context_recall"}
    scores = {m: prior.get(m) for m in METRICS}
    gold = str(row.get("gold", ""))
    wanted = [m for m in METRICS if scores[m] is None
              and not (m in ("answer_correctness", "verdict") and not gold.strip())]
    if wanted:
        sample = SingleTurnSample(user_input=row["question"], response=row["answer"],
                                  reference=gold,
                                  retrieved_contexts=_cap_contexts(row.get("retrieved_chunks") or []))
        try:
            result = evaluate(dataset=EvaluationDataset(samples=[sample]),
                              metrics=[factories[m]() for m in wanted], llm=llm,
                              embeddings=embeddings, raise_exceptions=False,
                              run_config=RunConfig(timeout=240, max_workers=4)).to_pandas()
            for col in result.columns:
                name = ragas_names.get(col, col)
                val = result.iloc[0][col]
                if name in METRICS and val == val and val is not None:      # skip NaN
                    scores[name] = float(val)
        except Exception as e:
            print(f"  ! scoring failed: {type(e).__name__}: {str(e)[:120]}")
    return {"financebench_id": row["financebench_id"], **scores}


def score(output: Path, judge_provider: str = JUDGE[0], judge_model: str = JUDGE[1],
          sample: Optional[int] = None) -> dict:
    """RAGAS-score the answers in `output` and write the report next to it."""
    from tqdm import tqdm

    # A batch job should wait out a per-minute limit, not give up after one wait.
    os.environ.setdefault("LLM_MAX_INLINE_WAIT_S", "90")
    os.environ.setdefault("LLM_MAX_WAIT_RETRIES", "6")

    rows = [r for r in json.loads(output.read_text()) if r.get("answer") and not r.get("error")][:sample]
    scores_path = output.with_name(output.stem + "_ragas.json")
    scored = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    llm, embeddings = _ragas_clients(judge_provider, judge_model)

    empty_in_a_row = 0
    for row in tqdm(rows, desc=f"RAGAS ({judge_model})"):
        prior = scored.get(row["financebench_id"], {})
        if all(prior.get(m) is not None for m in METRICS):
            continue
        new = _score_row(row, prior, llm, embeddings)
        gained = sum(1 for m in METRICS if new[m] is not None and prior.get(m) is None)
        scored[row["financebench_id"]] = new
        scores_path.write_text(json.dumps(scored, indent=2))
        # Several questions in a row with nothing scored means the judge's daily
        # quota is gone. RAGAS turns that into empty scores, not an error.
        empty_in_a_row = 0 if gained else empty_in_a_row + 1
        if empty_in_a_row >= 3:
            print("\nThe judge returned nothing for 3 questions in a row: its daily "
                  "quota is probably used up. Re-run to resume.")
            break
    return report(output, scored)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _mean(values: list) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def report(output: Path, scored: dict) -> dict:
    """Write `<output>_report.json` and `.md`: behaviour, numeric accuracy, RAGAS."""
    rows = json.loads(output.read_text())
    n = len(rows)
    errors = [r for r in rows if r.get("error")]
    refused = [r for r in rows if (r.get("answer") or "").startswith(REFUSAL_PREFIX)]
    latencies = sorted(r["latency_s"] for r in rows if r.get("latency_s") is not None)
    recoverable = (set(json.loads(RECOVERABLE_IDS.read_text()))
                   if RECOVERABLE_IDS.exists() else set())

    def ragas(ids) -> dict:
        picked = [scored[i] for i in ids if i in scored]
        return {m: {"mean": _mean([s.get(m) for s in picked]),
                    "n": sum(1 for s in picked if s.get(m) is not None)} for m in METRICS}

    all_ids = [r["financebench_id"] for r in rows]
    from finagent.evaluation.retrieval import load_questions
    not_indexed = {q["financebench_id"] for q in load_questions()
                   if q.get("doc_name") in FILING_NOT_INDEXED}
    out = {
        "questions": n,
        "answer_rate": round((n - len(errors) - len(refused)) / n, 4) if n else None,
        "refusal_rate": round(len(refused) / n, 4) if n else None,
        "error_rate": round(len(errors) / n, 4) if n else None,
        "numeric_accuracy": numeric_accuracy(rows),
        "verdict_vs_numeric": verdict_agreement(rows, scored),
        "latency_s": {"p50": latencies[len(latencies) // 2], "p95": latencies[int(len(latencies) * .95)]}
        if latencies else None,
        "ragas": ragas(all_ids),
        "ragas_recoverable": ragas([i for i in all_ids if i in recoverable]),
        "filing_not_indexed": len(not_indexed & set(all_ids)),
        "ragas_filing_indexed": ragas([i for i in all_ids if i not in not_indexed]),
        "ragas_by_type": {t: ragas([r["financebench_id"] for r in rows if r.get("qtype") == t])
                          for t in sorted({r.get("qtype") for r in rows if r.get("qtype")})},
    }
    output.with_name(output.stem + "_report.json").write_text(json.dumps(out, indent=2))

    na = out["numeric_accuracy"]
    lines = [
        "# Answer evaluation", "", f"`{output}`, {n} questions.", "",
        "| behaviour | value |", "|---|---|",
        f"| answer rate | {out['answer_rate']} |", f"| refusal rate | {out['refusal_rate']} |",
        f"| error rate | {out['error_rate']} |",
        f"| numeric accuracy (gold figure in the answer, 1% tolerance) | "
        f"{na['correct']}/{na['n']} = {na['accuracy']} |",
        f"| verdict agrees with numeric accuracy | "
        f"{out['verdict_vs_numeric']['agree']}/{out['verdict_vs_numeric']['n']} |",
        f"| latency p50 / p95 (s) | "
        + (f"{out['latency_s']['p50']} / {out['latency_s']['p95']}" if out["latency_s"] else "n/a")
        + " |", "",
        f"| RAGAS metric | all | rows scored | evidence-recoverable subset | without the "
        f"{out['filing_not_indexed']} questions whose filing is not in the index |",
        "|---|---|---|---|---|",
        *(f"| {m} | {out['ragas'][m]['mean']} | {out['ragas'][m]['n']} | "
          f"{out['ragas_recoverable'][m]['mean']} | {out['ragas_filing_indexed'][m]['mean']} |"
          for m in METRICS), "",
        "By question type:", "",
        "| type | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS),
        *(f"| {t} | " + " | ".join(str(v[m]["mean"]) for m in METRICS) + " |"
          for t, v in out["ragas_by_type"].items()), "",
        "`answer_correctness` counts every extra true statement against a one-line "
        "gold answer, so a correct but long answer scores about 0.4. Groundedness "
        "and faithfulness stay high for an honest non-answer. Read numeric accuracy "
        "and context recall next to them.", "",
    ]
    md = output.with_name(output.stem + "_report.md")
    md.write_text("\n".join(lines))
    print(f"\n-> {md}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Answer evaluation on FinanceBench.")
    ap.add_argument("step", choices=["run", "score", "report"])
    ap.add_argument("--output", type=Path, required=True, help="the answers JSON")
    ap.add_argument("--sample", type=int, help="only the first N questions")
    ap.add_argument("--writer", help="a Claude model (sonnet, haiku) that writes and "
                                     "fact-checks through the local `claude` CLI")
    ap.add_argument("--judge-provider", default=JUDGE[0],
                    help="a provider from finagent.llm, or claude-cli")
    ap.add_argument("--judge-model", default=JUDGE[1])
    args = ap.parse_args()
    if args.step == "run":
        run(args.output, args.sample, args.writer)
    elif args.step == "score":
        score(args.output, args.judge_provider, args.judge_model, args.sample)
    else:
        scores_path = args.output.with_name(args.output.stem + "_ragas.json")
        report(args.output, json.loads(scores_path.read_text()) if scores_path.exists() else {})


if __name__ == "__main__":
    main()
