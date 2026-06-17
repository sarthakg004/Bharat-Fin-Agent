"""
parallel.py  ·  finagent/evaluation/financebench/parallel.py

Scale the FinanceBench eval across every configured Groq key.

The serial `runner.run_agent_outputs` answers one question at a time on one
key — ~150 questions take hours and a single rate limit stalls everything.
This module shards the question set across N worker *processes*, giving each
worker the key pool in a different rotation order (worker i starts on key i),
so the whole 8-key pool is saturated and a 429 on one worker never blocks the
others. Workers reuse the serial runner per shard, so resumability and the
output-row shape are identical; the orchestrator merges the shards back into
one outputs file.

Safety on shared resources:
  * each worker opens its own read-only view of the Chroma store and runs
    with PERSIST_DYNAMIC_FETCH=false, so no process ever WRITES the store
    (concurrent Chroma writes are the known segfault class here);
  * each worker is a separate process, so the non-thread-safe native stacks
    (tokenizers, hnswlib) never share state.

Usage
-----
    # 1. answer all dev+heldout questions across 4 workers
    python -m finagent.evaluation.financebench.parallel \
        --workers 4 --output results/financebench_full_outputs.json

    # 2. RAGAS-score the merged outputs and write the final metrics report
    python -m finagent.evaluation.financebench.parallel \
        --score --output results/financebench_full_outputs.json

Final metrics land in `results/final_metrics.json` + `.md` — one aggregate
list (overall + per question type), not per-question noise.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

DEFAULT_OUTPUT = "results/financebench_full_outputs.json"
DEFAULT_METRICS_JSON = "results/final_metrics.json"
DEFAULT_METRICS_MD = "results/final_metrics.md"
_SHARD_DIR = "results/shards"

# The agent's explicit refusal opener (see graph.agent.REFUSAL_TEMPLATE).
_REFUSAL_PREFIX = "I don't have enough information to answer this"


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

def _worker_env(keys: list[str], worker_idx: int) -> dict:
    """Environment for one worker: the full key pool, rotated so worker i
    STARTS on key i (workers spread across keys but can still rotate off a
    rate-limited one)."""
    env = dict(os.environ)
    # Drop every numbered key first so the rotation order is exactly ours
    # (fixed range — the numbering may have gaps or start at 1).
    for i in range(1, 33):
        env.pop(f"GROQ_API_KEY{i}", None)
    rotated = keys[worker_idx % len(keys):] + keys[:worker_idx % len(keys)]
    env["GROQ_API_KEY"] = rotated[0]
    for j, k in enumerate(rotated[1:], start=2):
        env[f"GROQ_API_KEY{j}"] = k
    # Read-only corpus: fetched filings stay in-memory per question.
    env["PERSIST_DYNAMIC_FETCH"] = "false"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _default_workers(n_keys: int) -> int:
    """Memory-aware worker default. Each worker process holds torch + the
    encoder/reranker models + a BM25 index over the corpus (~2.5 GiB), so on a
    dev laptop too many workers OOMs the *desktop session* (systemd-oomd kills
    the user slice at 50% pressure — observed killing GNOME mid-run). Budget
    one worker per ~3 GiB of MemAvailable, capped at 4 and the key count.

    This is local tooling only — it never runs on Cloud Run (the serve path
    never imports `finagent.evaluation`)."""
    try:
        with open("/proc/meminfo") as f:
            kb = int(next(l for l in f if l.startswith("MemAvailable")).split()[1])
        by_mem = max(1, kb // (3 * 1024 * 1024))
    except Exception:
        by_mem = 2
    return max(1, min(n_keys, 4, by_mem))


def run_parallel(
    questions,
    output_path: Union[str, Path] = DEFAULT_OUTPUT,
    workers: Optional[int] = None,
    provider: str = "groq",
    synth_model: Optional[str] = None,
    top_k: int = 5,
) -> list[dict]:
    """Answer `questions` (a DataFrame) across N worker processes; merge to
    `output_path`. Resumable: a worker shard that already has answers skips
    them (the serial runner's behaviour)."""
    from finagent.llm import collect_provider_keys

    keys = collect_provider_keys(provider)
    if not keys:
        raise RuntimeError(f"No {provider} API keys configured in .env")
    workers = workers or _default_workers(len(keys))
    print(f"[parallel] using {workers} worker(s)")

    output_path = Path(output_path)
    shard_dir = Path(_SHARD_DIR)
    shard_dir.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem

    # Resume across re-sharding: answers already in the merged output OR in any
    # previous shard files count as done, even if the worker count (and thus
    # the shard cut) changed since the interrupted run.
    done: dict[str, dict] = {}
    for prev in [output_path, *sorted(shard_dir.glob(f"{stem}_shard*_outputs.json"))]:
        if prev.exists():
            for row in json.loads(prev.read_text()):
                if not row.get("error"):     # errored rows are retried
                    done.setdefault(row.get("financebench_id", row.get("question")), row)
    if done:
        print(f"[parallel] resuming: {len(done)} answers carried over")

    # Round-robin shard so slow question types spread evenly across workers.
    rows = questions.reset_index(drop=True)
    procs: list[subprocess.Popen] = []
    shard_outs: list[Path] = []
    for w in range(workers):
        shard = rows.iloc[w::workers]
        if shard.empty:
            continue
        shard_in = shard_dir / f"{stem}_shard{w}.jsonl"
        shard_out = shard_dir / f"{stem}_shard{w}_outputs.json"
        shard.to_json(shard_in, orient="records", lines=True)
        # Pre-fill this shard's output with its already-answered rows so the
        # serial runner skips them.
        prefill = [done[i] for i in (
            shard.get("financebench_id", shard.get("question"))) if i in done]
        shard_out.write_text(json.dumps(prefill, indent=2))
        shard_outs.append(shard_out)
        cmd = [sys.executable, "-m", "finagent.evaluation.financebench.parallel",
               "--worker", str(shard_in), "--output", str(shard_out),
               "--provider", provider, "--top-k", str(top_k)]
        if synth_model:
            cmd += ["--synth-model", synth_model]
        procs.append(subprocess.Popen(cmd, env=_worker_env(keys, w)))
        print(f"[parallel] worker {w}: {len(shard)} questions → {shard_out}")

    failures = 0
    for w, p in enumerate(procs):
        rc = p.wait()
        if rc != 0:
            failures += 1
            print(f"[parallel] worker {w} exited with {rc} (its shard file "
                  f"keeps whatever it completed; re-run to resume)")
    merged = merge_shards(shard_outs, output_path)
    if failures:
        print(f"[parallel] {failures} worker(s) failed — merged what completed; "
              f"re-run the same command to resume the gaps.")
    return merged


def merge_shards(shard_outs: list[Path], output_path: Union[str, Path]) -> list[dict]:
    """Concatenate shard outputs (deduped by financebench_id) into one file."""
    merged: dict[str, dict] = {}
    for sp in shard_outs:
        if not Path(sp).exists():
            continue
        for row in json.loads(Path(sp).read_text()):
            merged[row.get("financebench_id", row.get("question"))] = row
    out = list(merged.values())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    print(f"[parallel] merged {len(out)} answers → {output_path}")
    return out


# --------------------------------------------------------------------------- #
# Final metrics — ONE aggregate list for the whole system
# --------------------------------------------------------------------------- #

def summarize_outputs(outputs_path: Union[str, Path]) -> dict:
    """System-level behaviour metrics computed from the outputs alone (no
    judge): coverage, refusals, errors, confidence."""
    rows = json.loads(Path(outputs_path).read_text())
    n = len(rows)
    refused = [r for r in rows
               if (r.get("answer") or "").strip().startswith(_REFUSAL_PREFIX)]
    errors = [r for r in rows if r.get("error")]
    answered = n - len(refused) - len(errors)
    confs = [r["confidence"] for r in rows if isinstance(r.get("confidence"), (int, float))]

    by_type: dict[str, dict] = {}
    for r in rows:
        qt = r.get("qtype") or "untyped"
        b = by_type.setdefault(qt, {"questions": 0, "answered": 0,
                                    "refused": 0, "errors": 0})
        b["questions"] += 1
        if r.get("error"):
            b["errors"] += 1
        elif (r.get("answer") or "").strip().startswith(_REFUSAL_PREFIX):
            b["refused"] += 1
        else:
            b["answered"] += 1

    return {
        "questions": n,
        "answered": answered,
        "answer_rate": round(answered / n, 4) if n else None,
        "refused": len(refused),
        "refusal_rate": round(len(refused) / n, 4) if n else None,
        "errors": len(errors),
        "error_rate": round(len(errors) / n, 4) if n else None,
        "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
        "by_type": by_type,
    }


def final_report(
    outputs_path: Union[str, Path] = DEFAULT_OUTPUT,
    ragas_scores: Optional[dict] = None,
    metrics_json: Union[str, Path] = DEFAULT_METRICS_JSON,
    metrics_md: Union[str, Path] = DEFAULT_METRICS_MD,
) -> dict:
    """Assemble the single final-metrics record (behaviour + RAGAS) and write
    it as JSON + a readable markdown table."""
    behaviour = summarize_outputs(outputs_path)
    report = {"outputs": str(outputs_path), "behaviour": behaviour,
              "ragas": ragas_scores or {}}

    Path(metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_json).write_text(json.dumps(report, indent=2))

    lines = [
        "# FinAgent — final evaluation metrics",
        "",
        f"Outputs: `{outputs_path}` · {behaviour['questions']} questions",
        "",
        "## System behaviour",
        "",
        "| metric | value |",
        "|---|---|",
        f"| answer rate | {behaviour['answer_rate']} |",
        f"| refusal rate | {behaviour['refusal_rate']} |",
        f"| error rate | {behaviour['error_rate']} |",
        f"| mean confidence | {behaviour['mean_confidence']} |",
        "",
    ]
    if ragas_scores:
        overall = ragas_scores.get("overall", {})
        lines += ["## RAGAS (overall)", "", "| metric | score |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in overall.items()]
        by_type = ragas_scores.get("by_type", {})
        if by_type:
            metrics = sorted({m for d in by_type.values() for m in d})
            lines += ["", "## RAGAS by question type", "",
                      "| qtype | " + " | ".join(metrics) + " |",
                      "|---|" + "---|" * len(metrics)]
            for qt, d in sorted(by_type.items()):
                lines.append(f"| {qt} | " + " | ".join(
                    str(d.get(m, "—")) for m in metrics) + " |")
    lines += ["", "## Coverage by question type", "",
              "| qtype | questions | answered | refused | errors |",
              "|---|---|---|---|---|"]
    for qt, b in sorted(behaviour["by_type"].items()):
        lines.append(f"| {qt} | {b['questions']} | {b['answered']} | "
                     f"{b['refused']} | {b['errors']} |")
    Path(metrics_md).write_text("\n".join(lines) + "\n")
    print(f"[parallel] final metrics → {metrics_json} / {metrics_md}")
    return report


# --------------------------------------------------------------------------- #
# CLI (orchestrator, worker, scorer)
# --------------------------------------------------------------------------- #

def _load_all_questions():
    """The full tagged FinanceBench set (dev + heldout — the eval now covers
    every question instead of a slice, since the 8-key pool makes it cheap)."""
    from finagent.evaluation.dataset import load_eval_dataset

    return load_eval_dataset()


def main() -> None:
    p = argparse.ArgumentParser(description="Parallel multi-key FinanceBench eval.")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--workers", type=int, default=None,
                   help="default: min(n_keys, 4)")
    p.add_argument("--provider", default="groq")
    p.add_argument("--synth-model", default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--sample", type=int, default=None,
                   help="limit to the first N questions (smoke runs)")
    p.add_argument("--worker", default=None, metavar="SHARD_JSONL",
                   help="(internal) run as a worker over one shard file")
    p.add_argument("--score", action="store_true",
                   help="RAGAS-score --output and write the final metrics report")
    p.add_argument("--judge-model", default=None)
    args = p.parse_args()

    if args.worker:
        # Worker mode: serial runner over one shard, single process.
        import pandas as pd
        from finagent.evaluation.financebench.runner import run_agent_outputs

        shard = pd.read_json(args.worker, lines=True)
        run_agent_outputs(shard, output_path=args.output,
                          provider=args.provider, synth_model=args.synth_model,
                          top_k=args.top_k)
        return

    if args.score:
        from finagent.evaluation.financebench.runner import score_answers
        from finagent.llm import AllKeysExhaustedError

        try:
            scores = score_answers(
                outputs_path=args.output,
                scores_csv=str(Path(args.output).with_suffix("")) + "_ragas.csv",
                judge_provider=args.provider, judge_model=args.judge_model,
                max_workers=2, timeout=300, batch_size=8,
            )
        except AllKeysExhaustedError:
            print("\nLIMIT EXHAUSTED — re-run this command once your daily quota resets.")
            print("Already-scored rows are saved; the run will resume from where it stopped.")
            raise SystemExit(1)
        final_report(args.output, ragas_scores=scores)
        return

    qs = _load_all_questions()
    if args.sample:
        qs = qs.head(args.sample)
    run_parallel(qs, output_path=args.output, workers=args.workers,
                 provider=args.provider, synth_model=args.synth_model,
                 top_k=args.top_k)
    final_report(args.output)


if __name__ == "__main__":
    main()
