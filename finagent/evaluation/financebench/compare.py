"""
compare.py  ·  finagent/evaluation/financebench/compare.py

Build the **before/after comparison table** that tracks how the FinanceBench
metrics move as we change the system.

The workflow this supports
--------------------------
1. The V1 metrics are frozen as `results/baseline_metrics.json` (the "before"
   anchor the user compares against; it survives version churn).
2. After implementing changes, the eval is re-run into a NEW version dir, e.g.
   `results/v2/` → `results/v2/final_metrics.json`.
3. `python -m finagent.evaluation.financebench.compare` reads the baseline and
   the newest `results/vN/final_metrics.json`, computes the delta on every
   metric, and writes `results/comparison.md` — one table that makes regressions
   and wins obvious at a glance. Pass `--current results/v3/final_metrics.json`
   to compare a specific run.

Metrics compared (each overall + per question type where available):
    RAGAS        faithfulness · answer_relevancy · context_precision · context_recall
    Correctness  numeric_accuracy (deterministic, judge-free — see answer_match.py)
    Behaviour    answer_rate · refusal_rate · mean_confidence
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Union

# The frozen "before" anchor (V1 metrics + numeric accuracy). Lives at the top
# of results/ so it survives version churn under results/v*/.
DEFAULT_BASELINE = "results/baseline_metrics.json"
DEFAULT_OUT = "results/comparison.md"
RESULTS_DIR = "results"

_RAGAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def latest_version_metrics(results_dir: Union[str, Path] = RESULTS_DIR) -> Optional[Path]:
    """The `final_metrics.json` of the highest-numbered results/vN/ run, or None.

    So `compare` with no --current automatically picks up the newest rerun
    (results/v2, then v3, …) without the user passing a path each time.
    """
    versions: list[tuple[int, Path]] = []
    for d in Path(results_dir).glob("v*"):
        m = re.fullmatch(r"v(\d+)", d.name)
        fm = d / "final_metrics.json"
        if d.is_dir() and m and fm.exists():
            versions.append((int(m.group(1)), fm))
    if not versions:
        return None
    return max(versions, key=lambda t: t[0])[1]


def _g(d: dict, *path):
    """Safe nested get; returns None if any key is missing."""
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


def _delta(new, old) -> str:
    if not isinstance(new, (int, float)) or not isinstance(old, (int, float)):
        return "—"
    d = new - old
    arrow = "▲" if d > 0 else ("▼" if d < 0 else "▬")
    return f"{arrow} {d:+.4f}"


def _row(label: str, new, old) -> str:
    return f"| {label} | {_fmt(old)} | {_fmt(new)} | {_delta(new, old)} |"


def build_comparison(
    baseline_path: Union[str, Path] = DEFAULT_BASELINE,
    current_path: Optional[Union[str, Path]] = None,
    out_path: Union[str, Path] = DEFAULT_OUT,
    baseline_label: str = "Baseline (v1)",
    current_label: Optional[str] = None,
) -> str:
    # No explicit current → use the newest results/vN/ run.
    if current_path is None:
        current_path = latest_version_metrics()
        if current_path is None:
            raise FileNotFoundError(
                "No results/vN/final_metrics.json found — run the eval first "
                "(e.g. --output results/v2/financebench_full_outputs.json --score) "
                "or pass --current explicitly.")
    current_path = Path(current_path)
    if current_label is None:
        # Label by the version directory name when we can ("v2"), else generic.
        current_label = (current_path.parent.name
                         if re.fullmatch(r"v\d+", current_path.parent.name)
                         else "After changes")
    base = json.loads(Path(baseline_path).read_text())
    cur = json.loads(current_path.read_text())

    lines: list[str] = [
        "# FinAgent — FinanceBench metrics: before vs. after",
        "",
        f"**{baseline_label}** = `{baseline_path}` · "
        f"**{current_label}** = `{current_path}`",
        "",
        "Higher is better for every row except *refusal rate* (an explicit "
        "abstention is healthier than a confident wrong answer, so read its delta "
        "in context, not as a pure regression).",
        "",
        "## RAGAS (overall)",
        "",
        f"| metric | {baseline_label} | {current_label} | Δ |",
        "|---|---|---|---|",
    ]
    for m in _RAGAS:
        lines.append(_row(m, _g(cur, "ragas", "overall", m),
                          _g(base, "ragas", "overall", m)))

    # Numeric accuracy (deterministic).
    lines += [
        "",
        "## Correctness (deterministic, judge-free)",
        "",
        f"| metric | {baseline_label} | {current_label} | Δ |",
        "|---|---|---|---|",
        _row("numeric_accuracy (1% tol)",
             _g(cur, "behaviour", "numeric_accuracy"),
             _g(base, "behaviour", "numeric_accuracy")),
    ]

    # System behaviour.
    lines += [
        "",
        "## System behaviour",
        "",
        f"| metric | {baseline_label} | {current_label} | Δ |",
        "|---|---|---|---|",
        _row("answer_rate", _g(cur, "behaviour", "answer_rate"),
             _g(base, "behaviour", "answer_rate")),
        _row("refusal_rate", _g(cur, "behaviour", "refusal_rate"),
             _g(base, "behaviour", "refusal_rate")),
        _row("mean_confidence", _g(cur, "behaviour", "mean_confidence"),
             _g(base, "behaviour", "mean_confidence")),
    ]

    # Per-qtype RAGAS faithfulness (the headline diagnostic).
    base_bt = _g(base, "ragas", "by_type") or {}
    cur_bt = _g(cur, "ragas", "by_type") or {}
    qtypes = sorted(set(base_bt) | set(cur_bt))
    if qtypes:
        lines += [
            "",
            "## RAGAS faithfulness by question type",
            "",
            f"| qtype | {baseline_label} | {current_label} | Δ |",
            "|---|---|---|---|",
        ]
        for qt in qtypes:
            lines.append(_row(qt, _g(cur_bt, qt, "faithfulness"),
                              _g(base_bt, qt, "faithfulness")))

    text = "\n".join(lines) + "\n"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text)
    print(f"[compare] wrote {out_path}")
    return text


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build the before/after metrics table.")
    ap.add_argument("--baseline", default=DEFAULT_BASELINE,
                    help="frozen 'before' metrics (default: results/baseline_metrics.json)")
    ap.add_argument("--current", default=None,
                    help="'after' metrics (default: newest results/vN/final_metrics.json)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    build_comparison(args.baseline, args.current, args.out)


if __name__ == "__main__":
    main()
