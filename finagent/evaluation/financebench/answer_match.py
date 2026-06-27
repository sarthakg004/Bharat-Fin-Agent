"""
answer_match.py  ·  finagent/evaluation/financebench/answer_match.py

A **deterministic answer-correctness** signal to sit alongside the RAGAS judge
metrics.

Why this exists
---------------
RAGAS `faithfulness` measures whether the answer's claims are grounded in the
retrieved context — not whether the answer is *right*. On FinanceBench we found a
systematic gap: numerically-correct answers were being scored unfaithful because
the synthesizer padded them with provenance boilerplate the context didn't
literally contain (e.g. "$5,409M — sourced from the FY2019 10-K, us-gaap:
InventoryNet"). The figure matched gold exactly; faithfulness was 0.

`numeric_accuracy` closes that blind spot: for every numeric question it checks
whether the gold value actually appears in the answer (within a relative
tolerance), independent of how the answer was phrased or grounded. Read the two
together:

    high numeric_accuracy + low faithfulness  → answer is RIGHT but over-claims
                                                 (a phrasing/prompt problem)
    low numeric_accuracy                       → answer is WRONG (a retrieval or
                                                 computation problem)

It is judge-free (no LLM, no API quota), so it's cheap to run on every iteration
and is the natural "did the change actually help?" gauge for the numeric set.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Union

# A signed number with optional thousands separators and decimals: 1,577 / -3.2 / 0.40
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Default relative tolerance: 1%. Tight enough to reject a wrong line item,
# loose enough to accept rounding ($394,328M ≈ $394.3B) and the two-decimal
# rounding FinanceBench ratio questions ask for.
DEFAULT_TOL = 0.01


def _numbers(text: str) -> list[float]:
    """Every numeric literal in `text` as floats (commas stripped)."""
    out: list[float] = []
    for m in _NUM_RE.findall(str(text).replace(",", "")):
        try:
            out.append(float(m))
        except ValueError:
            continue
    return out


def numeric_match(gold: str, answer: str, tol: float = DEFAULT_TOL) -> Optional[bool]:
    """Does the gold figure appear in `answer` within relative tolerance `tol`?

    Returns True/False, or None when the gold answer carries no number (i.e. the
    question isn't really numeric — excluded from the accuracy denominator).
    Percent-style golds are matched both as written and against the same value
    scaled by 100 (so a gold of "0.40" matches an answer that says "40%").
    """
    g_nums = _numbers(gold)
    if not g_nums:
        return None
    a_nums = _numbers(answer)
    if not a_nums:
        return False
    for g in g_nums:
        targets = {g}
        # Percentage/ratio dual form: gold 0.40 ↔ answer 40(%); gold 40 ↔ 0.40.
        targets.add(g * 100.0)
        if g != 0:
            targets.add(g / 100.0)
        for t in targets:
            for a in a_nums:
                if t == 0:
                    if abs(a) < 1e-9:
                        return True
                elif abs(a - t) / abs(t) <= tol:
                    return True
    return False


def numeric_accuracy(
    rows: list[dict],
    tol: float = DEFAULT_TOL,
    qtypes: tuple = ("numeric",),
    gold_key: str = "gold",
    answer_key: str = "answer",
) -> dict:
    """Fraction of `qtypes` questions whose gold value appears in the answer.

    Scored only over rows whose gold answer actually contains a number, so a
    mis-tagged narrative question can't dilute the denominator. Returns the rate,
    the n, and the list of misses (for eyeballing) — overall and per qtype.
    """
    scored: list[dict] = []
    for r in rows:
        if qtypes and (r.get("qtype") not in qtypes):
            continue
        verdict = numeric_match(r.get(gold_key, ""), r.get(answer_key, ""), tol)
        if verdict is None:
            continue                       # no numeric gold → not a numeric item
        scored.append({**r, "_match": verdict})

    def _rate(items: list[dict]) -> dict:
        n = len(items)
        hits = sum(1 for x in items if x["_match"])
        return {
            "n": n,
            "correct": hits,
            "accuracy": round(hits / n, 4) if n else None,
        }

    by_type: dict[str, dict] = {}
    for x in scored:
        by_type.setdefault(x.get("qtype") or "untyped", []).append(x)

    misses = [
        {"question": x.get("question", "")[:160],
         "company": x.get("company", ""),
         "gold": x.get(gold_key, ""),
         "answer": (x.get(answer_key, "") or "")[:160]}
        for x in scored if not x["_match"]
    ]
    return {
        "overall": _rate(scored),
        "by_type": {qt: _rate(items) for qt, items in by_type.items()},
        "tolerance": tol,
        "misses": misses,
    }


def numeric_accuracy_from_file(
    outputs_path: Union[str, Path],
    tol: float = DEFAULT_TOL,
) -> dict:
    """Convenience wrapper: load a financebench outputs JSON and score it."""
    rows = json.loads(Path(outputs_path).read_text())
    return numeric_accuracy(rows, tol=tol)


if __name__ == "__main__":  # quick CLI: python -m ...answer_match results/...json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "results/v1/financebench_full_outputs.json"
    res = numeric_accuracy_from_file(path)
    ov = res["overall"]
    print(f"numeric_accuracy: {ov['correct']}/{ov['n']} = {ov['accuracy']} "
          f"(tol={res['tolerance']})")
    for qt, d in res["by_type"].items():
        print(f"  {qt}: {d['correct']}/{d['n']} = {d['accuracy']}")
    print(f"  misses: {len(res['misses'])}")
