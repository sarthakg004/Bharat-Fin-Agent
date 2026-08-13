"""Load the FinanceBench question set, tag each question by type, and carve a
held-out slice that is never inspected while iterating.

The type tag is what lets a later phase prove a TARGETED win ("XBRL lifted
numeric accuracy by X") rather than only "overall went up a bit":

    numeric        — the answer is a reported figure or derived from figures
    comparison     — two or more entities/periods pitted against each other
    cross-document — the evidence spans more than one filing
    narrative      — everything else (qualitative / extraction)

Priority when several rules match: cross-document > comparison > numeric >
narrative. The rarer, more specific buckets win over the broad numeric one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Union

import pandas as pd

# Default location of the FinanceBench open-source split (cloned in §1.1c).
DEFAULT_QUESTIONS = "data/us/eval/financebench/data/financebench_open_source.jsonl"
# Where the deterministic held-out split is recorded, so the *same* ~50
# questions stay held out across every notebook run and phase.
DEFAULT_SPLIT_PATH = "results/financebench_split.json"

_COMPARISON_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|relative to|compared (?:to|with)|"
    r"higher than|lower than|greater than|less than|difference between|"
    r"more than|change (?:in|from)|year[- ]over[- ]year|yoy|growth)\b",
    re.IGNORECASE,
)
_NUMERIC_REASONING_RE = re.compile(r"numerical|logical reasoning", re.IGNORECASE)


def load_questions(path: Union[str, Path] = DEFAULT_QUESTIONS) -> pd.DataFrame:
    """Read the FinanceBench JSONL into a DataFrame, one row per question."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FinanceBench questions not found at {path}. Run the §1.1c clone "
            "cell first."
        )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def _evidence_docs(evidence) -> set:
    """Distinct doc_names cited in a question's evidence list."""
    if not isinstance(evidence, list):
        return set()
    return {e.get("doc_name", "") for e in evidence if isinstance(e, dict)}


def classify_question(row: pd.Series) -> str:
    """Return the single question-type bucket for one FinanceBench row."""
    # 1. Cross-document — evidence spans more than one filing.
    if len(_evidence_docs(row.get("evidence"))) > 1:
        return "cross-document"

    question = str(row.get("question", ""))
    # 2. Comparison — explicit comparative phrasing.
    if _COMPARISON_RE.search(question):
        return "comparison"

    # 3. Numeric — FinanceBench's generated-metric bucket, or numeric reasoning.
    q_type = str(row.get("question_type", ""))
    reasoning = str(row.get("question_reasoning") or "")
    if q_type == "metrics-generated" or _NUMERIC_REASONING_RE.search(reasoning):
        return "numeric"

    # 4. Everything else is narrative.
    return "narrative"


def tag_question_types(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `qtype` column (numeric/comparison/cross-document/narrative)."""
    out = df.copy()
    out["qtype"] = out.apply(classify_question, axis=1)
    return out


def restrict_to_served_docs(df: pd.DataFrame, corpus_dir=None) -> pd.DataFrame:
    """Keep only questions whose evidence docs are ALL available as served HTML.

    The eval corpus is the original SEC HTML (what production serves); 8-K /
    earnings docs are excluded (their evidence is in exhibits, not the primary
    filing), so their questions have no gold chunk. Dropping them here keeps the
    eval honest — a question with no retrievable evidence would otherwise count
    as a retrieval failure that isn't one. A question survives only if every
    evidence doc has an HTML file on disk.
    """
    from .indexing import corpus_path

    def served(row) -> bool:
        docs = _evidence_docs(row.get("evidence")) or {row.get("doc_name", "")}
        docs.discard("")
        return bool(docs) and all(corpus_path(d, corpus_dir).exists()
                                  if corpus_dir else corpus_path(d).exists()
                                  for d in docs)

    return df[df.apply(served, axis=1)].reset_index(drop=True)


def split_heldout(
    df: pd.DataFrame,
    n_heldout: int = 50,
    seed: int = 13,
    split_path: Union[str, Path] = DEFAULT_SPLIT_PATH,
    reuse: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (dev, heldout), stratified by `qtype`, and persist the ids.

    The held-out slice is the RAGAS answer-eval set we *never look at* while
    iterating — that keeps it an honest generalisation check. The split is
    written to `split_path` and reused on later calls so the boundary is stable
    across phases.

    Returns
    -------
    (dev_df, heldout_df)
    """
    if "qtype" not in df.columns:
        df = tag_question_types(df)

    split_path = Path(split_path)
    if reuse and split_path.exists():
        ids = json.loads(split_path.read_text())
        heldout_ids = set(ids["heldout"])
        heldout = df[df["financebench_id"].isin(heldout_ids)]
        dev = df[~df["financebench_id"].isin(heldout_ids)]
        return dev.reset_index(drop=True), heldout.reset_index(drop=True)

    # Stratified sample: take a proportional share of each qtype.
    heldout_parts = []
    frac = n_heldout / len(df)
    for _, grp in df.groupby("qtype"):
        k = max(1, round(len(grp) * frac))
        heldout_parts.append(grp.sample(n=min(k, len(grp)), random_state=seed))
    heldout = pd.concat(heldout_parts)
    # Trim/pad to exactly n_heldout for a stable count.
    if len(heldout) > n_heldout:
        heldout = heldout.sample(n=n_heldout, random_state=seed)
    heldout_ids = set(heldout["financebench_id"])
    dev = df[~df["financebench_id"].isin(heldout_ids)]

    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(
        {"heldout": sorted(heldout_ids), "dev": sorted(set(dev["financebench_id"]))},
        indent=2,
    ))
    return dev.reset_index(drop=True), heldout.reset_index(drop=True)


def type_breakdown(df: pd.DataFrame) -> pd.Series:
    """Count of questions per qtype — handy for a one-line sanity print."""
    if "qtype" not in df.columns:
        df = tag_question_types(df)
    return df["qtype"].value_counts()
