"""
ledger.py  ·  finagent/evaluation/financebench/ledger.py

The append-only experiment ledger. One row per phase, carrying its full config
and every metric, so the notebook tells a chronological story of the system
improving and any change can be traced to its measured effect.

Rule of the whole project: **one change, one measurement, one ledger row.** Never
bundle two changes into a single row, or you can't tell which one helped.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd

DEFAULT_LEDGER = "results/ledger.csv"


def _flatten(prefix: str, d: dict) -> dict:
    """Recursively flatten a nested metrics dict into `a.b.c` columns."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(f"{key}.", v))
        else:
            flat[key] = v
    return flat


def append_row(
    phase: str,
    config: Optional[dict] = None,
    metrics: Optional[dict] = None,
    ledger_path: Union[str, Path] = DEFAULT_LEDGER,
    notes: str = "",
) -> pd.DataFrame:
    """Append one phase row to the ledger CSV and return the full ledger.

    Parameters
    ----------
    phase    : phase id, e.g. "baseline_v0".
    config   : the knobs that defined this run (collection, reranker, k's, ...).
               Stored verbatim as a JSON string in a `config` column.
    metrics  : nested dict of metrics. Top-level scalars become columns; one
               level of nesting becomes `parent.child` columns (e.g.
               `retrieval.pool_recall@50`, `ragas.overall.faithfulness`).
    notes    : free-text, e.g. the failure-bucket summary.

    Re-running the same `phase` overwrites that phase's previous row, so the
    ledger never accumulates duplicate phases.
    """
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    row: dict = {
        "phase": phase,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
        "config": json.dumps(config or {}, sort_keys=True),
    }
    for top, payload in (metrics or {}).items():
        if isinstance(payload, dict):
            row.update(_flatten(f"{top}.", payload))
        else:
            row[top] = payload

    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
        ledger = ledger[ledger["phase"] != phase]            # replace same phase
        ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
    else:
        ledger = pd.DataFrame([row])

    # Keep identity columns first; sort the rest for stable diffs.
    lead = ["phase", "timestamp", "notes", "config"]
    rest = sorted(c for c in ledger.columns if c not in lead)
    ledger = ledger[[c for c in lead if c in ledger.columns] + rest]

    ledger.to_csv(ledger_path, index=False)
    return ledger


def load_ledger(ledger_path: Union[str, Path] = DEFAULT_LEDGER) -> pd.DataFrame:
    """Return the ledger as a DataFrame (empty if it doesn't exist yet)."""
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return pd.DataFrame()
    return pd.read_csv(ledger_path)
