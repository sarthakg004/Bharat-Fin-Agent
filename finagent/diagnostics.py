"""
diagnostics.py  ·  finagent/diagnostics.py

System health check for the reference notebook's Section 0. Confirms the Chroma
collections exist and are populated, required API keys are loaded, the eval
dataset is present, and reports key library versions. Returns a DataFrame of
(check, status, detail) so the notebook can display it and the reader can see at
a glance whether the environment is ready.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Optional

import pandas as pd


def _check_collection(collection: str, chroma_dir: str) -> tuple[str, str]:
    try:
        from finagent.vectorstore import build_store

        store = build_store(collection, chroma_dir=chroma_dir)
        n = store._collection.count()
        if n == 0:
            return "WARN", f"{collection!r} exists but is EMPTY (run indexing)"
        return "OK", f"{collection!r}: {n:,} chunks"
    except Exception as e:  # surface loudly
        return "FAIL", f"{collection!r}: {type(e).__name__}: {e}"


def system_health_check(
    collections: Optional[list[str]] = None,
    chroma_dir: Optional[str] = None,
    eval_path: str = "data/us/eval/financebench/data/financebench_open_source.jsonl",
    required_keys: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Run all environment checks; return a (check, status, detail) DataFrame."""
    from dotenv import load_dotenv

    load_dotenv()

    collections = collections or ["us_filings", "financebench_eval"]
    chroma_dir = chroma_dir or os.getenv("CHROMA_DIR", "data/chroma")
    required_keys = required_keys or ["GROQ_API_KEY"]

    rows: list[tuple[str, str, str]] = []

    # Chroma collections
    for col in collections:
        status, detail = _check_collection(col, chroma_dir)
        rows.append((f"chroma:{col}", status, detail))

    # API keys
    for key in required_keys:
        present = bool(os.getenv(key))
        rows.append((f"env:{key}", "OK" if present else "FAIL",
                     "loaded" if present else f"{key} missing from .env"))

    # Eval dataset
    p = Path(eval_path)
    rows.append((
        "eval_dataset",
        "OK" if p.exists() else "FAIL",
        f"{eval_path} ({p.stat().st_size/1e3:.0f} KB)" if p.exists()
        else f"missing: clone FinanceBench into {eval_path}",
    ))

    # Library versions
    for lib in ["langgraph", "langchain", "chromadb", "ragas", "sentence_transformers"]:
        try:
            mod = importlib.import_module(lib)
            rows.append((f"lib:{lib}", "OK", getattr(mod, "__version__", "?")))
        except Exception:
            rows.append((f"lib:{lib}", "WARN", "not importable"))

    return pd.DataFrame(rows, columns=["check", "status", "detail"])
