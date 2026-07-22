"""
gold.py  ·  finagent/evaluation/financebench/gold.py

Build the gold-chunk mapping the retrieval eval scores against.

FinanceBench gives a *verified evidence span* (the exact text in the filing that
answers the question) plus its `doc_name` and page. Our retriever, though, works
over fixed-size chunks — so "did we retrieve the evidence?" only makes sense once
we know which chunk *contains* that evidence. For each evidence span we:

    1. restrict to chunks from the same filing (metadata `local_path` whose stem
       is the evidence `doc_name`), and
    2. pick the chunk most similar to the evidence text (dense similarity over
       the same BGE embeddings used everywhere else).

That best-matching chunk is the gold chunk. A question can have several evidence
spans, so it can have several gold chunks; the retrieval eval counts a "hit" if
*any* gold chunk is surfaced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from finagent.vectorstore import build_store

from .dataset import load_questions, tag_question_types
from .indexing import DEFAULT_PDF_DIR, EVAL_COLLECTION

DEFAULT_GOLD_PATH = "results/financebench_gold.json"


def build_gold_map(
    questions: Optional[pd.DataFrame] = None,
    collection_name: str = EVAL_COLLECTION,
    pdf_dir: Union[str, Path] = DEFAULT_PDF_DIR,
    gold_path: Union[str, Path] = DEFAULT_GOLD_PATH,
    reuse: bool = True,
) -> dict:
    """Map every question to its gold chunk text(s); persist to `gold_path`.

    Returns a dict keyed by `financebench_id`:
        {
          "<id>": {
            "question": ...,
            "qtype": ...,
            "gold_chunks": ["<chunk text>", ...],
            "matches": [{"doc_name", "page", "sim"}, ...],   # diagnostics
          },
          ...
        }
    """
    gold_path = Path(gold_path)
    if reuse and gold_path.exists():
        return json.loads(gold_path.read_text())

    if questions is None:
        questions = tag_question_types(load_questions())
    elif "qtype" not in questions.columns:
        questions = tag_question_types(questions)

    store = build_store(collection_name)
    pdf_dir = Path(pdf_dir)

    gold: dict = {}
    n_unmatched = 0
    for _, row in questions.iterrows():
        fb_id = row["financebench_id"]
        gold_chunks, matches = [], []
        for ev in (row.get("evidence") or []):
            if not isinstance(ev, dict):
                continue
            doc_name = ev.get("doc_name", "")
            evidence_text = ev.get("evidence_text", "") or ev.get(
                "evidence_text_full_page", ""
            )
            if not doc_name or not evidence_text:
                continue
            local_path = str(pdf_dir / f"{doc_name}.pdf")
            # Restrict the search to the chunks of this one filing.
            hits = store.similarity_search_with_relevance_scores(
                evidence_text[:2000],
                k=1,
                filter={"local_path": local_path},
            )
            if not hits:
                continue
            doc, sim = hits[0]
            if doc.page_content not in gold_chunks:
                gold_chunks.append(doc.page_content)
            matches.append({
                "doc_name": doc_name,
                "page": doc.metadata.get("page"),
                "sim": round(float(sim), 4),
            })

        if not gold_chunks:
            n_unmatched += 1
        gold[fb_id] = {
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "qtype": row.get("qtype", ""),
            "company": row.get("company", ""),
            "gold_chunks": gold_chunks,
            "matches": matches,
        }

    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.write_text(json.dumps(gold, indent=2))
    print(f"Gold map: {len(gold)} questions -> {gold_path}"
          + (f"  ({n_unmatched} with no matched chunk)" if n_unmatched else ""))
    return gold
