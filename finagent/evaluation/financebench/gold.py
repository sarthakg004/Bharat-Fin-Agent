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

from finagent.vectorstore import DEFAULT_EMBED_MODEL, build_store

from .dataset import load_questions, tag_question_types
from .indexing import DEFAULT_CORPUS_DIR, EVAL_COLLECTION, corpus_path

# Gold chunks are the *text* of the best-matching chunk, so the map is only
# valid for the collection it was built from — a re-index with different chunk
# geometry (or the §12 context headers, which prefix every chunk) makes every
# stored text a stale string that nothing will ever match. Name the file after
# the collection so switching corpora rebuilds instead of silently scoring 0.
DEFAULT_GOLD_PATH = (
    "results/financebench_gold.json"
    if EVAL_COLLECTION == "financebench_eval"
    else f"results/financebench_gold_{EVAL_COLLECTION}.json"
)


def _one_filing_filter(local_path: str):
    """Restrict the gold search to the chunks of one filing."""
    from qdrant_client import models

    from finagent.vectorstore import META_KEY

    return models.Filter(must=[models.FieldCondition(
        key=f"{META_KEY}.local_path",
        match=models.MatchValue(value=local_path))])


def build_gold_map(
    questions: Optional[pd.DataFrame] = None,
    collection_name: str = EVAL_COLLECTION,
    corpus_dir: Union[str, Path] = DEFAULT_CORPUS_DIR,
    gold_path: Union[str, Path] = DEFAULT_GOLD_PATH,
    reuse: bool = True,
    embedding_model: str = DEFAULT_EMBED_MODEL,
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

    # Only questions whose evidence is a served HTML filing (10-K/10-Q). 8-K /
    # earnings docs have no served HTML, so their questions are dropped rather
    # than counted as retrieval misses they aren't.
    from .dataset import restrict_to_served_docs
    before = len(questions)
    questions = restrict_to_served_docs(questions, corpus_dir)
    dropped = before - len(questions)
    if dropped:
        print(f"gold map: {len(questions)} served-doc questions "
              f"({dropped} dropped — 8-K/earnings not served as HTML)")

    # The embedder MUST match the one the collection was built with: this
    # embeds the evidence text to find its chunk, and a bge query vector
    # against a Gemini collection is a dimension mismatch at best and silent
    # nonsense at worst. It defaulted to bge-large regardless of collection.
    store = build_store(collection_name, embedding_model=embedding_model)

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
            local_path = str(corpus_path(doc_name, corpus_dir))
            # Restrict the search to the chunks of this one filing.
            hits = store.similarity_search_with_relevance_scores(
                evidence_text[:2000],
                k=1,
                filter=_one_filing_filter(local_path),
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
