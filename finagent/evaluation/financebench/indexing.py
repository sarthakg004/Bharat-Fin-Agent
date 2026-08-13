"""Index the FinanceBench filings into a dedicated eval collection through the
EXACT production `CorpusIngester`, over the same original SEC HTML production
serves (fetched by `html_source.py`, not the FinanceBench PDFs). Keeping both
parser and chunker identical to production is the point: the eval has to measure
the retrieval a user actually gets.

Scope is the filings the question set references, restricted to 10-K/10-Q — the
docs that map to one served HTML filing. Excluding 8-K/earnings drops ~23 of 150
questions; see `html_source.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from finagent.config import settings
from finagent.ingestion.ingest import CorpusIngester

from .dataset import load_questions
from .html_source import HTML_DIR, fetch_html_for_docs

# The eval corpus is the ORIGINAL SEC HTML (what production serves), fetched on
# demand from EDGAR — not the FinanceBench PDFs. See `html_source.py`. 8-K /
# earnings docs are excluded there (evidence lives in exhibits, not the primary
# filing), so the eval covers the 10-K / 10-Q questions.
DEFAULT_CORPUS_DIR = HTML_DIR
DEFAULT_DOC_INFO = "data/us/eval/financebench/data/financebench_document_information.jsonl"
DEFAULT_MANIFEST = "data/us/eval/financebench/eval_manifest.json"
# One source of truth for "which collection is the eval corpus" — the same
# FINANCEBENCH_COLLECTION the answer runner reads through `settings`. The gold
# mapper and the retrieval eval used to hardcode the name, so pointing the run
# at a re-ingested corpus silently scored answers on one index and retrieval on
# another.
EVAL_COLLECTION = settings.financebench_collection


def corpus_path(doc_name: str, corpus_dir: Union[str, Path] = DEFAULT_CORPUS_DIR) -> Path:
    """On-disk path of one filing's HTML. Its stem == the FinanceBench doc_name,
    which is how gold-chunk mapping filters payloads to a single filing."""
    return Path(corpus_dir) / f"{doc_name}.htm"


def _referenced_doc_names(questions) -> set:
    """Every doc_name a question relies on (question doc + evidence docs)."""
    docs: set = set()
    for _, row in questions.iterrows():
        docs.add(row.get("doc_name", ""))
        for e in (row.get("evidence") or []):
            if isinstance(e, dict):
                docs.add(e.get("doc_name", ""))
    docs.discard("")
    return docs


def build_manifest(
    questions_path: Union[str, Path] = None,
    corpus_dir: Union[str, Path] = DEFAULT_CORPUS_DIR,
    doc_info_path: Union[str, Path] = DEFAULT_DOC_INFO,
    manifest_path: Union[str, Path] = DEFAULT_MANIFEST,
    fetch: bool = True,
) -> list[dict]:
    """Build (and persist) an ingestion manifest for the referenced filings.

    Fetches each referenced 10-K/10-Q's SEC HTML from EDGAR (idempotent — cached
    files are reused), then writes one manifest record per filing that landed on
    disk. Docs that can't be represented as a single served HTML (8-K / earnings,
    or an EDGAR miss) are simply absent, so the eval runs on what production can
    actually serve. Each record's `local_path` stem == the FinanceBench doc_name,
    which is how the gold-chunk mapper filters chunks to one filing.
    """
    questions = (
        load_questions(questions_path) if questions_path else load_questions()
    )
    wanted = _referenced_doc_names(questions)

    if fetch:
        fetched = fetch_html_for_docs(wanted, html_dir=corpus_dir,
                                      doc_info_path=doc_info_path)
        n_ok = sum(1 for v in fetched.values() if v["status"] == "ok")
        n_skip = sum(1 for v in fetched.values() if v["status"] == "skipped")
        n_err = sum(1 for v in fetched.values() if v["status"] == "error")
        print(f"HTML acquisition: {n_ok} ok, {n_skip} skipped (8-K/earnings), "
              f"{n_err} error")

    # doc_name -> company / sector / period, for richer point metadata.
    doc_info = {}
    for line in Path(doc_info_path).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            doc_info[rec["doc_name"]] = rec

    records, missing = [], []
    for doc_name in sorted(wanted):
        html = corpus_path(doc_name, corpus_dir)
        if not html.exists():
            missing.append(doc_name)
            continue
        info = doc_info.get(doc_name, {})
        records.append({
            "local_path": str(html),
            "source_url": info.get("doc_link", str(html)),
            "company": info.get("company", doc_name.split("_")[0]),
            "ticker": info.get("company", doc_name.split("_")[0]),
            "year": str(info.get("doc_period", "")),
            "sector": info.get("gics_sector", ""),
            "filing_type": info.get("doc_type", "10k"),
            "doc_name": doc_name,
            "status": "ok",
        })

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(records, indent=2))
    print(f"Manifest: {len(records)} filings -> {manifest_path}"
          + (f"  ({len(missing)} not on disk: {missing[:5]})" if missing else ""))
    return records


def index_eval_corpus(
    manifest_path: Union[str, Path] = DEFAULT_MANIFEST,
    collection_name: str = EVAL_COLLECTION,
    reset: bool = False,
) -> dict:
    """Ingest the manifest into the `financebench_eval` collection.

    Uses the production `CorpusIngester` with its default chunking (1000/200)
    and BGE-small embeddings, so retrieval here mirrors production. Idempotent:
    re-running skips already-indexed filings unless `reset=True`.
    """
    if not Path(manifest_path).exists():
        build_manifest(manifest_path=manifest_path)

    ing = CorpusIngester(
        corpus_dir=DEFAULT_CORPUS_DIR,
        collection_name=collection_name,
        market="us",
    )
    if reset:
        ing.reset_collection()

    stats = ing.ingest_all(manifest_path=str(manifest_path))
    return stats.as_dict()
