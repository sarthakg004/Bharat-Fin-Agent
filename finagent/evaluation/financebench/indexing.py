"""
indexing.py  ·  finagent/evaluation/financebench/indexing.py

Index the FinanceBench filings into a dedicated `financebench_eval` Qdrant
collection using the *exact* production ingestion pipeline (same chunking, same
BGE embeddings as `CorpusIngester`). Keeping the pipeline identical is the whole
point: the eval then measures the retrieval the user actually gets, while the
dedicated collection keeps these eval docs out of the live corpus.

Scope: we index only the filings referenced by the open-source question set
(the evidence `doc_name`s — 84 documents). That is the representative corpus for
this eval; indexing the full FinanceBench PDF dump would add hundreds of 10-Ks
that no question touches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from finagent.ingestion.ingest import CorpusIngester

from .dataset import load_questions

DEFAULT_PDF_DIR = "data/us/eval/financebench/pdfs"
DEFAULT_DOC_INFO = "data/us/eval/financebench/data/financebench_document_information.jsonl"
DEFAULT_MANIFEST = "data/us/eval/financebench/eval_manifest.json"
EVAL_COLLECTION = "financebench_eval"


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
    pdf_dir: Union[str, Path] = DEFAULT_PDF_DIR,
    doc_info_path: Union[str, Path] = DEFAULT_DOC_INFO,
    manifest_path: Union[str, Path] = DEFAULT_MANIFEST,
) -> list[dict]:
    """Build (and persist) an ingestion manifest for the referenced filings.

    Each record carries the metadata `CorpusIngester` propagates into the payload —
    crucially `local_path`, whose stem equals the FinanceBench `doc_name`, so the
    gold-chunk mapper can filter chunks to a single filing.
    """
    questions = (
        load_questions(questions_path) if questions_path else load_questions()
    )
    wanted = _referenced_doc_names(questions)

    # doc_name -> company / sector / period, for richer point metadata.
    doc_info = {}
    for line in Path(doc_info_path).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            doc_info[rec["doc_name"]] = rec

    pdf_dir = Path(pdf_dir)
    records, missing = [], []
    for doc_name in sorted(wanted):
        pdf = pdf_dir / f"{doc_name}.pdf"
        if not pdf.exists():
            missing.append(doc_name)
            continue
        info = doc_info.get(doc_name, {})
        records.append({
            "local_path": str(pdf),
            "source_url": info.get("doc_link", str(pdf)),
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
          + (f"  ({len(missing)} missing PDFs: {missing[:5]})" if missing else ""))
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
        corpus_dir=DEFAULT_PDF_DIR,
        collection_name=collection_name,
        market="us",
    )
    if reset:
        ing.reset_collection()

    stats = ing.ingest_all(manifest_path=str(manifest_path))
    return stats.as_dict()
