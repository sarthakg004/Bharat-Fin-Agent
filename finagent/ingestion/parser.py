"""
parser.py  ·  finagent/ingestion/parser.py

Thin wrapper around `unstructured.io` document partitioning, exposed so the
notebook can *show* what parsing a real filing produces: a list of typed
elements (Title, NarrativeText, Table, ListItem, ...) each carrying metadata
(page number, section). This is the same partitioner the HTML branch of
`CorpusIngester` uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union


def parse_document(path: Union[str, Path], strategy: str = "fast") -> list:
    """Partition a filing (HTML or PDF) into unstructured Element objects.

    Parameters
    ----------
    path : the document to parse.
    strategy : "fast" (no OCR, default) or "hi_res" (layout + OCR, slow but
        better at tables).
    """
    from unstructured.partition.auto import partition

    return partition(filename=str(path), strategy=strategy)


def element_summary(elements: list) -> dict:
    """Count elements by type, e.g. {"NarrativeText": 412, "Title": 88, ...}."""
    counts: dict[str, int] = {}
    for el in elements:
        name = type(el).__name__
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
