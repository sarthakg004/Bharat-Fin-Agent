"""
filters.py  ·  finagent/retrieval/filters.py

Deterministic company/year metadata filtering for retrieval.

Every indexed chunk carries `company` / `year` metadata, but unfiltered
hybrid search makes a question about one company compete against EVERY other
filing's near-identical accounting language (the financebench_eval collection
is ~129k chunks across 84 filings — pool recall@100 was 0.48 unfiltered).
Restricting the candidate pool to the company the question names (and the
fiscal year when it names one) removes that distractor mass before BM25 /
dense / rerank ever run. No LLM call: the company vocabulary comes from the
collection's own metadata and is matched against the question text.
"""

from __future__ import annotations

import re
from typing import Optional

# Common spellings that don't substring-match the canonical metadata name.
# Keys and values are in NORMALISED form (see _norm).
COMPANY_ALIASES: dict[str, str] = {
    "amex": "american express",
    "jnj": "johnson and johnson",
    "j and j": "johnson and johnson",
    "aes": "aes corporation",
    "pepsi": "pepsico",
    "jp morgan": "jpmorgan",
    "jpm": "jpmorgan",
    "coke": "coca cola",
    "walmart inc": "walmart",
    "the coca cola company": "coca cola",
}

_YEAR_RE = re.compile(r"\b(?:FY\s?)?((?:19|20)\d{2})\b", re.I)


def _norm(s: str) -> str:
    """Lowercase; '&'→' and '; punctuation→space; collapse whitespace."""
    s = (s or "").lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Legal-form suffixes stripped to make a short variant ("GENERAL MILLS INC" →
# "general mills") so the name a question actually uses matches the metadata.
_LEGAL_SUFFIXES = (
    "incorporated", "international", "corporation", "company", "holdings",
    "group", "corp", "inc", "plc", "ltd", "llc", "co", "sa", "nv", "ag",
)


def _name_variants(name: str) -> list[str]:
    """Normalised name plus progressively suffix-stripped variants."""
    n = _norm(name)
    out = [n]
    words = n.split()
    while len(words) > 1 and words[-1] in _LEGAL_SUFFIXES:
        words = words[:-1]
        out.append(" ".join(words))
    return out


def build_company_vocab(metadatas) -> tuple[dict, dict]:
    """From chunk metadata, build:
       vocab          — normalised company name (and variants) → canonical value
       years_by_co    — canonical company → set of metadata `year` strings

    Metadata `company` values range from clean names ("3M") through legal names
    ("BEST BUY CO INC") to bare tickers ("AAPL"); ticker values get their SEC
    title registered as an alias so "Apple's revenue" still matches 'AAPL'.
    """
    vocab: dict[str, str] = {}
    years: dict[str, set] = {}
    tickerish: set[str] = set()
    for m in metadatas:
        c = (m or {}).get("company") or ""
        if not c:
            continue
        if c not in years:
            for v in _name_variants(c):
                vocab.setdefault(v, c)
            if c.isupper() and 1 < len(c) <= 5 and " " not in c:
                tickerish.add(c)
        y = str((m or {}).get("year") or "")
        years.setdefault(c, set()).add(y) if y else years.setdefault(c, set())

    # Bare-ticker company values: alias the SEC registrant title (and its
    # suffix-stripped variants) to the ticker, via the cached
    # company_tickers.json the resolver already maintains. Best-effort — a
    # missing cache or offline run just means ticker-only matching.
    if tickerish:
        try:
            from finagent.tools.resolver import TickerCIKResolver
            res = TickerCIKResolver()
            res._ensure_loaded()
            by_ticker = getattr(res, "_by_ticker", None) or {}
            for t in tickerish:
                entry = by_ticker.get(t.upper()) or {}
                title = entry.get("title") or ""
                for v in _name_variants(title):
                    if v:
                        vocab.setdefault(v, t)
        except Exception:
            pass
    return vocab, years


def infer_filter(question: str, vocab: dict, years_by_co: dict) -> Optional[dict]:
    """{"companies": [...], "years": [...]} for the question, or None.

    Companies: every vocab name (or alias) appearing in the normalised
    question, longest first so "American Water Works" beats "American".
    Years: only when exactly ONE company matched — the question's latest year
    if that company has a filing for it, else the following year (a fiscal-
    FY2019 answer often lives in the 2020-filed report's comparatives). Both
    are kept when both exist. No matching year → company-only filter.
    """
    qn = f" {_norm(question)} "
    matched: list[str] = []
    for norm_name in sorted(vocab, key=len, reverse=True):
        if norm_name and f" {norm_name} " in qn:
            canon = vocab[norm_name]
            if canon not in matched:
                matched.append(canon)
    for alias, target in COMPANY_ALIASES.items():
        if f" {alias} " in qn and target in vocab:
            canon = vocab[target]
            if canon not in matched:
                matched.append(canon)
    if not matched:
        return None

    flt: dict = {"companies": matched}
    if len(matched) == 1:
        years_avail = years_by_co.get(matched[0], set())
        yrs = [int(y) for y in _YEAR_RE.findall(question)]
        if yrs and years_avail:
            target = max(yrs)
            keep = [str(y) for y in (target, target + 1) if str(y) in years_avail]
            if keep:
                flt["years"] = keep
    return flt


def chroma_where(flt: Optional[dict]) -> Optional[dict]:
    """Translate an inferred filter into a Chroma `where` clause."""
    if not flt:
        return None
    comps = flt.get("companies") or []
    clauses: list[dict] = []
    if len(comps) == 1:
        clauses.append({"company": comps[0]})
    elif comps:
        clauses.append({"company": {"$in": comps}})
    yrs = flt.get("years") or []
    if yrs:
        clauses.append({"year": {"$in": yrs}} if len(yrs) > 1 else {"year": yrs[0]})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def metadata_matches(meta: dict, flt: Optional[dict]) -> bool:
    """Python-side equivalent of `chroma_where` (for BM25 index masking)."""
    if not flt:
        return True
    comps = flt.get("companies") or []
    if comps and (meta or {}).get("company") not in comps:
        return False
    yrs = flt.get("years") or []
    if yrs and str((meta or {}).get("year") or "") not in yrs:
        return False
    return True
