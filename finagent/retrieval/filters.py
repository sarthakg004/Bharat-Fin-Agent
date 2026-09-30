"""Company and year filter, inferred from the question text without an LLM.

Every chunk carries `company` and `year`. Without a filter, a question about one
company competes with every other filing's near-identical accounting language.
The company names come from the collection's own metadata and are matched
against the question.
"""

from __future__ import annotations

import re
from typing import Optional

# Nicknames that do not appear inside the stored company name (normalised form).
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
    "mgm": "mgm resorts",       # too short for the auto single-token alias
}

# "FY2022", "2022" and "FY22" are all years. A bare "22" is a quantity.
_YEAR_RE = re.compile(r"\bFY\s?(\d{2})\b|\b(?:FY\s?)?((?:19|20)\d{2})\b", re.I)


def parse_years(text: str) -> list[int]:
    """Every fiscal year named in `text`, in order. "FY22" -> 2022."""
    out: list[int] = []
    for m in _YEAR_RE.finditer(text or ""):
        short, full = m.group(1), m.group(2)
        if full:
            out.append(int(full))
        else:
            # FY98 means 1998, not 2098.
            n = int(short)
            out.append(2000 + n if n < 80 else 1900 + n)
    return out

# Section names in the question -> the 10-K item the chunk was tagged with.
# Only unambiguous names; matched on the normalised question ("MD&A" -> "md and a").
_ITEM_PHRASES: dict[str, str] = {
    "risk factor": "1A",
    "unresolved staff comment": "1B",
    "legal proceeding": "3",
    "management s discussion": "7",
    "md and a": "7",
    "quantitative and qualitative disclosure": "7A",
    "market risk": "7A",
    "controls and procedures": "9A",
    "internal control over financial reporting": "9A",
    "executive compensation": "11",
}

# Words that mean "the newest data" when no year is named.
_RECENT_RE = re.compile(
    r"\b(latest|most recent|current|year over year|yoy|prior fiscal year"
    r"|last quarter|this quarter|past year)\b")


def _norm(s: str) -> str:
    """Lowercase; '&' becomes ' and '; punctuation becomes a space."""
    s = (s or "").lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Legal suffixes stripped so "GENERAL MILLS INC" also matches "general mills".
_LEGAL_SUFFIXES = (
    "incorporated", "international", "corporation", "company", "holdings",
    "group", "corp", "inc", "plc", "ltd", "llc", "co", "sa", "nv", "ag",
)

# Too generic to name a company alone ("general" must not match General Mills).
_GENERIC_TOKENS = {
    "american", "general", "united", "national", "international", "first",
    "best", "new", "the", "and", "company", "water", "works", "health",
    "communications", "financial", "global", "world", "home", "air", "buy",
    "motors", "mills", "express", "depot", "foot", "locker", "resorts",
} | set(_LEGAL_SUFFIXES)


def _name_variants(name: str) -> list[str]:
    """The normalised name, plus versions with legal suffixes removed."""
    n = _norm(name)
    out = [n]
    words = n.split()
    while len(words) > 1 and words[-1] in _LEGAL_SUFFIXES:
        words = words[:-1]
        out.append(" ".join(words))
    return out


def build_company_vocab(metadatas) -> tuple[dict, dict]:
    """From chunk metadata, build:
       vocab        normalised company name or alias -> the stored company value
       years_by_co  stored company value -> the years it has filings for

    Both the `company` and the `ticker` field become aliases, because they
    differ: a live-fetched filing stores "APPLIED MATERIALS INC /DE" as the
    company and "AMAT" as the ticker, and a question may use either.
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
        t = str((m or {}).get("ticker") or "").strip()
        if t and _norm(t):
            vocab.setdefault(_norm(t), c)
        y = str((m or {}).get("year") or "")
        years.setdefault(c, set()).add(y) if y else years.setdefault(c, set())

    # When the stored company is a bare ticker ("AAPL"), also accept its SEC
    # company name, read from the resolver's cached ticker list.
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

    # A distinctive word owned by exactly one company becomes an alias for it
    # ("verizon" -> Verizon Communications Inc.). Generic words never qualify, and
    # matching is whole-word, so a short alias cannot fire inside another word.
    token_owner: dict[str, set] = {}
    for name, canon in list(vocab.items()):
        for tok in name.split():
            if len(tok) >= 3 and tok not in _GENERIC_TOKENS:
                token_owner.setdefault(tok, set()).add(canon)
    for tok, owners in token_owner.items():
        if len(owners) == 1 and tok not in vocab:
            vocab[tok] = next(iter(owners))
    return vocab, years


def infer_filter(question: str, vocab: dict, years_by_co: dict) -> Optional[dict]:
    """{"companies": [...], "years": [...]} for the question, or None.

    Years are added only when exactly one company matched: the question's
    latest year and the year after it (a fiscal-2019 figure often sits in the
    report filed in 2020), whichever of the two are indexed.
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
    items = sorted({item for phrase, item in _ITEM_PHRASES.items()
                    if f" {phrase}" in qn})
    if items:
        # Dropped by search() if it matches nothing.
        flt["items"] = items
    if len(matched) == 1:
        years_avail = years_by_co.get(matched[0], set())
        yrs = parse_years(question)
        if yrs and years_avail:
            target = max(yrs)
            keep = [str(y) for y in (target, target + 1) if str(y) in years_avail]
            if keep:
                flt["years"] = keep
        elif years_avail and _RECENT_RE.search(qn):
            # No year named but the question wants the latest: keep the two newest
            # indexed years (the newest filing also carries the prior year's figures).
            digit = sorted((y for y in years_avail if str(y).isdigit()), key=int)
            if digit:
                flt["years"] = digit[-2:]
    return flt


def qdrant_filter(flt: Optional[dict]):
    """An inferred filter as a Qdrant `Filter`: clauses are ANDed, values within a clause ORed."""
    if not flt:
        return None
    from qdrant_client import models

    from finagent.vectorstore import META_KEY

    def clause(field: str, values: list[str]):
        return models.FieldCondition(
            key=f"{META_KEY}.{field}",
            match=(models.MatchValue(value=values[0]) if len(values) == 1
                   else models.MatchAny(any=list(values))),
        )

    must = [clause(field, vals)
            for field, vals in (("company", flt.get("companies") or []),
                                ("year", flt.get("years") or []),
                                ("item", flt.get("items") or []))
            if vals]
    return models.Filter(must=must) if must else None


