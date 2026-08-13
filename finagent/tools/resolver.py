"""Ticker or company name → the SEC's zero-padded 10-digit CIK, which every SEC
data API requires. Shared plumbing under the XBRL, dynamic-fetch and full-text
search tools.

Source of truth is SEC's `company_tickers.json`, downloaded once, cached on disk
(refreshed past `ttl_days`) and served from memory:

    ticker match  exact, case-insensitive ("AAPL" / "aapl")
    name exact    after normalising away corporate suffixes ("Apple Inc." →
                  "apple"), so "Apple" resolves
    name fuzzy    `difflib` close-match fallback ("Microsft" → "Microsoft Corp")

SEC asks every request to identify itself; `SEC_UA_NAME` / `SEC_UA_EMAIL` fill
the User-Agent, falling back to a generic contact string.
"""

from __future__ import annotations

import json
import os
import re
import time
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Optional

import requests

from finagent.tools.base import BaseTool

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_CACHE = "data/cache/company_tickers.json"
DEFAULT_TTL_DAYS = 7

# Major delisted/acquired filers. SEC's company_tickers.json only lists CURRENT
# registrants, so these vanish from it — but their historical XBRL facts and
# filings remain fully available by CIK. Without this map, "ATVI" fuzzy-matched
# to "ATI INC" and the agent answered with the WRONG COMPANY's figures.
# Live listings always win (applied with setdefault); this only fills gaps.
DELISTED: dict[str, str] = {
    # ticker: (cik, registrant title)
    "ATVI": ("0000718877", "Activision Blizzard, Inc."),
    "TWTR": ("0001418091", "Twitter, Inc."),
    "VMW":  ("0001124610", "VMware, Inc."),
    "XLNX": ("0000743988", "Xilinx, Inc."),
    "SPLK": ("0001353283", "Splunk Inc."),
    "SGEN": ("0001060736", "Seagen Inc."),
    "FRC":  ("0001132979", "First Republic Bank"),
    "SIVB": ("0000719739", "SVB Financial Group"),
}

# Corporate suffixes / filler stripped before name matching so "Apple" matches
# "Apple Inc." and "Microsoft" matches "MICROSOFT CORP".
_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies",
    "ltd", "limited", "llc", "lp", "plc", "sa", "nv", "ag", "holdings", "holding",
    "group", "the", "and", "&", "class", "common", "stock", "trust", "fund",
}


def _build_user_agent() -> str:
    name = (os.getenv("SEC_UA_NAME") or "").strip()
    email = (os.getenv("SEC_UA_EMAIL") or "").strip()
    if name and email:
        return f"finagent/1.0 ({name}; {email})"
    # SEC blocks blank/library UAs; ship a generic but identifiable fallback.
    return "finagent/1.0 (contact: finagent@example.com)"


def _normalize_name(name: str) -> str:
    """Lowercase, drop punctuation, and strip corporate-suffix tokens."""
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    tokens = [t for t in cleaned.split() if t and t not in _SUFFIXES]
    return " ".join(tokens)


def pad_cik(cik: int | str) -> str:
    """Zero-pad a CIK to the 10-digit form the SEC data APIs require."""
    digits = re.sub(r"\D", "", str(cik))
    return digits.zfill(10)


class TickerCIKResolver(BaseTool):
    """Resolve a ticker symbol or company name to its SEC CIK.

    Usage:
        r = TickerCIKResolver()
        r.cik("AAPL")        -> "0000320193"
        r.cik("Apple")       -> "0000320193"
        r.run("Microsoft")   -> {"cik": "0000789019", "ticker": "MSFT", ...}
    """

    name = "ticker_cik_resolver"
    description = "Resolve a ticker symbol or company name to its SEC CIK."

    def __init__(
        self,
        cache_path: str | Path = DEFAULT_CACHE,
        ttl_days: int = DEFAULT_TTL_DAYS,
        user_agent: Optional[str] = None,
        fuzzy_cutoff: float = 0.84,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.ttl_seconds = ttl_days * 86400
        self.user_agent = user_agent or _build_user_agent()
        self.fuzzy_cutoff = fuzzy_cutoff

        # Built lazily on first lookup (see `_ensure_loaded`).
        self._by_ticker: dict[str, dict] = {}
        self._by_name: dict[str, dict] = {}       # normalized name -> entry
        self._norm_names: list[str] = []          # keys of _by_name, for fuzzy

    # --- mapping load / cache ------------------------------------------------

    def _cache_fresh(self) -> bool:
        return (
            self.cache_path.exists()
            and (time.time() - self.cache_path.stat().st_mtime) < self.ttl_seconds
        )

    def _load_raw(self) -> dict:
        """Return the raw `company_tickers.json` payload, cache-first.

        Uses a fresh on-disk cache if present; otherwise downloads from SEC and
        writes the cache. If the download fails but a (possibly stale) cache
        exists, fall back to it rather than erroring.
        """
        if self._cache_fresh():
            return json.loads(self.cache_path.read_text())

        try:
            resp = requests.get(
                SEC_TICKERS_URL, headers={"User-Agent": self.user_agent}, timeout=30
            )
            resp.raise_for_status()
            payload = resp.json()
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(payload))
            return payload
        except Exception:
            if self.cache_path.exists():   # network down → serve stale cache
                return json.loads(self.cache_path.read_text())
            raise

    def _ensure_loaded(self) -> None:
        if self._by_ticker:
            return
        raw = self._load_raw()
        # `raw` is a dict keyed by stringified row index; values hold the fields.
        rows = raw.values() if isinstance(raw, dict) else raw
        for row in rows:
            ticker = str(row.get("ticker", "")).upper().strip()
            title = str(row.get("title", "")).strip()
            cik = pad_cik(row.get("cik_str", row.get("cik", "")))
            entry = {"cik": cik, "ticker": ticker, "title": title}
            if ticker and ticker not in self._by_ticker:
                self._by_ticker[ticker] = entry
            norm = _normalize_name(title)
            # First writer wins so the canonical/primary listing keeps the name
            # (share classes like GOOG/GOOGL share a normalized title).
            if norm and norm not in self._by_name:
                self._by_name[norm] = entry
        # Delisted filers: fill gaps only — a live listing that reuses the
        # ticker keeps priority.
        for ticker, (cik, title) in DELISTED.items():
            entry = {"cik": cik, "ticker": ticker, "title": title}
            self._by_ticker.setdefault(ticker, entry)
            norm = _normalize_name(title)
            if norm:
                self._by_name.setdefault(norm, entry)
        self._norm_names = list(self._by_name.keys())

    # --- lookup --------------------------------------------------------------

    def resolve(self, query: str) -> dict:
        """Resolve `query` (ticker or company name) to a CIK entry.

        Returns ``{"query", "cik", "ticker", "title", "match", "score"}``. On no
        match, ``cik`` is ``None`` and ``match`` is ``"none"``. ``match`` is one
        of ``ticker`` / ``name_exact`` / ``name_fuzzy`` / ``none``.
        """
        self._ensure_loaded()
        q = (query or "").strip()
        miss = {"query": query, "cik": None, "ticker": None, "title": None,
                "match": "none", "score": 0.0}
        if not q:
            return miss

        # 1. Exact ticker (the common, unambiguous case).
        hit = self._by_ticker.get(q.upper())
        if hit:
            return {"query": query, **hit, "match": "ticker", "score": 1.0}

        # 2. Exact normalized-name match.
        norm = _normalize_name(q)
        if norm and norm in self._by_name:
            return {"query": query, **self._by_name[norm], "match": "name_exact",
                    "score": 1.0}

        # A ticker-shaped query (short, uppercase, single token) that missed the
        # exact-ticker and exact-name lookups must NOT fall through to prefix /
        # fuzzy NAME matching — that's how "ATVI" landed on "ATI INC" and the
        # agent answered with another company's figures. A miss here correctly
        # hands the question to retrieval / web instead.
        if q.isupper() and q.isalpha() and len(q) <= 5:
            return miss

        # 3. Prefix match: a short query is often a prefix of the full
        #    normalized title — "Amazon"→"amazon com", "JPMorgan"→"jpmorgan
        #    chase", "Pepsi"→"pepsico". Pick the closest such title (highest
        #    similarity) so prefixes resolve before we fall to noisy fuzzy. The
        #    len>=4 guard keeps tiny queries ("co", "the") from matching broadly.
        if len(norm) >= 4:
            cands = [n for n in self._norm_names if n.startswith(norm)]
            if cands:
                best = max(cands, key=lambda n: SequenceMatcher(None, norm, n).ratio())
                score = SequenceMatcher(None, norm, best).ratio()
                return {"query": query, **self._by_name[best],
                        "match": "name_prefix", "score": round(score, 3)}

        # 4. Fuzzy name fallback.
        if norm:
            close = get_close_matches(norm, self._norm_names, n=1,
                                      cutoff=self.fuzzy_cutoff)
            if close:
                score = SequenceMatcher(None, norm, close[0]).ratio()
                return {"query": query, **self._by_name[close[0]],
                        "match": "name_fuzzy", "score": round(score, 3)}

        return miss

    def cik(self, query: str) -> Optional[str]:
        """Convenience: return just the zero-padded CIK, or ``None``."""
        return self.resolve(query).get("cik")

    def run(self, query: str) -> dict:  # BaseTool interface
        return self.resolve(query)
