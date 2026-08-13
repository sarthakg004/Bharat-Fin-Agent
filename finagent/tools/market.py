"""Market data via yfinance (no API key). All functions are pure — no LangChain
decorators — so `market_data_node` dispatches them by name from a structured LLM
intent.

    get_quote         spot + day stats + 52-week metrics
    get_history       OHLCV history + a chart spec the frontend renders
    get_company_info  name, sector, industry, summary
    get_news          recent headlines with links
    compare           side-by-side quotes for N tickers

Every result is `{ok: bool, data: {...}, error: str | None}`, so callers
short-circuit on errors without try/except gymnastics. If Yahoo throws, each tool
degrades to an explicit error in that dict rather than raising.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Any


# --------------------------------------------------------------------------- #
# Ticker resolution
# --------------------------------------------------------------------------- #

def normalise_ticker(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    if not s:
        return s
    return s                               # US-listed symbols pass through as-is


def _ok(data: Any) -> dict:
    return {"ok": True, "data": data, "error": None}


def _err(msg: str) -> dict:
    return {"ok": False, "data": None, "error": msg}


# --------------------------------------------------------------------------- #
# Public tools
# --------------------------------------------------------------------------- #

def get_quote(symbol: str) -> dict:
    """Latest spot + day-range + 52-week metrics for a ticker."""
    import yfinance as yf

    sym = normalise_ticker(symbol)
    if not sym:
        return _err("Empty ticker.")
    try:
        info = dict(yf.Ticker(sym).fast_info)
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}")
    if not info:
        return _err(f"No data for {sym!r} (delisted? wrong suffix?).")
    keys = (
        "lastPrice", "previousClose", "open", "dayHigh", "dayLow",
        "currency", "marketCap", "yearHigh", "yearLow", "yearChange",
        "fiftyDayAverage", "twoHundredDayAverage", "lastVolume",
    )
    out = {k: info.get(k) for k in keys}
    out["symbol"] = sym
    return _ok(out)


def get_history(symbol: str, period: str = "1y", interval: str = "1d") -> dict:
    """Historical OHLCV + a frontend-ready chart spec.

    `period`   1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    `interval` 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo  (intraday capped at 60d)
    """
    import yfinance as yf

    sym = normalise_ticker(symbol)
    try:
        df = yf.Ticker(sym).history(period=period, interval=interval)
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}")
    if df is None or df.empty:
        return _err(f"No history for {sym!r} (period={period}, interval={interval}).")

    df = df.reset_index()
    time_col = "Date" if "Date" in df.columns else "Datetime"

    # lightweight-charts wants UNIX seconds as `time`, and float OHLC + volume.
    candles: list[dict] = []
    volume: list[dict] = []
    for _, row in df.iterrows():
        ts = int(row[time_col].value // 1_000_000_000)
        candles.append({
            "time": ts,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low":  float(row["Low"]),
            "close": float(row["Close"]),
        })
        volume.append({
            "time": ts,
            "value": float(row.get("Volume") or 0),
            # green when close >= open, red otherwise — coloured by client too,
            # but this keeps the JSON self-describing.
            "color": "#00D08470" if row["Close"] >= row["Open"] else "#F8717170",
        })

    first, last = df.iloc[0], df.iloc[-1]
    pct = float((last["Close"] - first["Close"]) / first["Close"] * 100) if first["Close"] else 0.0

    # Volume stats so questions like "is there a rapid increase in volume?" can be
    # answered from the summary (the synthesizer reads this, not the chart). We
    # compare the most-recent window against the prior window to flag a surge.
    vol = df["Volume"].astype(float) if "Volume" in df.columns else None
    vol_stats: dict = {}
    if vol is not None and len(vol) > 0:
        win = min(5, len(vol))
        recent = float(vol.iloc[-win:].mean())
        prior = float(vol.iloc[-2 * win:-win].mean()) if len(vol) >= 2 * win else float(vol.iloc[:-win].mean() or 0)
        vol_change = ((recent - prior) / prior * 100) if prior else 0.0
        vol_stats = {
            "last_volume": int(vol.iloc[-1]),
            "avg_volume": int(vol.mean()),
            "recent_avg_volume": int(recent),
            "prior_avg_volume": int(prior),
            "volume_change_pct": round(vol_change, 1),     # recent window vs prior window
            "volume_surge": bool(vol_change >= 50),         # heuristic "rapid increase" flag
        }
    summary = {
        "symbol": sym,
        "period": period,
        "interval": interval,
        "start": str(first[time_col].date()),
        "end":   str(last[time_col].date()),
        "first_close": float(first["Close"]),
        "last_close":  float(last["Close"]),
        "high":  float(df["High"].max()),
        "low":   float(df["Low"].min()),
        "pct_change": round(pct, 2),
        "points": len(candles),
        **vol_stats,
    }
    chart = {
        "type": "candlestick",
        "symbol": sym,
        "period": period,
        "interval": interval,
        "candles": candles,
        "volume": volume,
    }
    return _ok({"summary": summary, "chart": chart})


def get_company_info(symbol: str) -> dict:
    """Company-profile basics — useful when the question is about the entity."""
    import yfinance as yf

    sym = normalise_ticker(symbol)
    try:
        info = yf.Ticker(sym).info or {}
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}")
    if not info:
        return _err(f"No info for {sym!r}.")
    out = {
        "symbol": sym,
        "long_name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "website": info.get("website"),
        "employees": info.get("fullTimeEmployees"),
        "summary": (info.get("longBusinessSummary") or "")[:1500],
    }
    return _ok(out)


def get_news(symbol: str, limit: int = 5) -> dict:
    """Recent ticker-specific news headlines."""
    import yfinance as yf

    sym = normalise_ticker(symbol)
    try:
        items = yf.Ticker(sym).news or []
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}")
    out = []
    for item in items[:limit]:
        # yfinance 0.2.40+ flattens fields onto `content`; older versions
        # have them at the top level. Cover both shapes.
        content = item.get("content") or item
        out.append({
            "title": content.get("title") or item.get("title", ""),
            "url": (content.get("canonicalUrl") or {}).get("url")
                   or item.get("link", ""),
            "publisher": content.get("provider", {}).get("displayName")
                         or item.get("publisher", ""),
            "published_at": content.get("pubDate") or item.get("providerPublishTime"),
        })
    return _ok({"symbol": sym, "articles": out})


def compare(symbols: list[str]) -> dict:
    """Side-by-side quotes for N tickers (helpful for "compare X vs Y"-style Qs)."""
    rows: list[dict] = []
    for s in symbols[:6]:
        res = get_quote(s)
        if res["ok"]:
            rows.append(res["data"])
    if not rows:
        return _err("None of the symbols resolved.")
    return _ok({"rows": rows})


# --------------------------------------------------------------------------- #
# Tool registry — dispatched by `market_data_node` from the LLM intent
# --------------------------------------------------------------------------- #

TOOLS: dict[str, dict] = {
    "get_quote": {
        "fn": get_quote,
        "description": "Latest price + day range + 52-week metrics for one ticker.",
        "args": ["symbol"],
    },
    "get_history": {
        "fn": get_history,
        "description": (
            "OHLCV history + a candlestick chart spec. Use period in "
            "{1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max} and "
            "interval in {1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo}."
        ),
        "args": ["symbol", "period", "interval"],
    },
    "get_company_info": {
        "fn": get_company_info,
        "description": "Sector / industry / business summary for one ticker.",
        "args": ["symbol"],
    },
    "get_news": {
        "fn": get_news,
        "description": "Recent ticker-specific news headlines (Yahoo aggregated).",
        "args": ["symbol", "limit"],
    },
    "compare": {
        "fn": compare,
        "description": "Quote snapshot for 2-6 tickers, returned side-by-side.",
        "args": ["symbols"],
    },
}


# --------------------------------------------------------------------------- #
# Bounded TTL cache (#12)
# --------------------------------------------------------------------------- #
# yfinance calls are slow (1-3s) and rate-limit-prone, and identical calls
# recur within a session (follow-ups, the get_history safety-net, multi-lane
# questions). A short-TTL, hard-capped LRU cache removes the redundant network
# round-trips. Memory is bounded by `_CACHE_MAX` entries (a history result with
# its chart is at most tens of KB → a few MB worst case), so it's safe on a
# small instance. Set MARKET_CACHE_TTL=0 to disable entirely.
_CACHE_TTL = float(os.getenv("MARKET_CACHE_TTL", "90"))    # seconds; 0 disables
_CACHE_MAX = int(os.getenv("MARKET_CACHE_MAX", "48"))       # hard entry cap (LRU)
_cache: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(name: str, kwargs: dict) -> str:
    return name + "|" + repr(sorted(kwargs.items()))


def call_tool(name: str, **kwargs: Any) -> dict:
    """Dispatch by name, with a bounded TTL cache. Unknown tool → error dict."""
    tool = TOOLS.get(name)
    if tool is None:
        return _err(f"Unknown tool {name!r}. Available: {list(TOOLS)}.")

    key = _cache_key(name, kwargs)
    if _CACHE_TTL > 0:
        now = time.time()
        with _cache_lock:
            hit = _cache.get(key)
            if hit is not None and (now - hit[0]) < _CACHE_TTL:
                _cache.move_to_end(key)            # mark most-recently-used
                return hit[1]

    try:
        res = tool["fn"](**kwargs)
    except TypeError as e:
        return _err(f"Bad args for {name!r}: {e}")
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}")

    # Cache only successes, so a transient failure never sticks.
    if _CACHE_TTL > 0 and isinstance(res, dict) and res.get("ok"):
        with _cache_lock:
            _cache[key] = (time.time(), res)
            _cache.move_to_end(key)
            while len(_cache) > _CACHE_MAX:        # evict least-recently-used
                _cache.popitem(last=False)
    return res
