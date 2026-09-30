"""Add the statement line items a derived metric is computed from.

A user asks for a "quick ratio". A filing never prints that phrase; it prints
"Total current assets", "Inventories" and "Total current liabilities" on a page
titled "Consolidated Balance Sheets". Appending those words helps both halves of
the search: the embedding moves toward statement language and BM25 gets words
that are really in the filing. Measured effect: evidence in the candidate pool
for 79 of 99 questions instead of 74 (results/RETRIEVAL_EXPERIMENTS.md, section 13).
"""

from __future__ import annotations

import re

_BS = ("total current assets total current liabilities consolidated balance "
       "sheets consolidated statements of financial position")
_IS = ("consolidated statements of income consolidated statements of "
       "operations")
_CF = "consolidated statements of cash flows"

# derived metric -> the statement line items that actually carry the numbers
METRIC_TERMS: dict[str, str] = {
    "quick ratio": f"cash and cash equivalents accounts receivable inventories {_BS}",
    "current ratio": _BS,
    "working capital": _BS,
    "liquidity": f"cash and cash equivalents {_BS}",
    "operating margin": f"operating income total revenues net sales {_IS}",
    "gross margin": f"gross profit cost of sales net revenue {_IS}",
    "net margin": f"net income total revenues {_IS}",
    "profit margin": f"net income total revenues {_IS}",
    "effective tax rate": f"provision for income taxes income before income taxes {_IS}",
    "return on assets": f"net income total assets {_IS} {_BS}",
    "roa": f"net income total assets {_IS} {_BS}",
    "return on equity": f"net income total stockholders equity {_IS} {_BS}",
    "roe": f"net income total stockholders equity {_IS} {_BS}",
    "inventory turnover": f"cost of goods sold cost of sales inventories {_IS} {_BS}",
    "days inventory": f"cost of goods sold inventories {_IS} {_BS}",
    "asset turnover": f"net sales total revenues total assets property plant and equipment {_IS} {_BS}",
    "free cash flow": f"net cash provided by operating activities capital spending purchases of property plant and equipment capital expenditures {_CF}",
    "fcf": f"net cash provided by operating activities capital spending capital expenditures {_CF}",
    "capex": f"capital spending purchases of property plant and equipment capital expenditures {_CF}",
    "capital expenditure": f"capital spending purchases of property plant and equipment additions to property plant and equipment {_CF}",
    "ebitda": f"operating income depreciation and amortization {_IS} {_CF}",
    "dividend payout": f"dividends declared per share dividends paid net income {_IS} {_CF}",
    "dividends paid": f"dividends paid to shareowners cash dividends paid {_CF}",
    "interest coverage": f"interest expense operating income {_IS}",
    "debt to equity": f"total debt long-term debt total stockholders equity {_BS}",
    "leverage ratio": f"total debt long-term debt total stockholders equity {_BS}",
    "book value": f"total stockholders equity shares outstanding {_BS}",
    "days sales outstanding": f"accounts receivable net sales {_IS} {_BS}",
    "dso": f"accounts receivable net sales {_IS} {_BS}",
    "days payable": f"accounts payable cost of goods sold {_IS} {_BS}",
    "dpo": f"accounts payable cost of goods sold {_IS} {_BS}",
    # Not ratios, but the same gap: statements go by several titles.
    "property plant and equipment": f"property plant and equipment net {_BS}",
    "balance sheet": _BS,
    "income statement": _IS,
    "statement of income": _IS,
    "cash flow statement": _CF,
}

# Longest phrase first, so "return on assets" wins over a bare "roa".
_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in
                      sorted(METRIC_TERMS, key=len, reverse=True)) + r")\b",
    re.I)


def expand_query(query: str) -> str:
    """`query` plus the line items of any derived metric it names. A query
    naming none is returned unchanged."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _PATTERN.findall(query):
        terms = METRIC_TERMS[match.lower()]
        if terms not in seen:
            seen.add(terms)
            out.append(terms)
    return f"{query} {' '.join(out)}" if out else query
