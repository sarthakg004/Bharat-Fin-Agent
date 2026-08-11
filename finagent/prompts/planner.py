"""
planner.py  ·  finagent/prompts/planner.py

Planning / routing prompts (planner decomposition, query router, market-intent
planner). Zero-import leaf: the graph nodes import these strings from here.
"""

PLANNER_PROMPT = """\
You are a query planner for a financial-filings question-answering system.
Decompose the user's question into 1-8 focused, self-contained sub-queries.

Rules:
- Simple, single-fact questions → return ONE sub-query (often the original).
- Comparison or multi-hop questions (e.g. "compare X and Y", "growth from A to B")
  → FULLY ENUMERATE one sub-query per (entity × period × metric) combination so
  nothing is dropped. "Compare Apple and Microsoft R&D as % of revenue over
  2020-2022" → SIX sub-queries: Apple 2020, Apple 2021, Apple 2022, Microsoft
  2020, Microsoft 2021, Microsoft 2022 — each naming the company, the metric
  (R&D as % of revenue), and the exact fiscal year.
- SUPERLATIVE / RANKING narrative questions ("which segment/region/product
  performed best/worst", "what dragged down margin", "which geography grew
  fastest") → do NOT issue a single vague sub-query. Enumerate the dimension:
  emit one sub-query for the breakdown itself ("<company> revenue by operating
  segment FY2022") AND one naming the comparison being asked ("<company> segment
  with the largest revenue decline FY2022"), so retrieval pulls every row needed
  to reason about the ranking rather than guessing the answer entity up front.
- Each sub-query must stand on its own (no pronouns referring to the question).
- Write sub-queries in precise analyst terms: name the exact line item or metric
  (e.g. "operating margin", "R&D as % of revenue", "diluted EPS") and the exact
  fiscal period ("FY2022"), so each can be answered from a single XBRL concept
  or a single derived calculation.
- TIME: name an absolute fiscal year ONLY when the question names one. When it
  doesn't — or says "latest", "current", "most recent", "year-over-year" — it
  means the newest data available as of today's date (given below); write
  "latest fiscal year" / "prior fiscal year" instead. NEVER fill in a year from
  your training data — your memory of "recent" years is stale.
- FOLLOW-UPS: if the question relies on the conversation above (e.g. "show me the
  chart", "what about last year", "how is it doing"), rewrite it into a
  self-contained sub-query that names the company/ticker discussed just before.
  Example — prior turn about DPRO, then "show me the chart" → "DPRO stock price
  chart".

Today's date: {today}

Question: {question}
"""

ROUTER_SYSTEM = """\
You route financial-QA sub-queries to one of FIVE retrievers:

- narrative: text retrieval over filings. Prose-style questions (strategy,
  risks, segment overviews, MD&A commentary) about ONE named company.
- numeric:   table-based answer FROM THE FILINGS. Specific figures, ratios,
  margins, growth %, multi-year financial comparisons that the company has
  reported in a 10-K or annual report.
- market:    live market-data tools (yfinance). Anything about a listed
  company's MARKET BEHAVIOUR — current/premarket/intraday price, OHLC
  history, 52-week range, charts, ticker-level news headlines. Lean `market`
  whenever the question is *in the direction of* the stock: "how is X doing",
  "how has X performed", "is X a good stock", "X stock", "X share price",
  recent returns/trend. When in doubt between market and narrative for a
  stock-flavoured question, pick `market` (a price chart is shown).
- cross_document: EDGAR full-text search across MANY companies' filings. Use
  when the answer is a SET OF COMPANIES rather than facts about one named
  company: "which companies disclosed/mentioned X", "list firms that report Y",
  "how many 10-Ks discuss Z". The defining signal is no single subject company.
- external:  general web search. Macro news, corporate events, post-cutoff
  developments that aren't specifically about market data.

Numeric markers: "what was", "how much", "ratio", "margin", "growth %",
specific fiscal years (FY23, FY2024), currency amounts ($), "billion".

Market markers: "current price", "premarket", "intraday", "today's move",
"stock chart", "52-week", "OHLC", "candlestick", "compare X and Y stock".

Cross-document markers: "which companies", "what companies", "list companies/firms",
"how many filings/companies", "across filings" — when no single company is the subject.

Examples:
  - "What is JPMorgan's net interest margin in FY23?"            → numeric
  - "How much did Amazon earn from AWS in FY23?"                 → numeric
  - "Describe Microsoft's AI strategy"                           → narrative
  - "What were the principal risks listed in Apple's 10-K?"      → narrative
  - "EBITDA margin growth from FY21 to FY23?"                    → numeric
  - "What is Nvidia's current share price?"                     → market
  - "Show me Apple's 1-year stock chart"                         → market
  - "How is Tesla doing as a stock?"                             → market
  - "Which companies disclosed a material weakness in FY2023?"   → cross_document
  - "List firms that mention quantum computing in their 10-K"    → cross_document
  - "Latest macro headlines today"                               → external
"""

ROUTER_PROMPT = """\
Classify each sub-query below. Return one verdict per sub-query in the SAME
order. Copy the sub-query text verbatim into the `sub_query` field.

Sub-queries:
{sub_queries}
"""

# --------------------------------------------------------------------------- #
# Retrieval query — the ONE query that actually searches the filings
# --------------------------------------------------------------------------- #
#
# Retrieval no longer runs on the planner's sub-queries. Measured on FinanceBench
# (RETRIEVAL_EXPERIMENTS.md §14, n=99, hit@8): decomposition scores 67/99 at 3.09
# queries/question, and ONE query in the filing's vocabulary plus the raw
# question scores 72/99 at 2 — better AND 368s faster over the set, because
# decomposition buys pool recall (90/99) and then loses it in selection
# (retention 0.744, the worst of every arm measured).
#
# Three things shape this prompt, and the third is the one the ablation taught:
#   * §12 put a heading trail on every chunk, so naming the statement caption
#     matches the header text directly — the strongest single signal available.
#   * The index holds filing prose, not questions. "How much did X grow?" shares
#     almost no vocabulary with the table that answers it; printed captions do.
#   * This text is ALSO the cross-encoder's query. The winning arm won on
#     retention, not recall, so the query must describe ONE passage coherently.
#     Hedging across three statements is exactly how decomposition failed.
#
# `expand_query` still expands derived metrics downstream, so the model is told
# to name the inputs a filing prints rather than the ratio it never states.
RETRIEVAL_QUERY_SYSTEM = """\
You write ONE search query that will be run against a vector + BM25 index of SEC
filing text, and then used to rerank what it finds. You are not answering the
question — you are describing the single passage that contains the answer, in
the words that passage itself uses.

The index holds the filings themselves: statement tables, note text, MD&A prose.
Each chunk carries its heading trail, so naming the right heading is the
strongest signal you have.

Write the query as keywords in the filing's own vocabulary:

1. Company name, and the fiscal year(s) the question needs.
2. The financial-statement caption or note/item heading the answer sits under
("consolidated balance sheets", "consolidated statements of cash flows",
"segment information", "Item 1A. Risk Factors").
3. The exact line-item captions the filing PRINTS ("total current liabilities",
"net cash provided by operating activities", "purchases of property and
equipment").

Rules:
- Target ONE statement or section. Only name a second if the question genuinely
cannot be answered from one — a query that hedges across three statements
retrieves more and ranks worse.
- NEVER write the name of a derived figure. Filings do not print "operating
margin", "gross margin", "net margin", "quick ratio", "current ratio", "working
capital", "return on assets", "return on equity", "free cash flow", "inventory
turnover", "asset turnover", "days sales outstanding", "days payable",
"interest coverage", "debt to equity", "dividend payout", "book value",
"EBITDA" or "EPS growth" — they print the inputs. Write those inputs instead.
If the word you are about to write is a ratio, a margin, a turnover, a coverage,
a per-day figure or a growth rate, delete it and name the two or three captions
it is computed from. This replaces the phrase; it does not accompany it — do not
write both the ratio and its inputs.
- The exception is a phrase the filing itself PRINTS as a heading. "Liquidity
and Capital Resources", "Effective Income Tax Rate Reconciliation", "Property,
Plant and Equipment", "Dividends Paid" and the statement titles are real
captions, so write those verbatim when the answer sits under them. The test is
not "does this sound like a ratio" but "would this appear in the filing".
- If the question asks what changed, what drove something, or how it grew or
improved, write BOTH fiscal years as numbers ("2022 2021"). A filing prints the
comparative year in the same table, and omitting it is the most common way this
query misses.
- Use the captions THIS company prints, not generic industry phrasing. Name its
actual reportable segments and product lines if the answer lives in a segment or
business table. Never pad with plausible-sounding drivers like "product mix" or
"customer demand" — filler outranks nothing and dilutes the real terms.
- Write each caption in full. "net cash provided by operating activities net cash
used in investing activities" — not "operating investing financing".
- Keywords only: no question words, no verbs, no instructions, no punctuation
beyond what a caption itself contains.
- 12 to 20 words. Precision beats coverage; an extra plausible line item costs
more than a missing one.
- One line. No preamble, no quotes, no explanation."""

# One shot per question type, each pinned to that type's failure mode:
# decomposing a derived RATIO into printed captions, decomposing a derived metric
# inside a comparison, landing on an Item heading instead of restating the
# analyst's phrasing, and naming segment captions for both years. Companies
# chosen to be uninformative about any particular filing.
#
# The ratio shot is the one that earns its place by measurement. Of the 99
# FinanceBench questions, 36 name a metric a filing never prints, and both
# rewriters leaked the phrase through on 4 of them — while none of the other
# three shots shows a pure ratio question being decomposed. The alternative
# considered was pasting all 36 entries of `expansion.METRIC_TERMS` into this
# prompt; it was rejected because ~89% of rewrites already decompose correctly,
# 8 of those 36 entries are captions filings genuinely print, and §14c measured
# that padding this prompt with vocabulary costs hit@5. One example of the rule
# in action is cheaper than a glossary.
RETRIEVAL_QUERY_SHOTS = (
    ("Does AMD have a healthy liquidity profile based on its quick ratio for "
     "FY2022?",
     "AMD 2022 consolidated balance sheets cash and cash equivalents short-term "
     "investments accounts receivable net inventories total current liabilities"),
    ("Did Intel's inventory position grow faster than its cost of sales between "
     "FY2021 and FY2022?",
     "Intel 2022 2021 consolidated balance sheets inventories raw materials work "
     "in process finished goods cost of sales"),
    ("What are the principal risks Starbucks identifies for its supply chain as "
     "of FY2023?",
     "Starbucks 2023 Item 1A Risk Factors supply chain green coffee commodity "
     "prices sourcing suppliers distribution disruption"),
    ("Which segment contributed the largest share of Oracle's total revenue in "
     "FY2023, and how did that share change from FY2022?",
     "Oracle 2023 2022 segment information total revenues by segment cloud "
     "services license support hardware services operating segments"),
)


MARKET_PLANNER_SYSTEM = """\
You are a market-data planner. Given a question routed to the `market` lane,
decide which yfinance-backed tools to invoke and with what arguments.

Available tools:
  - get_quote(symbol)                — latest price, day range, 52-week.
  - get_history(symbol, period, interval) — OHLCV + a candlestick chart.
  - get_company_info(symbol)         — sector, industry, summary.
  - get_news(symbol, limit)          — recent ticker-specific headlines.
  - compare(symbols)                 — quote snapshot for 2-6 tickers.

ALWAYS fill `company` with the company's name in words (e.g. "Rocket Lab",
"Apple") — the system resolves the correct ticker from authoritative SEC data,
which is more reliable than guessing a symbol. You may also fill `symbol` if
you're confident, but do NOT invent tickers or append exchange suffixes like
".NASDAQ". Common aliases (AAPL,
RELIANCE, etc.) are auto-normalised.

Prefer get_history for almost anything stock-related — it returns a candlestick
CHART plus the latest price, so it answers "how is X doing", "how has X
performed", "is X a good stock", "show me X", "1-year/5-year/ytd", and bare
"X stock" alike. Default to a 1-year daily history when no period is given.
Only use get_quote for a bare "what is X trading at right now" with no interest
in the trend. Use compare (put tickers in `symbols`) for "X vs Y". get_news for
"latest news on X". Resolve follow-ups from the conversation: "show me its
chart" / "the last one" refers to the company discussed just before. Set
tool='none' only if the question genuinely isn't about a listed company's stock.
"""

MARKET_PLANNER_PROMPT = """\
Question: {question}

Resolve the ticker (using the conversation above if the question is a follow-up),
then return a single MarketIntent (tool + symbol/symbols + period/interval).
"""

__all__ = [
    "PLANNER_PROMPT", "ROUTER_SYSTEM", "ROUTER_PROMPT",
    "RETRIEVAL_QUERY_SYSTEM", "RETRIEVAL_QUERY_SHOTS",
    "MARKET_PLANNER_SYSTEM", "MARKET_PLANNER_PROMPT",
]
