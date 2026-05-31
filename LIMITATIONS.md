# Known limitations

An honest list of failure modes the system still has. Recruiters appreciate
this kind of disclosure more than over-claiming.

## Corpus gaps

* **TCS, Infosys, and a handful of other India PDFs failed to download** in
  the Week-1 fetch (HTTP 403 from the issuer's IR site). They are absent from
  the `india_filings` collection. Any question about them returns "no
  matching information" honestly.
* **HDFCBank 2023 and 2024 PDFs are corrupt** (broken xref / missing `/Root`).
  We skip them during ingestion; nothing about HDFC Bank for those years is
  available.
* **US 10-K coverage** is the first three filings per ticker from EDGAR via
  `sec_edgar_downloader` — no exhibits, no proxy statements, no quarterly
  reports.

## Citation semantics

* Citation tags name the **report's** fiscal year, not the figure's fiscal
  year. Annual reports carry prior-year comparatives, so e.g. an FY23 figure
  can legitimately appear cited as `[Reliance AR 2024, p. 100]` because the
  source document is the FY2024 annual report. The answer text says which
  fiscal year the figure refers to; the tag identifies the document.

## Numeric reasoning

* The 8B generator occasionally **misreads Indian digit grouping** (lakh /
  crore: `9,14,472 crore` is *nine* lakh-crore, not three hundred). Using
  `llama-3.3-70b-versatile` for the synth role (the default) fixes most of
  this; an OpenAI/Anthropic key would do better.
* The critic catches inverted comparisons (it correctly flagged `9,14,472 <
  3,45,967` in our test) and the numeric verifier rejects un-grounded
  figures, but neither can heal a fundamentally bad number — they only
  prevent the agent from confidently asserting it.

## Table extraction

* `unstructured` with `strategy="hi_res"` is the slow step (multiple minutes
  per ~100-page PDF on CPU). For first-pass smoke tests use `limit=1`.
* Tables that **span multiple pages** are extracted as two separate tables —
  often without re-printed headers on page 2.
* **Merged cells / multi-line headers** in scanned PDFs can confuse the
  layout model, leaving the resulting DataFrame with `NaN`s in the header
  row.
* The table embedding text (title + columns + first/last row) is short — if
  the actual relevant rows are in the middle of a 500-row table, retrieval
  can miss them.

## Code execution sandbox

* `exec` runs in a curated-builtins dict (no `import`, `open`, `eval`, etc.).
  That is **enough for a portfolio project**, not enough for any production
  exposure to user-supplied questions. For production: subprocess + seccomp,
  gVisor, or a WASM runtime.

## Web search

* Tavily free tier rate-limits aggressively. Without a key we fall back to
  the local `news` Chroma collection (built from `kdave/Indian_Financial_News`),
  which is **stale** the moment it is downloaded.
* External-route questions about events after the news dataset's last
  refresh will be answered with the closest stale article (or refused if
  retrieval scores zero results).

## Bilingual

* Detection is by `langdetect` which is unreliable on short queries (<8
  chars treated as English). Mixed-language ("Hinglish") questions usually
  detect as Hindi but translate badly.
* LLM translation preserves digits and proper nouns well but loses
  emphasis/idiom. The round-trip can subtly alter the question — the
  retriever sees the *translated* version, not the original.

## Rate limits and quotas

* Groq's free tier enforces **per-org, per-model TPD** (tokens-per-day).
  Cycling across multiple API keys from the SAME organisation does not
  help; cycling across keys from **different** orgs gives each its own
  bucket (`RotatingChatModel` does this).
* The 70B model has a much smaller TPD bucket than the 8B model. For
  heavy RAGAS judge runs, dropping the judge to `llama-3.1-8b-instant`
  buys ~5× headroom.

## Eval coverage

* Comparison numbers in `results/comparison.csv` reflect small samples
  (5-50 questions). Headline metrics on the full FinanceBench (150) and
  BhashaBench-Finance (3705 in the train split) will move; treat the
  smaller sample numbers as directionally meaningful only.

## Known graph behaviour

* The corrective loops (rewrite + critic-retry) can cap out without
  converging on a good answer; in that case the refusal node fires
  ("I don't have enough information to answer this …") instead of
  fabricating. That is intentional but counts as a "failure" from the
  user's point of view.
* `low_confidence: True` is informational — we surface it on the returned
  state and you can choose to surface it to the user as a hedge in the
  UI.
