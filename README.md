# FinAgent

FinAgent answers questions about US public companies. It splits a question into
parts, sends each part to the source that can actually answer it, writes a cited
answer, and has a second model fact-check every claim before you see it.

The idea behind the design: a language model should not be asked to remember a
number. Figures come from the SEC's structured data, ratios are computed in
Python, prices come from Yahoo Finance, and explanations come from the filings.
The models decide what to look up and how to explain it.

## What it can do

| Capability | Source |
|---|---|
| Answers grounded in filing text, with `[N]` citations | 10-K filings in Qdrant, hybrid search + rerank |
| Exact reported figures | SEC XBRL company-facts API |
| Margins, ratios, growth, CAGR, working-capital days | A Python calculator over the XBRL figures |
| Companies that are not indexed yet | Their 10-K is fetched from EDGAR and indexed on the spot |
| Price, volume, charts | Yahoo Finance |
| "Which companies disclosed X?" | EDGAR full-text search |
| News and events after the latest filing | Tavily web search |

## How a question is answered

```mermaid
flowchart TD
    Q([Question]) --> P[plan<br/>split into parts, tag each part,<br/>write one search query]
    P -->|a part needs filing text| F[fetch filing<br/>company missing? download its 10-K]
    F --> R[retrieve<br/>hybrid search, rerank, keep best 8]
    P -->|numbers, market or web only| X
    R --> X[XBRL<br/>exact filed figures]
    X --> C[calculator<br/>ratios computed in Python]
    C --> M[market data]
    C --> W[web search]
    C --> E[EDGAR search]
    M --> G[gather]
    W --> G
    E --> G
    G --> S[write<br/>cited answer]
    S --> K{critic<br/>is every claim<br/>in the evidence?}
    K -->|yes| A([Answer])
    K -->|draft overstated| S
    K -->|evidence missing| R2[retrieve again<br/>on the failed claims] --> S
    K -->|draft admits it cannot answer| W2[web search] --> S
    K -->|still under half supported<br/>after the one retry| X2([Refuse])
```

The critic gets exactly one recovery per question. It chooses which one: rewrite
the draft, search again for the claims that failed, or go to the web.

### Which model does which step

| Step | Model | Why |
|---|---|---|
| Plan, search-query rewrite, market plan | `qwen/qwen3.8-27b` on Groq | Fast, free, 1,000 requests a day per key |
| Structured extraction (company, XBRL lookup, formula, EDGAR phrase) | `qwen/qwen3.8-27b` on Groq | One short structured answer each |
| Write the answer | `gemini-3.5-flash` | Groq's free tier caps a request at 8,000 tokens, which would cut the evidence |
| Fact-check (critic) | `gemini-3.6-flash` | A different model from the writer, with its own daily quota |
| Embeddings | `gemini-embedding-2`, 1536 dimensions | The index was built with it |
| Rerank | Cohere `rerank-v4.0-pro` | Falls back to a local `bge-reranker-v2-m3` if Cohere is down |

This table lives in one place, `finagent/runtime.py`. The frontend reads it from
`GET /api/config`. Only the writer can be changed from the UI; OpenAI and
Anthropic models need your own API key, which stays in your browser.

### Inside retrieval

```mermaid
flowchart LR
    subgraph index [Building the index, offline]
        H[SEC HTML filing] --> PA[parent passages<br/>up to 2,500 chars<br/>tables kept whole]
        PA --> CH[child chunks<br/>600 chars, each prefixed<br/>company + year + section]
        CH --> V[(Qdrant<br/>dense vector + BM25 vector)]
    end
    subgraph search [Answering, online]
        Q2[rewritten query<br/>+ the raw question] --> EX[add line items for<br/>derived metrics]
        EX --> FI[company and year filter]
        FI --> HY[hybrid search<br/>48 candidates]
        HY --> PR[swap children<br/>for their parents]
        PR --> RR[rerank]
        RR --> CAP[keep the best 8,<br/>scored against the question]
    end
    V -.-> HY
```

Search matches on small chunks, because a short chunk matches a query precisely.
The model is then given the larger parent passage, because it needs the context.

## Project layout

```
finagent/
  runtime.py           which model does which job
  llm.py               chat models, key rotation, error classification
  vectorstore.py       Qdrant client and Gemini embeddings
  config.py            settings from the environment
  agent/
    agent.py           FinAgent: tools, graph wiring, routing
    state.py           the shared state and structured-output schemas
    prompts.py         every prompt
    plan.py            planner and search-query rewrite
    retrieve.py        10-K fetch gate, search, the 8-passage cap
    numeric.py         XBRL and calculator steps
    external.py        market data, web search, EDGAR search steps
    answer.py          write, fact-check, refuse
  retrieval/           hybrid.py  filters.py  expansion.py  reranker.py
  tools/               xbrl.py  calculator.py  sec_fetch.py  resolver.py
                       market.py  web_search.py  edgar_search.py
  ingestion/ingest.py  SEC HTML -> chunks -> Qdrant
  api/                 main.py (routes, streaming, errors)  service.py  models.py
  evaluation/          retrieval.py  answers.py
frontend/              React, TypeScript, Vite, Tailwind
tests/                 no network; providers and stores are stubbed
```

Each step of the graph is a plain function `step(agent, state) -> changed keys`.
`FinAgent` owns the tools and wires the steps together.

## Errors and retries

Provider errors are classified by HTTP status first and message second, because
providers reuse one status for different problems.

| What happened | How it is recognised | What the app does |
|---|---|---|
| Per-minute rate limit | 429 with a short reset | Tries the next key, waits if all are limited, then the UI retries with a countdown |
| Daily quota used up | 429 naming a per-day quota | Stops at once and says when it resets |
| Provider overloaded | 500, 502, 503, 529 | Retries |
| Prompt too large | 413 | Tells you to pick another writer or start a new chat |
| Key rejected | 401, 403 | Drops that key from the pool, or asks for a valid one |
| Model removed | 404 | Says so |
| Connection dropped | The stream ends without a `done` event | The UI retries twice, then shows a Retry button |

A step that fails on its own (for example the fact-check) does not lose the
answer. The answer is built from what was gathered and a notice says what was
skipped.

## Evaluation

Measured on FinanceBench: 150 questions, of which 127 have a filing available
as SEC HTML, of which 99 have evidence the HTML parser can recover.

```bash
# Does the evidence reach the writer? Three reranker arms: none, Cohere, local.
python -m finagent.evaluation.retrieval

# Answer every question with the production agent, then score it.
python -m finagent.evaluation.answers run   --output results/v7/answers.json
python -m finagent.evaluation.answers score --output results/v7/answers.json
```

Retrieval reports pool recall, evidence coverage and hit rate at 5 and 8, figure
recall, mean reciprocal rank and retention. Answers are scored with six RAGAS
metrics, a yes/no `verdict` (does the answer reach the gold answer, whatever its
length) and a judge-free check that the gold figure appears in the answer.

Both steps resume where they stopped. Add `--sample 10` for a quick run; a
sample run writes to its own files. The eval corpus is kept on a local Qdrant
(`QDRANT_EVAL_URL`), not on the served cluster.

What has been measured so far, all on the 99 questions:

| Change | Evidence in the top 8 |
|---|---|
| Starting point | 37 |
| Prefix every chunk with company, year and section | 57 |
| Re-score the final 8 against the question, expand derived metrics, search the raw question too | 67 |
| Search on one rewritten query instead of the sub-queries | 75 |
| A stronger model writing that query (Gemini) | 79 |

Those rows used the earlier local embedder (`bge-large`) and reranker. The
current stack (Gemini embeddings, Qwen search-query rewrite) on the same 99
questions:

| Reranker | Evidence found by search | In the top 5 | In the top 8 | MRR | Seconds per question |
|---|---|---|---|---|---|
| None | 95 | 60 | 78 | 0.45 | 0.02 |
| Cohere `rerank-v4.0-pro` (served) | 95 | 82 | **89** | 0.63 | 2.5 |
| Local `bge-reranker-v2-m3` (fallback) | 95 | 72 | 82 | 0.53 | 7.9 |

With Cohere, by question type: numeric 57 of 60, narrative 22 of 27, comparison
10 of 12. The embedder, the reranker and the rewrite model all changed between
79 and 89, so the gain is not attributed to any one of them. Full output is in
`results/retrieval_eval.md`; `results/RETRIEVAL_EXPERIMENTS.md` records every
earlier experiment, including the ones that lost.

### Answer quality

The last full run (`results/v7/`) answered all 127 questions with the production
agent, with Claude Haiku as the writer and critic (production uses Gemini for
both), and scored them with Claude Sonnet 5 as the judge.

The numbers below leave out 15 questions whose filing is not in the eval index:
the match from FinanceBench to SEC documents picked the wrong period for 8
filings (for example the Q3 10-Q for a Q2 question). Those questions cannot be
answered from the index. The list is `FILING_NOT_INDEXED` in
`finagent/evaluation/answers.py`; `results/v7/answers_report.md` has every
question too.

| Metric (112 questions) | Score |
|---|---|
| **verdict: answer is correct** | **0.85** |
| numeric questions correct | 58 of 69 |
| comparison questions correct | 12 of 13 |
| narrative questions correct | 25 of 30 |
| groundedness | 0.97 |
| answer_relevancy | 0.82 |
| context_recall | 0.79 |
| faithfulness | 0.77 |
| answer_correctness | 0.49 |
| context_precision | 0.38 |
| errors / refusals | 0 / 0 |
| latency median / p95 | 55 s / 236 s |

The verdict and the judge-free figure check agree on 55 of 65 numeric
questions. Of the 10 disagreements, the judge was right in 6, too strict about
rounding in 3 (8.74 against a gold of 8.70) and too lenient in 1.

Why three of the scores are low:

- **faithfulness (0.77)** checks every statement against the passages. A figure
  from the SEC's structured data reaches the judge as a fact line such as
  `capital expenditure (FY2018) = $1,577,000,000 (...)`, which does not name the
  company (the writer's copy of the same fact does). The judge cannot confirm the "3M's" in "3M's FY2018 capital
  expenditure was $1,577 million", so exactly right answers score 0.
  Groundedness, which grades the answer as a whole, is 0.97.
- **context_precision (0.38)** asks the judge, for each piece of evidence in
  order, whether it helped reach the gold answer, then weights the yes answers
  by rank. Extra evidence after the useful piece costs nothing; the score is low
  because 58 of 127 answers scored 0, meaning the judge found no piece useful,
  and 39 of those 58 answers were correct. 44 of the 58 start with structured
  facts. For "3-year average capex as % of revenue" the evidence was exactly
  the six inputs (capex and revenue for three years), the gold answer was
  "1.9%", and the score was 0: the judge does not connect `capital expenditures
  (FY2017) = $155,000,000` to a one-number gold answer, and the fact line does
  not name the company.
- **answer_correctness (0.49)** is mostly F1 over statements against a
  one-line gold answer. Every extra true statement (the inputs of a ratio, a
  driver of a change) counts as a false positive, so a correct answer of three
  sentences scores about 0.4. The verdict was added to measure correctness
  without that length penalty.

## Run it locally

You need Python 3.11+, Node 20+, a Qdrant cluster, and Groq and Gemini API keys.
Cohere and Tavily keys are recommended.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # fill in the keys

uvicorn finagent.api.main:app --reload --port 8000
cd frontend && npm install && npm run dev      # http://localhost:5173
```

To add filings in bulk, ingest a manifest (a JSON list of filing records):

```bash
python -m finagent.ingestion.ingest --manifest data/us/pdfs/rebuild_manifest.json \
    --collection us_filings_v5_gemini
```

Ingestion can be re-run safely. Point ids are derived from the filing, the
position and the text, so a re-run overwrites instead of duplicating, and
embeddings are cached on disk so it costs no API quota.

### Chunking strategy

Filings are chunked with parent-document retrieval (`finagent/ingestion/ingest.py`):

| Piece | Size | Role |
|---|---|---|
| Parent passage | a section, up to 2,500 characters | What the reranker scores and the writer reads |
| Child chunk | 600 characters, 100 overlap | What is embedded and matched |
| Table | kept whole | Its own parent and child, so rows stay together |
| Context header | `<company> <year> · <section caption>` | Prefixed to every chunk |

Small children match a query precisely; the larger parent gives the model the
context around the match. The header exists because the chunker separates a
statement's heading from the table below it, which left the table with no
company, year or statement name to match on. Adding it took the evidence hit
rate from 40 to 57 of 99 questions.

A chunk's id is derived from its text. The index in Qdrant was built with this
chunking, so a change to chunk text means re-embedding the corpus;
`tests/test_retrieval.py` checks the chunker's output against a recorded digest.

Run the tests with `pytest`.

## Configuration

| Variable | Purpose |
|---|---|
| `GROQ_API_KEYS` | Planner and extractors. One key or several, comma separated. |
| `GEMINI_API_KEYS` | Writer, critic and embeddings. |
| `COHERE_API_KEYS` | Reranker. Without it the local cross-encoder is used. |
| `QDRANT_URL`, `QDRANT_API_KEY` | The cluster holding the filings. |
| `TAVILY_API_KEY` | Web search. |
| `US_COLLECTION` | The collection to serve. |
| `RERANKER_MODEL` | Defaults to `cohere:rerank-v4.0-pro`. |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | Tracing. Optional. |

With several keys for a provider, a key that hits a limit is swapped for the
next one. Keep each pool in a single variable: in Google Secret Manager that is
one secret instead of one per key.

## Deployment

Pushing to `main` runs `.github/workflows/deploy.yml`: tests, then the backend
to Cloud Run (one image serving the API and the built frontend, scaling to zero)
and the frontend to Firebase Hosting. Auth to Google Cloud is keyless through
Workload Identity Federation.

The app runs on free tiers, which sets its limits: about 20 writer requests per
Gemini key per day, 1,000 embedded texts per key per day (one live 10-K fetch
uses most of a key), and 1,000 Cohere calls per key per month. It answers one
question at a time.

## Observability

With the `LANGFUSE_*` keys set, every question produces one trace with a span
per step and every model call with its prompt, output and token count. Each
answer also carries its own summary: tokens, seconds per step, the fact-check
score, and any step that was skipped.
