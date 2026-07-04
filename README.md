# FinAgent · Agentic RAG over financial filings

FinAgent answers questions about **SEC filings and listed companies**. A
LangGraph agent plans and routes the query in one pass, retrieves from the
filings, pulls **exact XBRL figures** from SEC company-facts, computes derived
metrics **deterministically**, runs a numeric **table agent**, fetches **live
market data** (Yahoo Finance) and **web results** (Tavily) when needed, and
**verifies every figure against the evidence** — refusing rather than
fabricating when it can't ground a claim. A React UI streams the answer with
live agent progress and the source chunks shown alongside.

---

## Features

- **Grounded answers over filings** — hybrid retrieval (BM25 + dense `bge-small` +
  cross-encoder rerank) with company/year metadata filtering and inline `[N]`
  citations linking to the source chunk.
- **Exact numbers, not paraphrases** — numeric questions hit SEC **XBRL
  company-facts** first (the figure as filed), then a deterministic
  **calculator** for derived metrics (margins, ratios, growth, CAGR, working-
  capital days), then the pandas **table agent** as a fallback.
- **Corrective RAG** — each chunk is graded 1–5; weak ones are dropped and the
  query is rewritten + retried when retrieval is poor.
- **Self-expanding corpus** — a company missing from the index gets its 10-K
  fetched from EDGAR on the fly (in-memory on Cloud Run, ingested locally),
  walking back enough filings to cover the fiscal year asked about.
- **Cross-document search** — "which companies disclosed X" runs EDGAR
  full-text search across all filers.
- **Live market data + charts** — price / return / market-cap questions hit
  Yahoo Finance; history requests render an inline candlestick chart.
- **Web search fallback** — Tavily (trusted finance domains, recency windows),
  with automatic escalation when retrieval comes back empty or the draft
  admits it can't answer.
- **Anti-hallucination** — a claim-checking critic, a deterministic numeric
  verifier (every figure must trace to the evidence or derive from it), and a
  blended **confidence gate** that answers, caveats, or refuses.
- **Per-session memory** — multi-chat threads kept in the browser; the agent
  gets the last few turns for follow-ups and pronouns.
- **Bring-your-own-key** — Groq by default; switch to OpenAI / Anthropic /
  Gemini per-request from the model picker (keys stay in your browser).

---

## Agent architecture

The pipeline is a LangGraph `StateGraph`, built as an inheritance ladder where each
layer adds a capability ([`finagent/graph/`](finagent/graph/)):

| File | Class | Adds |
|---|---|---|
| `base.py` | `AgenticRAG` | planner → retrieve → synthesize → critic |
| `corrective.py` | `AgenticRAGv2` | hybrid retrieval, relevance grader, rewrite loop |
| `full.py` | `AgenticRAGv3` | fused **plan+route** call, table agent |
| `agent.py` | `AgenticRAGv4` | XBRL facts, calculator, dynamic SEC fetch, EDGAR FTS, market data, web search, numeric verifier, confidence gate |

`AgenticRAGv4` is the deployed agent. The runtime graph:

```
START → planner(+routes) → router ─┬─ retrieval path: fetch_filing → retrieve → grader → {rewrite ↺ | proceed}
                                   └─ tools path (no narrative sub-query): skip retrieval
      → xbrl → calculator → table_agent → market_data ∥ web_search ∥ edgar_search   (parallel fan-out)
      → evidence_builder → synthesize → critic → verify_numbers
      → confidence → {answer | answer + caveat | low-confidence caveat}
                   ↘ refuse (ungrounded figures)                         → END
```

**One planning call does everything up front.** The planner decomposes the
question into sub-queries *and* tags each as
`narrative | numeric | market | external | cross_document` in a single
structured-output call; each lane then self-selects, so a lane only does work
when the question needs it. Purely numeric/market questions skip retrieval
entirely.

**Numbers are verified deterministically.** Every figure in the draft must
match the evidence (scale-insensitively) or be derivable from it in one
arithmetic step; an LLM verifier runs only as a rescue when figures fail to
ground. Ungrounded figures re-route, then refuse.

Retrieval uses an on-disk **Chroma** store (`finagent/vectorstore.py`);
embeddings and the reranker run on GPU when available, else CPU
(`finagent/device.py`). The FastAPI layer (`finagent/api/`) streams the answer
over SSE — with live per-node progress events driving the UI's thinking trace —
and serves the React SPA.

### Models (Groq, free tier)

Each role runs the cheapest model that holds its quality bar; roles start on
different keys of the rotating pool so rate limits don't hit in lockstep.

| Role | Model | Why |
|---|---|---|
| plan+route, tool extraction (XBRL/calc/EDGAR/market/gate) | `openai/gpt-oss-120b` | tool selection and decomposition sit on the quality path; weak models mis-route |
| grader, rewriter | `qwen/qwen3.6-27b` | structured scoring; separate quota bucket from the 120B roles (replaced the deprecated `llama-3.3-70b`; reasoning is stripped server-side so `<think>` never leaks) |
| synthesizer, critic, verifier, table-agent codegen | `openai/gpt-oss-120b` | long-form writing, claim checking, pandas codegen |

```
finagent/
  graph/         the LangGraph agent (base → corrective → full → agent)
  tools/         XBRL, calculator, SEC fetch, EDGAR FTS, market, web search
  retrieval/     hybrid retriever (BM25 ∪ dense + rerank), filters, reranker
  api/           FastAPI: SSE streaming + agent service
  vectorstore.py · chroma_client.py · device.py · llm.py
  ingestion/     corpus builders (PDF parse, chunk, embed)
  evaluation/    FinanceBench + RAGAS harness (incl. parallel multi-key runner)
frontend/        React + TypeScript + Vite + Tailwind SPA
notebooks/       experimentation.ipynb (system reference notebook)
```

---

## Evaluation

The agent is measured end-to-end on **FinanceBench** (150 open-source
questions over US filings), scored with RAGAS plus system-behaviour metrics.
With 12 Groq keys the full set runs in parallel:

```bash
# answer all 150 questions across the key pool (resumable)
python -m finagent.evaluation.financebench.parallel --workers 3 \
    --output results/financebench_full_outputs.json

# RAGAS-score the outputs and write the final metrics report
python -m finagent.evaluation.financebench.parallel --score \
    --output results/financebench_full_outputs.json
```

This produces **one aggregate metrics list** — `results/final_metrics.json` /
`.md` — covering answer/refusal/error rates, mean confidence, RAGAS
(faithfulness, answer relevancy, context precision/recall) overall and per
question type (numeric / comparison / narrative).

The eval answers from the dedicated **`financebench_eval`** Chroma collection —
the benchmark's own filings, ingested through the production pipeline and covering
the historical years the questions ask about. (Production serves `us_filings`
plus on-demand SEC fetch; the eval pins the corpus so it measures retrieval, not
EDGAR availability.) See **Bottleneck analysis & fixes** in `STRUCTURE.md` for
how a corpus-wiring bug — the eval previously searched the recent-only
`us_filings` and refused ~⅓ of questions — was found and fixed.

### Results — v3 (150 questions; v1 baseline in parentheses)

**Correctness & behaviour**

| Metric | v3 | v1 baseline |
|---|---|---|
| Numeric accuracy (gold figure in answer, 1% tol, judge-free) | **74.6%** | 61.2% |
| Answer rate | 74.7% | 100% |
| Refusal rate (explicit "insufficient evidence" abstentions) | 25.3% | 0% |
| Error rate | 0% | 0% |

v1's 100% answer rate was an anti-feature: it never abstained, so a third of
its "answers" were confident fabrications. v3 trades those for explicit
abstentions — numeric accuracy (+13.4 pts) is the honest correctness signal.

**RAGAS scores** (judge: `qwen/qwen3.6-27b`)

| Question type | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| Numeric (71) | 0.63 | 0.53 | 0.60 | 0.64 |
| Comparison (22) | 0.53 | 0.44 | 0.29 | 0.41 |
| Narrative (57) | 0.51 | 0.36 | 0.38 | 0.29 |
| **Overall (150)** | **0.57** | **0.45** | **0.50** | **0.47** |

RAGAS scores non-answers as zeros by construction, so the abstentions cap the
overall numbers. On the **79 plainly-answered questions** the same judge scores
faithfulness **0.75**, relevancy **0.65**, precision **0.60**, recall **0.72** —
the gap between those two views is the refusal calibration, tracked as the
main open item. `results/comparison.md` has the full v1 → v3 delta table, and
`STRUCTURE.md` → *Bottleneck analysis & fixes* documents each root cause found
along the way (corpus wiring, historical-year fetch, refusal confidence,
mis-routing, web-escalation pollution — all fixed cost-neutrally).

Runs are resilient to the free tier: per-minute limits rotate across the key
pool, revoked keys are dropped from rotation, and when *every* key is
exhausted the run stops with a clear `LIMIT EXHAUSTED` message — re-running
the same command resumes where it stopped (the RAGAS scorer is also
resumable: already-scored rows are skipped on the next run). The UI surfaces
the same condition as "Limit exhausted" within seconds instead of hanging.

### Why not a "true" multi-agent system?

FinAgent already runs specialised LLM roles (planner-router, grader, synth,
critic, verifier, market planner, codegen) orchestrated by a LangGraph state
machine — the supervisor/specialist pattern without free-form agent chatter.
A conversational multi-agent layer was evaluated and rejected: it multiplies
LLM calls (latency + free-tier quota) for no measured quality gain, since the
failure modes here (retrieval misses, ungrounded figures) are addressed by
deterministic tools and verification, not by more agent dialogue.

---

## Run it locally

**Prerequisites:** Python ≥ 3.11, Node ≥ 20, and a free [Groq](https://console.groq.com) API key.

```bash
# 1. Clone
git clone https://github.com/<your-user>/FinAgent.git
cd FinAgent

# 2. Install the Python package (editable) + dev extras
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. Configure secrets
cp .env.example .env          # then set GROQ_API_KEY (TAVILY_API_KEY optional)

# 4. Start the backend (FastAPI, port 8000)
uvicorn finagent.api.main:app --reload --port 8000

# 5. Start the frontend (Vite, port 5173 — proxies /api to :8000)
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173** and ask a question.

**Vector store:** the prebuilt `data/chroma` is not committed (size). Build it from
source filings with the ingestion extra:

```bash
pip install -e ".[ingest]"
python -m finagent.ingestion.fetchPDFs
python -m finagent.ingestion.ingest        # parse + chunk + embed into data/chroma
python -m finagent.ingestion.table_ingest
```

Point `CHROMA_DIR` at an existing store if you already have one.

**Tests:** `pytest`

---

## Deployment

CI/CD deploys on every push to `main` (`.github/workflows/deploy.yml`): the
backend builds into a single Cloud Run image (SPA + API + baked Chroma store +
baked encoder models, scale-to-zero) and the frontend deploys to Firebase
Hosting. The image stays CPU-only and within the existing 8 GiB / 2 vCPU
service shape — none of the agent's lanes add resident memory beyond the
shared encoder models.

## Configuration

Set in `.env` (see `.env.example`):

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` (+ `…2`–`…8`) | Default LLM provider; roles stagger across keys, rotate on rate-limit, drop revoked keys, and fail fast with "limit exhausted" when the whole pool is spent |
| `TAVILY_API_KEY` | Web search (optional) |
| `CHROMA_DIR` | Chroma directory (default `data/chroma`) |
| `PERSIST_DYNAMIC_FETCH` | `false` on Cloud Run: fetched filings stay in-memory |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` | Optional; usually set in the UI |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | [Langfuse](https://langfuse.com) tracing (optional; open source) |
| `LANGCHAIN_*` | LangSmith tracing (optional) |

## Observability

With `LANGFUSE_*` keys set, every query is traced to
[Langfuse](https://langfuse.com) (open source): one `finagent-query` trace per
question with nested per-node spans, every LLM generation (model, prompt,
completion, token usage), retrieval inputs/outputs, and latency. Chat threads
map to Langfuse **sessions** (follow-up turns group together); traces are
tagged with the provider and collection, so production traffic and
FinanceBench eval runs are separable in the UI. Traces are flushed
synchronously before each response returns, so tracing survives Cloud Run's
CPU throttling and scale-to-zero. Without keys, tracing is a no-op.

The answer payload separately carries in-app metrics — token totals per model,
per-node latencies, tool-lane health, and the confidence/verification audit —
independent of any tracing backend.
