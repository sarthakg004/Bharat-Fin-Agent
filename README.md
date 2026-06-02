# FinAgent · Agentic RAG over financial filings

FinAgent answers questions about **SEC 10-Ks (US)** and **Indian annual reports**.
A LangGraph agent plans the query, retrieves from the filings, runs a numeric
**table agent**, pulls **live market data** (Yahoo Finance) and **web results**
(Tavily) when needed, answers in **English or Hindi** with inline `[N]` citations,
and **verifies every number against the evidence** — refusing rather than
fabricating when it can't ground a claim. A React UI streams the answer with the
source chunks shown alongside.

---

## Features

- **Grounded answers over filings** — hybrid retrieval (BM25 + dense `bge-small` +
  cross-encoder rerank) with inline `[N]` citations linking to the source chunk.
- **Corrective RAG** — each chunk is graded 1–5; weak ones are dropped and the
  query is rewritten + retried when retrieval is poor.
- **Numeric table agent** — writes and runs sandboxed pandas over extracted tables
  instead of guessing numbers from prose.
- **Live market data + charts** — price / return / market-cap questions hit Yahoo
  Finance; history requests render an inline candlestick chart.
- **Web search fallback** — Tavily (trusted finance domains, recency windows) for
  things the corpus doesn't cover.
- **Bilingual** — Hindi questions are detected, translated in, answered, translated out.
- **Anti-hallucination** — a critic plus a numeric verifier gate the answer.
- **Per-session memory** — multi-chat threads kept in the browser for the session;
  the agent gets the last few turns for follow-ups and pronouns.
- **Bring-your-own-key** — Groq by default; switch to OpenAI / Anthropic / Gemini
  per-request from the model picker (keys stay in your browser).

---

## Agent architecture

The pipeline is a LangGraph `StateGraph`, built as an inheritance ladder where each
layer adds a capability ([`finagent/graph/`](finagent/graph/)):

| File | Class | Adds |
|---|---|---|
| `base.py` | `AgenticRAG` | planner → retrieve → synthesize → critic |
| `corrective.py` | `AgenticRAGv2` | hybrid retrieval, relevance grader, rewrite loop |
| `full.py` | `AgenticRAGv3` | sub-query **router**, table agent |
| `agent.py` | `AgenticRAGv4` | market data, web search, bilingual, numeric verify, memory |

`AgenticRAGv4` is the deployed agent. The runtime graph:

```
START → detect_lang → translate_in → planner → router → retrieve → grader
      → { rewrite → retrieve | table_agent }
      → table_agent → market_data → web_search → synthesize → critic
      → verify_numbers → { retrieve | refuse | translate_out } → END
```

**The router classifies, it doesn't branch.** It tags each sub-query as
`narrative | numeric | market | external` in shared state; each lane then
self-selects (retrieve = narrative, table_agent = numeric, market_data = market,
web_search = external), so a lane only does work when the question needs it.

Retrieval uses an on-disk **Chroma** store (`finagent/vectorstore.py`); embeddings
and the reranker run on GPU when available, else CPU (`finagent/device.py`). The
FastAPI layer (`finagent/api/`) streams the answer over SSE and serves the React SPA.

```
finagent/
  graph/         the LangGraph agent (base → corrective → full → agent + tools)
  api/           FastAPI: SSE streaming + agent service
  vectorstore.py · chroma_client.py · device.py · llm.py
  ingestion/     corpus builders (PDF parse, SEC/NSE/BSE)
  evaluation/    RAGAS harness
frontend/        React + TypeScript + Vite + Tailwind SPA
notebooks/       experimentation.ipynb (end-to-end build notebook)
```

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

## Configuration

Set in `.env` (see `.env.example`):

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` (+ `…2`–`…8`) | Default LLM provider; extra keys rotate on rate-limit |
| `TAVILY_API_KEY` | Web search (optional) |
| `CHROMA_DIR` | Chroma directory (default `data/chroma`) |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` | Optional; usually set in the UI |
| `LANGCHAIN_*` | LangSmith tracing (optional) |
