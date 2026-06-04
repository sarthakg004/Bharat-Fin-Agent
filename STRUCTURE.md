# FinAgent — Codebase Structure

This is the map a new contributor reads first. It explains what each top-level
module under `finagent/` contains and the **dependency direction** between them.

The codebase was reorganised into clean layers. To protect the deployed service,
the migration was done as a **structural move with backwards-compatibility
shims**: old import paths keep working, and the agent's tightly-coupled class
chain was *relocated/re-exported, not rewritten*. Two scoped deviations from a
"pure" layout are called out below.

## The Cloud Run contract (do not break)

The container entry point is:

```
uvicorn finagent.api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
```

So `finagent/api/main.py` must stay at that path and expose a module-level
`app`. The Dockerfile copies `finagent/` wholesale (`COPY finagent/ ./finagent/`),
so internal reorganisation is invisible to the build as long as that import
string resolves. `requirements.txt`, `pyproject.toml`, and `data/chroma/` paths
are also load-bearing and unchanged.

## Dependency direction

Strictly downward — a module may import from those above it, never below:

```
config.py        ← imported by everything; imports nothing from finagent
llm.py           ← config
corpus/          ← config, (low-level) chroma_client, vectorstore
ingestion/       ← config, corpus
retrieval/       ← config, corpus, llm
tools/           ← config, llm, retrieval (some tools)
prompts/         ← (ideally) nothing — see deviation note
agents/          ← config, llm, retrieval, tools, prompts, corpus
api/             ← config, agents
evaluation/      ← config, retrieval, agents
```

## Top-level modules

**`config.py`** — Centralised settings (`Settings` + a module-level `settings`).
All environment variables (`GROQ_API_KEY`, `STATELESS`, `ALLOWED_ORIGINS`,
`RERANKER_MODEL`, `CHROMA_DIR`, …) are read here and nowhere else; other modules
import `settings`. Built on `pydantic.BaseModel` (not `pydantic_settings`) to
avoid adding a dependency. Multi-key LLM rotation (`GROQ_API_KEY2`, …) is the one
deliberate exception — it stays dynamic in `llm.py`.

**`llm.py`** — Builds a chat model per provider (Groq/Gemini/OpenAI/Anthropic),
resolves keys, and provides `RotatingChatModel` (swaps keys on rate-limit). Kept
at this path; imports only config-level concerns.

**`corpus/`** — Read/inspect views over the Chroma vector store. `ChromaClient`
(typed handle: counts, stores, paged fetch), `stats` (corpus summary, per-company
counts), `inspect` (sample a stored chunk). Wraps the low-level `chroma_client.py`
/ `vectorstore.py` helpers, which remain the single source of "where Chroma lives".

**`ingestion/`** — The parse → chunk → embed → store pipeline. `parse_document`
(unstructured), `chunk_text` (production splitter config), `embed_text` (BGE),
and `pipeline.ingest_corpus` / `CorpusIngester` orchestrating them into Chroma.

**`retrieval/`** — The retrieval stack behind a `BaseRetriever` ABC:
`HybridRetriever` (BM25 ∪ dense + cross-encoder rerank — the one the agent uses),
standalone `BM25Retriever` / `DenseRetriever`, and `CrossEncoderReranker`.
`HybridRetriever` was moved here from `graph/corrective.py`; that path keeps a
re-export shim.

**`tools/`** — Agent tools behind a `BaseTool` ABC + `ToolRegistry`. Live today:
`market` (yfinance) and `web_search` (Tavily), moved here from `graph/` with
shims left behind. Stubs for roadmap tools — `resolver` (Phase 2), `xbrl`
(Phase 3), `calculator` (Phase 4), `sec_fetch` (Phase 5), `edgar_search`
(Phase 6) — are `BaseTool` subclasses whose `run()` raises `NotImplementedError`.

**`prompts/`** — System prompts grouped by role (`planner`, `synthesizer`,
`critic`). The canonical import surface for prompt strings.

**`agents/`** — The LangGraph agent. Canonical public namespace exposing
`AgentState`, `build_graph` / `build_agent`, `run_agent` + trace helpers, and the
agent class chain `AgenticRAG → AgenticRAGv2 → AgenticRAGv3 → AgenticRAGv4`
(the deployed agent). `AgentState` physically lives here (`agents/state.py`);
the rest of the chain is re-exported from `finagent.graph.*` (see deviation).

**`api/`** — FastAPI layer. `main.py` holds the `app` (Cloud Run entry point),
the `/api/query` SSE stream, chat endpoints, and static SPA hosting.
`rag_service.py` builds/caches the agent and normalises its output;
`history.py` / `models.py` are the chat store and Pydantic schemas. (Endpoint
handlers could later be extracted into an `api/routes/` package; not done yet.)

**`evaluation/`** — Dev-time only (never imported by the serve path). Canonical
modules: `dataset` (load + tag FinanceBench), `retrieval` (pool-recall / Hit@k /
MRR), `ragas_eval` (RAGAS + `RAGASEvaluator`), `ledger` (append/load/plot the
results ledger). `ragas.py` and `reranker_ab.py` remain as the underlying
implementations and `python -m` CLIs; `financebench/` is the harness the
wrappers delegate to.

## Backwards-compatibility shims (transitional)

These old paths still resolve via thin re-exports; remove them only after a
passing Docker build, once every consumer imports from the canonical path:

| Old path | Re-exports from |
|---|---|
| `graph/state.py` | `agents/state.py` |
| `graph/corrective.py` (`HybridRetriever`, `_get_shared_reranker`) | `retrieval/` |
| `graph/market_tools.py` | `tools/market.py` |
| `graph/web_search.py` | `tools/web_search.py` |

## Scoped deviations (deliberate, to protect production)

1. **The agent class chain was not decomposed into per-node function modules.**
   The nodes are bound methods on the `AgenticRAG*` classes; extracting them is a
   rewrite, not a move. `agents/` re-exports the intact chain from `graph/`; a
   future `agents/nodes/` package would hold that decomposition.

2. **`prompts/` re-exports prompt constants** from their `graph/*` definition
   sites rather than owning them. This means `prompts` currently imports from
   `graph` (against the ideal "prompts is a zero-import leaf"). Flipping the
   dependency — moving the string bodies into `prompts/` and importing them into
   the nodes — is a follow-up tied to deviation (1).
