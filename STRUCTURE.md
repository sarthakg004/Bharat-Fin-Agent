# Module map

What lives where. The README covers what the system does and why; this file is
the index. Retrieval decisions and their measurements are in
`results/RETRIEVAL_EXPERIMENTS.md`.

## The served path

A request enters at `api/main.py`, gets a per-request `RuntimeContext`, and runs
one compiled LangGraph:

```
START → planner(+routes) → router ─┬─ narrative/default → fetch_filing → retrieve ─┐
                                   └─ numeric/market/external/cross_doc ───────────┤
                                                                                   ▼
                                                                                 xbrl
                                                                                   ↓
                                                                              calculator
                                                                                   ↓
                                 ┌───────── market_data ∥ web_search ∥ edgar_search
                                 ↓
                          evidence_builder ─┬─ (all lanes empty) → fetch_filing ↺
                                            └─ synthesize → critic ─┬─ END
                                                                    ├─ refuse → END
                                                                    ├─ synthesize ↺
                                                                    ├─ retrieve ↺
                                                                    └─ web_search ↺
```

Both branches converge on `xbrl`: the retrieval path adds structured figures to
the passages it found rather than choosing between them. The three network lanes
after `calculator` run in parallel and fan back into `evidence_builder`.

The critic is the only gate. It extracts each claim from the draft, checks it
against the evidence, and names the fix: a re-draft when the draft overstated
good evidence, a re-retrieve when the evidence is thin, a web escalation when
the lane went unused. One retry. A draft still mostly unsupported after that is
refused (`REFUSE_BELOW_SUPPORT`).

## `finagent/graph/` — the agent

`AgenticRAGv4` in `agent.py` is the only class ever instantiated. It is assembled
from four node mixins plus three base layers that exist only to be inherited;
their `super()` calls chain through in MRO order.

| File | What it contributes |
|---|---|
| `agent.py` | `AgenticRAGv4`: constructor, tool resources, routers, graph wiring |
| `nodes/fetch.py` | corpus gate, dynamic SEC fetch, the retrieval node |
| `nodes/numeric.py` | XBRL extraction, the deterministic calculator |
| `nodes/external.py` | market data, web search, EDGAR full-text |
| `nodes/synthesis.py` | evidence builder, drafting, critic, refusal |
| `full.py` | `AgenticRAGv3`: fused plan+route, the retrieval-query rewrite |
| `corrective.py` | `AgenticRAGv2`: hybrid retrieval, pool cap, critic retry |
| `base.py` | `AgenticRAG`: planner, critic, `run()` |
| `state.py` | `AgentState` plus every structured-output schema |

Only `AgenticRAGv4._build_graph` runs. Method resolution:
`planner_node` → v3 → base; `hybrid_retrieve_node` → fetch → v3 → v2;
`critic_node` → synthesis → v2 → base; `synthesize_node` → synthesis.

## The rest

| Path | Contents |
|---|---|
| `api/main.py` | FastAPI app, SSE streaming, upload, SPA serving |
| `api/rag_service.py` | per-collection agent singletons, `run_agentic` |
| `api/models.py` | request/response schemas |
| `retrieval/hybrid.py` | `HybridRetriever` — RRF-fused BM25 + dense, reranked |
| `retrieval/reranker.py` | Cohere adapter with a local cross-encoder fallback |
| `retrieval/filters.py` | company/year metadata vocabulary and filters |
| `retrieval/expansion.py` | ratio name → the line items a filing actually prints |
| `tools/` | XBRL, calculator, SEC fetch, EDGAR search, market data, web search, CIK resolver |
| `ingestion/ingest.py` | `CorpusIngester` — parse, chunk, embed, upsert |
| `ingestion/fetchPDFs.py` | EDGAR/CSV filing downloader, writes the manifest |
| `ingestion/upload.py` | user uploads → ephemeral in-memory chunks |
| `vectorstore.py` | Qdrant client, collections, Gemini embeddings over REST |
| `runtime.py` | `RuntimeContext`, per-provider model defaults, `ROUTER_PIN` |
| `llm.py` | provider clients, key rotation, rate-limit classification |
| `config.py` | `settings` |
| `prompts/` | planner and critic prompts (the synthesizer's live in `nodes/synthesis.py`) |
| `research/` | Deep Research: specialist registry + orchestrator |

## `finagent/evaluation/` — dev-time only

Never imported by the serve path.

| Path | Contents |
|---|---|
| `evaluate_retrieval.py` | the retrieval sweep: geometry × embedder × reranker |
| `financebench/` | the end-to-end harness — dataset, gold map, indexing, runner, sharded `parallel.py`, `answer_match.py` |
| `ragas.py` | `RAGASEvaluator`, six judge metrics, resumable |
| `custom.py` | a hand-written question set for relative-period failures |
| `research_eval.py` | judge-free structural metrics for Deep Research reports |

## Elsewhere

```
scripts/    overnight.sh (metered index build), run.sh, score_v6.sh, watch_v6.sh
tests/      pytest, no network — test_smoke.py is the bulk of it
results/    every measurement behind every retrieval claim
frontend/   React + TypeScript + Vite + Tailwind
```
