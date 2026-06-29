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
wrappers delegate to. Inside `financebench/`: `parallel` (the orchestrator —
sharded multi-key run → merge → RAGAS → `final_metrics`), `answer_match`
(deterministic numeric-accuracy metric, judge-free), and `compare` (the
before/after metrics table). See **The evaluation & improvement loop** below.

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

## The agent pipeline (what actually runs per question)

The deployed agent is `AgenticRAGv4` (class chain `AgenticRAG → v2 → v3 → v4`,
built in `api/rag_service._build_agent`). Its LangGraph (`agent._build_graph`):

```
planner → router ─┬─ narrative/default ─→ fetch_filing → retrieve → grader ─┬─ rewrite ↺
                  │                                                          └─→ ┐
                  └─ purely numeric/market/external ───────────────────────────→ xbrl
xbrl → calculator → table_agent ─┬─ market_data ┐
                                 ├─ web_search   ├─→ evidence_builder → synthesize
                                 └─ edgar_search ┘                          │
synthesize → critic ─┬─ resynthesize ↺ (over-claimed, re-draft on same evidence)
                     ├─ websearch (insufficient draft → gather web, re-draft)
                     └─ verify_numbers ─┬─ retrieve ↺ (re-ground a figure)
                                        ├─ refuse  (figures mostly ungrounded)
                                        └─ confidence ─┬─ answer            → END
                                                       ├─ warn  → +caveat   → END
                                                       └─ low   → abstain OR → END
                                                                  +low-conf caveat
```

Key roles: **planner** decomposes into sub-queries (enumerating comparisons and
*superlative narrative* breakdowns); **router** tags each sub-query
narrative/numeric/market/external; **retrieve** is the hybrid stack (below);
**grader** scores 1-5 and drops off-entity chunks; the **xbrl → calculator →
table_agent** chain is the deterministic numeric path; **verify_numbers**
deterministically grounds every figure in the draft; the **confidence** gate
blends retrieval/verification/citation/critic sub-scores into one score and bands
it answer / warn / low.

## The retrieval stack (`retrieval/` + the agent's retrieve node)

Per sub-query, each `HybridRetriever` (one per filings collection) does:
BM25 (lexical) **∪** dense (MMR-diversified embeddings) → company/year metadata
filter inferred from the query → cross-encoder rerank → top `final_top_k`. The
agent's `hybrid_retrieve_node` (`graph/corrective.py`) merges every sub-query's
hits, drops non-English noise, then **`_cap_pool`** runs a *second, global*
cross-encoder rerank of the merged pool against the original question and keeps
the `retrieve_cap` best passages. This last step is what stops a 6-sub-query
question from carrying 25 noisy passages into synthesis — fewer, higher-precision
passages measurably raised RAGAS faithfulness.

Retrieval/synthesis knobs worth knowing (all on `AgenticRAGv4`, set in
`rag_service._build_agent`):

| knob | default | effect |
|---|---|---|
| `bm25_top_k` / `dense_top_k` | 12 / 12 | first-stage pool depth per retriever (wider ⇒ better narrative recall) |
| `final_top_k` | 5 | passages kept per sub-query after per-query rerank |
| `retrieve_cap` | 8 | **global** cap after merging all sub-queries' hits (`None` disables) |
| `analyst_voice` | True | analyst synth + critic prompts (incl. the "don't invent provenance" rule) |
| `strict_numeric` / `refuse_below_grounding` | True / 0.6 | hard-refuse only when most figures fail to ground |
| `confidence_answer` / `confidence_warn` | 0.80 / 0.60 | confidence bands |
| `abstain_on_insufficient` | True | promote a low-band *soft-refusal* draft to an explicit "Insufficient evidence" abstention |

The synthesizer (`SYNTH_ANALYST_SYSTEM` in `graph/agent.py`) enforces: cite by
`[N]` only; **never invent XBRL tags, filing dates, or "as filed" provenance**
the evidence doesn't state; and an **extractive mode** for numeric-only questions
(lead with the figure + unit + period + one `[N]`, nothing else).

## The evaluation & improvement loop (`evaluation/financebench/`)

Run the deployed agent over FinanceBench and score it, then iterate.

**Results are versioned.** Each run lives in its own `results/vN/` directory so a
rerun never clobbers an earlier one — every artifact (outputs, RAGAS csv, metrics,
shards, log) is derived from `--output`'s directory. Point `--output` at the next
version dir:

```
# 1. Answer all 150 questions with the production agent (sharded across keys,
#    resumable). Writes results/v2/financebench_full_outputs.json + metrics.
python -m finagent.evaluation.financebench.parallel \
    --output results/v2/financebench_full_outputs.json

# 2. RAGAS-score those outputs + assemble the final metrics (behaviour + RAGAS +
#    numeric accuracy). Writes results/v2/final_metrics.{json,md} + score.log.
python -m finagent.evaluation.financebench.parallel \
    --score --output results/v2/financebench_full_outputs.json

# 3. Deterministic numeric correctness on its own (judge-free, no quota):
python -m finagent.evaluation.financebench.answer_match \
    results/v2/financebench_full_outputs.json

# 4. Before/after table — frozen baseline vs the NEWEST results/vN/ run
#    (auto-detected). Writes results/comparison.md:
python -m finagent.evaluation.financebench.compare
```

### Results directory layout

```
results/
  financebench_gold.json        # shared input (FinanceBench gold answers)
  financebench_split.json       # shared input (question split + qtype tags)
  baseline_metrics.json         # frozen V1 metrics — the "before" comparison anchor
  comparison.md                 # generated before/after Δ table
  v1/                           # first full run (the current/baseline results)
    financebench_full_outputs.json
    financebench_full_outputs_ragas.csv
    final_metrics.{json,md}
  v2/                           # next rerun (created by --output results/v2/...)
    ...
```

`vN/shards/` and `vN/score.log` are gitignored per-run intermediates (the merged
output is the artifact; shards are auto-deleted after a clean merge).

**Metrics tracked** (in `results/final_metrics.md`):
- **RAGAS** — `faithfulness` (claims grounded in context), `answer_relevancy`,
  `context_precision`, `context_recall` — overall and per question type.
- **`numeric_accuracy`** (new, `answer_match.py`) — does the gold figure appear
  in the answer within 1% tolerance? Judge-free, scored over numeric questions.
  This is the **answer-is-RIGHT** signal RAGAS misses: a correct-but-verbose
  numeric answer scores high here even when faithfulness penalises its phrasing.
- **Behaviour** — answer / refusal / error rates, mean confidence. An explicit
  insufficient-evidence abstention counts as a refusal (healthier than a
  confident wrong answer).

**Comparison table.** `results/baseline_metrics.json` is the frozen "score before
these changes"; `compare.py` diffs it against the current `final_metrics.json`
into `results/comparison.md` with a Δ per metric. Re-run steps 1-2, then step 4,
to populate the *After changes* column.

Diagnostic decoder when iterating: **low faithfulness + high numeric_accuracy** =
the answer is right but over-claims (a prompt/phrasing fix — see the
"don't invent provenance" rule); **low numeric_accuracy** = the answer is wrong
(a retrieval or computation fix); **low `context_recall`** = first-stage
retrieval missed the gold passage (widen the pool / fix decomposition).

## Bottleneck analysis & fixes (2026-06)

A full v2 run improved `numeric_accuracy` (0.61 → 0.73) but RAGAS `faithfulness`
and `answer_relevancy` stayed flat (~0.59 / ~0.42). Drilling into the per-row
scores showed the cause was **~49 "no information" non-answers** (67 rows scored
0 on `answer_relevancy`, 86 scored 0 on `context_recall`). Those weren't a
prompt problem — the agent had no usable evidence. Root causes and fixes:

1. **Wrong eval corpus (the big one).** The eval ran the production agent
   against `us_filings` (recent FY2022-2026 only, partial company set), while
   the purpose-built `financebench_eval` collection (all 32 companies at the
   asked-about historical years, ingested through the *same* production
   pipeline) sat unused. Half the questions found no filing and fell through to
   web-search noise. **Fix:** the eval runner now targets
   `settings.financebench_collection`; `run_agentic`/`get_agentic`/`_build_agent`
   take a `collection` override. Production still serves `us_filings`. The eval
   also sets `DISABLE_DYNAMIC_FETCH=1` (the eval corpus is complete, so live
   EDGAR fetch is wasted work and the gate would misfire on financebench_eval's
   name-as-ticker metadata).

2. **Dynamic SEC fetch was broken for historical years.** On the cloud/ephemeral
   path (`persist_fetch=False`) an `already_indexed` company was never
   year-deepened, so "MSFT FY2016" found only the recent filings in the index
   and refused. The walk-back depth was also capped at 5 filings, making any
   year >5 back unreachable. **Fix:** year-deepening now runs on the ephemeral
   path too (pulls older filings in-memory as `fetched_chunks`), and the cap is
   raised to 12 (`graph/agent.fetch_filing_node`).

3. **Refusals scored as maximally confident.** A soft-refusal draft ("No
   relevant evidence provided…") had no figures to verify and no claims for the
   critic to refute, so the critic passed vacuously and the blend landed at
   `confidence=1.0`, status `answered` — a content-free non-answer counted as a
   confident answer (this is why v2's mean confidence jumped to 0.85). **Fix:**
   `_confidence_components` returns `{}` (→ `conf=0`, → clean abstention) when
   the draft matches `_SOFT_REFUSAL_RE`, and that regex was widened to catch the
   "no … evidence/data provided" phrasings it was missing.

4. **Mis-routing wasted the tools.** The planner tagged qualitative/judgement
   questions as `numeric` (inventing a metric like "retention rate" that no
   filing reports) and was willing to route a named in-corpus company to the
   web. **Fix:** the `PLAN_ROUTE_SYSTEM` prompt now routes purely-qualitative
   questions to `narrative`, ratio/figure judgements ("healthy quick ratio?") to
   `numeric` (so the calculator computes the ratio from XBRL), and never sends a
   named, in-corpus company to `external`. Same number of LLM calls — zero added
   cost.

5. **Web escalation buried the filing.** A low grade on *on-company* chunks
   (a hard narrative question whose answer is in the MD&A prose, not a chunk
   that restates the question) triggered a web search whose generic IR/marketing
   pages then dominated synthesis and tanked faithfulness. **Fix:** the retrieve
   node records `company_in_corpus` (from the no-LLM metadata-vocab filter); the
   grader keeps the best filing chunks instead of dropping them, and web
   escalation is skipped when the company is in-corpus and chunks are in hand.
   This also *removes* a Tavily call per affected question — a cost reduction.

6. **One small model removed.** Table-title extraction used
   `llama-3.1-8b-instant`, which mislabelled financial tables. Bumped to
   `llama-3.3-70b-versatile`; no sub-70B model is used anywhere now. (On Groq's
   free tier every model is $0, so model choice is cost-neutral for the cloud
   bill, which is Cloud Run compute.)

7. **Ratio questions computed against the wrong (year-end) denominator.** The
   turnover/return ratios are flow ÷ stock, so convention — and FinanceBench's
   gold ("average PP&E between FY2018 and FY2019") — AVERAGES the balance-sheet
   denominator over (t-1, t); the calculator used the single year-end value, so
   fixed-asset turnover came out 25.65 vs gold 24.26 (and a wrong 0.73 in v2
   before the corpus fix). **Fix:** `AVG_DENOMINATOR_RATIOS` in
   `tools/calculator.py` averages the denominator for fixed-asset / inventory /
   asset turnover and ROA / ROE; the numerator stays single-period. Also added a
   finished-goods-only inventory fallback in `tools/xbrl.py` for brands that
   outsource manufacturing (Nike reports `InventoryFinishedGoodsNetOfReserves`
   and no `InventoryNet`). Now exact: ATVI 24.26, CVS 17.98, Nike 3.46.

**Cost.** Every fix is cost-neutral or cost-reducing: the eval reads local
Chroma (no EDGAR load), fewer questions escalate to Tavily, and Groq free-tier
model choice doesn't change the cloud bill. Nothing here adds an LLM call.

**Validation.** Spot-checked end-to-end on previously-failing questions (all
refusals or badly wrong in v2): Microsoft FY2016 COGS, 3M FY2018 capex, Adobe
FY2017 operating cash flow ratio (0.83), Activision/CVS fixed-asset turnover
(24.26 / 17.98) and Nike inventory turnover (3.46) all now answer **correctly**;
American Express card-member retention answers from the filing; soft-refusals
score `conf=0`. A full 150-question RAGAS re-run is the remaining step — run
steps 1-2 then 4 above once Groq daily quota is comfortable. Known long-tail
gap: a few ratios still miss on concept *attribution* rather than formula (e.g.
AES ROA uses total `NetIncomeLoss`; gold uses net income attributable to the
parent) — a concept-mapping refinement for later.
