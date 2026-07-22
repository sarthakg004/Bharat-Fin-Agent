# Retrieval experiments — July 2026

Everything here was measured on the FinanceBench eval set (150 questions, 84
10-K filings). Numbers that were *not* measured are marked as such. Where an
earlier conclusion was overturned by a later experiment, the retraction is kept
rather than edited out — the sequence is the point.

Raw data: `results/*.json`. The generated metrics report is
`results/retrieval_report.md`.

---

## 0. Summary — what changed and why

| Change | Effect | Needs reindex |
|---|---|---|
| `use_mmr=False` (hybrid RRF actually runs) | +0.067 hit@8, ~40% faster | no |
| Reranker `bge-reranker-base` → `bge-reranker-v2-m3` | **+0.147 hit@8** | no |
| `pool_top_k` stays 48 | — (a proposal to lower it was withdrawn) | no |
| Embedder `bge-small` → `bge-large-en-v1.5` | +0.042 cov@8 | **yes** |
| Parent window 1500 → 2500 chars | +0.070 cov@8 | **yes** |
| Child window stays 600 | — (no measurable effect) | — |

Shipped now: the first three. The last two are a migration, gated on the
reindex — see §7.

---

## 1. The bug that invalidated the premise: hybrid search was never running

`HybridRetriever.__init__` had `use_mmr: bool = True` while the comment beside
it said "off by default", and nothing in the codebase ever passed the flag.
LangChain's `max_marginal_relevance_search` is **dense-only** — it issues a
`NearestQuery` with `using="dense"` and no prefetch. Traced live:

```
production   (use_mmr=True)    using='dense'  prefetch=0  NearestQuery
measurement  (use_mmr=False)   using=None     prefetch=2  FusionQuery
```

So BM25 sparse vectors were written for all 105,833 points at ingest and **never
read at query time**. The `modifier=IDF` config, server-side lexical scoring and
RRF fusion were all inert.

It was invisible because retrieval still returned plausible chunks, and because
the eval harness called `store.similarity_search` directly — the fused path —
so **the benchmark measured a code path production did not run**. Every recall
number reported before this point attributed gains to fusion that fusion had
never performed.

Locked by `tests/test_hybrid_is_hybrid.py`, which asserts on the query that
actually reaches Qdrant.

---

## 2. Pool depth × retrieval mode (`results/funnel_sweep.json`)

150 questions, parent-identity matching, `bge-reranker-base`.

| depth | MMR pool → final@5 / @8 | Hybrid pool → final@5 / @8 |
|---|---|---|
| 24 | 0.4400 → 0.3000 / 0.3533 | **0.5200 → 0.3267 / 0.4200** |
| 48 | 0.5333 → 0.2933 / 0.3600 | 0.6800 → 0.3267 / 0.4000 |
| 96 | 0.5733 → 0.2867 / 0.3200 | 0.7867 → 0.3067 / 0.3800 |
| 192 | 0.7533 → 0.2933 / 0.3200 | 0.8667 → 0.2933 / 0.3667 |
| 384 | 0.8533 → 0.3000 / 0.3400 | 0.9000 → 0.2800 / 0.3467 |

Hybrid beats MMR at every depth **and** is ~40% faster (95.9 s vs 161.9 s at
depth 24 — MMR fetches 2× candidates to diversify).

### A conclusion drawn here that was wrong

With `base`, pool recall climbs 0.52 → 0.90 while final@8 *falls* 0.4200 →
0.3467. Retention collapses from 81% to 39%. The conclusion drawn was "pool
depth is a dead lever; drop `pool_top_k` 48 → 24."

**That was an artifact of the weak reranker.** See §3.

---

## 3. Reranker A/B (`results/reranker_ab.json`, `reranker_depth_m3.json`)

Identical pools — `pool_recall` matches exactly at each depth, so this isolates
ranking quality.

| reranker | depth | pool | final@5 | final@8 | retention@8 |
|---|---|---|---|---|---|
| base | 24 | 0.5267 | 0.3267 | 0.4267 | 0.8101 |
| base | 48 | 0.6800 | 0.3267 | 0.4000 | 0.5882 |
| v2-m3 | 24 | 0.5267 | 0.4467 | 0.4733 | 0.8987 |
| **v2-m3** | **48** | 0.6800 | **0.4867** | **0.5467** | 0.8039 |
| v2-m3 | 96 | 0.7867 | 0.5067 | 0.5600 | 0.7119 |
| v2-m3 | 192 | 0.8667 | 0.5267 | 0.5800 | 0.6692 |

Depth 24 → 48 moves `base` **−0.027** and v2-m3 **+0.073**. The two changes are
multiplicative: neither pays off alone, and testing them independently produced
the wrong answer in §2. `pool_top_k` stays at **48** — the value it already had.

### Why depth stops at 48

Marginal return per doubling, against CPU cost on Cloud Run's 2 vCPU
(v2-m3 measured at 1,414 ms/pair, `results/reranker_bench.json`):

| depth | parents after collapse | final@8 | rerank/question, 2 vCPU |
|---|---|---|---|
| 24 | 21 | 0.4733 | 89 s |
| **48** | 41 | **0.5467** | **174 s** |
| 96 | 77 | 0.5600 | 327 s |
| 192 | 140 | 0.5800 | 594 s |

Depth 48 captures 82% of the total available gain for 29% of the compute of
depth 192. Beyond it, quality flattens while cost stays linear.

### Cost of v2-m3

| | base | v2-m3 |
|---|---|---|
| params | 278 M | 568 M |
| ms/pair (2 vCPU, fp32) | 410 | 1,414 |
| RSS | +437 MB | +1,103 MB |
| context window | 512 tok | 8,192 tok |

The window matters for §5: `_collapse_to_parents` runs *before* `_rerank`, so
the cross-encoder scores the **parent**. At 512 tokens base truncates 69.6% of
2,500-char parents — it cannot see what a larger window buys. v2-m3 is
therefore a **prerequisite** for the parent-size change, not an independent win.

---

## 4. Chunk containment (`results/shipped_containment.json`)

`chunk_ablation.md` reported `parent_doc = 0.8007`, but measured a **2500/300**
window while `CorpusIngester.PARENT_CHUNK_SIZE` ships **1500/200**. Re-run at
the shipped sizes, same 12 docs / same seed / same macro-average:

| strategy | containment | chunks/doc |
|---|---|---|
| child 600/100 — *what we embed* | 0.0521 | 1,314 |
| fixed 1000/200 — old flat default | 0.4653 | 834 |
| **parent 1500/200 — shipped** | **0.7243** | 545 |
| parent 2500/300 — what 0.8007 measured | 0.8007 | 329 |

This is the clearest statement of why parent-document retrieval exists: the unit
we **embed** contains the whole answer 5% of the time; the unit we **return**
contains it 72% of the time. A flat 1000-char chunk doing both jobs got 47%.

Containment is a ceiling: at 1500/200, 28% of gold evidence is split across two
parents and cannot be returned whole by any retriever.

---

## 5. Parent unit — is a paragraph better? (`results/parent_strategies.json`)

| parent unit | containment | mean chars | % over base's 512-tok window | ctx @ cap 8 |
|---|---|---|---|---|
| parent 1500/200 (shipped) | 0.7243 | 1,086 | 0% | 8,687 |
| parent 2500/300 | 0.8007 | 1,785 | 69.6% | 14,283 |
| parent 4000/400 | 0.9417 | 3,097 | 86.5% | 24,774 |
| paragraph (raw) | 0.2944 | 173 | 7.2% | 1,383 |
| paragraph merged→1500 | 0.5819 | 1,614 | 20.2% | 12,909 |
| paragraph merged→2500 | 0.4965 | 2,278 | 73.5% | 18,222 |
| page (whole page) | 1.0000 | 3,442 | 79.9% | 27,540 |

**Structure-aware chunking is worse than a fixed window here**, and the reason
is instructive: merged paragraphs have *no overlap*, so evidence straddling a
boundary is lost permanently, while `RecursiveCharacterTextSplitter` at 1500/200
slides a 200-char window over every boundary and catches it in the neighbour.
**The overlap is doing the work, not the structure.** Merged→2500 scoring below
merged→1500 confirms it — bigger no-overlap blocks mean fewer but more damaging
boundaries.

Raw paragraphs fail for a second reason: pypdf emits line-broken text, so a
"paragraph" in a 10-K averages 173 characters — a table row or heading, not a
prose block.

`page = 1.0000` is **a leak, not a result**: FinanceBench evidence spans were
extracted from single pages, so a page-level chunk contains its own gold by
construction. Its real cost is 79.9% reranker overflow and ~6.9k tokens at cap 8.

The HTML ingest path already *is* structure-aware (`chunk_by_title`, tables kept
intact) and should stay that way — HTML has real structure. PDFs arrive as a
character stream with the structure already destroyed.

---

## 6. Geometry grid (`results/geometry_study_*.json`)

Ground truth here is **not** the gold-chunk map: that map picks "the chunk most
similar to the evidence" by dense similarity, which changes with both the chunk
geometry and the embedder under test. Scoring instead against FinanceBench's
verified evidence span:

```
coverage@k = fraction of the evidence span's content lines (≥25 chars)
             present in the union of the top-k parents given to the LLM
hit@k      = coverage >= 0.5
```

### 6a. 9-cell grid — 20 docs, 74 questions (`geometry_study_geometry.json`)

| child | parent | cov@8 | hit@8 |
|---|---|---|---|
| 600 | 4000 | 0.6111 | 0.6351 |
| 900 | 4000 | 0.6082 | 0.6216 |
| 300 | 4000 | 0.5909 | 0.5946 |
| 600 | 2500 | 0.5888 | 0.6081 |
| 900 | 2500 | 0.5813 | 0.5811 |
| 300 | 2500 | 0.5767 | 0.5811 |
| 900 | 1500 | 0.4926 | 0.5135 |
| 600 | 1500 | 0.4862 | 0.5000 |
| 300 | 1500 | 0.4727 | 0.5000 |

Perfectly stratified by parent size; child size spread is ≤0.02 with rank flips,
i.e. inside noise.

**Caveat on this subset, recorded because it was not caught before quoting it:**
selecting the top-20 docs by question count over-weighted narrative questions
(51.4% vs 38.0% corpus-wide) and covered only 15 of 32 companies. The finalists
were therefore re-run on all 150.

### 6b. Finalists — 84 docs, all 150 questions (`geometry_study_final.json`, `geometry_study_large.json`)

| child | parent | embedder | cov@8 | hit@8 | cov@5 | hit@5 |
|---|---|---|---|---|---|---|
| 600 | 1500 | bge-small | 0.5223 | 0.5503 | 0.4613 | 0.4765 |
| 600 | 2500 | bge-small | 0.5927 | 0.6040 | 0.5281 | 0.5101 |
| 600 | 4000 | bge-small | 0.6071 | 0.6174 | 0.5293 | 0.5369 |
| **600** | **2500** | **bge-large** | **0.6347** | **0.6577** | **0.5678** | **0.5772** |

Parent 1500 → 2500 is **+0.070**; 2500 → 4000 only **+0.014**. The knee is at
2500, and 4000 additionally costs +266 MB of payload and ~6.2k prompt tokens
against Groq's per-request cap. **Parent = 2500.**

bge-large gains **+0.042 cov@8 / +0.054 hit@8** over bge-small at the same
geometry (threshold set beforehand: +0.03). Largest gain is **hit@5 +0.067** —
the metric that matters most, since `final_top_k=5` is each sub-query's
contribution before `_cap_pool` trims to 8.

> These coverage numbers are an **optimistic ceiling**: the harness indexes one
> filing per collection and searches only that filing, so cross-company
> distractors cannot occur. `results/retrieval_report.md` measures the live
> cluster with all 84 filings present.

---

## 7. Storage — why the eval collection moves off the managed cluster

Qdrant free tier is 1 GB. At 176,960 points:

```
                                vectors  + parent_text  =  total
current  (p1500, bge-small)      272 MB      265 MB        537 MB
target   (p2500, bge-small)      272 MB      442 MB        714 MB
target   (p2500, bge-LARGE)      725 MB      442 MB      1,167 MB   ✗ over
```

`parent_text` is stored on **every child**, so it is duplicated ~(parent/child)
times — enlarging parents grows payload super-linearly. That, not the vectors,
is what the parent-size change costs.

`financebench_eval` (105,833 points, 60% of the total) is referenced only by
`finagent/evaluation/**`. Production's `_build_agent` defaults to `us_filings`;
no served path touches it. Moving it to a local Qdrant container leaves:

```
us_filings only (71,127 pts, p2500 + bge-large):
  vectors 291 MB + parent_text 178 MB ≈ 469 MB   ✓ fits, ~50% headroom
```

Headroom matters because `PERSIST_DYNAMIC_FETCH=true` grows `us_filings` at
runtime.

Use a local **server** (`docker run -p 6333:6333 qdrant/qdrant`), not embedded
`path=` mode: embedded takes an exclusive file lock (the parallel eval runner
spawns worker processes) and silently ignores payload indexes, degrading every
company/year filter to a linear scan. Wired as `QDRANT_EVAL_URL`.

---

## 8. Rejected

- **MMR** — worse than RRF fusion at every depth, in two independent harnesses,
  and slower. Kept as an opt-in flag, off by default.
- **`pool_top_k` 48 → 24** — proposed under `base`, withdrawn once v2-m3 showed
  deeper pools pay.
- **Parent 4000 / page-level** — +0.014 over 2500 for +266 MB and ~6.2k tokens.
- **Structure-aware (paragraph) parents for PDFs** — measurably worse; the
  overlap in a fixed window is what preserves boundary-straddling evidence.
- **int8 dynamic quantisation of v2-m3** — benchmark was started to make fp32
  latency viable, then abandoned when latency was accepted as a non-issue for a
  portfolio deployment. Not measured; do not cite.
- **`--max-instances 3` for more RAM** — memory is per-instance and does not
  pool; it would also break `_UPLOADS`, which lives in one instance's memory.

---

## 9. Reproducing

```bash
conda activate finagent

# metrics + failure analysis on the live cluster
python scripts/retrieval_report.py

# chunk geometry sweep (local in-memory Qdrant; no cluster writes)
python scripts/geometry_study.py --docs 84 --stage final
python scripts/geometry_study.py --docs 84 --stage large --child 600 --parent 2500

# reranker / pool-depth A/B against the cluster
M3_ONLY=1 DEPTHS=24,48,96 python scripts/reranker_ab.py
```
