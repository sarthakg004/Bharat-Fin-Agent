# Retrieval experiments — July 2026

Everything here was measured on the FinanceBench eval set (150 questions, 84
10-K filings). Numbers that were *not* measured are marked as such. Where an
earlier conclusion was overturned by a later experiment, the retraction is kept
rather than edited out — the sequence is the point.

Raw data: `results/*.json`. The generated metrics report is
`results/retrieval_report.md`.

---

## Start here — the whole thing in plain English

**What is being measured.** For every question, the agent hands the LLM 8
passages from the filings. FinanceBench tells us the exact passage that answers
each question. We check whether that evidence is actually among the 8. If it
isn't, the LLM cannot possibly answer correctly — so this number is the ceiling
on everything downstream.

**Where it went: 37 of 99 → 67 of 99.**

**The five things that mattered, in plain terms:**

1. **The chunks didn't know what they were** (§12, the big one, +17 questions).
   A balance sheet was stored as a bare grid of numbers. The words "Apple",
   "2022" and "Consolidated Balance Sheet" sit in the *heading*, and the
   chunker puts the heading in a different chunk from the table. So every
   company's balance sheet looked identical to the search engine. Fix: paste a
   one-line label onto every chunk.

2. **We were ranking the final 8 the wrong way** (§13, +3). We had switched to
   ranking passages by how well they matched the narrow sub-query that found
   them. Switching back to scoring them against the user's actual question is
   better — but *only* after fix 1. See "the lesson" below.

3. **We searched for words the filings never use** (§13, +2). The planner asks
   for "quick ratio". No filing contains that phrase; it says "Total current
   assets" and "Total current liabilities". So we now add those real words to
   the search.

4. **Nobody searched for the user's actual question** (§13, +3). The planner
   splits a question into sub-queries and we only searched those. The original
   question was thrown away. Now it gets searched too.

5. **A fix in step 1 quietly broke a de-duplicator** (§13, +3). The code that
   removes duplicate passages compared the first 80 characters of the text. Once
   every chunk began with the same label from fix 1, different parts of one
   statement looked identical and were thrown away as duplicates. Comparing
   identities instead of text fixed it. This only showed up because the shipped
   code was measured against the experiment harness and they disagreed.

**The lesson worth remembering, because it caught us twice.** A retrieval
experiment is only valid for the chunks it ran on. Fix 1 changed what a chunk
contains — and that flipped *two* earlier conclusions we had already measured
and shipped: which embedding model is best (§12d) and how to rank the final 8
(§13a). Both had been correct when measured. Neither survived the chunk change.
**After any change to chunking, re-run the ranking experiments; do not inherit
them.**

**What did NOT work** (all measured, all rejected — §13c): pulling in
neighbouring chunks, making the candidate pool 8× deeper, and dropping BM25 for
pure semantic search.

**Honest limits.** 67/99 is 68%, not the 90% a good system should reach. About
8 more questions are reachable by better ranking; ~25 need better retrieval or
a better PDF/HTML parser. There is no measured path to 90% on this corpus.

Sections 1-9 are the earlier work (a bug, geometry, rerankers, storage).
§10-13 are the July deep-dive. Read §0 for the table, §12 and §13 for the
reasoning.

---

## 0. Summary — what changed and why

**Where it ended up: the evidence reaches the synthesizer for 67 of 99
questions, up from 37 when the deep-dive started.** The headline steps:

| Step | hit@8 | Section |
|---|---|---|
| starting point | 37/99 | §10 |
| rank the cap on sub-query scores | 41/99 | §11 |
| **give every chunk a context header** | **57/99** | **§12** |
| revert §11 (it reversed once chunks had headers) | 61/99 | §13 |
| expand derived metrics to line items | 63/99 | §13 |
| also retrieve on the user's original question | 66/99 | §13 |
| fix a dedupe key §12 had silently broken | **67/99** | §13 |

Individual knobs:

| Change | Effect | Needs reindex |
|---|---|---|
| `use_mmr=False` (hybrid RRF actually runs) | +0.067 hit@8, ~40% faster | no |
| Reranker `bge-reranker-base` → `bge-reranker-v2-m3` | **+0.147 hit@8** | no |
| `pool_top_k` stays 48 | — (a proposal to lower it was withdrawn) | no |
| Embedder `bge-small` → `bge-large-en-v1.5` | +0.042 cov@8 | **yes** |
| Parent window 1500 → 2500 chars | +0.070 cov@8 | **yes** |
| Child window stays 600 | — (no measurable effect) | — |
| **Chunk context headers** | **+17 questions** | **yes** |
| Cap re-scores against the question | +3 questions | no |
| Derived-metric query expansion | +2 questions | no |
| Retrieve on the original question (2 slots) | +3 questions | no |
| Dedupe on `parent_id`, not a header-prefixed text slice | +3 questions | no |

**The single most important lesson, learned twice:** a retrieval decision is
only valid for the chunk representation it was measured on. §12 changed what a
chunk contains, and that reversed both the embedder verdict (§12d) and the cap
ranking rule (§13a). Any experiment predating a chunking change has to be
re-run, not inherited.

Everything through §11 is shipped. §12 needs a re-index; §13 does not.

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

# reranker / pool-depth A/B against the cluster
M3_ONLY=1 DEPTHS=24,48,96 python scripts/reranker_ab.py
```

`scripts/geometry_study.py` (§4-6) was deleted with the PDF corpus; §10 replaces
it with `finagent/evaluation/evaluate_retrieval.py`.

---

## 10. Re-measured on the SERVED HTML pipeline — July 2026

Everything above §9 was measured on the **PDF** pipeline. Production now serves
SEC HTML, so those numbers describe a pipeline nobody runs. Three independent
reasons they cannot be carried over:

1. **Different chunker, not just a different parser.** The HTML path groups
   `partition_html` elements with `chunk_by_title(max_characters=…, overlap=…)`
   and keeps tables whole; the PDF path slid a `RecursiveCharacterTextSplitter`
   over pypdf page text. "Parent 2500" means a different thing in each.
2. **The old harness indexed one filing per collection**, so no cross-company
   distractor could occur — exactly what separates a good reranker from a weak
   one. Measured here: on 3 filings *no reranker at all* beat both rerankers; at
   72 filings both beat it by ~0.05 hit@8.
3. **The gold-chunk map cannot compare configurations.** It picks "the chunk most
   similar to the evidence" by dense similarity, so it moves with both the
   chunker and the embedder under test. §10 scores against FinanceBench's
   verified evidence span instead (10-word shingle recall over the returned
   parents, alphanumeric-normalised so unstructured's `a | b | c` table
   rendering isn't scored as missing content).

Harness: `python -m finagent.evaluation.evaluate_retrieval --stage {parent,child,embedder}`.
It runs the **real** `CorpusIngester` and `HybridRetriever` over all 72 filings
in ONE local-Qdrant collection, and reproduces the served retrieval path
end to end: planner decomposition → per-sub-query retrieval (`narrative`/`numeric`
routes only) → merge/dedupe → `_cap_pool` rerank vs the ORIGINAL question with
the per-sub-query floor. Sub-query plans are generated once and cached
(`results/financebench_subqueries.json`) so every configuration is compared on an
identical decomposition. Full table: `results/html_retrieval_sweep.md`.

**Question set: 99, not 150.** 10-K/10-Q only (127), minus 28 whose evidence
`partition_html` never recovers from the primary filing — mostly table layout.
That 22% is a **parser ceiling no retriever can beat**; counting it would
penalise every configuration by the same constant and compress the deltas.
**n = 99, so one question = 0.0101. Treat anything under ~0.02 as noise.**

### 10a. Parent window — the knee is still 2500 (child 600, bge-large, v2-m3)

| parent | pool_recall | cov@8 | hit@8 | hit@5 | retention | ctx@8 |
|---|---|---|---|---|---|---|
| 1500 | 0.6667 | 0.3155 | 0.3030 | 0.2626 | 0.455 | 7,378 |
| **2500** | 0.7071 | 0.3785 | 0.3737 | **0.3434** | **0.529** | 11,177 |
| 4000 | 0.7576 | 0.3787 | 0.3838 | 0.3232 | 0.507 | 16,359 |

Pool recall keeps climbing but delivered quality does not: 2500 → 4000 buys
**+0.0002 cov@8** and one question of hit@8 while *losing* two questions of hit@5
and 0.022 retention, for **+46% context**. `final_top_k=5` is each sub-query's
actual contribution, so hit@5 is the metric that matters — and 2500 wins it.

### 10b. Child window — 600, on cost (parent 2500, bge-large, v2-m3)

| child | pool_recall | cov@8 | hit@8 | hit@5 | points | vectors |
|---|---|---|---|---|---|---|
| 300 | 0.7172 | 0.3853 | 0.3737 | 0.3434 | 77,911 | 319 MB |
| **600** | 0.7071 | 0.3785 | **0.3737** | **0.3434** | 44,542 | 182 MB |
| 900 | 0.7071 | 0.3661 | 0.3636 | 0.3333 | 33,483 | 137 MB |

300 ties 600 *exactly* on both hit metrics for **75% more points and storage**;
900 is a genuine step down. The child choice is economics, not quality — which
is a stronger argument than "it scored highest".

### 10c. Embedder — dimension buys POOL RECALL, not top-8 (parent 2500, child 600, v2-m3)

| embedder | dim | pool_recall | cov@8 | hit@8 | hit@5 | vectors | ingest |
|---|---|---|---|---|---|---|---|
| bge-small-en-v1.5 | 384 | 0.5859 | 0.3538 | 0.3535 | 0.3232 | 68 MB | 240 s |
| bge-base-en-v1.5 | 768 | 0.6364 | 0.3831 | **0.3838** | 0.3333 | 137 MB | 360 s |
| **bge-large-en-v1.5** | 1024 | **0.7071** | 0.3785 | 0.3737 | **0.3434** | 182 MB | 998 s |
| FinLang/finance-embeddings-investopedia | 768 | 0.5556 | 0.3699 | 0.3636 | 0.3131 | 137 MB | 374 s |

Pool recall is cleanly monotonic in dimension (0.586 → 0.636 → 0.707), but the
cross-encoder flattens most of that by the time the synthesizer is fed: bge-base
even edges bge-large on hit@8 by one question. **bge-large is justified by pool
recall and hit@5, not by a top-8 gap** — and 768-d would be defensible if storage
or CPU latency mattered more. A better embedder buys *candidates*; the reranker
cannot retrieve what the pool never contained.

**Finance-domain fine-tuning REJECTED.** `FinLang/finance-embeddings-investopedia`
is a fine-tune *of* bge-base, so it isolates domain adaptation from model size —
and it **loses to its own base model** on every metric, worst of all on pool
recall (0.5556 vs 0.6364, eight questions). Do not re-propose domain-tuned
embedders without evidence on this harness.

### 10d. Reranker — the choice is CONDITIONAL on parent size

hit@8, child 600, bge-large:

| parent | none (RRF) | bge-reranker-base | bge-reranker-v2-m3 | m3 − base |
|---|---|---|---|---|
| 1500 | 0.2626 | **0.3131** | 0.3030 | **−0.0101** |
| 2500 | 0.2727 | 0.3535 | **0.3737** | +0.0202 |
| 4000 | 0.2828 | 0.3333 | **0.3838** | +0.0505 |

`base` caps at 512 tokens (~2000 chars), so it degrades precisely as parents
outgrow its window, while v2-m3's 8k window does not. **§3's "+0.147 for v2-m3"
was never a property of the reranker alone — it is conditional on the geometry.**
At parent 1500 base is *better* and 6× faster. Testing the two knobs
independently gives the wrong answer for either one.

`none` is last at every parent size, so reranking earns its place — but note the
cost: v2-m3 is **4,088 ms/question vs base's 691 ms** on this GPU. On Cloud Run's
2 vCPU that gap is the live latency argument for keeping `base` as an option.

### 10e. Verdict

**parent 2500 · child 600 · bge-large-en-v1.5 · bge-reranker-v2-m3** — the
configuration production already serves, now justified on the pipeline it
actually runs rather than on the PDF pipeline it retired. It is also the top row
by hit@5 across all 24 measured rows.

---

## 11. Why only 37 of 99 — and the fix (July 2026)

§10 settled *geometry*. It left the real number embarrassing: with the winning
configuration, only **37 of 99** questions had at least half their evidence in
the 8 passages the synthesizer receives. This section asks why, and it turns out
none of the usual suspects is responsible.

Everything below is on the §10 winner (parent 2500 / child 600 / bge-large /
v2-m3), n = 99, so **one question = 0.0101**. Full table:
`results/retrieval_selection_sweep.md`.

### 11a. Establish the ceilings before touching anything

Three measurements, all with the retriever switched off — just greedy selection
over parsed chunks:

| selector | hits |
|---|---|
| best 8 parents in the filing (oracle) | **99 / 99** |
| best 2 parents in the filing (oracle) | 98 / 99 |
| best 8 parents **selectable from the retrieved pool** | **70 / 99** |
| same, if each pooled parent could also pull its neighbours | **94 / 99** |
| pool union covers ≥ 0.5 | 71 / 99 |

This kills three hypotheses at once:

- **The metric is not too strict.** A perfect picker scores 99/99, so `hit@8`
  is satisfiable for every question that survived the parsing ceiling. The
  ≥ 0.5-of-evidence bar is not what caps us at 37.
- **The chunk geometry is not the constraint.** Evidence spans are *smaller*
  than one parent (median 1,306 chars vs a 2,500-char parent); only 13 of 127
  spans exceed one parent. Re-cutting chunks cannot be the answer.
- **First-stage recall is not the dominant loss.** The pool already holds the
  evidence for 70–71 of 99. Production converts that into 37.

So the entire gap is **selection**, and it is bounded: 70/99 from the current
pool, 94/99 only if neighbouring chunks can be pulled in.

Supporting diagnosis — where the answer-bearing parent actually sits in the
pool, and what the second reranker pass does to it:

| rank of the best parent | ≤ 0 | ≤ 2 | ≤ 4 | ≤ 7 | ≤ 15 |
|---|---|---|---|---|---|
| in RRF pool order | 5/55 | 19/55 | 23/55 | 29/55 | 36/55 |
| after re-ranking vs the ORIGINAL question | 10/55 | **16/55** | 24/55 | 29/55 | 39/55 |

Median rank 6 before, 6 after. The second pass is **not** surfacing evidence.

### 11b. What shipped: rank on the sub-query score, not the question

`_cap_pool` used to re-score the merged pool against the **original question**,
discarding the score each chunk had already earned against **its own
sub-query**. Dropping that second pass, on byte-identical candidate pools:

| | hit@5 | hit@8 | hits | ctx chars |
|---|---|---|---|---|
| re-score vs the original question (old) | 0.3434 | 0.3737 | 37/99 | 11,167 |
| **rank on the sub-query score (shipped)** | **0.3636** | **0.4141** | **41/99** | **11,021** |

**+4 questions for strictly less context, and it deletes a whole cross-encoder
pass from every query.** Confirmed three times: independently under
parent-scoring, under child-scoring (36 → 41), and end-to-end through the
harness after the change landed (hit@8 0.4040, hit@5 0.3636, retention
0.529 → **0.580**, narrative 12/27 → **15/27**).

The cause is plain once stated: a sub-query is a *retrieval key* — "3M total
assets 2022" — while the question that produced it is multi-hop and often shares
no vocabulary with the balance sheet that answers it. Re-scoring demoted exactly
the chunks the first pass had correctly found. The per-sub-query floor is kept;
it costs nothing (41/99 with and without) and still protects comparisons.

### 11c. What was tried and rejected — 51 rows, mostly negative

- **Blind neighbour expansion — rejected, 8 variants.** The oracle says
  neighbours lift the ceiling 70 → 94, so this looked like the big win. At a
  fixed 8-passage budget it *loses*: 25/99 (r=1) and 24/99 (r=2) vs 37/99. The
  reason is in the rank table above — the top seed is right only ~18% of the
  time, so spending 3 slots on its neighbourhood is a bad bet. Expansion only
  pays if the top hit is reliable, and ours is not.
- **Split-table stitching — rejected, and built on a wrong premise.** Sharper
  version of the same idea: expand *only* across a table→table boundary, on the
  belief that `chunk_by_title` was cutting financial statements into consecutive
  `TableChunk`s. It loses (38/99) — and the premise was checked afterwards and
  is mostly false. In `3M_2022_10K` only **4 of 45** tables exceed
  `max_characters=2500`; the balance sheet is a single 2,012-char chunk running
  intact from "Assets" to "Total liabilities and equity". Oversized tables *are*
  divided (17 `TableChunk`s vs 31 whole `Table`s), so "tables are never split"
  is also too strong — but the primary statements are not the ones splitting.
  The real geometry problem is the opposite of splitting, and §12 names it:
  keeping a table whole means giving it its OWN chunk, which separates it from
  the caption that identifies it. The "numeric questions go 38/60 → 60/60 with
  one more parent" oracle number is real; the second parent is usually the
  **caption chunk or a second statement**, not the continuation of a split table.
- **Scoring the child instead of the parent — no effect.** The cross-encoder
  scores a 2,500–4,000 char parent whose bulk is unrelated to the query, so
  scoring the focused ~600-char child that actually matched should help. It
  ties exactly (41 vs 41); it trades numeric (22 → 24) for narrative (15 → 12).
- **Unconditional table prior — rejected.** Tables are 45% of the pool but
  **73% of evidence-bearing candidates**, and 46 of 49 numeric questions have
  their best evidence in a table. Boosting tables regardless of route wrecks
  narrative questions (15/27 → 7/27) for a net loss. Gated to numeric-routed
  sub-queries it is 42/99 at 10% *less* context — a real but noise-width
  (+1 question) gain, **not shipped**; revisit with a larger question set.
- **Bigger budgets work but are not free.** cap 12 → 40/99, cap 16 → 42/99,
  per-sub 10 + cap 12 → 49/99 at 17,217 ctx chars (+54%). Buying hits with
  context is always available; it is a cost decision, not a retrieval insight.

### 11d. Honest limits

- **40/99 is not 90/99.** The pool caps any reranker at 70/99, and we reach 40.
  The remaining ranking headroom is real but not reachable by reordering rules —
  the next lever is the *pool*, not the sort: sub-queries that name the line
  item ("total assets") rather than the derived metric ("return on assets"),
  which is a planner change, and a first stage that retrieves the continuation
  of a split table at all.
- **This metric is evidence coverage, not answer accuracy.** 60 of 99 questions
  are numeric and production answers those primarily from XBRL facts; this
  harness measures the prose lane only.
- **The re-score removal is unvalidated for faithfulness.** Its original
  justification was "fewer, higher-precision passages measurably raised
  faithfulness". The passage *count* is unchanged (still 8) — only which 8 —
  so the original argument is not disturbed, but this needs an end-to-end eval
  run to confirm faithfulness does not regress.
- **§10's 24 rows predate this change** and were measured with the old
  re-scoring cap. The geometry conclusions should be unaffected (the cap applied
  identically to every row), but they have not been re-measured under §11b.

---

## 12. The chunk did not know what it was — July 2026

§11 got hit@8 from 37 to 41 of 99 by fixing *selection*. §10 had already settled
geometry. Both were tuning around the real defect, which is upstream of both:
**the chunk holding the answer does not say what it is.**

### 12a. What was actually wrong

`chunk_by_title` keeps a table whole by giving it its **own chunk**, never merged
with surrounding text. That is the right call for the table — but it means the
heading naming that table always lands in a *different* chunk:

```
[129] CompositeElement  len=195   '...3M Company and Subsidiaries / Consolidated Balance Sheet / At December 31'
[130] TableChunk        len=2012  '(Dollars in millions) | 2022 | 2021 / Assets / Cash and cash equivalents...'
```

So the chunk carrying "Total assets" has **no company, no year, no statement
name**. Every company's balance sheet embeds to nearly the same vector, and the
sub-query "3M total assets 2022" has nothing to match on but the metadata
filter. Measured: **18 of the 28 questions whose evidence never reached the pool
were numeric**, i.e. table-bearing.

Note this is the *opposite* of the failure §11c assumed. Tables are not being
split (only 4 of 45 in a 3M 10-K exceed `max_characters`); they are being
isolated. That is why §11c's table-stitching experiment failed.

### 12b. The fix

Prefix every chunk — the embedded child **and** `parent_text` — with the identity
it cannot supply itself, built from manifest metadata plus a caption recovered by
walking back through the raw elements (`_element_captions`; SEC HTML has no
`Title` elements, so a heading is recognised by shape):

```
3M 2022 10-K · 3M Company and Subsidiaries · Consolidated Balance Sheet · At December 31
(Dollars in millions, except per share amount) | 2022 | 2021 | Assets | ...
```

It lands on the parent too, because the reranker and the LLM both read
`parent_text` and both were seeing an untitled grid of numbers.

### 12c. Results — and the honesty check that matters

The header is a real string from the filing, but **67 of the 99 gold spans open
with the statement caption**, so pasting it on could satisfy the evidence metric
without retrieving anything better. Every configuration is therefore scored
twice on one index, one pool set and one rerank pass
(`--strip-headers-scoring`): *as served*, and with the injected header removed.

| config | scoring | pool_recall | hit@5 | hit@8 |
|---|---|---|---|---|
| bge-large, no header | — | 0.6970 (69) | 0.3636 (36) | 0.4040 (40) |
| **bge-large + header** | as served | 0.7879 (78) | 0.5556 (55) | **0.5960 (59)** |
| **bge-large + header** | **stripped** | **0.7475 (74)** | **0.5354 (53)** | **0.5758 (57)** |
| e5-large, no header | — | 0.7475 (74) | 0.3737 (37) | 0.4242 (42) |
| e5-large + header | as served | 0.7374 (73) | 0.5051 (50) | 0.5354 (53) |
| e5-large + header | stripped | 0.6869 (68) | 0.4646 (46) | 0.4949 (49) |

**Only 2 of the 19 questions were metric inflation** (59 → 57 stripped). On the
strictest scoring the header is worth **+17 questions on both hit@5 and hit@8** —
roughly four times every other intervention in §10 and §11 combined, at *less*
context (11,047 → 10,456 chars). Retention rises 0.580 → 0.756: the reranker
improves too, because it can finally see that a passage is a balance sheet.

The gain is genuine because `pool_recall` also moved on the stripped scoring,
69 → 74. Note this is the *stripped* number: the header inflates pool recall too
(78 as served), because pool recall is measured by the same text matching. What
the header cannot explain is the stripped 74 > 69 — that gap can only come from
different chunks being retrieved.

### 12c-i. The inflation check in plain English

**How a question is scored.** FinanceBench ships the exact passage that answers
each question — the gold span. Chop it into overlapping 10-word phrases and
count how many appear in the text we hand the synthesizer. ≥50% of the phrases
present = the question is a hit.

**Why the header was suspicious.** A real gold span:

```
3M Company and Subsidiaries Consolidated Balance Sheet At December 31
(Dollars in millions) 2022 2021 Assets Cash and cash equivalents 3,655 ...
```

It *opens with the statement caption* — and the caption is exactly what §12b
pastes onto the chunk:

```
3M 2022 10-K · 3M Company and Subsidiaries · Consolidated Balance Sheet · At December 31   <- added by us
(Dollars in millions) | 2022 | 2021 | Assets | Cash and cash equivalents | 3,655 ...        <- the real chunk
```

So a chunk that previously matched only the second half of the gold span now
matches the first half too — **without retrieving anything different**. The
score rises because we wrote the answer's opening words onto the page. 67 of
the 99 gold spans start this way, so most of the +19 could have been fake.

**The test.** Score one run twice — same index, same pools, same rerank order.
The only difference is that the second scoring deletes the header before
counting phrases. If the gain were fake the stripped score would fall back
toward 40; if it were real the stripped score holds up, because deleting a
prefix cannot change *which* chunks were retrieved.

**The result.** 59 as served, 57 stripped. Only **2** of the 19 extra questions
came from the pasted caption; **17 are real retrieval**.

**Why it is real, mechanically.** The header is not decoration at scoring time —
it is *embedded*. The chunk's vector changes. A balance sheet that used to look
identical to every other company's balance sheet now encodes "3M, 2022, balance
sheet", so the query `3M total assets 2022` can find it at all.

**Generalisable rule: any text you add to a document to help retrieval must be
checked against the text you grade with.** Otherwise you ship a system that
scores better and retrieves no better, and you find out in production.

### 12d. The embedder verdict REVERSES

§12's other lesson: **an embedder comparison is only valid for the chunks it was
run on.** Without headers e5-large-v2 beat bge-large on pool recall (74 vs 69);
with headers bge-large wins on every metric (74 vs 68 stripped, 57 vs 49 hit@8).
The §12b non-BGE sweep would have shipped the wrong embedder had the chunk fix
landed after it. **bge-large-en-v1.5 stays.**

`vectorstore._EMBED_PROMPTS` (query/document prefixes for e5/nomic/arctic) is
kept regardless: those models score near random without it, so any future
comparison that omits it is measuring the missing prefix, not the model.

### 12e. Standing numbers and what is left

hit@8 **40 → 57 of 99** (stripped), hit@5 **36 → 53**. Still short of the ceilings
§11a measured: 74/99 selectable from the current pool, 94/99 with neighbours,
99/99 from the filing. The remaining levers, in order:

1. **Query-side symmetry** — sub-queries say "return on assets"; filings say
   "Total assets" and "Net income". A planner change, and probably the next
   largest win.
2. **Pool depth** — 48 per sub-query today; raising it lifts the 74/99 ceiling.
3. **Near-misses** — 8 of the pool misses sit at coverage 0.41-0.49, just under
   the bar (counted on the pre-header pool of §11a; not re-measured since).

**25 of the 99 questions are unwinnable as the pipeline stands** — their
evidence never enters the pool, so no reranker or selection change can recover
them. Nothing downstream of the pool matters for those; only recall does
(levers 1 and 2, plus the neighbour ceiling of 94/99 from §11a).

**Operationally: this changes what is stored, so shipping it requires a full
re-index of the served corpus.** `CorpusIngester(context_headers=...)` defaults
to on; every collection written before this section lacks the headers.

---

## 13. Three more fixes, and a decision that reversed — July 2026

Plain summary first, because this section is the one worth reading.

**The score.** For each question we hand the LLM 8 passages. We check whether
the official evidence is actually among them. That number was **58 of 99** at
the start of this section and is **66 of 99** at the end.

### 13a. What we found, in order

**1. A decision from §11 had quietly become wrong.**

§11 changed how the final 8 get chosen: rank them by how well each matched the
*sub-query* that found it, rather than re-scoring everything against the
*user's actual question*. That measured +4 and shipped.

It is now worse. Re-running the identical A/B on one index, one pool set and
one scorer:

| rule | hit@5 | hit@8 |
|---|---|---|
| rank by sub-query score (what §11 shipped) | 0.5455 (54) | 0.5859 (58) |
| **re-score against the question** (pre-§11) | 0.5556 (55) | **0.6162 (61)** |

The §11 result was an artefact of chunks that had no identity. Re-scoring an
untitled grid of numbers against a multi-hop question was hopeless; once §12
made the chunk say `3M 2022 10-K · Consolidated Balance Sheet`, the question —
the only thing that knows what the user actually wants — can match it.

**This is the second §12-induced reversal**, after the embedder (§12d). The
lesson is now established rather than anecdotal: **a retrieval decision is only
valid for the chunk representation it was measured on.** Any change to what a
chunk contains invalidates every ranking experiment that came before it.

**2. The query vocabulary does not match the filing's vocabulary.**

The planner asks for `AMD quick ratio FY2022`. No filing contains that phrase —
it prints `Total current assets`, `Inventories`, `Total current liabilities`.
Checked against all 99 gold spans:

| phrase used by sub-queries | appears in ANY gold evidence |
|---|---|
| quick ratio, current ratio | **no** |
| return on assets, asset turnover | **no** |
| inventory turnover | **no** |
| free cash flow, EBITDA | **no** |
| dividend payout, liquidity | **no** |
| operating margin, gross margin, effective tax rate | yes |

32 of 99 questions have a sub-query naming a derived metric. Semantic search
does not rescue this: an embedder *paraphrases* ("capital spending" ≈ "capital
expenditures") but does not *derive* ("quick ratio" → "(current assets −
inventories) ÷ current liabilities"). Semantically, "quick ratio" is nearest to
text that says "quick ratio" — a glossary, not a balance sheet. Six of the
sixteen questions whose evidence never reached the pool at ANY depth sat at
coverage **0.00-0.06** for this reason.

`finagent/retrieval/expansion.py` appends the line items the metric is built
from. It feeds both halves of the retriever — the dense vector moves toward
statement language, and BM25 gets tokens that are actually present. Each entry
also names the statement, which works *because* §12 put the caption inside the
chunk text. Pool recall **74 → 79**, hit@8 **61 → 63**, hit@5 **55 → 60**, at
no extra query and no extra context.

**3. Nobody was searching for what the user actually asked.**

The planner decomposes a question into sub-queries and only those were
retrieved on; the original question was discarded. Adding it back as one more
query is worth **+3**. It is gated on the planner's routing — when every
sub-query was routed to yfinance/web/EDGAR, the filings are not the source and
the question must not drag them back in. (On this eval that gate never fires:
0 of 99 questions were routed entirely away from filings.)

It gets a smaller slot budget than a sub-query, because at the full 5 it
crowds the sub-queries out of the cap:

| slots for the question | hit@5 | hit@8 | ctx chars |
|---|---|---|---|
| 1 | 60 | 64 | 10,685 |
| **2 (shipped)** | **55** | **66** | 11,144 |
| 3 | 53 | 66 | 11,628 |
| 5 | 50 | 67 | 12,418 |

Two is the only setting that beats the previous production on **both** metrics.
Five buys one more hit@8 and costs a hit@5 regression below where we started.

### 13b. The combination, end to end

n=99, header-stripped scoring, bge-large + bge-reranker-v2-m3, parent 2500 /
child 600.

| arm | hit@5 | hit@8 | ctx |
|---|---|---|---|
| A production before this section | 0.5455 (54) | 0.5859 (58) | 10,240 |
| B + revert §11 | 0.5556 (55) | 0.6162 (61) | 10,389 |
| C B + expansion | 0.6061 (60) | 0.6364 (63) | 10,334 |
| D B + question query | 0.4747 (47) | 0.6566 (65) | 12,460 |
| E B + dense-only (no BM25) | 0.5859 (58) | 0.6263 (62) | 10,314 |
| F B + expansion + question query (5 slots) | 0.5051 (50) | 0.6768 (67) | 12,418 |
| G everything incl. dense-only | 0.5152 (51) | 0.6768 (67) | 12,347 |
| F with 2 question slots (harness prediction) | 0.5556 (55) | 0.6667 (66) | 11,144 |
| **SHIPPED, verified end to end** | **0.5960 (59)** | **0.6768 (67)** | **11,158** |

The last row is the one that counts: it runs the actual production classes
(`HybridRetriever.search` + `AgenticRAGv2._cap_pool`), not the harness's
reconstruction of them. **Always verify the shipped path — the two disagreed
by 2 questions and the disagreement was a real bug** (§13d).

### 13c. What was rejected, with the evidence

**Dropping BM25 for pure semantic search.** Dense-only genuinely beats hybrid
RRF on pool recall (77 vs 74 of 99) — the lexical half dilutes the pool, and
not only on ratio queries as predicted, but evenly across all question types.
But **G == F exactly**: once expansion and the question query are in, removing
BM25 adds nothing. Hybrid stays. A headline feature is not worth deleting on a
signal that disappears in combination.

**Neighbouring-chunk expansion** (take the parents either side of each hit).
Re-tested because §11 rejected it when the reranker was blind, and §12 fixed
that. Still no: at cap 8 it is a disaster (50 vs 61) because each neighbour
displaces a seed, it only pays at cap 16, and it costs 15 points of hit@5 in
every variant. Its apparent +4 was partly measuring the ranking rule, not the
neighbours — under a fixed rule it ties (66 = 66).

**Deeper candidate pools.** Retrieving 8× as many children per sub-query:

| children / sub-query | pool recall |
|---|---|
| 48 (production) | 74/99 |
| 96 | 77/99 |
| 192 | 80/99 |
| 384 | 83/99 |

+9 questions for 8× the cross-encoder cost, and a hard ceiling at 83 — **16
questions are unreachable at any depth**. Not worth it.

### 13d. The harness and the shipped code disagreed — and the harness was right

The §13b table is a reconstruction of production inside the eval harness.
Running the REAL classes end to end gave **64/99**, not the 66 the harness
predicted. Chasing the 2-question gap found a genuine bug that §12 had created.

`hybrid_retrieve_node` de-duplicated merged results on
`(local_path, page, text[:80])`. Since §12 every chunk STARTS with its context
header, so `text[:80]` is frequently just
`Apple 2022 10-K · Consolidated Balance Sheet` — byte-identical across the
several parents of one long statement. Measured on the cached pools: **42 of
13,956 distinct parents (0.3%) were silently discarded as duplicates.** The
harness deduped on `parent_id` and never hit it, which is exactly why the two
numbers diverged.

Keying on `(local_path, page, parent_id)` — falling back to the text prefix only
for chunks with no parent — fixed it, and the shipped path then measured
**67/99 hit@8 / 59/99 hit@5**, better than the harness had predicted.

**The general point:** a metric computed by a harness that re-implements
production measures the harness. Three times this session a number only held up
because it was checked against a second implementation — the §12c
header-stripping check, the §11/§13 reversal, and this. Budget for the
cross-check; it is where the bugs are.

### 13e. A hypothesis that was wrong, recorded because it was expensive

The 16 hard misses looked like they must be questions whose evidence spans
several chunks (a whole income statement). Measured: median gold span is
**1,674 chars for misses vs 1,452 for hits** — both fit inside one 2,500-char
parent. The right parent simply never enters the pool, which is why coverage
sits at 0.01-0.13 rather than "partially covered". That killed the
split-statement theory a second time (§11c killed it once) and redirected the
work to vocabulary, where the wins actually were.

Also measured, and useful context for any future pool work: a 48-child pool
collapses to a median of **37 distinct parents out of ~229 per filing** — one
sub-query sees about **16% of the document**.

### 13f. Cost

The cross-encoder dominates retrieval latency. Per question it now scores
roughly **89 → 138 passages (+55%)**: +37 for the question query's own pool,
+12 for the restored cap re-score. Retrieval is therefore ~50% slower; end to
end the effect is smaller because synthesis dominates, but this has **not been
measured** — only counted. Context to the synthesizer rises 10,240 → 11,144
chars (+9%).

No re-index is needed for any of §13 — every change is query-side or
selection-side.

### 13g. Honest limits

- **67/99 is 68%**, against a ceiling of 74 from the current pool. ~8 questions
  are reachable by better ranking; ~25 need better retrieval or a better parser.
  There is no measured path to 90% on this corpus.
- **The expansion table was built by reading this eval's failures.** The terms
  are generic accounting vocabulary rather than question-specific, but it is
  fitted to the test set and should be validated on held-out questions before
  the number is quoted anywhere.
- **99 questions means ±3 is close to noise.** The large steps (§12's +17) are
  solid; the +2s and +3s here should be held loosely.
- The gain is in *"somewhere in the 8"*, not *"at the top"* — hit@5 moved 54 →
  55 while hit@8 moved 58 → 66. Lowering `retrieve_cap` to 5 would give most of
  it back.

## 14. Does decomposition help at all? — July 2026

Every section above was measured on the planner's sub-queries. The harness has
carried a `--mode question` control since §10 and it was **never run on the
shipped configuration**, so the value of decomposition itself had never been
measured. `scripts/query_mode_ablation.py` measures it: three arms, one index,
one reranker, one 8-passage budget — only the query text changes.

| arm | queries/q | pool_recall | hit@5 | hit@8 | retention | wall |
|---|---|---|---|---|---|---|
| subquery (served path) | 3.09 | 0.9091 (90) | 0.6061 (60) | 0.6768 (67) | 0.744 | 1122s |
| question only | 1 | 0.7677 (76) | 0.5859 (58) | 0.6970 (69) | 0.908 | 374s |
| **one rewritten query** | 1 | 0.8586 (85) | **0.7677 (76)** | **0.8081 (80)** | **0.941** | **340s** |

The subquery arm returns hit@8 0.6768 (67/99) — **identical to §13b's shipped
row**, so this is measuring the same thing the rest of the file measures.

**Decomposition buys recall and then throws it away.** Sub-queries pool the most
evidence (90/99) and retain the least (0.744). One rewritten query pools less
(85) and delivers 80. Question-only, from a pool 14 questions worse, still beats
the served path on hit@8. §11a said the constraint was selection rather than
recall; this is that finding at full strength — every extra sub-query adds five
more candidates competing for the same eight slots.

The rewrite is the question restated in the filing's vocabulary: company, fiscal
year, statement caption, and the line items a filing actually prints — no
question words, no derived-metric names. It is §13a-2's expansion idea applied
to the whole query rather than appended to it. The gain concentrates exactly
where §13a-2 predicted: numeric 41 -> 51 of 60.

Comparison questions are the one place decomposition earns its keep (7/12 vs
question-only 5/12) — the cross-document intuition is real but small, and a
single query naming both years and all segments still beats it (9/12).

**Caveat, and it is the important one:** the 99 rewrites were hand-authored, not
model-generated. They were written from the question and its sub-queries with
the gold spans never consulted, but by an author who knew §12 puts captions in
chunk text. Whether a production LLM rewriter reaches the same number is
UNTESTED and is the next thing to measure before shipping this.

Full table, per-type breakdown and limits: `results/query_mode_ablation.md`.
