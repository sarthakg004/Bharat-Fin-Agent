# Query mode ablation — does decomposing the question help retrieval?

One index (`qx_p2500_c600_bge-large-en-v1.5_hdr`), one reranker (`BAAI/bge-reranker-v2-m3`), n=99.
**Only the query text differs between arms** — same pool engine, same
parent geometry, same cross-encoder, same 8-passage budget.

- `subquery` — the served path: cached planner decomposition, retrieve per
  sub-query, merge, global cap. 3.09 queries/question (2.1 sub-queries + the
  original question, §13a-3).
- `question` — the user's question, unchanged.
- `rewrite` — ONE query per question, restated in the filing's vocabulary:
  company + fiscal year + statement caption + the line items a filing
  actually prints. Authored offline, cached in
  `results/financebench_rewrites.json`.

`expand_query` runs in EVERY arm — it lives inside `HybridRetriever.search`,
and removing it from one arm would confound the comparison. So `question` is
"question + line-item expansion", which makes it a harder baseline, not a
straw man.

`hit@k` = the verified evidence span is >=50% covered by the top-k parents
handed to the synthesizer. `retention` = hit@8 / pool_recall: how much of what
the pool found actually survives selection. One question = 0.0101.

| arm | queries/q | pool_recall | hit@5 | hit@8 | retention | ctx@8 | wall |
|---|---|---|---|---|---|---|---|
| subquery | 3.09 | 0.9091 (90) | 0.6061 (60) | 0.6768 (67) | 0.7444 | 11451 | 1122s |
| question | 1 | 0.7677 (76) | 0.5859 (58) | 0.6970 (69) | 0.9079 | 13955 | 374s |
| **rewrite** | 1 | 0.8586 (85) | 0.7677 (76) | 0.8081 (80) | 0.9412 | 12920 | 340s |

## By question type (hit@8)

| qtype | n | subquery | question | rewrite |
|---|---|---|---|---|
| numeric | 60 | 41 | 45 | 51 |
| narrative | 27 | 19 | 19 | 20 |
| comparison | 12 | 7 | 5 | 9 |

## What this says

**1. The subquery arm reproduces the published shipped number exactly.**
§13b's "SHIPPED, verified end to end" row is hit@8 0.6768 (67/99); this arm
returns 0.6768 (67/99). The harness is measuring the same thing, so the other
two arms are comparable to the whole of §10-13, not just to each other.

**2. Decomposition buys recall and then throws it away.** Sub-queries have the
BEST pool recall of the three (90/99) and the WORST retention (0.744). One
rewritten query pools less (85/99) and delivers far more (80/99, retention
0.941). This is §11a's finding made unavoidable: the constraint was never
finding the evidence, it was choosing it, and every extra sub-query adds five
more candidates competing for the same eight slots.

**3. Question-only already beats the served path on hit@8** (69 vs 67) from a
pool that is 14 questions worse. The entire benefit of decomposition is
consumed by the selection it makes harder.

**4. The win is concentrated exactly where §13a-2 predicted.** Numeric
questions go 41 -> 51 of 60. Those are the ones whose sub-queries name a
derived metric no filing prints; a rewrite that names `total current
liabilities` instead of `quick ratio` retrieves the page that has the number.

**5. Comparison questions are the one place decomposition earns its keep** —
7/12 vs question-only's 5/12. That is the cross-document intuition, and it is
real but small (n=12). The rewrite still beats both at 9/12, because a single
query naming both years and all segments covers the same ground.

**6. It is 3.3x faster.** 340s vs 1122s over 99 questions. Retrieval is linear
in query count, and this cuts 3.09 queries to 1. On the served path the same
ratio turned a traced 827s retrieve into what should be ~270s.

## Honest limits

- The 99 rewrites were **hand-authored, not model-generated**. They were
  written from the question and its sub-queries only — the gold evidence spans
  were never consulted — but the author knew the system design, including that
  §12 puts statement captions inside chunk text. A production LLM rewriter
  given the same rules would plausibly land close, but that is untested and is
  the single biggest caveat here.
- Cost moves from retrieval to planning: production would need one LLM call to
  produce the rewrite. That is far cheaper than the 2+ extra cross-encoder
  passes it removes, but it is not free, and it is a latency floor.
- `ctx@8` rises 11,451 -> 12,920 (+13%): one query fills all eight slots from
  one pool, so passages are less deduplicated across sub-queries.
- The sub-query plans also drive the XBRL / calculator / market routing.
  Replacing them for RETRIEVAL does not remove the planner.

## Reproducing

```
export QDRANT_EVAL_URL=http://localhost:6333
python scripts/query_mode_ablation.py
```
