# Retrieval report — FinanceBench eval collection

`financebench_eval` · 150 questions · hybrid RRF (dense + BM25 sparse) · pool 48 · reranker `BAAI/bge-reranker-v2-m3` · parent-document retrieval

Measured on the live cluster, so cross-filing distractors are present (the geometry harness searched one filing at a time and cannot show them).


## Ranking metrics

| k | hit@k | recall@k | precision@k | nDCG@k | coverage@k |
|---|---|---|---|---|---|
| 1 | 0.2400 | 0.2067 | 0.2400 | 0.2400 | 0.2236 |
| 3 | 0.4467 | 0.3967 | 0.1556 | 0.3274 | 0.3852 |
| 5 | 0.4867 | 0.4467 | 0.1067 | 0.3502 | 0.4500 |
| 8 | 0.5533 | 0.5233 | 0.0783 | 0.3779 | 0.4995 |
| 20 | 0.6333 | 0.6000 | 0.0363 | 0.3996 | 0.5822 |

**MRR** 0.3564  ·  **pool_recall** 0.6800 (gold in the pool before reranking)  ·  **retention@8** 0.8137 (share of retrievable questions the reranker delivers)


### What each metric means here

- **hit@k** — the question is *answerable*: at least one gold parent reached the LLM. Most questions have one gold parent (median 1), so this is the headline number.
- **recall@k** — of all gold parents, how many arrived. Differs from hit@k only for the 35 questions with 2–3 gold spans.
- **precision@k** — share of delivered chunks that are gold. It is *mechanically low*: with 1 gold parent, precision@8 cannot exceed 0.125. Read it as noise ratio, not quality.
- **nDCG@k** — rewards ranking gold higher, not merely including it. This is what `_cap_pool` and the grader actually consume.
- **coverage@k** — fraction of FinanceBench's verified evidence *lines* present in the top-k. Independent of chunk geometry, so it is the metric that survives a reindex.


## By question type

| type | n | hit@8 | recall@8 | MRR |
|---|---|---|---|---|
| comparison | 22 | 0.4091 | 0.3864 | 0.2983 |
| narrative | 57 | 0.5439 | 0.5351 | 0.3397 |
| numeric | 71 | 0.6056 | 0.5563 | 0.3878 |

## Failure analysis

67 of 150 questions (44.7%) fail at k=8: no gold parent reached the synthesizer. Each is assigned ONE reason by the first matching rule below — every rule is a measurement on that question, not a judgement call.

| Failure reason | Count | % of failures | % of all questions |
|---|---|---|---|
| Embedding semantic miss | 29 | 43.3% | 19.3% |
| Reranker removed relevant chunk | 15 | 22.4% | 10.0% |
| Chunk boundary split evidence | 14 | 20.9% | 9.3% |
| Gold annotation ambiguity | 8 | 11.9% | 5.3% |
| Cross-document retrieval (same company, wrong filing) | 1 | 1.5% | 0.7% |

### Classification rules (in order)

1. **Chunk boundary split evidence** — no single parent in the correct filing contains ≥50% of the evidence lines. The chunker destroyed it; no retriever could return it whole. *Fixed by a larger parent window.*
2. **Reranker removed relevant chunk** — a gold parent WAS in the pool, and the cross-encoder ranked it below k. *Fixed by a stronger reranker.*
3. **Cross-company retrieval** — every returned parent came from another filing AND no company filter was applied (`infer_filter` matched no indexed company). *Fixed by company-vocabulary coverage.*
4. **Cross-document retrieval** — right company, wrong filing/year.
5. **Table retrieval failure** — never entered the pool and the evidence is >35% digits/currency characters, i.e. a dense numeric table, where embeddings carry little lexical signal.
6. **Gold annotation ambiguity** — the evidence span has ≤2 content lines; too thin to attribute the miss to retrieval.
7. **Embedding semantic miss** — residual: retrievable, in the right filing, prose, and still never surfaced.

