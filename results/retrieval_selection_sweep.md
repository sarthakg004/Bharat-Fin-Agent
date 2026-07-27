# Retrieval selection sweep — what reaches the synthesizer

> **Superseded in part (§13).** The winning row here — ranking the cap on
> each chunk's sub-query score — was re-measured after §12 context headers
> and REVERSED (58 vs 61 of 99). Production re-scores against the question
> again. The ceilings and the rejected strategies below still stand.

Fixed: parent 2500 / child 600 / bge-large-en-v1.5 / bge-reranker-v2-m3, 72 SEC 10-K/10-Q filings, one Qdrant collection.
n = 99 questions. 1 question = 0.0101; deltas under ~0.02 are noise.

`hit@top` = at least half the verified evidence span is inside the passages the synthesizer receives (`cap` of them, 8 unless stated). `ctx_chars` is the mean total characters handed over — the budget each row spends.

## Ceilings (measured, not assumed)

| selector | hits |
|---|---|
| oracle@8, any parent in the filing | 99/99 |
| oracle@2, any parent in the filing | 98/99 |
| best k parents selectable from the retrieved pool | 70/99 |
| same, if each pooled parent could pull its neighbours | 94/99 |
| pool union covers >=0.5 | 71/99 |

A perfect picker reaches 99/99, so the >=0.5 bar is satisfiable and the chunk geometry is not the constraint — selection is. The pool caps any reranker at 70/99; only neighbouring chunks lift that to 94/99.

## Every strategy tried

Round 1's `neighbours *` rows emitted the final set in document order rather than rank order, so their **hit@5 is not meaningful** (marked `n/a`); hit@top is unaffected because it reads the whole set. Round 5 re-tested expansion with rank order preserved and it still lost.

| round | varied | strategy | hit@5 | hit@top | hits | cov@top | ctx |
|---|---|---|---|---|---|---|---|
| 1 | budget, per-sub depth, blind neighbours | baseline (production) | 0.3434 | **0.3737** | 37/99 | 0.3785 | 11167 |
| 1 | budget, per-sub depth, blind neighbours | no per-sub floor | 0.3434 | **0.3737** | 37/99 | 0.3827 | 11239 |
| 1 | budget, per-sub depth, blind neighbours | cap 12 | 0.3434 | **0.404** | 40/99 | 0.4114 | 12471 |
| 1 | budget, per-sub depth, blind neighbours | cap 16 | 0.3636 | **0.4242** | 42/99 | 0.431 | 12738 |
| 1 | budget, per-sub depth, blind neighbours | per_sub 8 | 0.3131 | **0.404** | 40/99 | 0.4088 | 14409 |
| 1 | budget, per-sub depth, blind neighbours | no sub-rerank (RRF order) | 0.2424 | **0.2727** | 27/99 | 0.3005 | 9800 |
| 1 | budget, per-sub depth, blind neighbours | neighbours r=1 (cap 8) | n/a | **0.2525** | 25/99 | 0.2599 | 10374 |
| 1 | budget, per-sub depth, blind neighbours | neighbours r=1, no floor | n/a | **0.2525** | 25/99 | 0.2599 | 10374 |
| 1 | budget, per-sub depth, blind neighbours | neighbours r=2 (cap 8) | n/a | **0.2424** | 24/99 | 0.253 | 9966 |
| 1 | budget, per-sub depth, blind neighbours | neighbours r=1 (cap 12) | n/a | **0.3636** | 36/99 | 0.3701 | 15016 |
| 1 | budget, per-sub depth, blind neighbours | neighbours r=1 (cap 16) | n/a | **0.4444** | 44/99 | 0.4423 | 18839 |
| 2 | what orders the merged pool | baseline (production) | 0.3434 | **0.3737** | 37/99 | 0.3785 | 11167 |
| 2 | what orders the merged pool | per-sub score, no global rescore | 0.3636 | **0.4141** | 41/99 | 0.4217 | 11015 |
| 2 | what orders the merged pool | per-sub score + floor | 0.3636 | **0.4141** | 41/99 | 0.4133 | 11050 |
| 2 | what orders the merged pool | round-robin by sub | 0.3535 | **0.404** | 40/99 | 0.4099 | 11101 |
| 2 | what orders the merged pool | baseline + expand r1 top2 | 0.1919 | **0.303** | 30/99 | 0.3111 | 11266 |
| 2 | what orders the merged pool | baseline + expand r1 top1 | 0.2323 | **0.3535** | 35/99 | 0.3664 | 11844 |
| 2 | what orders the merged pool | per-sub + expand r1 top2 | 0.2121 | **0.3636** | 36/99 | 0.3709 | 10956 |
| 2 | what orders the merged pool | round-robin + expand r1 top2 | 0.1919 | **0.3434** | 34/99 | 0.3553 | 11070 |
| 2 | what orders the merged pool | per-sub, per_sub=10 | 0.3636 | **0.4545** | 45/99 | 0.4556 | 13893 |
| 2 | what orders the merged pool | per-sub, per_sub=10, cap 12 | 0.3636 | **0.4747** | 47/99 | 0.4814 | 18784 |
| 2 | what orders the merged pool | per-sub + expand r1 top2, cap 12 | 0.2121 | **0.4242** | 42/99 | 0.4308 | 14625 |
| 3 | what text the cross-encoder scores | parent-scored + global rescore (production) | 0.3434 | **0.3737** | 37/99 | 0.3827 | 11250 |
| 3 | what text the cross-encoder scores | parent-scored, no rescore (round2 best) | 0.3636 | **0.4141** | 41/99 | 0.4217 | 11021 |
| 3 | what text the cross-encoder scores | CHILD-scored, no rescore | 0.3333 | **0.4141** | 41/99 | 0.4271 | 10089 |
| 3 | what text the cross-encoder scores | CHILD-scored + global rescore | 0.3333 | **0.3636** | 36/99 | 0.3804 | 10380 |
| 3 | what text the cross-encoder scores | both max(child,parent) | 0.3535 | **0.404** | 40/99 | 0.4096 | 10974 |
| 3 | what text the cross-encoder scores | both mean(child,parent) | 0.3535 | **0.4242** | 42/99 | 0.4256 | 10517 |
| 3 | what text the cross-encoder scores | CHILD-scored, per_sub=10 | 0.3333 | **0.4343** | 43/99 | 0.4528 | 12804 |
| 3 | what text the cross-encoder scores | CHILD-scored, per_sub=10, cap 12 | 0.3333 | **0.4949** | 49/99 | 0.5016 | 17217 |
| 3 | what text the cross-encoder scores | parent-scored, per_sub=10 | 0.3636 | **0.4545** | 45/99 | 0.4556 | 13902 |
| 4 | a table prior on the reranker score | production (parent + global rescore) | 0.3434 | **0.3737** | 37/99 | 0.3827 | 11250 |
| 4 | a table prior on the reranker score | drop global rescore  [round2 best] | 0.3636 | **0.4141** | 41/99 | 0.4217 | 11021 |
| 4 | a table prior on the reranker score | + table boost 0.5 | 0.3535 | **0.3939** | 39/99 | 0.4023 | 9819 |
| 4 | a table prior on the reranker score | + table boost 1.0 | 0.3131 | **0.3636** | 36/99 | 0.3781 | 9408 |
| 4 | a table prior on the reranker score | + table boost 2.0 | 0.3131 | **0.3636** | 36/99 | 0.3781 | 9408 |
| 4 | a table prior on the reranker score | + table boost 5.0 | 0.3131 | **0.3636** | 36/99 | 0.3781 | 9408 |
| 4 | a table prior on the reranker score | + table boost 2.0, numeric subs only | 0.3737 | **0.4242** | 42/99 | 0.4326 | 9969 |
| 4 | a table prior on the reranker score | + table boost 1.0, child-scored | 0.2828 | **0.3434** | 34/99 | 0.3626 | 9029 |
| 4 | a table prior on the reranker score | + table boost 2.0, child-scored | 0.2828 | **0.3434** | 34/99 | 0.3626 | 9029 |
| 4 | a table prior on the reranker score | + table boost 2.0, per_sub=10 | 0.3131 | **0.3737** | 37/99 | 0.3906 | 12292 |
| 4 | a table prior on the reranker score | + table boost 2.0, cap 12 | 0.3131 | **0.3636** | 36/99 | 0.3781 | 10424 |
| 5 | split-table stitching | production today | 0.3434 | **0.3737** | 37/99 | 0.3827 | 11250 |
| 5 | split-table stitching | A: drop global rescore | 0.3636 | **0.4141** | 41/99 | 0.4217 | 11021 |
| 5 | split-table stitching | B: A + table boost (numeric subs) | 0.3737 | **0.4242** | 42/99 | 0.4326 | 9969 |
| 5 | split-table stitching | C: A + stitch 1 | 0.3131 | **0.3838** | 38/99 | 0.4025 | 11376 |
| 5 | split-table stitching | D: A + stitch 2 | 0.3131 | **0.3939** | 39/99 | 0.4062 | 11413 |
| 5 | split-table stitching | E: B + stitch 1 | 0.3333 | **0.3838** | 38/99 | 0.4078 | 10364 |
| 5 | split-table stitching | F: B + stitch 2 | 0.3333 | **0.3838** | 38/99 | 0.4024 | 10474 |
| 5 | split-table stitching | G: E, cap 10 | 0.3333 | **0.404** | 40/99 | 0.4226 | 11646 |
| 5 | split-table stitching | H: E, per_sub=8 | 0.3333 | **0.404** | 40/99 | 0.4284 | 12155 |

## By question type, for the rows that matter

| strategy | comparison | narrative | numeric |
|---|---|---|---|
| baseline (production) | 5/12 | 12/27 | 20/60 |
| baseline (production) | 5/12 | 12/27 | 20/60 |
| production today | 5/12 | 12/27 | 20/60 |
| A: drop global rescore | 4/12 | 15/27 | 22/60 |
| B: A + table boost (numeric subs) | 4/12 | 12/27 | 26/60 |
