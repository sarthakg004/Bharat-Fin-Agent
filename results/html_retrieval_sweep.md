# Retrieval sweep — served HTML pipeline

Corpus: reused SEC filings (FinanceBench 10-K/10-Q), one Qdrant collection, production `CorpusIngester` + `HybridRetriever`, pool depth 48, 5 per sub-query, capped to 8 against the original question.

Questions: **99** scored · 28 excluded because `partition_html` never recovers their evidence from the primary filing (parsing ceiling < 0.5, no retriever can return it) · 0 where the planner routed every sub-query away from the filings (scored as misses — production retrieves nothing for them either).

`mode` = `served` runs the production path (one retrieval query → per-sub-query retrieval → merge → global cap); `question` retrieves on the bare question and is a control, not what production does.

`coverage@k` = shingle recall of the verified evidence span in the top-k parents given to the synthesizer; `hit@k` = coverage ≥ 0.5; `retention` = hit@8 / pool_recall; `num@k` = fraction of the evidence's FIGURES present (order-insensitive, so it sees table evidence shingles miss); `mrr` = mean reciprocal rank of the first passage carrying the evidence.

| mode | rewriter | parent | child | embed | dim | reranker | n | subqueries | pool_recall | pool_num | cov@5 | hit@5 | cov@8 | hit@8 | num@8 | mrr | retention | points | vectors_mb | ctx_chars@8 | pool_ms | rerank_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| served |  | 2500 | 600 | bge-large-en-v1.5 | 1024 | BAAI/bge-reranker-v2-m3 | 99 | 2 | 0.8687 | 0.8912 | 0.6231 | 0.6566 | 0.7014 | 0.7374 | 0.8278 | 0.4297 | 0.8488 | 44542 | 182.4 | 14543 | 33 | 10586 |
| served | default | 2500 | 600 | bge-large-en-v1.5 | 1024 | BAAI/bge-reranker-v2-m3 | 99 | 2 | 0.9091 | 0.9048 | 0.6203 | 0.6566 | 0.708 | 0.7475 | 0.8238 | 0.444 | 0.8222 | 44542 | 182.4 | 13990 | 27 | 8411 |
| subquery |  | 1500 | 600 | bge-large-en-v1.5 | 1024 | BAAI/bge-reranker-base | 99 | 2.1 | 0.6667 |  | 0.3119 | 0.303 | 0.3286 | 0.3131 |  |  | 0.4697 | 46186 | 189.2 | 7165 | 20 | 659 |
| subquery |  | 1500 | 600 | bge-large-en-v1.5 | 1024 | BAAI/bge-reranker-v2-m3 | 99 | 2.1 | 0.6667 |  | 0.278 | 0.2626 | 0.3155 | 0.303 |  |  | 0.4545 | 46186 | 189.2 | 7378 | 20 | 3728 |
| subquery |  | 1500 | 600 | bge-large-en-v1.5 | 1024 | none (RRF order) | 99 | 2.1 | 0.6667 |  | 0.2771 | 0.2424 | 0.2959 | 0.2626 |  |  | 0.3939 | 46186 | 189.2 | 6565 | 20 | 0 |
| subquery |  | 2500 | 300 | bge-large-en-v1.5 | 1024 | BAAI/bge-reranker-base | 99 | 2.1 | 0.7172 |  | 0.3498 | 0.3232 | 0.3699 | 0.3434 |  |  | 0.4789 | 77911 | 319.1 | 10596 | 20 | 711 |
| subquery |  | 2500 | 300 | bge-large-en-v1.5 | 1024 | BAAI/bge-reranker-v2-m3 | 99 | 2.1 | 0.7172 |  | 0.3517 | 0.3434 | 0.3853 | 0.3737 |  |  | 0.5211 | 77911 | 319.1 | 11116 | 20 | 4095 |
| subquery |  | 2500 | 300 | bge-large-en-v1.5 | 1024 | none (RRF order) | 99 | 2.1 | 0.7172 |  | 0.3172 | 0.2727 | 0.3282 | 0.303 |  |  | 0.4225 | 77911 | 319.1 | 9497 | 20 | 0 |
| subquery |  | 2500 | 600 | bge-base-en-v1.5 | 768 | BAAI/bge-reranker-base | 99 | 2.1 | 0.6364 |  | 0.3268 | 0.303 | 0.3383 | 0.3232 |  |  | 0.5079 | 44542 | 136.8 | 10594 | 13 | 706 |
| subquery |  | 2500 | 600 | bge-base-en-v1.5 | 768 | BAAI/bge-reranker-v2-m3 | 99 | 2.1 | 0.6364 |  | 0.3427 | 0.3333 | 0.3831 | 0.3838 |  |  | 0.6032 | 44542 | 136.8 | 11124 | 13 | 4129 |
| subquery |  | 2500 | 600 | bge-base-en-v1.5 | 768 | none (RRF order) | 99 | 2.1 | 0.6364 |  | 0.2791 | 0.2525 | 0.294 | 0.2626 |  |  | 0.4127 | 44542 | 136.8 | 9088 | 13 | 0 |
| subquery |  | 2500 | 600 | bge-large-en-v1.5 | 1024 | BAAI/bge-reranker-base | 99 | 2.1 | 0.697 |  | 0.345 | 0.3232 | 0.3609 | 0.3333 |  |  | 0.4783 | 44542 | 182.4 | 10707 | 26 | 916 |
| subquery |  | 2500 | 600 | bge-large-en-v1.5 | 1024 | BAAI/bge-reranker-base | 99 | 2.1 | 0.7879 |  | 0.3288 | 0.303 | 0.3807 | 0.3636 |  |  | 0.4615 | 44542 | 182.4 | 10811 | 19 | 649 |
| subquery |  | 2500 | 600 | bge-large-en-v1.5 | 1024 | BAAI/bge-reranker-v2-m3 | 99 | 2.1 | 0.697 |  | 0.3887 | 0.3636 | 0.4079 | 0.404 |  |  | 0.5797 | 44542 | 182.4 | 11047 | 26 | 4748 |
| subquery |  | 2500 | 600 | bge-large-en-v1.5 | 1024 | BAAI/bge-reranker-v2-m3 | 99 | 2.1 | 0.7879 |  | 0.5624 | 0.5556 | 0.607 | 0.596 |  |  | 0.7564 | 44542 | 182.4 | 10456 | 19 | 5067 |
| subquery |  | 2500 | 600 | bge-large-en-v1.5 | 1024 | cross-encoder/ms-marco-MiniLM-L12-v2 | 99 | 2.1 | 0.697 |  | 0.2755 | 0.2626 | 0.2848 | 0.2727 |  |  | 0.3913 | 44542 | 182.4 | 10095 | 26 | 250 |
| subquery |  | 2500 | 600 | bge-large-en-v1.5 | 1024 | cross-encoder/ms-marco-MiniLM-L12-v2 | 99 | 2.1 | 0.7879 |  | 0.3421 | 0.3333 | 0.3678 | 0.3636 |  |  | 0.4615 | 44542 | 182.4 | 9477 | 19 | 318 |
| subquery |  | 2500 | 600 | bge-large-en-v1.5 | 1024 | mixedbread-ai/mxbai-rerank-base-v2 | 99 | 2.1 | 0.697 |  | 0.3227 | 0.2929 | 0.3618 | 0.3333 |  |  | 0.4783 | 44542 | 182.4 | 9430 | 26 | 1716 |
| subquery |  | 2500 | 600 | bge-large-en-v1.5 | 1024 | mixedbread-ai/mxbai-rerank-base-v2 | 99 | 2.1 | 0.7879 |  | 0.3248 | 0.3131 | 0.3299 | 0.3131 |  |  | 0.3974 | 44542 | 182.4 | 8863 | 19 | 2079 |
| subquery |  | 2500 | 600 | bge-large-en-v1.5 | 1024 | none (RRF order) | 99 | 2.1 | 0.697 |  | 0.2889 | 0.2323 | 0.305 | 0.2626 |  |  | 0.3768 | 44542 | 182.4 | 9675 | 26 | 0 |
| subquery |  | 2500 | 600 | bge-large-en-v1.5 | 1024 | none (RRF order) | 99 | 2.1 | 0.7879 |  | 0.3879 | 0.3636 | 0.4309 | 0.4141 |  |  | 0.5256 | 44542 | 182.4 | 9581 | 19 | 0 |
| subquery |  | 2500 | 600 | bge-small-en-v1.5 | 384 | BAAI/bge-reranker-base | 99 | 2.1 | 0.5859 |  | 0.3444 | 0.3232 | 0.3688 | 0.3535 |  |  | 0.6034 | 44542 | 68.4 | 10938 | 16 | 905 |
| subquery |  | 2500 | 600 | bge-small-en-v1.5 | 384 | BAAI/bge-reranker-v2-m3 | 99 | 2.1 | 0.5859 |  | 0.3327 | 0.3232 | 0.3538 | 0.3535 |  |  | 0.6034 | 44542 | 68.4 | 11068 | 16 | 5612 |
| subquery |  | 2500 | 600 | bge-small-en-v1.5 | 384 | none (RRF order) | 99 | 2.1 | 0.5859 |  | 0.235 | 0.202 | 0.2473 | 0.2121 |  |  | 0.3621 | 44542 | 68.4 | 9239 | 16 | 0 |
| subquery |  | 2500 | 600 | finance-embeddings-investopedia | 768 | BAAI/bge-reranker-base | 99 | 2.1 | 0.5556 |  | 0.3348 | 0.3131 | 0.3518 | 0.3333 |  |  | 0.6 | 44542 | 136.8 | 10815 | 14 | 691 |
| subquery |  | 2500 | 600 | finance-embeddings-investopedia | 768 | BAAI/bge-reranker-v2-m3 | 99 | 2.1 | 0.5556 |  | 0.3303 | 0.3131 | 0.3699 | 0.3636 |  |  | 0.6545 | 44542 | 136.8 | 11123 | 14 | 4209 |
| subquery |  | 2500 | 600 | finance-embeddings-investopedia | 768 | none (RRF order) | 99 | 2.1 | 0.5556 |  | 0.191 | 0.1616 | 0.2083 | 0.1717 |  |  | 0.3091 | 44542 | 136.8 | 8442 | 14 | 0 |
| subquery |  | 2500 | 600 | snowflake-arctic-embed-l-v2.0 | 1024 | BAAI/bge-reranker-base | 99 | 2.1 | 0.6263 |  | 0.3332 | 0.3131 | 0.3536 | 0.3333 |  |  | 0.5323 | 44542 | 182.4 | 10817 | 19 | 741 |
| subquery |  | 2500 | 600 | snowflake-arctic-embed-l-v2.0 | 1024 | BAAI/bge-reranker-v2-m3 | 99 | 2.1 | 0.6263 |  | 0.3783 | 0.3535 | 0.4073 | 0.4141 |  |  | 0.6613 | 44542 | 182.4 | 11117 | 19 | 4206 |
| subquery |  | 2500 | 600 | snowflake-arctic-embed-l-v2.0 | 1024 | cross-encoder/ms-marco-MiniLM-L12-v2 | 99 | 2.1 | 0.6263 |  | 0.2786 | 0.2626 | 0.2884 | 0.2727 |  |  | 0.4355 | 44542 | 182.4 | 10084 | 19 | 267 |
| subquery |  | 2500 | 600 | snowflake-arctic-embed-l-v2.0 | 1024 | mixedbread-ai/mxbai-rerank-base-v2 | 99 | 2.1 | 0.6263 |  | 0.3073 | 0.2929 | 0.3277 | 0.3131 |  |  | 0.5 | 44542 | 182.4 | 9800 | 19 | 1811 |
| subquery |  | 2500 | 600 | snowflake-arctic-embed-l-v2.0 | 1024 | none (RRF order) | 99 | 2.1 | 0.6263 |  | 0.2448 | 0.2121 | 0.2458 | 0.2222 |  |  | 0.3548 | 44542 | 182.4 | 9952 | 19 | 0 |
| subquery |  | 2500 | 600 | e5-large-v2 | 1024 | BAAI/bge-reranker-base | 99 | 2.1 | 0.7475 |  | 0.3467 | 0.3232 | 0.367 | 0.3434 |  |  | 0.4595 | 44542 | 182.4 | 10712 | 20 | 740 |
| subquery |  | 2500 | 600 | e5-large-v2 | 1024 | BAAI/bge-reranker-base | 99 | 2.1 | 0.7374 |  | 0.3346 | 0.3131 | 0.3697 | 0.3535 |  |  | 0.4795 | 44542 | 182.4 | 10993 | 22 | 649 |
| subquery |  | 2500 | 600 | e5-large-v2 | 1024 | BAAI/bge-reranker-v2-m3 | 99 | 2.1 | 0.7475 |  | 0.393 | 0.3737 | 0.4217 | 0.4242 |  |  | 0.5676 | 44542 | 182.4 | 11117 | 20 | 4225 |
| subquery |  | 2500 | 600 | e5-large-v2 | 1024 | BAAI/bge-reranker-v2-m3 | 99 | 2.1 | 0.7374 |  | 0.5108 | 0.5051 | 0.5443 | 0.5354 |  |  | 0.726 | 44542 | 182.4 | 10220 | 22 | 4114 |
| subquery |  | 2500 | 600 | e5-large-v2 | 1024 | cross-encoder/ms-marco-MiniLM-L12-v2 | 99 | 2.1 | 0.7475 |  | 0.2894 | 0.2727 | 0.2988 | 0.2828 |  |  | 0.3784 | 44542 | 182.4 | 10110 | 20 | 344 |
| subquery |  | 2500 | 600 | e5-large-v2 | 1024 | cross-encoder/ms-marco-MiniLM-L12-v2 | 99 | 2.1 | 0.7374 |  | 0.3357 | 0.3131 | 0.3683 | 0.3535 |  |  | 0.4795 | 44542 | 182.4 | 9265 | 22 | 241 |
| subquery |  | 2500 | 600 | e5-large-v2 | 1024 | mixedbread-ai/mxbai-rerank-base-v2 | 99 | 2.1 | 0.7475 |  | 0.2964 | 0.2828 | 0.3489 | 0.3434 |  |  | 0.4595 | 44542 | 182.4 | 9424 | 20 | 2246 |
| subquery |  | 2500 | 600 | e5-large-v2 | 1024 | mixedbread-ai/mxbai-rerank-base-v2 | 99 | 2.1 | 0.7374 |  | 0.2593 | 0.2424 | 0.2765 | 0.2525 |  |  | 0.3425 | 44542 | 182.4 | 8633 | 22 | 1655 |
| subquery |  | 2500 | 600 | e5-large-v2 | 1024 | none (RRF order) | 99 | 2.1 | 0.7475 |  | 0.3028 | 0.2828 | 0.3176 | 0.2929 |  |  | 0.3919 | 44542 | 182.4 | 9953 | 20 | 0 |
| subquery |  | 2500 | 600 | e5-large-v2 | 1024 | none (RRF order) | 99 | 2.1 | 0.7374 |  | 0.3092 | 0.2929 | 0.3189 | 0.303 |  |  | 0.411 | 44542 | 182.4 | 9534 | 22 | 0 |
| subquery |  | 2500 | 600 | mxbai-embed-large-v1 | 1024 | BAAI/bge-reranker-base | 99 | 2.1 | 0.6667 |  | 0.3535 | 0.3333 | 0.374 | 0.3434 |  |  | 0.5152 | 44542 | 182.4 | 10779 | 26 | 936 |
| subquery |  | 2500 | 600 | mxbai-embed-large-v1 | 1024 | BAAI/bge-reranker-v2-m3 | 99 | 2.1 | 0.6667 |  | 0.385 | 0.3535 | 0.4041 | 0.3939 |  |  | 0.5909 | 44542 | 182.4 | 11075 | 26 | 7606 |
| subquery |  | 2500 | 600 | mxbai-embed-large-v1 | 1024 | cross-encoder/ms-marco-MiniLM-L12-v2 | 99 | 2.1 | 0.6667 |  | 0.2795 | 0.2626 | 0.2889 | 0.2727 |  |  | 0.4091 | 44542 | 182.4 | 10022 | 26 | 257 |
| subquery |  | 2500 | 600 | mxbai-embed-large-v1 | 1024 | mixedbread-ai/mxbai-rerank-base-v2 | 99 | 2.1 | 0.6667 |  | 0.331 | 0.303 | 0.3541 | 0.3232 |  |  | 0.4848 | 44542 | 182.4 | 9502 | 26 | 1721 |
| subquery |  | 2500 | 600 | mxbai-embed-large-v1 | 1024 | none (RRF order) | 99 | 2.1 | 0.6667 |  | 0.2902 | 0.2525 | 0.3065 | 0.2727 |  |  | 0.4091 | 44542 | 182.4 | 9687 | 26 | 0 |
| subquery |  | 2500 | 600 | nomic-embed-text-v1.5 | 768 | BAAI/bge-reranker-base | 99 | 2.1 | 0.5051 |  | 0.3027 | 0.2727 | 0.3132 | 0.2828 |  |  | 0.56 | 43406 | 133.3 | 10744 | 25 | 949 |
| subquery |  | 2500 | 600 | nomic-embed-text-v1.5 | 768 | BAAI/bge-reranker-v2-m3 | 99 | 2.1 | 0.5051 |  | 0.3463 | 0.3232 | 0.3693 | 0.3737 |  |  | 0.74 | 43406 | 133.3 | 10800 | 25 | 5220 |
| subquery |  | 2500 | 600 | nomic-embed-text-v1.5 | 768 | cross-encoder/ms-marco-MiniLM-L12-v2 | 99 | 2.1 | 0.5051 |  | 0.2573 | 0.2424 | 0.2673 | 0.2525 |  |  | 0.5 | 43406 | 133.3 | 9867 | 25 | 269 |
| subquery |  | 2500 | 600 | nomic-embed-text-v1.5 | 768 | mixedbread-ai/mxbai-rerank-base-v2 | 99 | 2.1 | 0.5051 |  | 0.2582 | 0.2323 | 0.2782 | 0.2525 |  |  | 0.5 | 43406 | 133.3 | 9433 | 25 | 1803 |
| subquery |  | 2500 | 600 | nomic-embed-text-v1.5 | 768 | none (RRF order) | 99 | 2.1 | 0.5051 |  | 0.1729 | 0.1414 | 0.1897 | 0.1515 |  |  | 0.3 | 43406 | 133.3 | 9878 | 25 | 0 |
| subquery |  | 2500 | 900 | bge-large-en-v1.5 | 1024 | BAAI/bge-reranker-base | 99 | 2.1 | 0.7071 |  | 0.339 | 0.3232 | 0.3607 | 0.3434 |  |  | 0.4857 | 33483 | 137.1 | 10707 | 19 | 734 |
| subquery |  | 2500 | 900 | bge-large-en-v1.5 | 1024 | BAAI/bge-reranker-v2-m3 | 99 | 2.1 | 0.7071 |  | 0.3403 | 0.3333 | 0.3661 | 0.3636 |  |  | 0.5143 | 33483 | 137.1 | 11266 | 19 | 5389 |
| subquery |  | 2500 | 900 | bge-large-en-v1.5 | 1024 | none (RRF order) | 99 | 2.1 | 0.7071 |  | 0.3063 | 0.2626 | 0.3126 | 0.2828 |  |  | 0.4 | 33483 | 137.1 | 9866 | 19 | 0 |
| subquery |  | 4000 | 600 | bge-large-en-v1.5 | 1024 | BAAI/bge-reranker-base | 99 | 2.1 | 0.7576 |  | 0.3448 | 0.3232 | 0.3613 | 0.3333 |  |  | 0.44 | 44071 | 180.5 | 14885 | 26 | 902 |
| subquery |  | 4000 | 600 | bge-large-en-v1.5 | 1024 | BAAI/bge-reranker-v2-m3 | 99 | 2.1 | 0.7576 |  | 0.3294 | 0.3232 | 0.3787 | 0.3838 |  |  | 0.5067 | 44071 | 180.5 | 16359 | 26 | 5833 |
| subquery |  | 4000 | 600 | bge-large-en-v1.5 | 1024 | none (RRF order) | 99 | 2.1 | 0.7576 |  | 0.3101 | 0.2626 | 0.3256 | 0.2828 |  |  | 0.3733 | 44071 | 180.5 | 13988 | 26 | 0 |

## By question type (top-8, production reranker)

| config | qtype | n | cov@8 | hit@8 |
|---|---|---|---|---|
| p2500_c600_bge-large-en-v1.5_hdr_tbl-md | comparison | 12 | 0.5847 | 0.5833 |
| p2500_c600_bge-large-en-v1.5_hdr_tbl-md | narrative | 27 | 0.702 | 0.7407 |
| p2500_c600_bge-large-en-v1.5_hdr_tbl-md | numeric | 60 | 0.7245 | 0.7667 |
| p2500_c600_bge-large-en-v1.5_hdr_tbl-pipe | comparison | 12 | 0.6252 | 0.6667 |
| p2500_c600_bge-large-en-v1.5_hdr_tbl-pipe | narrative | 27 | 0.6989 | 0.7407 |
| p2500_c600_bge-large-en-v1.5_hdr_tbl-pipe | numeric | 60 | 0.7287 | 0.7667 |
| p1500_c600_bge-large-en-v1.5 | comparison | 12 | 0.3114 | 0.25 |
| p1500_c600_bge-large-en-v1.5 | narrative | 27 | 0.3255 | 0.3333 |
| p1500_c600_bge-large-en-v1.5 | numeric | 60 | 0.3119 | 0.3 |
| p2500_c300_bge-large-en-v1.5 | comparison | 12 | 0.4112 | 0.4167 |
| p2500_c300_bge-large-en-v1.5 | narrative | 27 | 0.4658 | 0.4444 |
| p2500_c300_bge-large-en-v1.5 | numeric | 60 | 0.3438 | 0.3333 |
| p2500_c600_bge-base-en-v1.5 | comparison | 12 | 0.375 | 0.4167 |
| p2500_c600_bge-base-en-v1.5 | narrative | 27 | 0.473 | 0.4815 |
| p2500_c600_bge-base-en-v1.5 | numeric | 60 | 0.3443 | 0.3333 |
| p2500_c600_bge-large-en-v1.5 | comparison | 12 | 0.3714 | 0.3333 |
| p2500_c600_bge-large-en-v1.5 | narrative | 27 | 0.5106 | 0.5556 |
| p2500_c600_bge-large-en-v1.5 | numeric | 60 | 0.369 | 0.35 |
| p2500_c600_bge-large-en-v1.5_hdr | comparison | 12 | 0.5719 | 0.5 |
| p2500_c600_bge-large-en-v1.5_hdr | narrative | 27 | 0.6544 | 0.6667 |
| p2500_c600_bge-large-en-v1.5_hdr | numeric | 60 | 0.5926 | 0.5833 |
| p2500_c600_bge-small-en-v1.5 | comparison | 12 | 0.4146 | 0.5 |
| p2500_c600_bge-small-en-v1.5 | narrative | 27 | 0.4136 | 0.4074 |
| p2500_c600_bge-small-en-v1.5 | numeric | 60 | 0.3146 | 0.3 |
| p2500_c600_finance-embeddings-investopedia | comparison | 12 | 0.412 | 0.4167 |
| p2500_c600_finance-embeddings-investopedia | narrative | 27 | 0.3967 | 0.4074 |
| p2500_c600_finance-embeddings-investopedia | numeric | 60 | 0.3495 | 0.3333 |
| p2500_c600_snowflake-arctic-embed-l-v2.0 | comparison | 12 | 0.3714 | 0.3333 |
| p2500_c600_snowflake-arctic-embed-l-v2.0 | narrative | 27 | 0.5106 | 0.5556 |
| p2500_c600_snowflake-arctic-embed-l-v2.0 | numeric | 60 | 0.368 | 0.3667 |
| p2500_c600_e5-large-v2 | comparison | 12 | 0.3719 | 0.3333 |
| p2500_c600_e5-large-v2 | narrative | 27 | 0.5285 | 0.5556 |
| p2500_c600_e5-large-v2 | numeric | 60 | 0.3836 | 0.3833 |
| p2500_c600_e5-large-v2_hdr | comparison | 12 | 0.5714 | 0.5 |
| p2500_c600_e5-large-v2_hdr | narrative | 27 | 0.6356 | 0.6667 |
| p2500_c600_e5-large-v2_hdr | numeric | 60 | 0.4979 | 0.4833 |
| p2500_c600_mxbai-embed-large-v1 | comparison | 12 | 0.3714 | 0.3333 |
| p2500_c600_mxbai-embed-large-v1 | narrative | 27 | 0.5106 | 0.5556 |
| p2500_c600_mxbai-embed-large-v1 | numeric | 60 | 0.3628 | 0.3333 |
| p2500_c600_nomic-embed-text-v1.5 | comparison | 12 | 0.3322 | 0.3333 |
| p2500_c600_nomic-embed-text-v1.5 | narrative | 27 | 0.468 | 0.4815 |
| p2500_c600_nomic-embed-text-v1.5 | numeric | 60 | 0.3324 | 0.3333 |
| p2500_c900_bge-large-en-v1.5 | comparison | 12 | 0.335 | 0.3333 |
| p2500_c900_bge-large-en-v1.5 | narrative | 27 | 0.4386 | 0.4444 |
| p2500_c900_bge-large-en-v1.5 | numeric | 60 | 0.3397 | 0.3333 |
| p4000_c600_bge-large-en-v1.5 | comparison | 12 | 0.3757 | 0.4167 |
| p4000_c600_bge-large-en-v1.5 | narrative | 27 | 0.4513 | 0.4815 |
| p4000_c600_bge-large-en-v1.5 | numeric | 60 | 0.3467 | 0.3333 |
