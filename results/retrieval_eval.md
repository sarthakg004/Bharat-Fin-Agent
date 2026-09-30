# Retrieval evaluation

Collection `sweep_p2500_c600_gemini-embedding-2_hdr_tbl-md`, embedder `gemini-embedding-2`, planner `qwen/qwen3.8-27b`. 99 questions scored; 28 dropped because the HTML parser never recovers their evidence.

| reranker | n | pool_recall | cov@5 | hit@5 | cov@8 | hit@8 | num@8 | mrr | retention | ms_per_question |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 99 | 0.9596 | 0.5787 | 0.6061 | 0.7172 | 0.7879 | 0.8178 | 0.4489 | 0.8211 | 22 |
| cohere:rerank-v4.0-pro | 99 | 0.9596 | 0.7778 | 0.8283 | 0.8129 | 0.899 | 0.886 | 0.6273 | 0.9368 | 2499 |
| BAAI/bge-reranker-v2-m3 | 99 | 0.9596 | 0.6757 | 0.7273 | 0.7659 | 0.8283 | 0.8661 | 0.5295 | 0.8632 | 7901 |

hit@8 by question type:

- `none`: {'comparison': '10/12', 'narrative': '20/27', 'numeric': '48/60'}
- `cohere:rerank-v4.0-pro`: {'comparison': '10/12', 'narrative': '22/27', 'numeric': '57/60'}
- `BAAI/bge-reranker-v2-m3`: {'comparison': '8/12', 'narrative': '21/27', 'numeric': '53/60'}
