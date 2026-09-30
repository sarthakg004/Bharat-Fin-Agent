# Retrieval evaluation

Collection `sweep_p2500_c600_gemini-embedding-2_hdr_tbl-md`, embedder `gemini-embedding-2`, planner `qwen/qwen3.8-27b`. 10 questions scored; 28 dropped because the HTML parser never recovers their evidence.

| reranker | n | pool_recall | cov@5 | hit@5 | cov@8 | hit@8 | num@8 | mrr | retention | ms_per_question |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 10 | 0.9 | 0.547 | 0.6 | 0.6876 | 0.7 | 0.8435 | 0.3093 | 0.7778 | 22 |
| cohere:rerank-v4.0-pro | 10 | 0.9 | 0.7787 | 0.7 | 0.8876 | 0.9 | 0.9244 | 0.5458 | 1.0 | 2563 |
| BAAI/bge-reranker-v2-m3 | 10 | 0.9 | 0.3942 | 0.4 | 0.6744 | 0.6 | 0.806 | 0.2001 | 0.6667 | 9303 |

hit@8 by question type:

- `none`: {'comparison': '1/1', 'narrative': '1/2', 'numeric': '5/7'}
- `cohere:rerank-v4.0-pro`: {'comparison': '1/1', 'narrative': '2/2', 'numeric': '6/7'}
- `BAAI/bge-reranker-v2-m3`: {'comparison': '0/1', 'narrative': '2/2', 'numeric': '4/7'}
