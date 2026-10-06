# Retrieval evaluation

Collection `sweep_p2500_c600_gemini-embedding-2_hdr_tbl-md`, embedder `gemini-embedding-2`, planner `qwen/qwen3.8-27b`. 99 questions scored; 28 dropped because the HTML parser never recovers their evidence.

| reranker | n | pool_recall | cov@5 | hit@5 | cov@8 | hit@8 | num@8 | mrr | retention | ms_per_question |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 99 | 0.9697 | 0.5829 | 0.6162 | 0.7048 | 0.7778 | 0.8082 | 0.4413 | 0.8021 | 19 |
| cohere:rerank-v4.0-pro | 99 | 0.9697 | 0.7867 | 0.8384 | 0.825 | 0.9091 | 0.8937 | 0.6462 | 0.9375 | 2541 |
| BAAI/bge-reranker-v2-m3 | 99 | 0.9697 | 0.6527 | 0.6869 | 0.7522 | 0.8081 | 0.8622 | 0.5012 | 0.8333 | 8401 |

hit@8 by question type:

- `none`: {'comparison': '8/12', 'narrative': '20/27', 'numeric': '49/60'}
- `cohere:rerank-v4.0-pro`: {'comparison': '10/12', 'narrative': '23/27', 'numeric': '57/60'}
- `BAAI/bge-reranker-v2-m3`: {'comparison': '8/12', 'narrative': '20/27', 'numeric': '52/60'}
