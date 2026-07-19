# Chunk-strategy ablation — single-chunk gold containment

12 FinanceBench docs · 29 gold passages · containment = fraction of gold passages that fit inside one chunk.

| strategy | containment | mean chunks/doc |
|---|---|---|
| fixed_500 | 0.0417 | 1595.2 |
| fixed_1000 | 0.4653 | 834.2 |
| fixed_1500 | 0.741 | 565.1 |
| semantic | 0.2458 | 621.1 |
| parent_doc | 0.8007 | 328.8 |

Notes: `parent_doc` = containment of a 2500-char parent window (children are matched, the parent is returned). `semantic` is a dependency-free line-merge proxy (SemanticChunker isn't installed) — it under-represents true embedding-based semantic chunking; read it as a floor, not a verdict.

Caveat: containment measures whether gold evidence fits in ONE chunk. Larger chunks raise it but exceed bge-small's ~2048-char (512-token) embed window and dilute retrieval precision — the tension parent-document retrieval resolves (embed small children, return the large parent).
