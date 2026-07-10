# Parsing evaluation — Docling (upload parser) vs pypdf (baseline)

3 FinanceBench PDFs · evidence recall = fraction of each gold passage's 10-word shingles found in the parsed text.

| parser | evidence_recall | chars_per_page | page_coverage | n_chunks | parse_s | tables/doc |
|---|---|---|---|---|---|---|
| pypdf | 1.0 | 3506.3333 | 0.9982 | 1031 | 14.0433 | — |
| docling | 0.9244 | 4732.4333 | 0.9982 | 1344.6667 | 88.84 | 119.3 |
