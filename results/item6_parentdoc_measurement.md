# Section-metadata (item 6) + parent-document retrieval — rebuild measurement

Collections rebuilt with parent-document chunking (small ~600-char children
embedded, ~1500-char parent returned) + 10-K `item` section tags:

- **us_filings** (production): 34,354 → **71,127** children. Deployed.
- **financebench_eval** (eval-only, temp dir): **105,838** children. Not deployed.

## Item 6 — section filter A/B (same rebuilt financebench_eval collection)

| pool_recall | @20 | @50 | @100 |
|---|---|---|---|
| section filter **ON** | 0.2733 | 0.5000 | 0.6533 |
| section filter **OFF** | 0.2733 | 0.5000 | 0.6533 |

**Identical** — because FinanceBench questions are numeric ("What was X's FY2022
COGS?") and essentially never name a 10-K section, so `infer_filter` adds no
`item` clause for them. The section filter is therefore **neutral on this
benchmark (no regression)** and only engages on section-language questions,
which FinanceBench doesn't contain.

It demonstrably works where it's meant to: on `us_filings`, the query *"What are
the principal risk factors Apple discloses in its 10-K?"* narrows the pool to
`item=1A` (Risk Factors) chunks and returns them — the targeted narrative-recall
lever the feature exists for.

## Parent-document retrieval

The child-level recall numbers above are **not a fair lens** for parent-doc: the
eval scores whether the exact small gold *child* is retrieved, which penalises
smaller chunks without crediting the larger *parent* that production actually
returns. The real parent-doc signal is the containment win already measured in
`chunk_ablation.md` (single-chunk gold containment **0.80 at a ~2500-char parent
vs 0.47 at 1000**). End-to-end: production matches on children, collapses to
parents, and hands synthesis the larger, evidence-complete parent.

Validated mechanically on the deployed `us_filings`: retrieval returns
parent-sized passages (855–1408 chars, above the 600 child cap) tagged with the
correct 10-K item.

**Follow-up:** a controlled parent-level A/B (score whether the retrieved
child's *parent_text* contains the gold evidence) and an end-to-end answer eval
would quantify the production gain beyond the containment proxy.
