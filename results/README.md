# Results

| Path | What it is |
|---|---|
| `answers/` | The latest answer evaluation: answers, RAGAS scores, report |
| `retrieval_eval.md`, `.json` | The latest retrieval evaluation |
| `RETRIEVAL_EXPERIMENTS.md` | Every retrieval experiment, including the ones that lost |
| `financebench_retrieval_queries.json` | Input: the 99 questions whose evidence survives HTML parsing |
| `financebench_plans_*.json` | Input: cached planner output for the retrieval eval |

The raw outputs of earlier runs were removed on 2026-10-02 and stay in git
history (commit `99da65d`). This page keeps their headline numbers and what
changed between them.

## Earlier runs

Each row lists what changed since the run before it. The latest row is the
code in this repository.

| Run | Date | What changed before this run |
|---|---|---|
| 1 | 2026-06-12 | First full run: 150 questions, filings read as PDFs. The agent answered everything. |
| 2 | 2026-06 | XBRL answers fixed for delisted tickers, three missing ratios added, a circuit breaker when every key is spent. |
| 3 | 2026-07-04 | The eval searched a collection holding every FinanceBench filing instead of recent filings only; SEC fetch reaches older years; a non-answer counts as an abstention; four RAGAS scoring bugs fixed. |
| 4 | 2026-07-10 | The confidence gate withheld more drafts (8 more abstentions). Not RAGAS-scored. |
| 5 | 2026-08-08 | Abstention and the confidence score removed. The eval moved from PDFs to SEC HTML (127 questions have an HTML filing). Retrieval rebuilt: company, year and section prefixed to every chunk, and one rewritten search query (evidence in the top 8 went from 37 to 67 of 99). |
| 6 | 2026-08-16 | Gemini embeddings, Cohere reranker, Gemini writer and critic, one critic recovery; the grader, verifier and period guard deleted. 122 of 127 answers scored. |
| latest | 2026-10-02 | Codebase simplified (about 18,000 to 7,500 lines). Numeric route: named quarters, several line items per sub-query, one period per ratio. Judgement questions also search the filings. A draft that admits a missing figure triggers a filing search. Web search only as a fallback. The writer leads with the answer. A yes/no `verdict` metric added. |

| Run | Questions | Answer rate | Figure check | Faithfulness | Groundedness | Relevancy | Context precision | Context recall | Correctness | Verdict | Latency p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 150 | 1.00 | | 0.59 | | 0.41 | 0.39 | 0.34 | | | |
| 2 | 150 | 0.94 | 0.73 | 0.59 | | 0.43 | 0.36 | 0.40 | | | |
| 3 | 150 | 0.75 | 0.75 | 0.57 | | 0.45 | 0.50 | 0.47 | | | |
| 4 | 150 | 0.69 | 0.72 | | | | | | | | 44 s |
| 5 | 127 | 0.96 | 0.82 | 0.76 | 0.93 | 0.66 | 0.52 | 0.66 | | | 22 s |
| 6 | 127 | 0.95 | 0.82 (0.71) | 0.75 | 0.95 | 0.70 | 0.45 | 0.66 | 0.34 | | 79 s |
| latest | 127 | 1.00 | 0.85 | 0.78 | 0.96 | 0.81 | 0.38 | 0.74 | 0.47 | 0.79 | 55 s |

The latest row is on all 127 questions so it lines up with runs 5 and 6. The
project README quotes the latest run without the 15 questions whose filing is
not in the eval index (verdict 0.85).

Read the table with these caveats:

- **Runs 1 to 4 are not comparable with 5 onwards.** They used 150 questions
  and PDF filings; later runs use the 127 questions with an SEC HTML filing.
- **The judge changed.** Runs 1 to 6 were scored by free-tier models (Groq,
  later Gemini); the latest by Claude Sonnet 5. Answer rate, the figure check and
  latency do not depend on the judge.
- **The figure check changed.** Before the latest run it counted year tokens
  as figures, so a non-answer mentioning "FY2024" could match. Run 6 recomputed
  with the current check is 0.71 (46 of 65); the latest is 0.85 (55 of 65).
- **The writer changed.** Runs 1 to 6 used the production writer of the time; the latest
  run used Claude Haiku as writer and critic, because the free tier cannot
  answer 127 questions in a day.
