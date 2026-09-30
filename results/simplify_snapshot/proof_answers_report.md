# Answer evaluation

`results/simplify_snapshot/proof_answers.json`, 2 questions.

| behaviour | value |
|---|---|
| answer rate | 1.0 |
| refusal rate | 0.0 |
| error rate | 0.0 |
| numeric accuracy (gold figure in the answer, 1% tolerance) | 2/2 = 1.0 |
| latency p50 / p95 (s) | 17.93 / 17.93 |

| RAGAS metric | all | rows scored | evidence-recoverable subset |
|---|---|---|---|
| faithfulness | 0.5 | 2 | 0.5 |
| groundedness | 1.0 | 2 | 1.0 |
| answer_relevancy | 0.6996 | 2 | 0.6996 |
| context_precision | 1.0 | 1 | 1.0 |
| context_recall | 0.5 | 2 | 0.5 |
| answer_correctness | 0.5625 | 2 | 0.5625 |

By question type:

| type | faithfulness | groundedness | answer_relevancy | context_precision | context_recall | answer_correctness |
|---|---|---|---|---|---|---|
| numeric | 0.5 | 1.0 | 0.6996 | 1.0 | 0.5 | 0.5625 |

`answer_correctness` counts every extra true statement against a one-line gold answer, so a correct but long answer scores about 0.4. Groundedness and faithfulness stay high for an honest non-answer. Read numeric accuracy and context recall next to them.
