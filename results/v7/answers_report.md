# Answer evaluation

`results/v7/answers.json`, 127 questions.

| behaviour | value |
|---|---|
| answer rate | 1.0 |
| refusal rate | 0.0 |
| error rate | 0.0 |
| numeric accuracy (gold figure in the answer, 1% tolerance) | 55/65 = 0.8462 |
| verdict agrees with numeric accuracy | 55/65 |
| latency p50 / p95 (s) | 54.91 / 235.53 |

| RAGAS metric | all | rows scored | evidence-recoverable subset | without the 15 questions whose filing is not in the index |
|---|---|---|---|---|
| faithfulness | 0.7826 | 127 | 0.7896 | 0.7683 |
| groundedness | 0.9606 | 127 | 0.9646 | 0.9665 |
| answer_relevancy | 0.814 | 127 | 0.8234 | 0.8158 |
| context_precision | 0.3837 | 127 | 0.4218 | 0.3817 |
| context_recall | 0.7369 | 127 | 0.7694 | 0.7909 |
| answer_correctness | 0.4666 | 127 | 0.4761 | 0.4897 |
| verdict | 0.7874 | 127 | 0.8283 | 0.8482 |

By question type:

| type | faithfulness | groundedness | answer_relevancy | context_precision | context_recall | answer_correctness | verdict |
|---|---|---|---|---|---|---|---|
| comparison | 0.8052 | 0.9412 | 0.84 | 0.2951 | 0.5882 | 0.3846 | 0.8235 |
| narrative | 0.8895 | 0.9359 | 0.8049 | 0.5244 | 0.6261 | 0.4511 | 0.6923 |
| numeric | 0.7184 | 0.9789 | 0.8128 | 0.3277 | 0.8333 | 0.4948 | 0.831 |

`answer_correctness` counts every extra true statement against a one-line gold answer, so a correct but long answer scores about 0.4. Groundedness and faithfulness stay high for an honest non-answer. Read numeric accuracy and context recall next to them.
