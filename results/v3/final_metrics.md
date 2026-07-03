# FinAgent — final evaluation metrics

Outputs: `results/v3/financebench_full_outputs.json` · 150 questions

## System behaviour

| metric | value |
|---|---|
| answer rate | 0.7467 |
| refusal rate | 0.2533 |
| error rate | 0.0 |
| mean confidence | 0.6423 |
| numeric accuracy (gold in answer, 1% tol) | 0.7463 |

## Numeric accuracy (deterministic, judge-free)

| qtype | correct | n | accuracy |
|---|---|---|---|
| numeric | 50 | 67 | 0.7463 |
| **overall** | 50 | 67 | 0.7463 |

## RAGAS (overall)

| metric | score |
|---|---|
| faithfulness | 0.5724 |
| answer_relevancy | 0.4494 |
| context_precision | 0.4999 |
| context_recall | 0.4706 |

## RAGAS by question type

| qtype | answer_relevancy | context_precision | context_recall | faithfulness |
|---|---|---|---|---|
| comparison | 0.4389 | 0.2941 | 0.4091 | 0.529 |
| narrative | 0.356 | 0.3798 | 0.2895 | 0.5051 |
| numeric | 0.5276 | 0.6006 | 0.635 | 0.6349 |

## Coverage by question type

| qtype | questions | answered | refused | errors |
|---|---|---|---|---|
| comparison | 22 | 15 | 7 | 0 |
| narrative | 57 | 35 | 22 | 0 |
| numeric | 71 | 62 | 9 | 0 |
