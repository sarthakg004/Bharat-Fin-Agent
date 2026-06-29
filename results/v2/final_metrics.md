# FinAgent — final evaluation metrics

Outputs: `results/v2/financebench_full_outputs.json` · 150 questions

## System behaviour

| metric | value |
|---|---|
| answer rate | 0.94 |
| refusal rate | 0.06 |
| error rate | 0.0 |
| mean confidence | 0.8547 |
| numeric accuracy (gold in answer, 1% tol) | 0.7313 |

## Numeric accuracy (deterministic, judge-free)

| qtype | correct | n | accuracy |
|---|---|---|---|
| numeric | 49 | 67 | 0.7313 |
| **overall** | 49 | 67 | 0.7313 |

## RAGAS (overall)

| metric | score |
|---|---|
| faithfulness | 0.586 |
| answer_relevancy | 0.4264 |
| context_precision | 0.3611 |
| context_recall | 0.3967 |

## RAGAS by question type

| qtype | answer_relevancy | context_precision | context_recall | faithfulness |
|---|---|---|---|---|
| comparison | 0.3834 | 0.1778 | 0.4545 | 0.6127 |
| narrative | 0.377 | 0.2968 | 0.2412 | 0.4553 |
| numeric | 0.4793 | 0.4492 | 0.5035 | 0.6826 |

## Coverage by question type

| qtype | questions | answered | refused | errors |
|---|---|---|---|---|
| comparison | 22 | 22 | 0 | 0 |
| narrative | 57 | 49 | 8 | 0 |
| numeric | 71 | 70 | 1 | 0 |
