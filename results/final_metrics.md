# FinAgent — final evaluation metrics

Outputs: `results/financebench_full_outputs.json` · 150 questions

## System behaviour

| metric | value |
|---|---|
| answer rate | 1.0 |
| refusal rate | 0.0 |
| error rate | 0.0 |
| mean confidence | 0.6252 |

## RAGAS (overall)

| metric | score |
|---|---|
| faithfulness | 0.594 |
| answer_relevancy | 0.4081 |
| context_precision | 0.4654 |
| context_recall | 0.3537 |

## RAGAS by question type

| qtype | answer_relevancy | context_precision | context_recall | faithfulness |
|---|---|---|---|---|
| comparison | 0.4457 | 0.3462 | 0.4211 | 0.7232 |
| narrative | 0.278 | 0.3204 | 0.2663 | 0.4507 |
| numeric | 0.499 | 0.572 | 0.3902 | 0.6448 |

## Coverage by question type

| qtype | questions | answered | refused | errors |
|---|---|---|---|---|
| comparison | 22 | 22 | 0 | 0 |
| narrative | 57 | 57 | 0 | 0 |
| numeric | 71 | 71 | 0 | 0 |
