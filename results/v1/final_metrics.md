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
| faithfulness | 0.5901 |
| answer_relevancy | 0.4112 |
| context_precision | 0.3857 |
| context_recall | 0.3361 |

## RAGAS by question type

| qtype | answer_relevancy | context_precision | context_recall | faithfulness |
|---|---|---|---|---|
| comparison | 0.483 | 0.3544 | 0.4545 | 0.7021 |
| narrative | 0.29 | 0.238 | 0.2135 | 0.4744 |
| numeric | 0.4862 | 0.4883 | 0.3979 | 0.6483 |

## Coverage by question type

| qtype | questions | answered | refused | errors |
|---|---|---|---|---|
| comparison | 22 | 22 | 0 | 0 |
| narrative | 57 | 57 | 0 | 0 |
| numeric | 71 | 71 | 0 | 0 |
