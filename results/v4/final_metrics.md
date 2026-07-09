# FinAgent — final evaluation metrics

Outputs: `results/v4/financebench_full_outputs.json` · 150 questions

## System behaviour

| metric | value |
|---|---|
| answer rate | 0.6933 |
| refusal rate | 0.3067 |
| error rate | 0.0 |
| mean confidence | 0.5742 |
| numeric accuracy (gold in answer, 1% tol) | 0.7164 |

## Latency & tokens

| metric | value |
|---|---|
| latency mean / p50 / p95 (s) | 60.92 / 43.81 / 157.66 |
| tokens mean per question / total | 14,314 / 2,147,064 |

## Numeric accuracy (deterministic, judge-free)

| qtype | correct | n | accuracy |
|---|---|---|---|
| numeric | 48 | 67 | 0.7164 |
| **overall** | 48 | 67 | 0.7164 |


## Coverage by question type

| qtype | questions | answered | refused | errors |
|---|---|---|---|---|
| comparison | 22 | 15 | 7 | 0 |
| narrative | 57 | 33 | 24 | 0 |
| numeric | 71 | 56 | 15 | 0 |
