# FinAgent — FinanceBench metrics: before vs. after

**Baseline (v1)** = `results/baseline_metrics.json` · **v1** = `results/v1/final_metrics.json`

Higher is better for every row except *refusal rate* (an explicit abstention is healthier than a confident wrong answer, so read its delta in context, not as a pure regression).

## RAGAS (overall)

| metric | Baseline (v1) | v1 | Δ |
|---|---|---|---|
| faithfulness | 0.5901 | 0.5901 | ▬ +0.0000 |
| answer_relevancy | 0.4112 | 0.4112 | ▬ +0.0000 |
| context_precision | 0.3857 | 0.3857 | ▬ +0.0000 |
| context_recall | 0.3361 | 0.3361 | ▬ +0.0000 |

## Correctness (deterministic, judge-free)

| metric | Baseline (v1) | v1 | Δ |
|---|---|---|---|
| numeric_accuracy (1% tol) | 0.6119 | — | — |

## System behaviour

| metric | Baseline (v1) | v1 | Δ |
|---|---|---|---|
| answer_rate | 1.0000 | 1.0000 | ▬ +0.0000 |
| refusal_rate | 0.0000 | 0.0000 | ▬ +0.0000 |
| mean_confidence | 0.6252 | 0.6252 | ▬ +0.0000 |

## RAGAS faithfulness by question type

| qtype | Baseline (v1) | v1 | Δ |
|---|---|---|---|
| comparison | 0.7021 | 0.7021 | ▬ +0.0000 |
| narrative | 0.4744 | 0.4744 | ▬ +0.0000 |
| numeric | 0.6483 | 0.6483 | ▬ +0.0000 |
