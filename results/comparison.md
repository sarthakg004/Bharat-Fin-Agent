# FinAgent — FinanceBench metrics: before vs. after

**Baseline (v1)** = `results/baseline_metrics.json` · **v3** = `results/v3/final_metrics.json`

Higher is better for every row except *refusal rate* (an explicit abstention is healthier than a confident wrong answer, so read its delta in context, not as a pure regression).

## RAGAS (overall)

| metric | Baseline (v1) | v3 | Δ |
|---|---|---|---|
| faithfulness | 0.5901 | 0.5724 | ▼ -0.0177 |
| answer_relevancy | 0.4112 | 0.4494 | ▲ +0.0382 |
| context_precision | 0.3857 | 0.4999 | ▲ +0.1142 |
| context_recall | 0.3361 | 0.4706 | ▲ +0.1345 |

## Correctness (deterministic, judge-free)

| metric | Baseline (v1) | v3 | Δ |
|---|---|---|---|
| numeric_accuracy (1% tol) | 0.6119 | 0.7463 | ▲ +0.1344 |

## System behaviour

| metric | Baseline (v1) | v3 | Δ |
|---|---|---|---|
| answer_rate | 1.0000 | 0.7467 | ▼ -0.2533 |
| refusal_rate | 0.0000 | 0.2533 | ▲ +0.2533 |
| mean_confidence | 0.6252 | 0.6423 | ▲ +0.0171 |

## RAGAS faithfulness by question type

| qtype | Baseline (v1) | v3 | Δ |
|---|---|---|---|
| comparison | 0.7021 | 0.5290 | ▼ -0.1731 |
| narrative | 0.4744 | 0.5051 | ▲ +0.0307 |
| numeric | 0.6483 | 0.6349 | ▼ -0.0134 |
