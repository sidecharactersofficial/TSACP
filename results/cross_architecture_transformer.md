# Cross-Architecture Comparison: Transformer (UNSW-NB15)

## Methodology
A 2-layer TabTransformer model (d_model=128) was trained on the same UNSW-NB15 dataset used for the MLP and XGBoost evaluations. The same budget=20 query limit and n=2000 correctly classified benign sample subset were used.
For this experiment, a logit margin loss objective was used to prevent BCE gradient vanishing on overconfident Transformer samples.

## Classifier Quality
| Metric | Transformer |
|---|---|
| Accuracy | 99.14% |
| F1 Score | 0.9658 |

## Attack Results (Budget=20)
| Condition | Transformer ASR | Mean Q |
|---|---|---|
| Undefended | 1.05% | 4.6 |
| TASCP (Naive) | 12.50% | 8.1 |
| TASCP (Full Adaptive) | 7.00% | 7.4 |
| TASCP (Ablation 1: Grad Mag) | 10.30% | 9.1 |
| TASCP (Ablation 2: Basis) | 1.05% | 5.0 |
| TASCP (Ablation 3: Proj Only) | 9.25% | 8.5 |

## Sanity Check
- **Naive vs Adaptive trajectory divergence:** Mean L2 diff = 0.060560
- **Status:** PASS (requires_grad flow is unbroken from $x_t$ through the rotation quantities).
