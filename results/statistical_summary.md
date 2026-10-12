# Statistical Summary for ASR Metrics (n=2000)

## 95% Wilson Score Intervals

The following intervals are appropriate for binomial proportions with a sample size of 2000. They can be formatted in LaTeX tables as `XX.XX% (95% CI: YY.Y-ZZ.Z%)` or as a footnote `\footnotesize{(95% CI: YY.Y-ZZ.Z%)}` if space is constrained.

| ASR Value | 95% Confidence Interval |
|-----------|-------------------------|
| 0.60%     | 0.60% (95% CI: 0.3-1.0%) |
| 0.85%     | 0.85% (95% CI: 0.5-1.4%) |
| 1.55%     | 1.55% (95% CI: 1.1-2.2%) |
| 1.75%     | 1.75% (95% CI: 1.3-2.4%) |
| 10.30%    | 10.30% (95% CI: 9.0-11.7%) |
| 10.50%    | 10.50% (95% CI: 9.2-11.9%) |
| 22.95%    | 22.95% (95% CI: 21.2-24.8%) |
| 34.75%    | 34.75% (95% CI: 32.7-36.9%) |
| 36.05%    | 36.05% (95% CI: 34.0-38.2%) |
| 41.45%    | 41.45% (95% CI: 39.3-43.6%) |
| 43.20%    | 43.20% (95% CI: 41.0-45.4%) |
| 44.30%    | 44.30% (95% CI: 42.1-46.5%) |
| 46.40%    | 46.40% (95% CI: 44.2-48.6%) |
| 53.60%    | 53.60% (95% CI: 51.4-55.8%) |
| 53.85%    | 53.85% (95% CI: 51.7-56.0%) |
| 53.95%    | 53.95% (95% CI: 51.8-56.1%) |
| 56.20%    | 56.20% (95% CI: 54.0-58.4%) |
| 56.95%    | 56.95% (95% CI: 54.8-59.1%) |
| 58.80%    | 58.80% (95% CI: 56.6-60.9%) |
| 59.90%    | 59.90% (95% CI: 57.7-62.0%) |
| 61.60%    | 61.60% (95% CI: 59.4-63.7%) |
| 95.70%    | 95.70% (95% CI: 94.7-96.5%) |

## Pairwise Comparisons (Two-Proportion Z-Test)

For claims made in Section 4/8, here are the corresponding significance tests (matching the existing basis-fixed vs undefended methodology).

### UNSW-NB15 (Exp 5 & 8)
- **TASCP Naive vs Undefended:** 22.95% vs 10.50% ($z = 10.549$, $p < 0.0001$)
- **Adaptive vs Naive:** 53.60% vs 22.95% ($z = 19.941$, $p < 0.0001$)
- **Angle-Rounded vs Full-Adaptive:** 95.70% vs 53.60% ($z = 30.604$, $p < 0.0001$)

### CICIDS2017 (Budget=100)
- **TASCP Naive vs Undefended:** 58.80% vs 56.20% ($z = 1.663$, $p = 0.0963$)
- **Adaptive vs Naive:** 53.85% vs 58.80% ($z = -3.156$, $p = 0.0016$)
- **Angle-Rounded vs Full-Adaptive:** 59.90% vs 53.85% ($z = 3.863$, $p = 0.0001$)

### CICIDS2017 (Budget=20)
- **TASCP Naive vs Undefended:** 43.20% vs 41.45% ($z = 1.120$, $p = 0.2627$)
- **Adaptive vs Naive:** 46.40% vs 43.20% ($z = 2.035$, $p = 0.0419$)

### XGBoost Cross-Architecture (Budget=20)
- **TASCP Naive vs Undefended:** 34.75% vs 0.85% ($z = 28.026$, $p < 0.0001$)
- **Adaptive vs Naive:** 1.55% vs 34.75% ($z = -27.239$, $p < 0.0001$)
- **Angle-Rounded vs Full-Adaptive:** 36.05% vs 1.55% ($z = 27.923$, $p < 0.0001$)

*(Note: P-values below 0.0001 are reported as $p < 0.0001$ in accordance with standard statistical reporting.)*
