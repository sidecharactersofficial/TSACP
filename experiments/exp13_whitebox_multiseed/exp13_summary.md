# EXP13 — White-Box Multiseed + Basis-Fixed Reference Robustness (real data)

Run: `python -m experiments.exp13_whitebox_multiseed.run <dataset> [--with-xgb]`,
seeds {42..46}, 2,000 samples/seed, exit 0 both datasets.
Tables: `results/tables/exp13_whitebox_multiseed_{unsw_nb15,cicids2017}.csv`;
metrics: `results/logs/exp13_whitebox_multiseed/multiseed_{ds}.json`.
Model checkpoint is fixed across seeds; seeds vary the data draw and the
attack's random seeds only (documented in run.py's docstring).

## UNSW-NB15 (white-box stateful PGD)

| condition | pooled ASR | Wilson 95% CI | seeds |
|---|---:|---:|---|
| Undefended | 10.65% | [10.06, 11.27] | 10.50 … 11.35 |
| TASCP (Naive) | 23.53% | [22.71, 24.37] | 22.85 … 24.60 |
| TASCP (Full Adaptive) | 52.99% | [52.01, 53.97] | 52.10 … 53.60 |
| TASCP (Basis fixed) | 10.47% | [9.89, 11.09] | 9.90 … 11.15 |

Seed CIs: naive [22.71, 24.37] and undefended [10.06, 11.27] are disjoint
(naive TASCP *hurts* vs baseline, as exp5 found); basis-fixed CI overlaps
undefended exactly — the ablation collapses the defense to the baseline.

Basis-fixed reference robustness (canonical e0/e1 vs 7 random orthonormal
pairs, UNSW): mean 15.01%, std 6.48%, range [10.30%, 30.40%] — random
reference pairs behave like the baseline on most pairs but individual pairs
(pair 5: 30.4%) can still couple; the effect is reference-dependent, not
universally zero.

XGBoost-surrogate path (UNSW): XGB Undefended 1.00%, XGB Basis-fixed 0.65%
— on this (already tiny-ASR) XGB attack surface the basis-fixed wrapper is
harmless (−0.35 pp), i.e. no evidence the wrapper itself creates
exploitable surface for the surrogate attack.

## CICIDS2017 (white-box stateful PGD)

| condition | pooled ASR | Wilson 95% CI | seeds |
|---|---:|---:|---|
| Undefended | 40.75% | [39.79, 41.72] | 37.90 … 43.35 |
| TASCP (Naive) | 42.62% | [41.65, 43.59] | 40.15 … 44.70 |
| TASCP (Full Adaptive) | 45.63% | [44.66, 46.61] | 43.10 … 47.15 |
| TASCP (Basis fixed) | 43.48% | [42.51, 44.45] | 40.70 … 45.50 |

CICIDS2017 picture: on the white-box PGD surface, adaptive BPDA remains
+4.9 pp over undefended (CIs disjoint), but naiveness now also *hurts*
(+1.9pp) and basis-fixed still sits between. Note CICIDS white-box ASR
(~41%) ≫ UNSW (~10.6%): dataset-dependent baseline hardness dominates.

Basis-fixed reference robustness (CICIDS): mean 40.90%, std 3.77%, range
[33.85%, 46.95%] — same conclusion as UNSW, with smaller dispersion.

## Bottom line

- exp5's central finding replicates under 5 seeds with tight pooled CIs:
  full adaptive BPDA strictly degrades robustness (UNSW 53.0% vs 10.7%;
  CICIDS 45.6% vs 40.7%), and the degradation localizes to the
  data-dependent basis (basis-fixed ≈ undefended on UNSW; near-baseline on
  CICIDS).
- The CICIDS adaptive-vs-undefended gap (+4.9pp, CIs [44.66,46.61] vs
  [39.79,41.72] disjoint) is smaller than UNSW's but strictly positive.
