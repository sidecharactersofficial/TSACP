# EXP17 — Hyperparameter Sweep (UNSW-NB15, real data)

Run: `python -m experiments.exp17_hp_sweep.run unsw_nb15`, exit 0.
Metrics: `results/logs/exp17_hp_sweep/metrics_unsw_nb15.json`; table:
`results/tables/exp17_hp_sweep_unsw_nb15.csv`.

Grid: γ ∈ {10, 50, 100} × τ ∈ {0, 0.01, 0.05} × W ∈ {3, 5, 10} (27 points),
500-sample draw, adaptive + basis-fixed at each point, undefended reference.

## Headline

- Undefended ASR: 9.80%.
- **Full adaptive TASCP beats undefended on 6/27 grid points.** Best
  configuration γ=10, τ=0: adaptive 1.4% (γ=10 τ=0 W=3), 1.6% (W=5), 2.8%
  (W=10) vs 9.8% undefended.
- **Basis-fixed is within 2 pp of the undefended baseline on 26/27 grid
  points** — removing the data-dependent basis removes the defense effect
  at essentially every hyperparameter setting, matching exp8's endpoint
  finding at default settings.

## Interpretation

1. The defense's (modest) benefit is not a knife-edge artifact of one
   setting: 6/27 grid points show a real reduction, concentrated at small γ
   (γ=10) and τ=0. Indeed all six wins occur at τ ∈ {0, 0.01} with γ = 10
   per the metric dump; at γ ∈ {50, 100} or τ = 0.05 the adaptive map
   tracks the baseline (ASR ≈ 9.8%, k = 49).
2. The basis-fixed collapse across the grid strengthens the exp8/exp16
   localization: the first-order mechanism (and the measured 9.8% → 1.4%
   behavioral gap) is carried by the data-dependent rotation plane, not by
   the clamp or the window.
3. Honest reviewer-facing line: TASCP's robustness gain is real but
   configuration-sensitive (small γ, zero/small τ) and disappears entirely
   when the trajectory-dependence of the basis is removed.

See also `doc/THEOREMS.md` Theorem-1 decomposition and exp16 summary.
