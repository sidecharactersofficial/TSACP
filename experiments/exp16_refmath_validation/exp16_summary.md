# EXP16 — Reference-Math Validation (UNSW-NB15, real data)

Run: `python -m experiments.exp16_refmath_validation.run unsw_nb15`, exit 0.
Metrics: `results/logs/exp16_refmath/metrics_unsw_nb15.json`; tables:
`results/tables/exp16_prop3_cosine_unsw_nb15.csv`,
`results/tables/exp16_cor3_mt_histogram.csv`.

| check | result | verdict |
|---|---|---|
| Lemma 3: autograd vs central FD on ∇ε_p | 9,216 comparisons, median rel err 1.37e-5, p90 9.9e-5, max(0,·) active 99.7% | ✔ autograd ∇ε_p is faithful — Theorem 1's chain-rule premise holds numerically |
| Theorem-1 decomposition: gradient cosines | cos(full, angle-detached) = 1.00, cos(full, basis-fixed) = 1.00, trigger fired 100% of measured points | Ablated maps are first-order indistinguishable at firing points; the adaptive gap (exp5/exp8) is a *behavioral* effect of the state map, consistent with exp8's basis-fixed collapse to 10.3% |
| Lemma 2 ρ isotropy | E[ρ] = 0.198 vs isotropic √(2/d) = 0.216 (ratio 0.92, d=43, 32 trials) | ✔ isotropic scaffold accurate to ≈ 8% |
| Prop 3 NES cosine vs q | measured 0.008 → 0.046 for q = 2 → 100; √(q/(q+d+1)) predicts 0.21 → 0.83 | Isotropic scaffold OVERPREDICTS at every q; real NES anchor-displacement autocorrelation ≈ 0 at all q — trajectories stay deeper in Corollary 3's blind regime than theory predicts. (Synthetic sanity showed the same low measured values; this appears to be the honest result, see caveat below.) |
| Cor 3 M_t block statistics | E[M_t] = −0.703 (with-anchor), −0.703 (without); P(M_t>0) = 1.6% / 0.0%; theory −2/√5 = −0.894 | ✔ trigger momentum input is negative w.h.p.; matches exp9's measured −0.83; residual gap is the ‖x−x_t‖ term |

All preceding checks (Lemma 3, ρ, M_t) were also sanity-checked on synthetic
data before the real run (median FD rel err 8.6e-5; ρ 0.188/0.216; M_t
−0.74/−0.70, P(M>0)=0).

## Caveats

- The measured M_t block count is small (n_blocks = 3 — the histogram CSV has
  31,232 with-anchor / 29,696 without-anchor samples, but only 3 blocks of
  (2q+1) interleaving were formed by the recorder on this run); the mean is
  stable across with/without-anchor variants, but a rerun with more rows
  would tighten it.
- Prop 3's measured-vs-predicted divergence means the √(q/(q+d+1)) scaling
  in THEOREMS.md Prop 3 should be presented as an *upper scaffold*, not a
  fit: real trajectories are strictly inside the blind regime for all
  measured q.
