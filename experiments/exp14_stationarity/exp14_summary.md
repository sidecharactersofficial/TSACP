# EXP14 — Stationarity (P0-5) + CICIDS Multiseed (Opt-B) (real data)

Run: `python -m experiments.exp14_stationarity.run all`, exit 0.
Metrics: `results/logs/exp14_stationarity/metrics_{unsw_nb15,cicids2017}.json`;
CICIDS multiseed table lands in
`results/logs/exp9_blackbox_cicids2017/multiseed.json` (per exp9 convention).

## Stationarity sweep (session probe on persistent-direction trajectories)

γ ∈ {5,10,25,50,100,200}, Δθ measured, ν_session = verdict-flip rate of a
fixed benign boundary probe evaluated through the *advancing* trajectory,
ν_final at steady state, undefended control, sin(Δθ/2) predicted width.

**UNSW-NB15** (benign boundary shard, steady ν_final caps at 10.55%):

| γ | Δθ | ν_session | ν_final | ctrl | pred halfwidth |
|---|---:|---:|---:|---:|---:|
| 5 | 0.0355 | 0.36% | 1.95% | 0.00% | 0.0030 |
| 10 | 0.0710 | 0.46% | 2.73% | 0.00% | 0.0060 |
| 25 | 0.1775 | 1.17% | 5.47% | 0.00% | 0.0149 |
| 50 | 0.3550 | 2.41% | 10.55% | 0.00% | 0.0297 |
| 100 | 0.7100 | 5.37% | 10.55% | 0.00% | 0.0580 |
| 200 | 1.2835 | 8.89% | 10.55% | 0.00% | 0.0991 |

ν grows monotonically with Δθ; control ν ≡ 0 at every γ. sin(Δθ/2) width
scaling: mean relative error 30.6% over 5 ratios — the sin-quadratic
prediction is the right shape (ν rises with γ, plateaus via max_theta
saturation) but is not a tight fit: the measured flip width saturates at
the probe's finite decision-bandwidth rather than continuing quadratically.

**CICIDS2017** (ν_final caps at 35.2%; decision boundary much closer to the
benign boundary shard):

| γ | Δθ | ν_session | ν_final | ctrl | pred halfwidth |
|---|---:|---:|---:|---:|---:|
| 5 | 0.0324 | 8.53% | 50.39% | 0.00% | 0.0038 |
| 10 | 0.0648 | 12.63% | 48.83% | 0.00% | 0.0075 |
| 25 | 0.1621 | 16.47% | 42.58% | 0.00% | 0.0187 |
| 50 | 0.3241 | 15.95% | 35.16% | 0.00% | 0.0373 |
| 100 | 0.6482 | 14.39% | 35.16% | 0.00% | 0.0731 |
| 200 | 1.1992 | 13.15% | 35.16% | 0.00% | 0.1284 |

CICIDS: ν_session peaks at γ=25 (16.5%) then *decreases* with γ — the
sin-scaling prediction fails beyond γ ≈ 25 (mean relative error 71.1%).
Honest reading: ν(Δθ) is non-monotone on real decision geometry; the
sin(Δθ/2) width ∝ scaling is a valid small-angle approximation only, and
the paper should present Prop 5's scaling as the small-Δθ regime.

Controls: undefended probe verdicts NEVER flip (ν ≡ 0 at all γ on both
datasets), so the entire ν budget is attributable to the rotation state —
Prop 5's attribution claim holds exactly.

## CICIDS multi-seed black-box (Opt-B; seeds 42/43/44, 2000 samples each)

| condition | pooled ASR | Wilson 95% CI |
|---|---:|---:|
| NES / Undefended | 20.83% | [19.82, 21.88] |
| NES / TASCP (Naive) | 20.75% | [19.74, 21.79] |
| NES / Undefended + Domain | 20.20% | [19.20, 21.23] |
| HSJ / Undefended | 15.28% | [14.40, 16.22] |
| HSJ / TASCP (Naive) | 88.15% | [87.31, 88.94] |

Replicates exp9 on 3 seeds with disjoint CIs: NES/undefended vs NES/TASCP
CIs overlap (no black-box NES effect — Corollary 3), HSJ/TASCP massively
disjoint (+72.9pp — the amplification). Seed-level spread is small (e.g.
HSJ TASCP 87.40/88.25/88.80%), so the amplification is not a seed artifact.
