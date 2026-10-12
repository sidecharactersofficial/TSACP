# Exp 20: Coupling Magnitude vs Adaptive Effect, Across State Functions (RQ2)

## Question
Does coupling magnitude predict exploitable effect? Theorem 1 establishes
the channel `dD/dh * dh/dx != 0` for TASCP's momentum rule; this experiment
generalizes the audit to SIX state functions x FIVE transformations
(30 cells) and tests the reviewer's hypothesis: higher C -> larger adaptive
ASR increase over the fixed-state baseline.

## Method
State functions (`family.STATE_FNS`): momentum (TASCP's eps_p), cosine
similarity of consecutive displacements, displacement magnitude, running
mean direction, EMA deviation, curvature. Transformations: tascp / tscale /
tshift / tshear / tnoise. Per cell: fixed-state control ASR, adaptive B2
replica ASR, coupling C_raw = ||dT/dh||_F and C_norm = C_raw relative to
the per-sample transform displacement (median-normalized, same-batch
geometry — the naive ||T(x)|| denominator is dominated by ||h|| for
large-mean feature spaces and hides the channel), and nu_session.
500 benign points, PGD eps 0.3 / alpha 0.01 / 20 iters.

## Results
- UNSW-NB15: 30/30 cells with an open state channel (C>0). Spearman
  rho(log C_norm, adaptive dASR) = **-0.126** (p=0.51); Pearson r = -0.17
  (p=0.37).
- NF-ToN-IoT-v2: Spearman rho = **+0.086** (p=0.65); Pearson r = +0.27
  (p=0.14).

## Finding (honest — a null result that reshapes the claim)
**Coupling magnitude does NOT predict adaptive exploitability.** What the
data supports:

1. **The channel is binary, not scalar-ranked.** C = 0 exactly identifies
   the closed channel (silent state -> identity transform -> ASR equal to
   the fixed-state control by construction, verified in
   `tests/test_trajectory_family.py`); C > 0 identifies an open channel.
   Within open channels, magnitude does not rank exploitability.
2. **Effect direction is geometry-dependent, not coupling-dependent.**
   Cells with comparable C_norm span both +42.4pp (tshift, amplification)
   and -48.8pp (tshear, suppression) on UNSW. What matters is whether the
   state-conditioned transform moves benign-space candidates toward or
   away from the boundary — a property of the transform's geometry, not of
   the Jacobian norm.
3. Consequence for the paper: Theorem 1's generalization is
   **"nonzero state coupling is a necessary audit signal; geometry of the
   state-conditioned transform determines the failure mode"** — not
   "higher coupling = more exploitable". This is a weaker but defensible
   claim, and it is what the data supports on both datasets.

## Artifacts
- results/tables/exp20_coupling_scatter_{unsw_nb15,ton_iot}.csv (RQ2 scatter data)
- results/logs/exp20_coupling_audit/metrics_{dataset}.json
- results/logs/runs_session5/exp20_{unsw,toniot}.log
