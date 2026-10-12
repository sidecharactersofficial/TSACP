# Experiment 9: Black-Box Query Attacks

Operational-deployment threat model (Threat Model B1, see `doc/THREAT_MODEL.md`):
score-based (NES) and decision-based (HopSkipJump) black-box attacks against
undefended vs. TASCP. No gradients, no weights — query access only.

**Cross-dataset replication (`run_cicids2017.py`):** Corollary 3's
antithetic-blindness prediction replicates exactly on CICIDS2017 (rotation
0.00%, NES ASR unchanged at 22.30%), while HSJ *amplifies* under TASCP there
(15.05% → 87.40%). Multi-seed CIs (`run_multiseed.py`): NES effect nil (CIs
overlap), HSJ friction significant (CIs disjoint). Full numbers in
`exp9_summary.md`.

## Scripts

| Script | Attacks | Description |
|--------|---------|-------------|
| `run.py` | NES + HopSkipJump | 2,000 UNSW-NB15 samples (seed 42); primary conditions + NES budget sweep [100, 250, 500, 1000] |
| `run_cicids2017.py` | NES + HopSkipJump | Cross-dataset replication on CICIDS2017 (streamed loading; 500-sample attack chunks) |
| `run_multiseed.py` | NES + HopSkipJump | Seeds 42/43/44 × primary pairs with Wilson 95% CIs |

## Attack Configuration

| Parameter | NES (score-based) | HSJ (decision-based) |
|-----------|-------------------|----------------------|
| Access | logits (scores) | predicted label only |
| Perturbation budget | l2 ball, epsilon = 0.1 (matches white-box PGD) | unconstrained; distance minimized over time |
| Queries per step | 2q + 1 = 21 (q = 10 antithetic pairs) | N·(b+1) + b + 1 ≈ 119 (N=16, b=6) |
| Default budget | 250 queries/sample | 1000 queries/sample |

## Key Findings (UNSW-NB15, MLP, 2,000 samples)

| Condition | ASR | Mean Q to success | Mean L2 |
|-----------|-----|-------------------|---------|
| NES / Undefended | 2.70% | 178.9 | 0.0995 |
| NES / TASCP (Naive) | 2.60% | 176.9 | 0.0995 |
| NES / Undefended + Domain-Constrained | 0.80% | 63.0 | 0.0958 |
| NES / TASCP (Naive) + Domain-Constrained | 0.80% | 63.0 | 0.0957 |
| HSJ / Undefended | 97.10% | 68.8 | 0.6373 |
| HSJ / TASCP (Naive) | 90.70% | 66.3 | 0.7121 |

NES budget sweep (ASR, Undefended vs TASCP):

```
budget=  100:  0.65% vs 0.65%
budget=  250:  2.70% vs 2.60%
budget=  500:  8.05% vs 7.85%
budget= 1000:  8.85% vs 8.80%
```

1. **TASCP is blind to score-based NES attacks — by construction.** Antithetic
   NES queries alternate direction (x + sigma·u, then x - sigma·u), so consecutive
   trajectory steps have cosine momentum M_t ~= -0.83 (negative), clipped to zero
   by the max(0, M_i) in the penetration integral. Measured over the full query
   trajectory: the Givens rotation triggers on **0.00%** of samples (max
   delta_theta = 0). ASR is statistically identical with and without TASCP
   (2.70% vs 2.60%).
2. **TASCP's trigger is direction-persistence-sensitive.** A persistent-direction
   control (fixed u, growing radius) triggers the rotation on **100%** of samples
   (mean max delta_theta = 0.833). Real decision-based HSJ queries persist long
   enough to trigger on 90.55% of samples at the final query.
3. **TASCP adds friction, not protection, against decision-based attacks.** HSJ
   ASR drops 97.10% -> 90.70% (-6.4pp) and the attacker pays +12% mean L2
   distance (0.637 -> 0.712) — but the attack still succeeds for 9 in 10
   samples within budget.
4. **Black-box attacks are query-expensive.** NES needs ~1,000 queries to match
   what 20-step white-box PGD achieves (10.50% ASR, exp5) — a ~50x query
   overhead. Decision-based HSJ succeeds almost always but at large distances
   (L2 ~ 0.64-0.71) and ~5x the queries.
5. **Domain constraints slash feasible attacks (P2 deliverable).** Snapping the
   16 discrete/binary fields back to observed encoding levels cuts NES ASR from
   2.70% to 0.80% (-70% relative) against the *undefended* model — the feasible
   perturbation space, not the defense, is doing the work here.

## Honest Caveats

- NES operates inside the same l2 ball as white-box PGD (epsilon = 0.1); HSJ
  minimizes distance without a ball. The two attack classes are compared on
  query economics, not distance budget. Do not compare HSJ and NES L2 directly.
- "Rotation triggered" is measured on the final query in `run.py`; the
  trajectory-level claims (0.00% / 100%) were verified separately over *all*
  queries (see summary).
- TASCP wrapper parameters match exp5 (window 5, max_theta pi/2, tau 0.0,
  gamma 50.0) for comparability with the white-box experiments.

## Outputs

- `exp9_summary.md` — narrative summary
- `results/logs/exp9_blackbox/metrics.json` — full metrics
- `results/logs/exp9_blackbox_run.log` — raw run log
- `results/tables/exp9_blackbox_asr.csv` — paper table
