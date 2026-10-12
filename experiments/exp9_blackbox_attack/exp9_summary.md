# Exp 9 Summary — Black-Box Query Attacks (Threat Model B1)

**Date:** 2026-10-07 · **Datasets:** UNSW-NB15 and CICIDS2017 (2,000 benign, correctly-classified samples each, seed 42) · **Architecture:** MLP

## Cross-Dataset Replication (Corollary 3 prediction confirmed)

Corollary 3 (`doc/THEOREMS.md`) predicts the antithetic-blindness mechanism is
dataset-independent — it depends only on query geometry. It replicates exactly:

| Condition | UNSW-NB15 (d=43) | CICIDS2017 (d=78) |
|---|---:|---:|
| NES / Undefended | 2.70% | 22.30% |
| NES / TASCP (Naive) | 2.60% (rotation **0.00%**) | 22.30% (rotation **0.00%**) |
| NES / Undefended + Domain-Constrained | **0.80%** (-70%) | 21.55% (-0.8pp) |
| HSJ / Undefended | 97.10% | 15.05% |
| HSJ / TASCP (Naive) | 90.70% | **87.40%** (rotation 87.35%) |
| NES sweep @1000 queries | 8.85% | 41.45% |

1. **NES blindness replicates**: rotation activation 0.00% on both datasets;
   ASR identical with/without TASCP. The mechanism-level prediction holds.
2. **NEW — TASCP amplifies decision-based attacks on CICIDS2017**: HSJ
   15.05% -> 87.40% (5.8x). The same rotation-as-adversarial-perturbation
   mechanism isolated in white-box exp5 (naive 22.95% vs 10.50%) is reachable
   by a label-only attacker: persistent HSJ queries trigger the rotation
   (87.35%), and the rotation's displacement dominates the classifier's
   verdict. On CICIDS2017 the defense is not friction — it is a catalyst.
3. **Domain constraints are dataset-dependent**: they crush the UNSW-NB15
   attack surface (-70% relative) but barely bind on CICIDS2017 (-0.8pp)
   because CICIDS's constrained fields (flags, protocol, bulk counters) are
   mostly at trivial levels — the realizable attack surface lives in the
   continuous flow statistics (IATs, byte counts), which a real attacker can
   vary freely. Tabular robustness claims must therefore state *which* fields
   are constrained, not just that constraints exist.

## UNSW-NB15 headline

TASCP neither detects nor meaningfully degrades black-box query attacks — but
for two *different* mechanistic reasons depending on the attack class:

| Attack | Access | Undefended ASR | TASCP ASR | Delta | Rotation fired |
|--------|--------|---------------:|----------:|------:|---------------:|
| NES (score-based, 250 queries) | logits | 2.70% | 2.60% | -0.10pp | 0.00% (0.00% over full trajectory) |
| HSJ (decision-based, 1000 queries) | label only | 97.10% | 90.70% | -6.40pp | 90.55% (final query) |

## Mechanism: why the rotation ignores NES

NES estimates gradients with **antithetic pairs** — each direction u is queried
as x + sigma·u and then x − sigma·u. Consecutive trajectory steps therefore
alternate: v_t ~= -2·sigma·u_j, v_{t+1} ~= +2·sigma·u_j + ... The cosine momentum
between successive steps is systematically **negative** (M_t ~= -0.83), and the
penetration integral clips it: eps_p = (1/|W|) * sum ||v_i|| * max(0, M_i) = 0.

With eps_p = 0 and tau = 0, delta_theta = gamma * max(0, eps_p - tau) = 0:
**the Givens rotation never fires** (verified over every query of the
trajectory, max delta_theta = 0.00e+00). A persistent-direction control
(fixed u, growing radius) triggers on 100% of samples (mean max
delta_theta = 0.833) — confirming the trigger is sensitive to direction
*persistence*, not to query volume or perturbation size.

Consequence: TASCP's threat model implicitly assumes the adversary walks a
persistent direction. Score-based black-box attackers — who look like noise by
design — are invisible to it. Decision-based attackers (HSJ) do persist and do
trigger the rotation, but it only adds friction: -6.4pp ASR and +12% mean L2
distance, with the attack still succeeding 90.70% of the time.

## Query economics (bifurcated threat model)

| Attacker | Access | Queries | ASR | Distance |
|----------|--------|--------:|----:|---------:|
| White-box PGD 20 steps (exp5) | gradients | 20 | 10.50% | <= 0.1 (l2 ball) |
| NES budget 1000 | scores | 1000 | 8.85% | <= 0.1 (l2 ball) |
| NES budget 250 | scores | 250 | 2.70% | <= 0.1 (l2 ball) |
| HSJ budget 1000 | labels | 1000 | 97.10% | ~0.64 (unconstrained, minimized) |

Matching 20-step white-box PGD costs a score-based attacker ~50x more queries.
Black-box query attacks are practically expensive even without any defense —
which is exactly why stateful *detection* (Blacklight-style fingerprinting)
and stateful *perturbation* (TASCP) both target this regime.

## Domain-constrained perturbations (P2)

Snapping the 16 discrete/binary UNSW-NB15 fields back to their observed
encoding levels after every NES step (proto/state/service, is_sm_ips_ports,
is_ftp_login, ct_*, trans_depth):

| Condition | ASR | Mean Q to success |
|-----------|----:|------------------:|
| NES / Undefended | 2.70% | 178.9 |
| NES / Undefended + Domain-Constrained | 0.80% | 63.0 |

A 70% relative reduction against the *undefended* model: most of the "attack
surface" in a naive l2 ball is not realizable in the actual data domain. Any
robustness claim for tabular NIDS should therefore be made against
domain-constrained adversaries, not free-form l2 perturbations.

## Seed Robustness (3 seeds × 2,000 samples, UNSW-NB15)

Primary paired conditions re-run across seeds 42/43/44 (pooled n = 6,000):

| Condition | Pooled ASR | Wilson 95% CI | Per-seed |
|---|---:|---|---|
| NES / Undefended | 2.70% | [2.32, 3.14] | 2.90 / 2.60 / 2.60 |
| NES / TASCP (Naive) | 2.53% | [2.17, 2.96] | 2.75 / 2.30 / 2.55 |
| HSJ / Undefended | 97.18% | [96.73, 97.57] | 97.05 / 97.05 / 97.45 |
| HSJ / TASCP (Naive) | 91.38% | [90.65, 92.07] | 90.80 / 91.55 / 91.80 |

- **NES: no TASCP effect** — CIs overlap (Δ = −0.17pp); the blindness claim is
  seed-robust.
- **HSJ: significant TASCP friction** — CIs are disjoint (−5.8pp); the
  decision-based friction claim is seed-robust.

(Pooled numbers supersede the single-seed table above for these four rows;
they differ by ≤0.7pp due to the chunked-RNG protocol introduced with the
cross-dataset refactor — same protocol, deterministic per seed.)

## Artifacts

- `results/logs/exp9_blackbox/metrics.json` — full metrics incl. budget sweep
- `results/logs/exp9_blackbox/multiseed.json` + `results/tables/exp9_multiseed.csv`
- `results/logs/exp9_blackbox_run.log`, `results/logs/exp9_blackbox_cicids_run.log` — raw run logs
- `results/tables/exp9_blackbox_asr.csv`, `results/tables/exp9_blackbox_asr_cicids2017.csv` — paper tables
