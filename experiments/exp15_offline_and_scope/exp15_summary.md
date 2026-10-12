# EXP15 — Offline Attacker & Corollary-3 Scope (UNSW-NB15, real data)

Run: `python -m experiments.exp15_offline_and_scope.run unsw_nb15`, exit 0.
Logs: `results/logs/exp15_offline_scope_unsw.log`; tables:
`results/tables/exp15_offline_unsw_nb15.csv`,
`results/tables/exp15_cor3_scope_unsw_nb15.csv`.

## Part A — offline attacker (P1-9)

| condition | bit-identical TASCP==DACM | max abs diff | submitted ASR |
|---|---|---|---|
| white-box offline (optimize vs undefended replica, submit 1 fresh flow) | True | 0.00e+00 | 10.50% |
| surrogate mismatch (optimize vs fresh surrogate MLP, submit 1 flow) | — | — | 0.55% |

The strongest threat model is offline: TASCP's verdict on a first/fresh query
is **bit-identical** to DACM's (and the undefended model's) because Δθ = 0
when the session has no history. A stateful defense adds nothing against the
single-submission attacker — the paper's honest negative result, formalized
as the offline corner of Theorem 1's threat model.

## Part B — Corollary-3 scope (P1-8): trigger-fire rate over ALL queries

| attack | fire % (Δθ>1e-5, all queries) | ASR % | mean Q-to-success |
|---|---:|---:|---:|
| NES antithetic (reference) | 9.93 | 0.75 | — |
| NES i.i.d. | 20.91 | 0.60 | 89.9 |
| SPSA | 64.90 | 1.25 | — |
| ZOO | 2.49 | 0.60 | 24.6 |
| SimBA | 23.30 | 0.85 | 25.8 |
| Square | 20.73 | 46.70 | 2.0 |
| Boundary | 41.01 | 75.80 | 8.0 |

**Findings.**
1. Every non-antithetic attacker fires the trigger on a nonzero fraction of
   queries — Corollary 3's antithetic statement does not label the whole
   ecosystem of score-based attacks as blind.
2. But firing ≠ success: gradient-estimator attacks fire up to 65% of
   queries yet land ASR ≤ 1.25%.
3. Only decision-based refinement attacks (Square 46.7%, Boundary 75.8%)
   achieve meaningful ASR, at 2–8 queries-to-success over ~240k total
   queries — high-volume boundary walkers, consistent with Prop 2's
   query-complexity picture. NES/TASCP at 2.60% (exp9) stays defensible
   against the canonical estimator; the practical defense failure mode is
   Boundary/Square-class attacks.

Paper wording: Corollary 3 holds for the canonical antithetic estimator and
*degrades* (does not vanish) for non-antithetic gradient estimators; the
honest scope line is "momentum-triangle blindness is estimator-specific,
not attack-class-universal."
