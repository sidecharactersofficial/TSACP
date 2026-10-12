# ADR-003: Independent Numerical Validation of the Reference Math

## Status

Accepted.

## Context

The paper makes analytical claims (rotation identities, projection
equivariance, analytical gradients, Proposition 6, antithetic-query momentum).
The plan is explicit: a script reproducing an ASR table does **not** validate
the explanation for that result. Using the same implementation as both the
system under test and the only oracle is not acceptable.

## Decision

Validate analytical claims with at least one independent numerical oracle:

1. **Gradients** — compare autograd against central finite differences on real
   trajectories. Implemented in `experiments/exp16_refmath_validation/run.py`;
   archived in `results/logs/exp16_refmath/metrics_unsw_nb15.json`
   (9,216 comparisons, median rel err 1.37e-5).
2. **Penetration** — drift test against the Rust prototype's expected values
   (`tests/test_reference_math_matches_prototype.py`). This is a *drift*
   detector, not an independent formula, and is labelled as such.
3. **Momentum / antithetic** — exp16 measures E[M_t] = −0.70, P(M>0) = 1.6%
   and the NES cosine at increasing q, independent of the attack's outcome.

## Consequences

### Positive
- The gradient-degradation explanation is checked numerically, not assumed.
- The antithetic-blindness claim has a distributional check.

### Negative
- Rotation orthogonality, norm preservation, and projection equivariance still
  lack independently written reference tests (`tests/test_rotation.py` is a
  stub). This is an open gap tracked in
  `REPRODUCIBILITY_DISCREPANCY_REPORT.md`.

### Neutral
- The finite-difference comparison is deterministic given the archived inputs.
