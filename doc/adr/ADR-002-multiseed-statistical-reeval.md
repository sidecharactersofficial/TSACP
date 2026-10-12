# ADR-002: Multi-Seed Statistical Re-Evaluation with Pre-Registered Tolerances

## Status

Accepted.

## Context

Single-seed white-box runs can overstate or understate a contrast. The paper's
central empirical claim (TASCP's data-dependent basis is the gradient leak)
must survive seed variance in both the data draw and the attack's randomness.

## Decision

Repeat every headline contrast across 5 seeds (`{42, 43, 44, 45, 46}`) at
2,000 samples per seed, with the **model checkpoint held fixed** and only the
data draw / attack randomness varying. Report per-seed ASR, per-seed success
counts (`per_seed_k`), pooled ASR, and Wilson 95% CIs.

Pre-register comparison tolerances for fresh runs
(`verification/compare_results.py`):

| Quantity | Tolerance |
|----------|-----------|
| pooled ASR | ±0.02 absolute |
| per-seed ASR | ±0.03 absolute |
| per-seed rotation-trigger fraction | ±0.05 absolute |

## Consequences

### Positive
- exp13 shows UNSW adaptive 52.99% [52.01, 53.97] vs undefended 10.65%
  [10.06, 11.27] with disjoint CIs; basis-fixed overlaps undefended.
- The leak localizes to the data-dependent basis across seeds.
- Archived per-seed counts make the numerator/denominator recoverable.

### Negative
- Seeds themselves are not stored in the JSON (only per-seed order); the seed
  list is fixed in `run.py`.
- Higher compute cost.

### Neutral
- NES/HSJ black-box multiseed uses the same machinery with 3 seeds.
