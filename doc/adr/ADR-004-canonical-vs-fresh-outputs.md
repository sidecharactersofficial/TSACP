# ADR-004: Canonical vs Fresh Reproduction Outputs

## Status

Accepted.

## Context

Reviewers need to reproduce shipped results from shipped code and checkpoints.
Stochastic attacks (adaptive BPDA, NES, HSJ) produce run-to-run variance, and a
naive re-run that overwrites the archive would destroy the original results and
make fresh-vs-archive comparison impossible.

## Decision

Keep two namespaces:

1. **Canonical results** — `results/` (and `experiments/*/exp*_summary.md`).
   Authoritative paper numbers. Reproduction drivers must never overwrite these.
2. **Fresh results** — `results_fresh/`, or a `.fresh` / `.repro` suffix.
   Written by reviewer runs, then compared against the archive.

Comparison uses pre-registered tolerances in
`verification/compare_results.py` (see ADR-002).

## File Naming Convention

| Namespace | Pattern | Example |
|-----------|---------|---------|
| Canonical | `results/logs/exp13_whitebox_multiseed/multiseed_unsw_nb15.json` | Archived paper result |
| Fresh | `results_fresh/exp13_whitebox_multiseed/multiseed_unsw_nb15.json` | Reviewer reproduction |

## Consequences

### Positive
- Archive is immutable; fresh runs are quantitatively comparable.
- `make clean` removes fresh outputs but preserves the archive.

### Negative
- Two namespaces increase disk usage.
- Some experiments (exp5/exp8/exp6/exp7) have **no** canonical machine-readable
  archive, so only qualitative comparison is possible (level R4).

### Neutral
- Aggregation over canonical JSONs is reproducible from the archive alone.
