# Reproducibility Levels

Not every result in this artifact has the same reproducibility guarantee. This
document classifies results into four levels so reviewers know what to expect.

## Level Definitions

| Level | Meaning | Examples in this artifact |
|-------|---------|---------------------------|
| R1 | Exact byte-identical reproduction | Checkpoint SHA-256, dataset SHA-256 |
| R2 | Deterministic numerical reproduction | exp16 reference-math metrics (autograd-vs-FD, ρ, M_t); exp14 stationarity tables; exp23 audit tables |
| R3 | Statistical / tolerance-based reproduction | exp13 white-box multi-seed, exp9 black-box multi-seed, exp5/exp6/exp8, exp18 Rust bench, exp19–exp22 synthetic-family runs |
| R4 | Archival verification only | Results with **no** archived machine-readable output (exp7 latency) — verify against `exp*_summary.md` and the shipped logs |
| NR | Not reproducible | No source and no archived output (exp1–exp4; see `doc/OPEN_SCIENCE.md`) |

## R1: Exact

- `manifest/checkpoint_sha256.json` — 5 shipped checkpoints.
- `manifest/dataset_sha256.txt` — 3 preprocessed parquet tensors.

## R2: Deterministic Numerical

- `results/logs/exp16_refmath/metrics_unsw_nb15.json` — autograd vs central
  finite differences, ρ, NES cosine, Cor-3 M_t.
- `results/tables/exp14_stationarity_*.csv`, `results/tables/exp23_defense_audit_*.csv`.

## R3: Statistical / Tolerance-Based

These depend on stochastic optimization and/or random data draws. Seeded
white-box runs (exp5/exp6/exp8) reproduce their archive **exactly**; the
tolerances below bound the unseeded NES/HSJ multi-seed sweeps. Pre-registered
tolerances (see `verification/compare_results.py`):

| Quantity | Tolerance |
|----------|-----------|
| pooled ASR / ASR | ±0.02 absolute |
| per-seed ASR | ±0.03 absolute |
| per-seed rotation-trigger fraction | ±0.05 absolute |
| mean queries to success | ±25% relative (secondary) |

| Experiment | Comparator |
|------------|-----------|
| exp5 / exp6 / exp8 white-box | `verification/compare_results.py` vs archived JSON (exact for seed 42) |
| exp13 white-box multi-seed | `verification/compare_results.py` vs archived JSON |
| exp9 black-box multi-seed | `verification/compare_results.py` vs archived JSON |
| exp18 Rust bench | `experiments/exp18_rust_bench/run.py --no-run` re-aggregates the archive |

## R4: Archival Verification Only

These outputs exist only as prose summaries or logs, not as structured results.
Reviewers can read the summary and the run log, but cannot automatically diff a
fresh run against a pinned archive.

- **exp7 latency** — `experiments/exp7_latency_benchmark/exp7_summary.md`.
  Reproducing it needs the Rust fast-loop HTTP server; the in-process cost of
  the same computation is archived by exp18.

## NR: Not Reproducible

No implementation and no archived output. The run.py files are explicit
placeholders that exit with code 2.

- **exp1, exp2, exp3** — momentum separation, gradient correlation, query
  budget (claims C1–C3).
- **exp4** — AT/RS baselines need checkpoints that are not shipped; the
  DACM/TASCP/undefended comparison is covered by exp5.

## Implications for Reviewers

- Do not assume R3 results are deterministic; use `verification/compare_results.py`.
- R1 can be checked with `make verify`; R2/R3 with `make smoke` and the
  `reproduce-*` targets; NR targets are reported, not run.
- R4 and NR are honest limitations, tracked in
  `REPRODUCIBILITY_DISCREPANCY_REPORT.md` and `doc/OPEN_SCIENCE.md`.
