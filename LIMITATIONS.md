# Known Limitations

This artifact is a research prototype and a **negative result**. It does not
provide production-grade robustness guarantees. Reviewers should understand
these boundaries before interpreting the results.

## Machine-readable outputs strengthened (was R4, now R3 for exp5/6/8)

**Status**: Resolved for exp5, exp6, exp8; exp7 remains prose (R4).

**Description**: The main white-box headline (exp5: undefended 10.50%, naive
22.95%, adaptive 53.60%), the exp8 ablations, and exp6 (FPR) are now archived
as JSON records (config/checkpoint/dataset SHA-256, denominator, numerator)
plus CSV tables under `results/logs/` and `results/tables/`. exp7 (latency)
remains prose-only because it requires the Rust HTTP server.

**Impact**: exp5/6/8 can be automatically diffed against a fresh run via
`verification/compare_results.py` (they reproduce exactly for seed 42). exp7
still requires reading `exp7_summary.md`.

**Mitigation**: `scripts/check_traceability.py` enforces the record schema in
CI; `REPRODUCIBILITY_LEVELS.md` classifies exp7 as R4 and exp1–exp4 as NR.

## exp1–exp4 are not implemented (NR)

**Status**: Known limitation, disclosed.

**Description**: `experiments/exp{1,2,3,4}_*/run.py` are placeholders with no
implementation and no archived output. They exit code 2.

**Impact**: Claims C1–C3 (momentum separation, gradient correlation, query
budget) and the AT/RS part of C16 cannot be independently reproduced from this
artifact; the reproducible component of exp4 is covered by exp5.

**Mitigation**: Documented in `CLAIM_MAP.md`, `REPRODUCIBILITY_LEVELS.md` (level
NR) and `doc/OPEN_SCIENCE.md`; `make reproduce-all` skips them.

## Stochastic attacks are unseeded by design

**Status**: Design choice with reproducibility impact.

**Description**: The adaptive BPDA, NES, and HSJ experiments use unseeded
random draws, matching the original evaluation protocol.

**Impact**: Exact byte-identical reproduction is impossible. Pooled ASR may
vary by roughly ±0.01–0.02 across runs.

**Mitigation**: Pre-registered tolerances in `verification/compare_results.py`
and multi-seed reporting (Wilson CIs) quantify this variance.

## Rotation math: independent oracles added

**Status**: Resolved (R3).

**Description**: `tests/test_reference_math_matches_prototype.py` remains a
**drift detector** against hard-coded Rust values. On top of it, independent
NumPy references now exist: `src/tascp_eval/{rotation,momentum,penetration,
metrics}.py` were written from the paper's equations (not imported from the
PyTorch implementation) and are exercised by `tests/test_rotation.py`,
`test_momentum.py`, `test_penetration.py`, `test_projection.py`,
`test_gradients.py`, `test_metrics.py`.

**Impact**: Orthogonality, norm preservation, projection equivariance,
autograd-vs-central-finite-difference (including boundary cases), and the
momentum/stationarity oracles are now checked by tests that do not depend on
the system under test.

**Mitigation**: exp16 (`results/logs/exp16_refmath/metrics_unsw_nb15.json`)
additionally cross-checks analytical gradients against central finite
differences on real trajectories.

## Dataset reconstruction

**Status**: Known limitation.

**Description**: Raw datasets (~2.3 GB) are not shipped. Processed parquet
files are pinned by SHA-256. From-scratch regeneration via public mirrors may
differ in row order or category frequencies.

**Impact**: Reviewers needing exact numbers must use the shipped parquet files.

**Mitigation**: `manifest/dataset_sha256.txt` and `dataset_links.md`.

## Hardware dependence of performance numbers

**Status**: Inherent limitation.

**Description**: The 7.08 ms HTTP latency (exp7) is hardware- and
transport-dependent. Only the in-process microbenchmark (exp18, 53.3 ns/query)
isolates the fast-loop math.

**Impact**: Reviewers on different hardware will not reproduce 7.08 ms.

**Mitigation**: exp18 reports the transport-independent quantity; both are
labelled with their measurement context.

## Non-differentiable architectures

**Status**: Methodological boundary.

**Description**: On XGBoost the basis-coupling gradient leak is not measurable;
the exp22 comparisons use surrogate gradients, and the Random-Forest arm is
trained on the fly (no RF checkpoint is shipped).

**Impact**: C13's coupling claim is explicitly differentiability-dependent.

**Mitigation**: exp22 output labels RF as trained-on-the-fly; documented in
`results/MANIFEST_session5_v5.md`.
