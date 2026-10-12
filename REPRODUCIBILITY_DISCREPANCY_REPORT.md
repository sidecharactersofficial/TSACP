# Reproducibility Discrepancy Report — TASCP Artifact

Audit date: 2026-10-09
Repo: `/root/workspace/workspace/03-Code/Projects/Legacy/Project Archimedes`
Scope: Compare the repository against the proposed reproducibility plan (claims matrix,
independently runnable artifact, independent math tests, per-number traceability,
experiment-ID mapping, clean-machine reproduction).

This document reports **gaps between the plan and the current repository**. It is not a
claim that any result is wrong — it records where the artifact does not yet let an
independent reviewer verify the claim. Severity: **BLOCKER** (cannot reproduce),
**HIGH** (misleading / incomplete evidence), **MEDIUM** (hardening needed), **LOW** (cosmetic).

---

## 0. Severity summary

| # | Discrepancy | Severity | Evidence |
|---|-------------|----------|----------|
| 1 | Canonical main result (exp5 adaptive, 53.60% ASR) has **no machine-readable output at all** | **BLOCKER** | no `results/**/exp5*`; only hand-typed table in `experiments/exp5_adaptive_bpda_attack/exp5_summary.md` |
| 2 | No per-number traceability records (config/checkpoint hashes, numerator, denominator) | **BLOCKER** | zero matches for `config_sha256`/`checkpoint_sha256`/`n_evaluated`/`n_successful` in `results/` |
| 3 | Core math tests are empty stubs | **BLOCKER** | `tests/test_rotation.py`, `tests/test_momentum.py`, `tests/test_penetration.py` are `pass` |
| 4 | Proposed independent math test suite absent (projection equivariance, gradients vs FD, Prop 6, antithetic, stationarity) | **BLOCKER** | no `tests/test_projection.py`, `tests/test_gradients.py`, `tests/test_metrics.py` |
| 5 | Quick Start install command cannot succeed | **BLOCKER** | `README.md:166` `pip install -e src/tascp_eval`; no `pyproject.toml`/`setup.py` in `src/tascp_eval` |
| 6 | No artifact scaffold: `environment/`, `manifests/`, `CITATION.cff`, lockfile, Dockerfile, hardware.md | **HIGH** | none exist at repo root |
| 7 | No CI to regenerate tables / flag discrepancies vs paper | **HIGH** | no `.github/`, no workflow; `scripts/generate_paper_tables.py` is a `pass` stub |
| 8 | `run_all_experiments.sh` only covers exp1–exp10; no orchestration for exp11–exp24 | **HIGH** | `scripts/run_all_experiments.sh` |
| 9 | exp18 and exp24 have outputs but **no runnable source** in `experiments/` | **HIGH** | `results/logs/exp18_rust_bench/`; `data/ton_iot/exp24_summary.md` |
| 10 | No experiment-ID → table/figure → script mapping | **HIGH** | README "Repository Structure" says exp1–exp17; repo has exp1–exp23 |
| 11 | Result JSONs store percentages but rarely seeds, configs, or raw predictions | **HIGH** | sampled `results/logs/**/*.json` |
| 12 | `make setup` references a non-existent `requirements.txt` | **HIGH** | `Makefile:setup`; no `requirements*.txt` at root |
| 13 | No numerical tolerances defined for any comparison | **HIGH** | none documented in README/doc |
| 14 | Paper PDF not in repo; open-science statement cannot be verified | **MEDIUM** | no `*.pdf` |
| 15 | Dataset dimensionality mismatch 49 vs 43 | **MEDIUM** | `configs/dataset/unsw_nb15.yaml` vs whitepaper |
| 16 | exp5 summary text is self-contradictory (stale narrative) | **MEDIUM** | `experiments/exp5_adaptive_bpda_attack/exp5_summary.md` |
| 17 | Python/dependency versions inconsistent and unlocked | **MEDIUM** | `pyproject.toml` `^3.11`; bytecode tagged `cpython-314` |
| 18 | `Makefile` targets reference unrelated `backend/`, `frontend/`, `devtrace/` | **LOW** | `Makefile` |

---

## 1. The reproducibility matrix is mostly unmet

The plan defines seven components. Current status:

### 1.1 Main TASCP results (undefended, naive, adaptive, ablations) — **BLOCKER**
- `experiments/exp5_adaptive_bpda_attack/` has `run.py`, `run_cicids2017*.py`, `run_xgboost.py`,
  `diagnose_xgboost_naive.py`, and `exp5_summary.md`.
- The headline numbers (Undefended 10.50%, TASCP naive 22.95%, adaptive 53.60%) appear **only**
  as a hand-written Markdown table in `exp5_summary.md`. There is:
  - no `results/logs/exp5*` directory,
  - no `results/tables/exp5_*.csv`,
  - no raw predictions or per-sample verdicts.
- Same for the ablation experiment (`experiments/exp8_adaptive_ablations/`): summary only, no
  machine-readable output in `results/`.
- **Cannot** reproduce "same ASR within a predefined tolerance" because neither the original
  numeric output nor a tolerance exists.

### 1.2 Multi-seed experiments — **HIGH (partial)**
- Present and reasonably structured: `results/logs/exp13_whitebox_multiseed/multiseed_{unsw_nb15,cicids2017}.json`
  contains `per_seed_asr`, `per_seed_k`, `pooled_asr`, `wilson95`.
- `results/logs/exp9_blackbox/multiseed.json` similarly.
- Gaps: the **seed values themselves** are not stored in the JSON (only per-seed order), and the
  denominator (expected 2000) is not recorded — only `k` counts. No tolerance for seed-to-seed
  aggregate agreement.

### 1.3 Cross-dataset results — **HIGH**
- UNSW-NB15 and CICIDS2017 outputs exist widely; NF-ToN-IoT-v2 is present via
  `data/processed/ton_iot/ton_iot_processed.parquet` and exp19–exp23 `*_ton_iot` tables.
- Gap: the plan names `configs/nf_ton_iot_v2.yaml`; only `configs/dataset/{unsw_nb15,cicids2017}.yaml`
  exist. No config file for the third dataset. Train/test split and file hashes are not recorded.

### 1.4 Trigger audits — **MEDIUM**
- `results/tables/exp21_trigger_matrix_{unsw_nb15,ton_iot}.csv` exist.
- `results/logs/exp9_blackbox/metrics.json` records `rotation_triggered_frac` (e.g. NES
  0.0, HSJ 0.9055).
- Gap: benign-calibrated thresholds are not stored as structured output for auditing.

### 1.5 Verdict stationarity — **MEDIUM**
- `results/tables/exp14_stationarity_{unsw_nb15,cicids2017}.csv` and exp19/exp23 stationarity
  metrics exist.
- Gap: no independent verification of numerator/denominator logic; no hand-constructed-history
  test (see §3).

### 1.6 Performance — **HIGH (partial)**
- exp18 Rust bench: `results/logs/exp18_rust_bench/bench_summary.json`, `estimates.json` exist.
- exp7 latency (7.08 ms HTTP) has **no stored output** (`results/logs` has no `exp7`; no CSV/JSON).
  The benchmark protocol and hardware are not recorded (`hardware.md` absent).

### 1.7 Mathematical results — **BLOCKER**
- Only `results/logs/exp16_refmath/metrics_unsw_nb15.json` exists (autograd-vs-FD error, ρ,
  NES cosine, M_t).
- No independent tests for rotation identities, norm preservation, projection equivariance,
  Prop 6 bound, antithetic-query momentum, or stationarity metric logic. See §3.

---

## 2. The artifact is not yet independently runnable

The proposed `tascp-artifact/` layout maps as follows:

| Proposed path | Present? | Notes |
|---|---|---|
| `README.md` | yes | but Quick Start install is broken (§5) |
| `CITATION.cff` | **no** | absent |
| `LICENSE` | yes | present |
| `environment/Dockerfile` | **no** | only `tascp-proto/{slow,fast}-loop/Dockerfile` (different component) |
| `environment/requirements-lock.txt` | **no** | no lockfile anywhere |
| `environment/hardware.md` | **no** | absent |
| `configs/unsw_nb15.yaml` | partial | `configs/dataset/unsw_nb15.yaml` (different path) |
| `configs/cicids2017.yaml` | partial | `configs/dataset/cicids2017.yaml` |
| `configs/nf_ton_iot_v2.yaml` | **no** | absent |
| `src/defense|attacks|audits|metrics` | partial | `src/tascp_eval/{defenses,attacks,...}`; no `audits/`, no `metrics/` package |
| `tests/test_rotation.py` | **stub** | `pass` only |
| `tests/test_projection.py` | **no** | absent |
| `tests/test_gradients.py` | **no** | absent |
| `tests/test_metrics.py` | **no** | absent |
| `scripts/verify_environment.py` | **no** | absent |
| `scripts/reproduce_main_results.sh` | **no** | nearest is `scripts/run_all_experiments.sh` (exp1–10 only) |
| `scripts/reproduce_ablation_results.sh` | **no** | absent |
| `scripts/reproduce_audits.sh` | **no** | absent |
| `manifests/datasets.json` | **no** | absent |
| `manifests/checkpoints.json` | **no** | absent (weights exist but are not manifested) |
| `manifests/expected_results.json` | **no** | absent |
| `results/raw/` | **no** | only `results/logs`, `results/tables`, empty `results/figures` |

### Three-level reproducibility
- **Level 1 (smoke test):** partially available — `make test` runs and `33 passed in 9.42s`.
  But the documented install step fails, and 3 of the math test files are empty, so the "smoke
  test" is not meaningful yet.
- **Level 2 (core reproduction):** not available — no single command reproduces exp5/exp8; no raw
  predictions are saved.
- **Level 3 (full reproduction):** not available — no orchestration for exp11–exp24, no runtimes,
  no hardware requirements documented.

---

## 3. Independent math verification is largely missing

Required check vs. what exists:

| Math claim | Required independent check | Status |
|---|---|---|
| `R_t^T R_t = I` | random bases/angles, orthogonality within tolerance | **absent** (`test_rotation.py` is a stub) |
| `R_t` preserves norms | compare ‖R_t x‖ vs ‖x‖ | **absent** |
| Projection equivariance | `R_t Π_C(R_t^T x)` vs reference | **absent** |
| Analytical gradients | autograd vs central finite differences at clip/threshold boundaries | **partial** — only inside exp16 run output (`metrics_unsw_nb15.json`), not a standalone test with boundary cases |
| Proposition 6 bound | numerical test across random inputs/params | **absent** |
| Antithetic-query result | simulate growing dims/query counts | **partial** — `exp16` `prop3_nes_cosine` measures it, but no test |
| Stationarity metrics | hand-constructed histories with known flips | **absent** as an oracle |

Implemented tests that do exist (good, but not the required set):
- `tests/test_reference_math_matches_prototype.py` — compares `compute_penetration` against
  hard-coded Rust-expected values (3 tests). This is a drift check, not an independent oracle.
- `tests/test_domain_constraints.py`, `tests/test_blackbox_attacks.py`, `tests/test_hybrid_detection.py`,
  `tests/test_new_attacks.py`, `tests/test_trajectory_family.py` — contract/invariant tests for
  attacks and the family wrapper.

Oracle-independence concern: the plan explicitly warns against using the system under test as its
only oracle. Today the rotation/projection/gradient math has **no** oracle at all, and the one
reference-math test imports the implementation under test (`from src.tascp_eval.reference_math
import compute_penetration`) rather than an independently written formula.

---

## 4. Paper numbers are not demonstrably traceable to outputs

The plan's canonical record schema (`experiment_id`, `dataset`, `defense`, `attacker`, `seed`,
`config_sha256`, `checkpoint_sha256`, `n_evaluated`, `n_successful`, `asr`) is not used anywhere.

Verified by search across `results/`:
- `config_sha256` / `config_sha` → **0 matches**
- `checkpoint_sha` → **0 matches**
- `n_evaluated` / `n_successful` → **0 matches**
- `sha256` in results → **0 matches**

What exists instead (best case, exp13):
```json
"Undefended": { "per_seed_asr": [...], "pooled_asr": 0.1065,
                "wilson95": [...], "per_seed_k": [210,227,212,205,211] }
```
This records per-seed success counts but not: the denominator, the seed identifiers, the dataset
hash, the checkpoint hash, or a config hash. The traceability chain
"paper table → aggregation script → raw result → config → checkpoint/dataset version" is broken at
**every** link except the last two JSON layers for a subset of experiments.

Additional gaps:
- No raw predictions anywhere (`results/raw/` does not exist); only aggregates.
- `scripts/generate_paper_tables.py` is a stub (`pass`), with a `TODO` to pull results into LaTeX.
- No CI that regenerates tables from saved results or flags discrepancies against the paper.

---

## 5. Clean-machine reproduction currently fails

Documented path (`README.md:163-169`):
```bash
pip install -e src/tascp_eval
python experiments/exp5_adaptive_bpda_attack/run.py
make test
```

Problems:
1. **`pip install -e src/tascp_eval` cannot work** — there is no `pyproject.toml` or `setup.py`
   under `src/tascp_eval/`. The root `pyproject.toml` is a Poetry config whose `packages`
   entry is `[{include = "tascp", from = "src"}]` — i.e. it names a package `tascp`, whereas the
   actual import package is `tascp_eval`. Installing the editable path in the README will error.
2. **`make setup` references `requirements.txt`** (`Makefile:setup`) which does not exist at the
   repo root (no `requirements*.txt` anywhere except inside `tascp-proto/slow-loop/`).
3. **Dependencies are not locked** — `pyproject.toml` uses caret ranges (`torch ^2.0.0`,
   `numpy ^1.24.0`, …). Exact bit-for-bit reproduction across hardware/library versions is
   impossible to bound.
4. **Python version inconsistency** — `pyproject.toml:10` declares `python = "^3.11"`, but
   compiled bytecode is tagged `cpython-314`, i.e. the suite was run under Python 3.14.
5. **No environment verification** (`scripts/verify_environment.py` absent), no Docker image, no
   hardware spec. The 7.08 ms HTTP latency number is hardware-dependent and uncontextualized.
6. The tests "pass" only because `tests/conftest.py` injects the repo root onto `sys.path`
   (`conftest.py:8-9`); they do **not** demonstrate that the documented install works.

### Fresh-machine checklist (as specified in the plan)
None of the required outputs exist: no independent collaborator report, no recorded environment,
runtime, failed steps, or fixes. This should be performed only **after** items 1–4 are fixed.

---

## 6. Experiment-ID mapping and open-science statement

- `README.md:70` describes `experiments/` as **exp1–exp17**, but the repository contains
  exp1–exp23 (plus exp_transformer). The doc is stale.
- The plan flags the paper's open-science statement (exp1–exp10 + cross-dataset/cross-architecture/
  black-box) versus a roadmap referencing experiments **up to 24**. This repo cannot resolve that
  because:
  - **exp18 has no source** — only `results/logs/exp18_rust_bench/{bench_summary,estimates}.json`.
    `README.md:127` cites exp18 as a criterion microbenchmark, but there is no
    `experiments/exp18_*` directory or Rust bench source in this tree.
  - **exp24 has no `experiments/exp24` directory** — only `data/ton_iot/{exp24_summary.md,
    prepare_ton_iot.py, train_ton_iot.py}`. The manifest (`results/MANIFEST_session5_v5.md`)
    lists exp24 under "exp22 arch split + exp24 third dataset" with data-prep scripts, but there
    is no runnable `exp24` experiment module.
  - **No README section maps experiment IDs → paper tables/figures → scripts.**
- Orchestration coverage is incomplete:
  - `scripts/run_all_experiments.sh` runs exp1–exp10 only.
  - `scripts/run_priority_experiments.sh` runs exp1, exp2, exp5, exp9 (+CICIDS), exp10 only.
  - No script covers exp11, exp12, exp13, exp14, exp15, exp16, exp17, exp18, exp19, exp20, exp21,
    exp22, exp23, exp24.
- The paper PDF (`tascp_sp (7).pdf`) is **not in the repository**, so the open-science statement
  and the exp1–10 vs up-to-24 mismatch cannot be checked against the artifact.

---

## 7. Analytical-claim specific discrepancies

The plan stresses that reproducing a result ≠ validating the explanation. Current mismatches:

1. **exp5 narrative contradicts itself.** `experiments/exp5_adaptive_bpda_attack/exp5_summary.md`
   contains two conflicting conclusions: one says TASCP "degrades robustness"; a later paragraph
   says it provides "meaningful protection" against the naive attacker and that an "implementation
   bug" was fixed. One of these is stale; a reader cannot tell which run produced the paper's
   53.60%.
2. **Gradient derivation not independently checked.** exp16 reports median rel err 1.4e-5 for
   `∇ε_p`, but the plan requires finite-difference validation **including points near clipping and
   threshold boundaries** — that boundary sweep is not present.
3. **Proposition 6 is not numerically tested** anywhere as a standalone check.
4. **Stationarity numerator/denominator is not verified against a hand-built oracle** — only
   end-to-end exp14/exp19/exp23 outputs exist.
5. **Dimensionality discrepancy:** `configs/dataset/unsw_nb15.yaml` documents `d_feature: 43`
   while noting "the paper originally states d=49". The paper and config disagree and the config
   merely explains it away; the artifact does not reconcile the paper text.

---

## 8. Recommended remediation order

**Must fix before any reproduction claim (BLOCKER):**
1. Persist exp5 and exp8 outputs as machine-readable JSON/CSV (denominator + numerator + seed +
   config hash + checkpoint hash), and add `scripts/reproduce_main_results.sh`.
2. Add per-number traceability records (the plan's schema) for every table/figure; make
   `scripts/generate_paper_tables.py` real.
3. Replace the three stub math tests; add `test_rotation.py` (orthogonality/norm), `test_projection.py`
   (equivariance vs independent reference), `test_gradients.py` (autograd vs central FD incl.
   boundaries), `test_metrics.py` (stationarity oracle), plus Prop 6 / antithetic checks.
4. Fix installation: add a real `src/tascp_eval/pyproject.toml` (or correct the root package name)
   and make the Quick Start actually run on a clean checkout.

**HIGH:**
5. Add `environment/` (Dockerfile, lockfile, hardware.md), `manifests/` (datasets, checkpoints,
   expected_results), `CITATION.cff`, and tolerance definitions.
6. Extend orchestration to exp11–exp24; add exp18 and exp24 runnable sources or document them as
   precomputed-only with provenance.
7. Add a CI job that regenerates tables from saved results and fails on discrepancy vs published
   numbers.
8. Store seed identifiers, denominators, and (where feasible) raw predictions.

**MEDIUM/LOW:**
9. Add the paper PDF (or an open-science mapping table) to the artifact and reconcile exp1–10 vs
   up-to-24.
10. Reconcile d=43 vs d=49; fix stale exp5 summary; align Python/dependency versions; remove
    unrelated `Makefile` targets.

---

## 9. What is already good (do not regress)

- `make test` → `33 passed in 9.42s` from a clean checkout (uses `conftest.py` path injection).
- Structured per-seed JSON for exp13 and exp9 with Wilson CIs.
- Tables and run logs for exp10–exp23, session-4/5 driver logs, and an honest limitations manifest
  (`results/MANIFEST_session5_v5.md`) that itself calls out several of the gaps above.
- Model weights are present (`src/tascp_eval/models/weights/*.pth`, `*.json`) and small enough to
  check in.
- `DATA PROVENANCE` for the third dataset (`data/ton_iot/PROVENANCE.md`).

---

## Addendum (2026-10-09): artifact scaffolding added

A reviewer-facing artifact scaffold has been added, modelled on a reference
reproducibility artifact. It does **not** fabricate any scientific result: every
number in the new documents is drawn from existing files in this repository, and
missing outputs are labelled as missing.

New files:
- `manifest/checkpoint_sha256.json`, `manifest/dataset_sha256.txt` (real hashes).
- `verification/verify_manifest.py`, `verify_datasets.py`, `compare_results.py`
  (all run; `verify_manifest` and `verify_datasets` currently PASS).
- `experiment_manifests/E-EXP01..E-EXP24-*.json` + `E-EXP-CROSS-ARCH-TRANSFORMER.json`
  (one per experiment; previously 15 claim-grouped manifests, now 25).
- `CLAIM_MAP.md`, `REVIEWER_GUIDE.md`, `REVIEW_CHECKLIST.md`, `VERIFY.md`,
  `REPRODUCE.md`, `REPRODUCIBILITY_LEVELS.md`, `EXPECTED_OUTPUTS.md`,
  `LIMITATIONS.md`, `PROVENANCE.md`, `CITATION_TO_ARTIFACT.md`, `dataset_links.md`.
- `doc/adr/ADR-001..004`.
- `requirements.txt` (pinned to the actual installed versions).
- Rewritten `Makefile` (`verify`, `test`, `smoke`, `reproduce-*`).
- `README.md` restructured with an Artifact Navigation section.

Still not resolved (from the sections above):
- **§1.1 / §4 — exp5, exp8, exp6, exp7 remain R4 (prose only).** The headline
  table still has no machine-readable archive.
- **§3 — rotation/projection/gradient independent tests still absent**
  (`tests/test_rotation.py` and friends are stubs); only the exp16 finite-difference
  oracle exists.
- **§6 — exp18 has no source in `experiments/`; exp24 has no driver.** Both are
  documented as archived/scripted rather than fully runnable.
- **§5 — CI and a true clean-machine run** have not been performed here.

---

## Addendum 2 (2026-10-09): findings addressed

This addendum records the disposition of all 18 findings plus one newly
discovered gap. No scientific number was fabricated: every value added to the
artifact was produced by a run on this machine and archived, or is an existing
file.

### Findings 1–18

| # | Severity | Status | Resolution |
|---|----------|--------|------------|
| 1 | BLOCKER | **FIXED** | exp5 and exp8 now persist machine-readable JSON + CSV: `results/logs/exp5_adaptive_bpda_attack/metrics_unsw_nb15.json`, `results/logs/exp8_adaptive_ablations/metrics_unsw_nb15.json` (+ `results/tables/…`). Both re-run and reproduce the paper's headline numbers exactly. |
| 2 | BLOCKER | **FIXED** | `src/tascp_eval/traceability.py` emits the canonical record (`experiment_id`, `dataset`, `defense`, `attacker`, `seed`, `config`+`config_sha256`, `checkpoint_sha256`, `dataset_sha256`, `n_evaluated`, `n_successful`, `asr`, `git_commit`, `environment`). exp5/exp6/exp8 records use it; `scripts/check_traceability.py` enforces it in CI. |
| 3 | BLOCKER | **FIXED** | `tests/test_rotation.py`, `test_momentum.py`, `test_penetration.py` implemented as independent NumPy oracles (not stubs). |
| 4 | BLOCKER | **FIXED** | `tests/test_projection.py`, `tests/test_gradients.py`, `tests/test_metrics.py` added (equivariance, autograd-vs-central-FD incl. boundaries, momentum/stationarity oracles). Suite: **67 passed**. |
| 5 | BLOCKER | **FIXED** | Root `pyproject.toml` rewritten (setuptools, package `tascp-eval`, `src/tascp_eval`); `pip install -e .` verified in a clean venv. README Quick Start uses `pip install -r requirements.txt`. |
| 6 | HIGH | **FIXED** | Added `environment/{Dockerfile,requirements.lock,hardware.md}`, `CITATION.cff`, `scripts/verify_environment.py`, `scripts/gen_lock.py`. |
| 7 | HIGH | **FIXED** | `.github/workflows/ci.yml`: integrity + traceability + environment + tests, plus a discrepancy job that regenerates exp5/exp8 and diffs against the archive with `verification/compare_results.py`. |
| 8 | HIGH | **FIXED** | `Makefile` now has targets for exp4/12/17/18/24/transformer; `reproduce-all` covers exp1–exp24 + transformer. |
| 9 | HIGH | **FIXED** | exp18 runnable driver `experiments/exp18_rust_bench/run.py` over `tascp-proto/fast-loop/benches/hot_path.rs`; exp24 driver `experiments/exp24_ton_iot_transfer/run.py`. |
| 10 | HIGH | **FIXED** | `CLAIM_MAP.md` + `doc/OPEN_SCIENCE.md`: full exp→claim→script→output table for exp1–24 + transformer. |
| 11 | HIGH | **FIXED (for exp5/6/8)** | Seed, config hash, checkpoint/dataset hash, denominator, numerator archived; exp5/exp8 store per-condition records. Remaining legacy JSONs are flagged by `scripts/check_traceability.py`. |
| 12 | HIGH | **FIXED** | `requirements.txt` at root (pinned). |
| 13 | HIGH | **FIXED** | Pre-registered tolerances in `verification/compare_results.py` and `REPRODUCIBILITY_LEVELS.md` (±0.02 pooled ASR, ±0.03 per-seed, ±0.05 rotation). |
| 14 | MEDIUM | **PARTIAL (honest)** | `doc/OPEN_SCIENCE.md` reconciles exp1–10 vs up-to-24 and maps every experiment. The paper PDF is **not** shipped (double-blind). |
| 15 | MEDIUM | **FIXED** | `doc/DIMENSIONALITY.md` reconciles d=49 (raw) vs d=43 (model input); `configs/dataset/ton_iot.yaml` added. |
| 16 | MEDIUM | **FIXED** | `exp5_summary.md` rewritten from the archived record; the contradictory stale paragraphs are explicitly marked superseded. |
| 17 | MEDIUM | **FIXED** | `environment/requirements.lock` (53-package closure, regenerable) + `scripts/gen_lock.py`; `pyproject.toml` pins deps and `requires-python = ">=3.11,<3.15"`. |
| 18 | LOW | **FIXED** | `Makefile` rewritten; no `backend/`/`frontend/`/`devtrace/` references. |

### New finding (not in the original 18)

| # | Severity | Status | Detail |
|---|----------|--------|--------|
| 19 | HIGH | **DISCLOSED, not implemented** | exp1–exp4 (`experiments/exp{1,2,3,4}_*/run.py`) are **placeholders** with no implementation and no archived output; they previously printed and exited 0. They now exit code 2 with a clear message, are marked **NR** in `CLAIM_MAP.md`/`REPRODUCIBILITY_LEVELS.md`/`doc/OPEN_SCIENCE.md`, and are skipped by `reproduce-all`. Claims C1–C3 (and the AT/RS part of C16) are therefore **not independently reproducible** from this artifact. Implementing them requires the paper's exact definitions, which are not shipped. |

### Verification performed

- `make smoke` → `environment: PASS`, `verify_manifest: PASS (5/5)`,
  `verify_datasets: PASS (3 files)`, `traceability: PASS`, `67 passed`.
- exp5/exp6/exp8 re-run from the shipped scripts; exp5/exp8/exp6 records match
  the archived JSON (compare_results PASS with 0.0 diff).
- `pip install -e .` tested in a throwaway venv (build + import OK).

### Still open

- Finding 14: the paper PDF is not in the repo (double-blind limitation).
- Finding 19: exp1–exp4 are unimplemented.
- Finding §2 `manifests/*.json` vs `manifest/`: the artifact uses `manifest/`
  (hashes) + `experiment_manifests/` (per-experiment) rather than the plan's
  `manifests/{datasets,checkpoints,expected_results}.json` layout; the content
  is equivalent and documented.

