# Reviewer Guide

This artifact accompanies a paper that is an **empirical autopsy** of a
proposed moving-target defense (TASCP) for DNN-based intrusion detection. The
headline is negative: TASCP degrades robustness and is structurally bypassable.
Reviewers do not need to read the repository linearly.

## Quick Navigation

| If you are reviewing... | Start here |
|---|---|
| The claim structure (3 core claims) | `doc/EVALUATION_GUIDE.md` |
| Threat model (B1 black-box vs B2 white-box) | `doc/THREAT_MODEL.md` |
| Formal results (Lemma 2, Prop 1, Cor 3) | `doc/THEOREMS.md` |
| Claim → evidence mapping | `CLAIM_MAP.md` |
| Known limits of the artifact | `LIMITATIONS.md` |
| Methodological history | `PROVENANCE.md` |
| Fast verification | `VERIFY.md` |
| Full reproduction | `REPRODUCE.md` |
| Experiment manifests | `experiment_manifests/` |
| Reviewer checklist | `REVIEW_CHECKLIST.md` |

## Suggested Review Paths

### 15-minute scientific sanity check
1. `CLAIM_MAP.md` — understand the claim/evidence structure.
2. `README.md` "Key Results" — the four ASR numbers.
3. `make verify && make smoke` — integrity + tests.
4. `LIMITATIONS.md` — boundaries.

### 30-minute methodology review
1. `doc/THREAT_MODEL.md` — B1/B2 split.
2. `doc/adr/ADR-001-stateful-evaluation.md` — how the defense sees queries.
3. `src/tascp_eval/defenses/tascp.py` — the defense under test.
4. `experiments/exp16_refmath_validation/run.py` — independent math checks.
5. `experiments/exp5_adaptive_bpda_attack/` — the white-box break.

### 2-hour reproducibility review
1. `VERIFY.md` — integrity + smoke.
2. `REPRODUCE.md` — full reproduction.
3. Run selected `reproduce-*` targets.
4. `verification/compare_results.py` — fresh vs archive tolerances.
5. `EXPECTED_OUTPUTS.md` — check outputs.

### 30-minute provenance review
1. `PROVENANCE.md` — including the exp5 narrative revision.
2. `doc/adr/` — decision rationale.
3. `results/MANIFEST_session5_v5.md` — session-5 honest limitations.

## Reproducibility Levels

See `REPRODUCIBILITY_LEVELS.md`: R1 exact, R2 deterministic, R3 statistical,
R4 archival-only, **NR** not reproducible. **exp5/exp6/exp8 are R3** — they ship
archived records that re-run exactly for seed 42. **exp7 is R4** (prose only).
**exp1–exp4 are NR** (placeholders, no source or archive; see
`doc/OPEN_SCIENCE.md`). Decide how much weight to give each accordingly.

## Environment

`environment/hardware.md` records the interpreter, GPU and RAM; install from
`requirements.txt` (direct) or `environment/requirements.lock` (full closure).
CI (`.github/workflows/ci.yml`) runs integrity, traceability, environment and
the test suite, and diffs a fresh exp5/exp8 run against the archive.

## Dataset Caveats

Raw datasets are not shipped. Use the pinned preprocessed parquet files
(`manifest/dataset_sha256.txt`) for exact numbers; see `dataset_links.md`.
