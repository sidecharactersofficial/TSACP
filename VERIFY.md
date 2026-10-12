# Verification

Fast verification of the artifact **without** re-running full experiments.
Verification confirms that the shipped files support the paper's claims.
Full regeneration is covered in `REPRODUCE.md`.

## Prerequisites

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Step 1: Verify Checkpoint + Dataset Integrity

```bash
make verify
# or
python3 verification/verify_manifest.py
python3 verification/verify_datasets.py
```

Expected: `verify_manifest: PASS (5/5 checkpoints)` and
`verify_datasets: PASS (3 files)`.

## Step 2: Run Smoke Tests

```bash
make smoke
```

This runs the test suite (fast invariants) plus the checkpoint/dataset
integrity checks. Expected runtime: <2 minutes on CPU.

```bash
make test    # pytest tests/ -q  -> 67 passed
```

## Step 3: Inspect Archived Results

```bash
ls results/logs/exp13_whitebox_multiseed/
ls results/logs/exp16_refmath/
ls results/tables/ | head
```

## Step 4: Regenerate Aggregate Statistics From the Archive

The multi-seed JSON archives contain per-seed numerators (`per_seed_k`), pooled
ASR, and Wilson CIs. Confirm the paper's pooled values:

```bash
python3 - <<'PY'
import json
d = json.load(open("results/logs/exp13_whitebox_multiseed/multiseed_unsw_nb15.json"))
for cond, v in d["summary"].items():
    print(f"{cond:26s} pooled={v['pooled_asr']:.4f}  k={v['per_seed_k']}")
PY
```

## Step 5: Verify Paper Tables (Time-Budgeted)

| Claim | Command | Expected |
|-------|---------|----------|
| C4 (adaptive) | `python -m experiments.exp5_adaptive_bpda_attack.run unsw_nb15 --n 2000 --seed 42` then `python verification/compare_results.py results/logs/exp5_adaptive_bpda_attack/metrics_unsw_nb15.json <fresh>` | archived: 53.60% adaptive vs 10.50% undefended; diff PASS |
| C5 (black-box) | `python -m experiments.exp9_blackbox_attack.run` | NES rotation fires 0.00%; HSJ trigger fraction ~0.90 |
| C7 (multi-seed) | `python -m experiments.exp13_whitebox_multiseed.run unsw_nb15` then `python verification/compare_results.py results/logs/exp13_whitebox_multiseed/multiseed_unsw_nb15.json <fresh>` | PASS within ±0.02 pooled |
| C9 (reference math) | `python -m experiments.exp16_refmath_validation.run` | median rel err ~1e-5; E[M_t] ~ -0.70 |
| C15 (performance) | see `experiments/exp7_latency_benchmark/` and `tascp-proto/fast-loop` | 7.08 ms HTTP mean; 53.3 ns/query in-process |

## What Not to Expect

- **Byte-identical reproduction** of stochastic attacks. Use `verification/compare_results.py`.
- **A machine-readable archive for exp7.** The latency number is prose-only
  (level R4). exp5/exp6/exp8 now have archived records (level R3); exp1–exp4 are
  not implemented (level NR, `doc/OPEN_SCIENCE.md`).
- **Dataset reconstruction** for CICIDS2017 / UNSW-NB15 / NF-ToN-IoT-v2 from raw
  sources with matching hashes; use the shipped parquet files for exact numbers.
