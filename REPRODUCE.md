# Reproduction

Full reproduction of the artifact's experiments from scratch. For fast
verification without re-running experiments, see `VERIFY.md`.

## Prerequisites

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

CPU is sufficient for every experiment except the Transformer architecture
comparison, which is faster with CUDA. All reported latency numbers were
measured on the reference machine (see `LIMITATIONS.md`).

## Dataset Acquisition

Raw datasets are **not** shipped (see `dataset_links.md` for sources). The
authoritative inputs are the preprocessed parquet files pinned in
`manifest/dataset_sha256.txt`:

```bash
python3 data/scripts/preprocess.py          # UNSW-NB15 + CICIDS2017
python3 data/ton_iot/prepare_ton_iot.py     # NF-ToN-IoT-v2
```

For exact reproduction, use the shipped parquet files; from-scratch
regeneration may differ (public-mirror drift).

## Model Training

Canonical checkpoints are shipped under `src/tascp_eval/models/weights/` and
pinned in `manifest/checkpoint_sha256.json`. Re-training is optional:

```bash
python src/tascp_eval/models/train.py                    # UNSW-NB15 MLP
python src/tascp_eval/models/train_cicids2017.py         # CICIDS2017 MLP
python src/tascp_eval/models/train_transformer.py        # UNSW-NB15 Transformer
python src/tascp_eval/models/xgboost_classifier.py       # UNSW-NB15 XGBoost + surrogate
python data/ton_iot/train_ton_iot.py                     # NF-ToN-IoT-v2 MLP
```

## Experiment Reproduction

| Target | Claim | Command | Runtime (ref.) |
|--------|-------|---------|----------------|
| `reproduce-mechanism` | C1–C3 | `python -m experiments.exp1_momentum_separation.run` (+exp2, exp3) | minutes |
| `reproduce-adaptive` | C4 | `python -m experiments.exp5_adaptive_bpda_attack.run` | minutes |
| `reproduce-ablations` | C4 | `python -m experiments.exp8_adaptive_ablations.run` | minutes |
| `reproduce-blackbox` | C5 | `python -m experiments.exp9_blackbox_attack.run` (+`run_cicids2017`, `run_multiseed`) | ~30 min |
| `reproduce-hybrid` | C6 | `python -m experiments.exp10_hybrid_detection_sim.run` (+exp11) | minutes |
| `reproduce-multiseed` | C7 | `python -m experiments.exp13_whitebox_multiseed.run unsw_nb15` | ~10 min |
| `reproduce-stationarity` | C8 | `python -m experiments.exp14_stationarity.run` | minutes |
| `reproduce-refmath` | C9 | `python -m experiments.exp16_refmath_validation.run` | minutes |
| `reproduce-offline` | C10 | `python -m experiments.exp15_offline_and_scope.run` | minutes |
| `reproduce-family` | C11 | `python -m experiments.exp19_trajectory_family.run unsw_nb15` (+exp20) | minutes |
| `reproduce-triggers` | C12 | `python -m experiments.exp21_trigger_matrix.run unsw_nb15` | minutes |
| `reproduce-arch` | C13 | `python -m experiments.exp22_arch_split.run unsw_nb15` | minutes |
| `reproduce-audit` | C14 | `python -m experiments.exp23_defense_audit.run unsw_nb15` | minutes |
| `reproduce-performance` | C15 | `python -m experiments.exp7_latency_benchmark.run` + Rust bench | seconds–minutes |

Fresh outputs are written under `results_fresh/` (or a `.fresh` suffix) and are
**never** allowed to overwrite the canonical archive in `results/`. Compare
fresh vs archived with `verification/compare_results.py` (R3 experiments).

## Full Reproduction

```bash
make reproduce-all     # all of the above except the longest black-box sweeps
```

## Cleanup

```bash
make clean             # removes results_fresh/ and *.fresh.* outputs
```

The canonical archive under `results/` is preserved.
