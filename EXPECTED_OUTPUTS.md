# Expected Outputs

What to expect when running each experiment. Use this to confirm the
environment is configured correctly.

## Checkpoint + Dataset Verification

```bash
make verify
```

Expected runtime: <1 min.

```
verify_manifest: PASS (5/5 checkpoints)
verify_datasets: PASS (3 files)
```

## Smoke / Test Suite

```bash
make smoke
```

Expected runtime: <2 min (CPU).

```
verify_manifest: PASS (5/5 checkpoints)
verify_datasets: PASS (3 files)
67 passed
```

## Reference Math (exp16, C9)

```bash
python -m experiments.exp16_refmath_validation.run
```

Expected file: `results/logs/exp16_refmath/metrics_unsw_nb15.json`.

Expected properties (real archived values):

- `lemma3_fd_vs_autograd.median_rel_err_all` ≈ 1.37e-5 over 9,216 comparisons
- `lemma2_rho.rho_mean` ≈ 0.198 vs isotropic 0.216
- `prop3_nes_cosine.*.measured` ≈ 0 at all q
- `cor3_mt.with_anchor.mean` ≈ −0.70, `P(M>0)` ≈ 1.6%

## White-Box Multi-Seed (exp13, C7)

```bash
python -m experiments.exp13_whitebox_multiseed.run unsw_nb15
```

Expected file: `results/logs/exp13_whitebox_multiseed/multiseed_unsw_nb15.json`.

Expected properties (5 seeds {42..46}, 2,000 samples/seed):

| condition | pooled ASR |
|---|---:|
| Undefended | 0.1065 |
| TASCP (Naive) | 0.2353 |
| TASCP (Full Adaptive) | 0.5299 |
| TASCP (Basis fixed) | 0.1047 |

Tolerance for a fresh run: pooled ±0.02, per-seed ±0.03.

## Black-Box Attacks (exp9, C5)

```bash
python -m experiments.exp9_blackbox_attack.run
```

Expected file: `results/logs/exp9_blackbox/metrics.json`.

Expected properties:

- NES `rotation_triggered_frac` = 0.0 (both datasets)
- HSJ UNSW: undefended 0.9710 → TASCP 0.9070
- HSJ CICIDS2017: undefended 0.1505 → TASCP 0.8740

## Adaptive BPDA (exp5, C4) — R3, archived

```bash
python -m experiments.exp5_adaptive_bpda_attack.run unsw_nb15 --n 2000 --seed 42
```

Expected files:

- `results/logs/exp5_adaptive_bpda_attack/metrics_unsw_nb15.json` (per-condition
  traceability records)
- `results/tables/exp5_summary_unsw_nb15.csv`

| Defense | ASR | n_successful | mean queries |
|---|---:|---:|---:|
| Undefended | 0.1050 | 210 | 10.68 |
| DACM (static) | 0.1050 | 210 | 10.68 |
| TASCP (Naive) | 0.2295 | 459 | 10.18 |
| TASCP (Adaptive BPDA) | 0.5360 | 1072 | 4.26 |

> Seeded (seed 42) and deterministic: a fresh run reproduces the archive
> exactly. Diff with
> `python verification/compare_results.py <archive> <fresh>`.

## Adaptive ablations (exp8, C4) — R3, archived

```bash
python -m experiments.exp8_adaptive_ablations.run unsw_nb15 --n 2000 --seed 42
```

Expected files: `results/logs/exp8_adaptive_ablations/metrics_unsw_nb15.json`,
`results/tables/exp8_ablations_unsw_nb15.csv`. Basis-fixed 10.30% vs undefended
10.50% (z = −0.207, p = 0.836, not significant).

## False-positive rate (exp6, C15) — R3, archived

```bash
python -m experiments.exp6_false_positive_rate.run unsw_nb15
```

Expected file: `results/logs/exp6_false_positive_rate/metrics_unsw_nb15.json`.
21,509 benign packets; 0.00% FPR; rotation triggered 1.21%.

## Stationarity (exp14, C8)

```bash
python -m experiments.exp14_stationarity.run
```

Expected files: `results/tables/exp14_stationarity_{unsw_nb15,cicids2017}.csv`.

Expected properties: control ν ≡ 0 at every γ; ν grows with γ; sin(Δθ/2)
scaling fit error 31% (UNSW) / 71% (CICIDS).

## Performance (C15)

```bash
python -m experiments.exp7_latency_benchmark.run     # 7.08 ms mean, 20.79 ms p99 (prose)
python -m experiments.exp18_rust_bench.run           # 53.3 ns/query (criterion, in-process)
```

exp18 bench archive: `results/logs/exp18_rust_bench/bench_summary.json`
(re-aggregate with `python -m experiments.exp18_rust_bench.run --no-run`).

## Third dataset (exp24, C18)

```bash
python -m experiments.exp24_ton_iot_transfer.run                # runs exp19/20/21/23 on ton_iot
python -m experiments.exp24_ton_iot_transfer.run --prepare --train
```

Expected files: `results/tables/exp{19,20,21,23}_*_ton_iot.csv` and
`data/ton_iot/exp24_summary.md`.
