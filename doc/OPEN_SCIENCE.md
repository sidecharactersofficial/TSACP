# Open-Science Experiment Inventory

This is the artifact-side counterpart to the paper's open-science statement.
The paper PDF itself is **not shipped** with the artifact (the manuscript is
redacted for double-blind review), so this table is the authoritative
experiment-ID → claim → code → output mapping.

## ID reconciliation

The paper's open-science statement enumerates the **core** experiments
(`exp1`–`exp10`) plus the cross-dataset / cross-architecture / black-box
studies. The reviewer-response roadmap references experiments **up to 24**.
Both are the same suite: `exp1`–`exp10` are the core manuscript experiments and
`exp11`–`exp24` (plus `exp_transformer`) are the reviewer-response extensions.
This artifact ships all 24 numbered manifests plus the Transformer manifest, so
the "exp1–10" statement and the "up to 24" roadmap no longer disagree — they
describe the core and the full suite respectively.

## Inventory

| Exp | Claim | What it tests | Code entry point | Archived output | Level |
|---|---|---|---|---|---|
| exp1 | C1 | Momentum separation (Assumptions 1–2) | `experiments/exp1_momentum_separation/run.py` (placeholder) | — | **NR** |
| exp2 | C2 | Gradient correlation (Lemma 2) | `experiments/exp2_gradient_correlation/run.py` (placeholder) | — | **NR** |
| exp3 | C3 | Query budget (Proposition 1) | `experiments/exp3_query_budget/run.py` (placeholder) | — | **NR** |
| exp4 | C16 | Defense comparison (AT/RS/DACM/TASCP) | `experiments/exp4_defense_comparison/run.py` (placeholder) | — | **NR** |
| exp5 | C4 | Adaptive BPDA white-box attack | `experiments/exp5_adaptive_bpda_attack/run.py` | `results/logs/exp5_adaptive_bpda_attack/metrics_unsw_nb15.json` | R3 |
| exp6 | C15 | False-positive rate on benign sessions | `experiments/exp6_false_positive_rate/run.py` | `results/logs/exp6_false_positive_rate/metrics_unsw_nb15.json` | R3 |
| exp7 | C15 | HTTP latency benchmark | `experiments/exp7_latency_benchmark/run.py` | `experiments/exp7_latency_benchmark/exp7_summary.md` | R4 |
| exp8 | C4 | Adaptive-attacker ablations | `experiments/exp8_adaptive_ablations/run.py` | `results/logs/exp8_adaptive_ablations/metrics_unsw_nb15.json` | R3 |
| exp9 | C5 | Black-box NES/HSJ attacks | `experiments/exp9_blackbox_attack/run.py` | `results/logs/exp9_blackbox/metrics.json` | R3 |
| exp10 | C6 | Hybrid detection simulation | `experiments/exp10_hybrid_detection_sim/run.py` | `results/logs/exp10_hybrid/metrics.json` | R3 |
| exp11 | C6 | Detector robustness sweep | `experiments/exp11_detector_robustness/run.py` | `results/tables/exp11_*.csv` | R3 |
| exp12 | C5 | CICIDS2017 disambiguation | `experiments/exp12_cicids_disambiguation/run.py` | `results/logs/exp12_cicids_disambiguation/metrics.json` | R3 |
| exp13 | C7 | 5-seed white-box replication | `experiments/exp13_whitebox_multiseed/run.py` | `results/logs/exp13_whitebox_multiseed/multiseed_*.json` | R3 |
| exp14 | C8 | Verdict stationarity | `experiments/exp14_stationarity/run.py` | `results/tables/exp14_stationarity_*.csv` | R2 |
| exp15 | C10 | Offline attacker / scope | `experiments/exp15_offline_and_scope/run.py` | `results/tables/exp15_*.csv` | R3 |
| exp16 | C9 | Reference-math validation | `experiments/exp16_refmath_validation/run.py` | `results/logs/exp16_refmath/metrics_unsw_nb15.json` | R2 |
| exp17 | C17 | Hyperparameter sweep | `experiments/exp17_hp_sweep/run.py` | `results/logs/exp17_hp_sweep/metrics_unsw_nb15.json` | R3 |
| exp18 | C15 | In-process Rust hot-path bench | `experiments/exp18_rust_bench/run.py` → `tascp-proto/fast-loop/benches/hot_path.rs` | `results/logs/exp18_rust_bench/bench_summary.json` | R3 |
| exp19 | C11 | Transformation family | `experiments/exp19_trajectory_family/run.py` | `results/tables/exp19_family_*.csv` | R3 |
| exp20 | C11 | Coupling-magnitude audit | `experiments/exp20_coupling_audit/run.py` | `results/tables/exp20_coupling_scatter_*.csv` | R3 |
| exp21 | C12 | Trigger × attack-family matrix | `experiments/exp21_trigger_matrix/run.py` | `results/tables/exp21_trigger_matrix_*.csv` | R3 |
| exp22 | C13 | Architecture split (MLP vs XGB/RF) | `experiments/exp22_arch_split/run.py` | `results/tables/exp22_arch_split_unsw_nb15.csv` | R3 |
| exp23 | C14 | Defense-class audit | `experiments/exp23_defense_audit/run.py` | `results/tables/exp23_defense_audit_*.csv` | R2 |
| exp24 | C18 | Third dataset (NF-ToN-IoT-v2) transfer | `experiments/exp24_ton_iot_transfer/run.py` | `data/ton_iot/exp24_summary.md`, `results/tables/*_ton_iot.csv` | R3 |
| exp_transformer | C13 | Cross-architecture Transformer | `experiments/exp_transformer/run.py` | `results/cross_architecture_transformer.md` | R3 |

## Known gaps (stated plainly)

- **exp1–exp4 (NR)** — the artifact contains no runner implementation and no
  archived output. Their run.py files are explicit placeholders that exit with
  code 2; `make reproduce-all` skips them. The reproducible portion of exp4
  (DACM / TASCP / undefended) is covered by exp5.
- **exp7 (R4)** — the 7.08 ms HTTP figure is prose-only; reproducing it needs
  the Rust fast-loop server (`tascp-proto/fast-loop`) and a network benchmark
  harness. The in-process cost of the same computation is archived by exp18.
- **Paper PDF** — not shipped (double-blind). Numbers in this artifact can be
  verified against the tables in `results/`, but the manuscript text is not.

## See also

- `CLAIM_MAP.md` — claim → evidence mapping.
- `REPRODUCIBILITY_LEVELS.md` — definition of R1–R4 (+NR).
- `REPRODUCIBILITY_DISCREPANCY_REPORT.md` — the audit this inventory responds to.
