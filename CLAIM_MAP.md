# Scientific Claim Map

Stable identifiers for every scientific claim in the TASCP artifact. Each claim
links to its machine-readable manifest(s), evidence, generating script, and
output. Every one of the 24 numbered experiments (`exp1`–`exp24`) plus the
cross-architecture `exp_transformer` has a one-to-one manifest under
`experiment_manifests/`. All numbers are drawn from files shipped in this
repository; where an output is **not** archived, that is stated explicitly.

## Claim → Evidence

| ID | Paper Claim | Manifest(s) | Evidence | Script | Output |
|---|---|---|---|---|---|
| C1 | Momentum separates benign from adversarial trajectories (Assumptions 1–2) | `E-EXP01-MOMENTUM-SEPARATION.json` | **Not implemented** (placeholder; no archived evidence) | `experiments/exp1_momentum_separation/run.py` (placeholder) | none (NR) |
| C2 | Gradient correlation decays as `cos∠(g_t,g_t') = cos(Δθ)` (Lemma 2) | `E-EXP02-GRADIENT-CORRELATION.json` | **Not implemented** (placeholder; no archived evidence) | `experiments/exp2_gradient_correlation/run.py` (placeholder) | none (NR) |
| C3 | Threshold-avoidant adversaries are query-bounded (Proposition 1) | `E-EXP03-QUERY-BUDGET.json` | **Not implemented** (placeholder; no archived evidence) | `experiments/exp3_query_budget/run.py` (placeholder) | none (NR) |
| C4 | Adaptive BPDA breaks the defense; the data-dependent basis is the leak | `E-EXP05-ADAPTIVE-BPDA-ATTACK.json`, `E-EXP08-ADAPTIVE-ABLATIONS.json` | ASR 53.60% adaptive vs 10.50% undefended; basis-fixed collapses to 10.30% (p=0.836) | `experiments/exp5_adaptive_bpda_attack/run.py`, `experiments/exp8_adaptive_ablations/run.py` | `results/logs/exp5_adaptive_bpda_attack/metrics_unsw_nb15.json`, `results/logs/exp8_adaptive_ablations/metrics_unsw_nb15.json` (R3) |
| C5 | Black-box trigger blindness (Corollary 3): NES never fires the rotation | `E-EXP09-BLACKBOX-HSJ.json`, `E-EXP12-CICIDS-DISAMBIGUATION.json` | NES rotation fires 0.00%; HSJ UNSW 97.10%→90.70%; HSJ CICIDS 15.05%→87.40%; placebo 33.75% ≠ 87.40% | `experiments/exp9_blackbox_attack/run.py`, `experiments/exp12_cicids_disambiguation/run.py` | `results/logs/exp9_blackbox/metrics.json`, `results/logs/exp12_cicids_disambiguation/metrics.json` |
| C6 | Per-session detection, not perturbation, is the viable stateful mechanism | `E-EXP10-HYBRID-DEFENSE.json`, `E-EXP11-DETECTOR-ROBUSTNESS.json` | Detector 0.00% benign FPR, flags 91–98% of attack queries; multi-identity flag rate 1.2–1.3% | `experiments/exp10_hybrid_detection_sim/run.py`, `experiments/exp11_detector_robustness/run.py` | `results/logs/exp10_hybrid/metrics.json`, `results/tables/exp11_*.csv` |
| C7 | The white-box contrast replicates across seeds (5-seed) | `E-EXP13-WHITEBOX-MULTISEED.json` | UNSW adaptive 52.99% [52.01,53.97] vs undefended 10.65% [10.06,11.27]; basis-fixed ≈ undefended | `experiments/exp13_whitebox_multiseed/run.py` | `results/logs/exp13_whitebox_multiseed/multiseed_{unsw_nb15,cicids2017}.json` |
| C8 | Verdict non-stationarity: ν grows with γ; controls ν ≡ 0 | `E-EXP14-STATIONARITY.json` | sin(Δθ/2) scaling is small-angle only (fit error 31–71%) | `experiments/exp14_stationarity/run.py` | `results/tables/exp14_stationarity_*.csv` |
| C9 | Reference math is faithful to the theory | `E-EXP16-REFMATH-VALIDATION.json` | Autograd vs central FD median rel err 1.37e-5 (9,216 comparisons); E[M_t] = −0.70 | `experiments/exp16_refmath_validation/run.py` | `results/logs/exp16_refmath/metrics_unsw_nb15.json` |
| C10 | A fresh offline flow defeats TASCP/DACM identically | `E-EXP15-OFFLINE-SCOPE.json` | TASCP verdict bit-identical to DACM/undefended (max abs diff 0.0) | `experiments/exp15_offline_and_scope/run.py` | `results/tables/exp15_*.csv` |
| C11 | The channel is not rotation-specific; coupling magnitude ≠ exploitability | `E-EXP19-TRANSFORMATION-FAMILY.json`, `E-EXP20-COUPLING-SWEEP.json` | rotation +10.4pp, translation +42.4pp adaptive-vs-fixed; ρ(log C_norm, dASR) = +0.086 (p=0.65) | `experiments/exp19_trajectory_family/run.py`, `experiments/exp20_coupling_audit/run.py` | `results/tables/exp19_family_*.csv`, `exp20_coupling_scatter_*.csv` |
| C12 | Stateful triggers are attack-family dependent | `E-EXP21-TRIGGER-MATRIX.json` | Momentum fires 87–100%; cosine/sign/displacement collapse to 0–1% on specific families | `experiments/exp21_trigger_matrix/run.py` | `results/tables/exp21_trigger_matrix_*.csv` |
| C13 | Coupling is differentiability-dependent; trigger blindness is not | `E-EXP22-ARCHITECTURE-SPLIT.json`, `E-EXP-CROSS-ARCH-TRANSFORMER.json` | XGB/RF coupling not measurable (undefended ASR 0.85%); trigger blindness + stationarity universal; Transformer undef 1.05% / naive 12.50% / adaptive 7.00% | `experiments/exp22_arch_split/run.py`, `experiments/exp_transformer/run.py` | `results/tables/exp22_arch_split_unsw_nb15.csv`, `results/cross_architecture_transformer.md` |
| C14 | Audits separate defense classes: perturbation fails all three | `E-EXP23-DEFENSE-AUDIT.json` | TASCP fails all 3; detector + rate limiter monotone-safe (ν_allow = 0) with benign cost; NF-ToN-IoT ν_allow 18.5% | `experiments/exp23_defense_audit/run.py` | `results/tables/exp23_defense_audit_*.csv` |
| C15 | The defense is deployable and still pointless | `E-EXP06-FALSE-POSITIVE-RATE.json`, `E-EXP07-LATENCY.json`, `E-EXP18-RUST-BENCH.json` | 0.00% FPR (21,509 benign packets); 7.08 ms HTTP mean; 53.3 ns/query in-process | `experiments/exp6_false_positive_rate/run.py`, `exp7_latency_benchmark/run.py`, `experiments/exp18_rust_bench/run.py` | `results/logs/exp6_false_positive_rate/metrics_unsw_nb15.json` (R3); exp18 `results/logs/exp18_rust_bench/bench_summary.json` (R3); exp7 prose only (R4) |
| C16 | TASCP underperforms dedicated defenses | `E-EXP04-DEFENSE-COMPARISON.json` | **Not reproducible** — AT/RS checkpoints absent; DACM/TASCP portion covered by exp5 | `experiments/exp4_defense_comparison/run.py` (placeholder) | none (NR) |
| C17 | The (modest) robustness gain is configuration-sensitive | `E-EXP17-HP-SWEEP.json` | Undefended 9.80%; adaptive beats undefended on 6/27 grid points (best 1.4% at γ=10, τ=0); basis-fixed within 2pp on 26/27 | `experiments/exp17_hp_sweep/run.py` | `results/logs/exp17_hp_sweep/metrics_unsw_nb15.json`, `results/tables/exp17_hp_sweep_unsw_nb15.csv` |
| C18 | The mechanism survives a different dataset (NF-ToN-IoT-v2) | `E-EXP24-THIRD-DATASET.json` | Test acc 0.9819 (F1 0.9860); family δASR +4.8pp; ν_allow 18.5% | `data/ton_iot/prepare_ton_iot.py`, `data/ton_iot/train_ton_iot.py` | `data/ton_iot/exp24_summary.md`, `results/logs/runs_session5/exp24_train.log` |

## Experiment → Manifest Index (24 numbered + transformer)

| Exp | Manifest (`experiment_manifests/`) | Claim | Level |
|---|---|---|---|
| exp1 | `E-EXP01-MOMENTUM-SEPARATION.json` | C1 | NR |
| exp2 | `E-EXP02-GRADIENT-CORRELATION.json` | C2 | NR |
| exp3 | `E-EXP03-QUERY-BUDGET.json` | C3 | NR |
| exp4 | `E-EXP04-DEFENSE-COMPARISON.json` | C16 | NR |
| exp5 | `E-EXP05-ADAPTIVE-BPDA-ATTACK.json` | C4 | R3 |
| exp6 | `E-EXP06-FALSE-POSITIVE-RATE.json` | C15 | R3 |
| exp7 | `E-EXP07-LATENCY.json` | C15 | R4 |
| exp8 | `E-EXP08-ADAPTIVE-ABLATIONS.json` | C4 | R3 |
| exp9 | `E-EXP09-BLACKBOX-HSJ.json` | C5 | R3 |
| exp10 | `E-EXP10-HYBRID-DEFENSE.json` | C6 | R3 |
| exp11 | `E-EXP11-DETECTOR-ROBUSTNESS.json` | C6 | R3 |
| exp12 | `E-EXP12-CICIDS-DISAMBIGUATION.json` | C5 | R3 |
| exp13 | `E-EXP13-WHITEBOX-MULTISEED.json` | C7 | R3 |
| exp14 | `E-EXP14-STATIONARITY.json` | C8 | R2 |
| exp15 | `E-EXP15-OFFLINE-SCOPE.json` | C10 | R3 |
| exp16 | `E-EXP16-REFMATH-VALIDATION.json` | C9 | R2 |
| exp17 | `E-EXP17-HP-SWEEP.json` | C17 | R3 |
| exp18 | `E-EXP18-RUST-BENCH.json` | C15 | R3 |
| exp19 | `E-EXP19-TRANSFORMATION-FAMILY.json` | C11 | R3 |
| exp20 | `E-EXP20-COUPLING-SWEEP.json` | C11 | R3 |
| exp21 | `E-EXP21-TRIGGER-MATRIX.json` | C12 | R3 |
| exp22 | `E-EXP22-ARCHITECTURE-SPLIT.json` | C13 | R3 |
| exp23 | `E-EXP23-DEFENSE-AUDIT.json` | C14 | R2 |
| exp24 | `E-EXP24-THIRD-DATASET.json` | C18 | R3 |
| exp_transformer | `E-EXP-CROSS-ARCH-TRANSFORMER.json` | C13 | R3 |

> Level **NR** = not reproducible: the artifact ships neither source nor
> archived output (exp1–exp4; see `doc/OPEN_SCIENCE.md`). Level **R4** = prose
> only (exp7; the Rust HTTP server path is documented but its numbers are not
> archived). Levels R1–R4 are defined in `REPRODUCIBILITY_LEVELS.md`.

## Claim Dependencies

```
C1,C2  (mechanism behaves as specified)  ─┐
C3     (query bound)                     ─┼─> C4 (white-box leak)
C9     (reference math faithful)         ─┘        │
                                                   ├─> C7  (multi-seed replication)
                                                   ├─> C11 (channel not rotation-specific)
                                                   ├─> C13 (architecture split)
                                                   └─> C17 (hyperparameter robustness)

C5,C6  (black-box blindness / detection) ──> C10 (offline bit-identity) ──> C14 (defense audit)
C8,C12 (stationarity, trigger scope)     ──> C14
C15    (deployable but pointless) validates the failure is structural, not operational
C16, C18 (baseline comparison, third dataset) bound the generality of the failure
```

## Cross-References

- Threat model: `doc/THREAT_MODEL.md`
- Formal results: `doc/THEOREMS.md`
- Evaluation structure: `doc/EVALUATION_GUIDE.md`
- Decision records: `doc/adr/`
- Provenance: `PROVENANCE.md`
- Limitations: `LIMITATIONS.md`
