# Reviewer Verification Checklist

Verify the artifact systematically. Each section maps to a claim or a
reproducibility requirement. Checked items are expected to pass on a clean
machine with the shipped preprocessed data.

## Environment Setup
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Environment pins match (`python3 scripts/verify_environment.py`)
- [ ] Checkpoint manifest passes (`python3 verification/verify_manifest.py`)
- [ ] Dataset manifest passes (`python3 verification/verify_datasets.py`)
- [ ] Traceability check passes (`python3 scripts/check_traceability.py`)
- [ ] Test suite passes (`make test` → 67 passed)

## Claim C1–C3 — The mechanism behaves as specified (NOT implemented)
- [ ] **Note:** exp1–exp4 are placeholders (exit code 2); no implementation or
      archive is shipped. See `doc/OPEN_SCIENCE.md`. Do not mark these as passed.

## Claim C4 — Adaptive BPDA breaks the defense (white-box)
- [ ] `experiments/exp5_adaptive_bpda_attack/run.py unsw_nb15 --n 2000 --seed 42` executes
- [ ] `results/logs/exp5_adaptive_bpda_attack/metrics_unsw_nb15.json` reports 53.60% adaptive vs 10.50% undefended
- [ ] exp8 basis-fixed collapses to ~10.30% (p = 0.836) in `results/logs/exp8_adaptive_ablations/metrics_unsw_nb15.json`
- [ ] Fresh run passes `verification/compare_results.py` (exact for seed 42)

## Claim C5 — Black-box trigger blindness (Corollary 3)
- [ ] `results/logs/exp9_blackbox/metrics.json` present
- [ ] NES `rotation_triggered_frac` = 0.0
- [ ] HSJ UNSW undefended 0.9710 → TASCP 0.9070
- [ ] HSJ CICIDS2017 undefended 0.1505 → TASCP 0.8740

## Claim C6 — Detection, not perturbation, is viable
- [ ] `results/logs/exp10_hybrid/metrics.json` present
- [ ] Detector benign FPR 0.00%
- [ ] Attack-query flag rate 91–98%
- [ ] Reject policy → 0.00% verified ASR

## Claim C7 — White-box contrast replicates (5 seeds)
- [ ] `results/logs/exp13_whitebox_multiseed/multiseed_unsw_nb15.json` present
- [ ] Undefended pooled 0.1065; adaptive pooled 0.5299
- [ ] Basis-fixed pooled 0.1047 (overlaps undefended)
- [ ] Fresh run passes `verification/compare_results.py` within ±0.02 pooled

## Claim C8 — Verdict stationarity
- [ ] `results/tables/exp14_stationarity_*.csv` present
- [ ] Control ν ≡ 0; ν grows with γ
- [ ] sin(Δθ/2) fit error 31% (UNSW) / 71% (CICIDS)

## Claim C9 — Reference math is faithful
- [ ] `results/logs/exp16_refmath/metrics_unsw_nb15.json` present
- [ ] `lemma3_fd_vs_autograd.median_rel_err_all` ≈ 1.37e-5 (9,216 comparisons)
- [ ] `lemma2_rho.rho_mean` ≈ 0.198
- [ ] `cor3_mt.with_anchor.mean` ≈ −0.70, P(M>0) ≈ 1.6%

## Claim C10 — Offline attacker bit-identity
- [ ] `results/tables/exp15_offline_unsw_nb15.csv` present
- [ ] TASCP vs DACM max abs diff = 0.0

## Claim C11 — Channel is not rotation-specific
- [ ] exp19 family table present (`results/tables/exp19_family_*.csv`)
- [ ] exp20 coupling scatter present; ρ(log C_norm, dASR) ≈ +0.086 (p ≈ 0.65, n.s.)

## Claim C12 — Trigger family dependence
- [ ] exp21 trigger matrix: momentum fires 87–100%; other triggers collapse (0–1%)

## Claim C13 — Architecture split
- [ ] exp22 arch split: coupling not measurable for XGB/RF (undefended ASR ≈ 0.85%)
- [ ] exp_transformer: undef 1.05% / naive 12.50% / adaptive 7.00% (`results/cross_architecture_transformer.md`)

## Claim C14 — Defense audit
- [ ] exp23 audit table present: TASCP fails all three; detector ν_allow = 0

## Claim C15 — Deployable and still pointless
- [ ] exp6: 0.00% FPR (`results/logs/exp6_false_positive_rate/metrics_unsw_nb15.json`, R3)
- [ ] exp7: 7.08 ms mean (prose; R4)
- [ ] exp18: `results/logs/exp18_rust_bench/bench_summary.json` ≈ 53.3 ns/query

## Claim C16 — Baseline comparison (exp4, NOT implemented)
- [ ] **Note:** AT/RS checkpoints are not shipped; exp4 is a placeholder. The
      DACM/TASCP/undefended comparison is covered by exp5.

## Claim C17 — Hyperparameter robustness (exp17)
- [ ] `results/logs/exp17_hp_sweep/metrics_unsw_nb15.json` present
- [ ] Adaptive beats undefended on 6/27 grid points (best 1.4%); basis-fixed within 2pp on 26/27

## Claim C18 — Third dataset (exp24, NF-ToN-IoT-v2)
- [ ] `data/ton_iot/exp24_summary.md` present; test acc 0.9819 / F1 0.9860
- [ ] Driver `experiments/exp24_ton_iot_transfer/run.py` runs exp19/20/21/23 on `ton_iot`

## Manifests
- [ ] `experiment_manifests/` contains exactly one manifest per experiment (exp1–exp24 + transformer = 25 files)
- [ ] Each manifest's `claim` matches `CLAIM_MAP.md`
- [ ] Regenerate with `python scripts/gen_experiment_manifests.py`

## Provenance & Limitations
- [ ] `PROVENANCE.md` documents the exp5 narrative revision
- [ ] `LIMITATIONS.md` understood (exp1–exp4 NR, exp7 R4, unseeded attacks, dataset reconstruction)
- [ ] `doc/OPEN_SCIENCE.md` maps every experiment ID → claim → output
- [ ] `doc/adr/` contains ADR-001 through ADR-004

## Artifact Integrity
- [ ] All 5 checkpoints verify against `manifest/checkpoint_sha256.json`
- [ ] All 3 datasets verify against `manifest/dataset_sha256.txt`
- [ ] No author-identifying information in committed files
