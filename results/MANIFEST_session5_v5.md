# MANIFEST — tascp_session_changes v5 (session 5, 2026-10-08)

Delta over v4: reviewer-response experiments implementing the Tier 1-3
roadmap (RQ1-RQ6). Zip base: git archive 841cc2f..HEAD + gitignored
logs/metrics appended (24 extras; 76 entries total, testzip clean).

## What's in the delta

| RQ | Experiment | Deliverables |
|----|------------|--------------|
| RQ1 | exp19 trajectory family | experiments/exp19_trajectory_family/{family.py,run.py,exp19_summary.md}; tables exp19_family_{unsw_nb15,ton_iot}.csv |
| RQ2 | exp20 coupling audit | experiments/exp20_coupling_audit/{run.py,exp20_summary.md}; tables exp20_coupling_scatter_{unsw_nb15,ton_iot}.csv |
| RQ3 | exp21 trigger matrix | experiments/exp21_trigger_matrix/{run.py,exp21_summary.md}; tables exp21_trigger_matrix_{unsw_nb15,ton_iot}.csv |
| RQ4 | stationarity | inside exp19 (replay) + exp23 (nu_allow); documented in both summaries |
| RQ5 | exp22 arch split + exp24 third dataset | experiments/exp22_arch_split/{run.py,exp22_summary.md}; data/ton_iot/{prepare_ton_iot.py,train_ton_iot.py,exp24_summary.md,PROVENANCE.md}; src/tascp_eval/models/weights/ton_iot_nids.pth; configs/attack/domain_constraints_toniot.yaml |
| RQ6 | exp23 defense audit | experiments/exp23_defense_audit/{run.py,exp23_summary.md}; tables exp23_defense_audit_{unsw_nb15,ton_iot}.csv |

Support: tests/test_trajectory_family.py (5 tests; suite 33/33 green),
exp9_common extended for ton_iot, doc/THEOREMS.md RQ1-RQ6 restructure,
doc/EVALUATION_GUIDE.md reviewer-objection rows, README.md RQ table.

## Headline results (real data)

- RQ1: channel not rotation-specific — rotation +10.4pp, translation
  +42.4pp, scaling +4.8pp adaptive-vs-fixed-state (UNSW); shear
  suppresses (-48.8pp); noise inert. Variation, not uniform failure.
- RQ2: honest null — coupling magnitude does NOT predict exploitability
  (Spearman -0.126 UNSW / +0.086 TON-IoT, p > 0.5/0.65). C=0 exactly
  identifies the closed channel; geometry sets the failure mode.
- RQ3: no trigger fires on >50% of sessions for every attack family
  (UNSW); trajectory statistics are attack-family dependent.
- RQ4: stationarity violations family-wide (5/5 variants on TON-IoT,
  up to 37.0%); toward-allow flips TASCP-only (36.2% UNSW / 18.5%
  TON-IoT) vs 0% for detectors.
- RQ5: coupling differentiable-only (XGB/RF: not measurable, not zero);
  trigger blindness + stationarity violations universal (all four
  architectures). Third dataset NF-ToN-IoT-v2: mechanism transfers
  (3,388,100 NetFlow flows, 64% attacks; classifier acc 0.9819).
- RQ6: audits separate defense classes — TASCP fails all three;
  Blacklight-style detector + rate limiter are monotone-safe
  (nu_allow = 0) with a documented benign-traffic cost.

## Honest limitations

1. exp20's coupling-magnitude-vs-effect correlation is NULL on both
   datasets — the paper must claim open/closed channel necessity, not
   scalar predictivity (reviewer point 2 answered with a refinement,
   not the hoped-for monotone figure).
2. exp19/exp23 family members are controlled synthetic defenses, not
   published systems (except the exp10-derived Blacklight-style
   simplification and a generic rate limiter in exp23).
3. blacklight_sim flags 100% of benign repeated-recheck sessions at
   k_min=5/tau=0.1 — the detect/reject safety carries this benign cost;
   exp10/exp11's distinct-query benign calibration is the complement.
4. Transformer: open coupling channel (C=7.9e7) but the B2 replica
   attack did not exploit it at this budget (dASR +0.0pp) — reported as
   measured, not smoothed.
5. NF-ToN-IoT-v2 is the dataset's test split (smaller download); IPs
   dropped; features log1p+min-max normalized (documented in
   data/ton_iot/PROVENANCE.md).
6. RF trained on the fly (no RF checkpoint exists in the repo) — labeled
   as such in exp22 outputs.

## Reproduction

python -m experiments.exp19_trajectory_family.run [unsw_nb15|ton_iot]
python -m experiments.exp20_coupling_audit.run [dataset]
python -m experiments.exp21_trigger_matrix.run [dataset]
python -m experiments.exp22_arch_split.run [unsw_nb15]
python -m experiments.exp23_defense_audit.run [dataset]
(data prep + training: python data/ton_iot/prepare_ton_iot.py &&
 python data/ton_iot/train_ton_iot.py)
Run logs: results/logs/runs_session5/*.log
