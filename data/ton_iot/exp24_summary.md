# Exp 24: Third Dataset — NF-ToN-IoT-v2 (RQ5 / Tier-1 D)

## Question
Does the mechanism survive a different feature schema, class distribution,
and flow-generation process?

## Dataset
**NF-ToN-IoT-v2** (test split), from Nora9029/NF-ToN-IoT-v2 (HuggingFace):
41 NetFlow-v9 numeric features (byte/packet counters, durations,
throughputs, TCP windows, per-direction retransmissions, DNS/ICMP/FTP
fields), 3,388,100 flows, 64.0% attacks — a genuinely different tabular
security dataset from UNSW-NB15 / CICIDS2017 (NetFlow schema vs
packet/flow hybrids; IoT attack traffic vs enterprise). Source IPs/ports
dropped; log1p + min-max normalization to [0,1] (the repo's other
pipelines operate in [0,1]; raw NetFlow scales 1e0-1e7 diverge in MLP
training). Classifier: same NIDSClassifier architecture and protocol as
the other datasets (2x64 MLP, Adam 1e-3, 5 epochs, stratified 80/20):
test accuracy 0.9819 (F1 0.9860), confusion in
`results/logs/runs_session5/exp24_train.log`.

## Protocol (held identical to UNSW/CICIDS runs, per the reviewer's
requirement): same PGD budget (eps 0.3, alpha 0.01, 20 it, L2), same
trigger calibration, same transformation family hyperparameters
(gamma=50, tau=0, window 5), same coupling calculation, same stationarity
replay, same B1 attack suite.

## Results (cross-reference the other summaries)
- exp19 family: adaptive-vs-fixed-state dASR = +4.8pp (tascp),
  +0.4pp (tscale), +10.6pp (tshift), -2.2pp (tshear), +0.0pp (tnoise);
  nu_session > 1% for 5/5 variants (37.0% TASCP).
- exp20 coupling sweep: Spearman rho(log C_norm, dASR) = +0.086 (p=0.65) —
  same null-magnitude finding as UNSW; open-channel/closed-channel
  distinction holds.
- exp21 trigger matrix: momentum fires 87-100% for every attack family
  here; cosine/sign/displacement still collapse on specific families
  (0-1%) — family dependence persists, with a dataset-dependent profile.
- exp23 defense audit: TASCP nu_allow 18.5% (adversarial verdicts flip
  back to benign), detectors/rate-limiter 0%.

## Finding
The trajectory-state mechanism (coupling channel, trigger blindness,
stationarity violation) transfers to a different schema, class balance,
and flow-generation process. The PROFILE differs (which trigger is
separable, which transform amplifies vs suppresses) — consistent with the
paper's claim that these are audit outcomes, not constants.

## Reproduction
- `data/ton_iot/prepare_ton_iot.py` (CSV -> processed parquet)
- `data/ton_iot/train_ton_iot.py` (-> src/tascp_eval/models/weights/ton_iot_nids.pth)
- `python -m experiments.exp19_trajectory_family.run ton_iot` etc.
- Data provenance + preparation documented in data/ton_iot/PROVENANCE.md

## Artifacts
- results/tables/exp19_family_ton_iot.csv, exp20_coupling_scatter_ton_iot.csv,
  exp21_trigger_matrix_ton_iot.csv, exp23_defense_audit_ton_iot.csv
- results/logs/runs_session5/exp24_prepare.log, exp24_train.log
