# Exp 22: Differentiability Split (RQ5 — architecture table)

## Question
Does the phenomenon depend on differentiability? The reviewer's matrix:
coupling should be differentiability-dependent; trigger blindness and
stationarity violations should NOT be.

## Method
Four architectures on UNSW-NB15, identical TASCP-family wrapper
(variant=tascp, gamma=50, tau=0), identical attack budget:
  mlp            NIDSClassifier checkpoint (differentiable)
  transformer    TransformerClassifier checkpoint (differentiable)
  xgboost        trained XGBClassifier checkpoint (non-differentiable,
                 margins via predict(output_margin=True))
  random_forest  sklearn RF, trained on the fly (80k labeled rows,
                 n_estimators=100) — NO RF checkpoint exists in this repo;
                 labeled as trained-on-the-fly in all outputs
Mechanisms: coupling C_norm (autograd — n/a for XGB/RF, reported as null,
NOT zero); naive/adaptive/fixed-state PGD (differentiable only);
stationarity replay (forward-only, ALL architectures); trigger fire rates
under black-box attacks (NES-independent, antithetic NES jitter, Square —
all architectures).

## Results
| architecture | coupling C | dASR    | nu_session | momentum fire (NES / antithetic) |
|--------------|-----------|---------|------------|----------------------------------|
| mlp          | 6.7e7     | +11.0pp | 6.62%      | 100.0% / 77.0%                   |
| transformer  | 7.9e7     | +0.0pp  | 19.75%     | 100.0% / 76.5%                   |
| xgboost      | n/a       | n/a     | 1.96%      | 100.0% / 74.5%                   |
| random_forest| n/a       | n/a     | 0.46%      | 99.5% / 78.5%                    |

## Findings (honest)
1. **Coupling is differentiability-dependent**: the C audit is only
   definable through autograd; for XGB/RF it is *not measurable*, not zero.
   This matches the reviewer's predicted table (coupling: differentiable
   bases only).
2. **Stationarity violations are NOT differentiability-dependent**:
   nu_session > 0 for ALL four architectures (transformer is the worst at
   19.75%; even the RF shows 0.46%). State evolution changes verdicts on
   non-differentiable models too.
3. **Trigger blindness is NOT differentiability-dependent**: momentum fire
   rates are 99.5-100% (NES) across all architectures.
4. Caveat: the transformer shows measurable coupling (C=7.9e7) but the B2
   replica attack did not convert it into an ASR increase (+0.0pp) at this
   budget — an open channel that was not exploited here; reported as-is,
   not smoothed.

## Artifacts
- results/tables/exp22_arch_split_unsw_nb15.csv
- results/logs/exp22_arch_split/metrics_unsw_nb15.json
- results/logs/runs_session5/exp22_unsw.log
