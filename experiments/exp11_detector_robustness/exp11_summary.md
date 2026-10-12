# EXP11 — Detector Robustness (UNSW-NB15 + CICIDS2017, real data)

Run: `python -m experiments.exp11_detector_robustness.run <dataset>`. All
computation completed on the first pass (metrics.json intact); a
summary-CSV parsing bug (`", k="` vs `,`) crashed the CSV write on BOTH
datasets and was fixed, then CSVs were regenerated from the saved metrics —
no results were lost. NOTE: the exp11 metrics.json holds the LATEST run
(CICIDS2017) — the UNSW numbers below come from the crash log
(`results/logs/exp11_detector_unsw.log`) and were also emitted as CSVs
before the file was overwritten; preserve this caveat when citing both
datasets from one file.

Metrics: `results/logs/exp11_detector_robustness/metrics.json`; tables:
`results/tables/exp11_detector_sweep.csv`, `exp11_roc.csv`,
`exp11_adaptive_evaders.csv`.

## Part A — adaptive evaders (P0-1, P0-2)

| evader | verified ASR | internal ASR | flag rate | client detected |
|---|---:|---:|---:|---|
| NES baseline (no evasion) | 0.00% | 0.00% | 97.8% | True |
| NES + query jitter (scale 0.12 > τ=0.1) | 0.00% | 0.00% | 78.1% | True |
| NES + multi-identity (5 ids) | 0.00% | 0.00% | **1.2%** | True |
| NES + multi-identity (10 ids) | 0.00% | 0.00% | **1.3%** | True |
| HSJ baseline | 0.00% | 97.1% | 91.1% | True |
| HSJ + min-distance (0.15) | 0.00% | 97.7% | 90.4% | True |

Read against exp10's verified-ASR metric: **the identity-rotation evader
works against the per-session detector** — with ≥5 rotating client ids,
queries are split into per-id windows with no cluster exceeding k_min, so
the per-session flag rate collapses to ≈1% (the detector never aggregates
across identities). NES baseline stays at zero verified ASR either way, so
on THIS dataset the evader gains attack cover without changing the outcome;
the hotspot-detection claim must be softened to "covers the blind spots of
momentum triggers *per identity*, not against identity-rotation."
(Exp10's verified-ASR = 0 under detection remains true; exp11 shows the
detection that enforces it is evadeable by identity splitting.)

Caveat on HSJ + min-distance: HSJ's bisection inherently re-queries points
near the boundary, so raising the min-distance gate reduces internal ASR
only marginally (97.1%→97.7% here — noise) and the flag rate barely moves
(91.1%→90.4%). Distance-gating is NOT an effective HSJ evader against this
detector; the summary CSV logs it for completeness.

## Part B — detector sweep + ROC (P0-3)

Full grid: τ ∈ {0.05,0.1,0.2,0.3,0.5} × k ∈ {3,5,8,12,20}; benign-boundary
shard (259 flows, smallest-margin benign) for FPR.

- Best operating point (τ=0.1-family): **NES TPR 96.8% @ FPR 0.1%, HSJ TPR
  89.8% @ FPR 0.1%** (k = 7-count threshold on the ROC).
- Benign-boundary FPR ≤ 5.3% across the grid (0% at most points; 5.3% at
  τ=0.1, k=5, i.e. the boundary shard is exactly where false alarms live).
- HSJ flag rate stays ≥ 81% at every grid point — HSJ's cluster structure
  survives aggressive τ.

## Part C — quantize-then-hash vs L2-density (P0-3)

| detector | best config | flag rate |
|---|---|---:|
| L2-density (reference) | τ=0.1 | NES 97.7% / HSJ 91.4% |
| Quantize-then-hash, best | q=0.1, r=20 | NES 97.4% / HSJ 84.3% |

Hash-based detection MATCHES the L2 proxy only at the loosest tolerance
(r=20 Hamming neighbors over wide quantization q=0.1); at tighter
tolerances it collapses (q=0.05, r=20: NES 90.9%; q=0.02: ≤ 9% everywhere).
Honest conclusion: quantize-then-hash (Blacklight-style) is a weaker
detector for NES-style probes on tabular flows than L2-density at matched
settings — real-valued query neighborhoods are not captured by coordinate
quantization at the scales tried, and wide-column quantization is required
to recover detectability.

## Bottom line for the paper

Per-identity L2-density detection is strictly stronger than
quantize-then-hash on these streams, but is defeated (as an aggregator) by
client-identity rotation: detection covers the *momentum-trigger blind
spot* (persistent probes are visible per-identity) yet does not provide a
security boundary against multi-identity adversaries. Soften exp10's
"detection closes the blind spots" to "detection closes the blind spots
per identity; identity rotation is the remaining bypass, addressable only
at the network level (per-row rate/identity correlation), out of scope for
the model-level defense."

## CICIDS2017 (second dataset — detector_sweep/ROC/hash numbers from the
## latest metrics.json)

| evader (CICIDS) | internal ASR | flag rate |
|---|---:|---:|
| NES baseline | 0.00% | 97.8% |
| NES + jitter (0.12) | 0.00% | 83.2% |
| NES + multi-identity (5 ids) | 0.00% | **45.5%** |
| NES + multi-identity (10 ids) | 0.00% | **45.2%** |
| HSJ baseline | 14.50% | 92.6% |
| HSJ + min-distance (0.15) | 15.15% | 91.8% |

ROC best points (τ = 0.1): NES TPR 96.8% @ FPR 0.1%, HSJ TPR 91.5% @ FPR
0.1% — cross-dataset consistent with UNSW.

Quantize-then-hash on CICIDS: NES best (q=0.1, r=20) flag 5.4% vs L2 proxy
45.0% — the hash detector is strictly worse here (and its benign worst-case
FPR over the grid is 84–88%, i.e. at the tolerances where it detects NES
probes it also flags benign boundary traffic). HSJ/hash best 91.2% ≈ L2
39.1%, but again with catastrophic benign FPR. Cross-dataset conclusion:
quantize-then-hash does not transfer to tabular NIDS streams at safe
operating points; L2-density remains the only detector with usable FPR.

The identity-rotation evader also degrades detection on CICIDS (97.8% →
45.5% at 5 ids), though not as completely as on UNSW (→ 1.2%): CICIDS's
dense benign rate keeps per-id windows above k_min more often. The
qualitative conclusion is dataset-independent: identity rotation attacks
the per-session aggregation assumption.
