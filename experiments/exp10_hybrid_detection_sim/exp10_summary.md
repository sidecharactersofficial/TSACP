# Exp 10 Summary — Hybrid Detection Simulation

**Date:** 2026-10-07 · **Dataset:** UNSW-NB15 (2,000 attack samples + 1,280 held-out benign) · **Architecture:** MLP (43-64-64-1)

## Headline

A per-session near-duplicate **density detector** (Blacklight-style, L2 proxy)
flags 97.8% (NES) and 91.1% (HSJ) of black-box attack queries at **0.00%
false-positive rate** on held-out benign sessions, and a reject policy drives
verified attack success to **0.00%** for both attack classes. TASCP adds
nothing measurable on top (hybrid ≈ detection-only) — and its own verified
contribution is *negative* in the decision-based regime (transient 90.70%
success collapses to 24.30% on re-query because the defense's verdicts are
non-stationary).

## Why this closes the argument

exp9 showed TASCP's trigger is structurally blind to antithetic NES queries
(0.00% rotation activation) and only taxes HSJ. exp10 shows the complementary
mechanism — similarity/density detection over the session's query window —
covers exactly those blind spots **per client session:**

| Attack | TASCP alone (verified) | Detection alone | Hybrid |
|---|---:|---:|---:|
| NES (250 queries) | 2.35% | **0.00%** | 0.00% |
| HSJ (1,000 queries) | 24.30%² | **0.00%** | 0.00% |

The two stateful mechanisms are complementary in *principle* (disjoint blind
spots, `doc/RELATED_WORK.md`) but asymmetric in *practice*: detection works at
0% FPR here; perturbation does not work at all. The paper's constructive
recommendation follows directly: **stateful NIDS defenses should detect
(query-fingerprint similarity/density), not perturb.**

## Scope update after exp11 (reviewer-response session 4)

exp11's adaptive evaders show the per-session detection above is defeated
(as an aggregator) by **client-identity rotation**: with ≥ 5 rotating client
ids the per-session flag rate collapses (UNSW 97.8% → 1.2%; CICIDS 97.8% →
45.5%) because no single session window ever exceeds k_min. Verified ASR
still stays 0.00% only because those NES probes never succeed anyway; the
robust claim is: detection closes the *momentum-trigger blind spot per
identity*, and identity rotation is the remaining bypass — addressable only
at the network level (identity/rate correlation), not at the model level.
Detection also remains the only mechanism with usable benign FPR:
exp11 shows quantize-then-hash either collapses (NES flag 5.4% on CICIDS)
or carries 84–88% benign FPR at its detection points. Do not overstate
"hybrid detection is secure" — it is strictly better than perturbation,
not a security boundary.

## The non-stationarity finding

Under TASCP-only, HSJ's during-attack success (90.70%, exp9) overstates what
the defender would actually act on: re-querying each final adversarial point
through the defense yields an adversarial verdict for only 24.30% of samples.
The stateful rotation makes classification non-idempotent — the same flow is
attack-class on one query and benign on the next. For an IDS, whose actions
(alerts, blocks) are irreversible, non-stationary verdicts are an operational
defect independent of any attack.

## Detector calibration (documented simplification)

- Per-session L2-density test: flag iff ≥ 5 of the last 50 session queries are
  within τ = 0.1 (above NES probe spacing σ·√2 ≈ 0.073, below benign
  inter-query distance).
- Benign control: 0/1,280 held-out benign queries flagged (0.00% FPR).
- This is a **mechanism proxy** for Blacklight's perceptual hashing; do not
  quote these detection rates as Blacklight's own (its reported rates are
  similar in spirit: detection within a handful of queries).

## Artifacts

- `results/logs/exp10_hybrid/metrics.json`
- `results/tables/exp10_hybrid_tradeoff.csv`
