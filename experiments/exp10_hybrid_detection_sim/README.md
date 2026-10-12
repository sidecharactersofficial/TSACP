# Experiment 10: Hybrid Detection Simulation (TASCP + Blacklight-style Detector)

Constructive counterpart to exp9's autopsy: does **detection-based** statefulness
(Blacklight, Li et al., USENIX Security 2022) cover the black-box blind spots
that **perturbation-based** statefulness (TASCP) leaves open?

## Detector (documented simplification)

Blacklight's exact probabilistic perceptual hashing is replaced by an
**L2 near-duplicate density test** per session:

- each sample = one client session with a W = 50 query ring buffer;
- a query is flagged when **≥ k = 5** of the session's windowed queries lie
  within τ = 0.1 L2 of it (τ sits above the NES probe spacing ≈ σ·√2 and
  below benign inter-query distances);
- policy: flagged queries are **rejected** (fail-closed, Blacklight's
  "detect and ban" semantics) — the attacker sees a benign default verdict.

This preserves the load-bearing mechanism (iterative attacks emit *dense
clusters of highly similar queries*; benign sessions do not) without
claiming Blacklight's exact detection rates. τ/k were validated on a held-out
benign control **before** the attack runs.

## Conditions (UNSW-NB15, 2,000 samples, seed 42)

| Condition | ASR (verified¹) | Queries flagged | Client detected |
|---|---:|---:|:---:|
| Benign control (no attack) | — | **0.00%** | — |
| NES + Detection only | **0.00%** | 97.8% | yes |
| NES + TASCP only (detector passive) | 2.35% | 97.8% | yes |
| NES + Hybrid (TASCP + detect & reject) | **0.00%** | 97.8% | yes |
| HSJ + Detection only | **0.00%** | 91.1% | yes |
| HSJ + TASCP only (detector passive) | 24.30%² | 91.3% | yes |
| HSJ + Hybrid (TASCP + detect & reject) | **0.00%** | 91.3% | yes |

¹ *Verified* success = the defender **served** an adversarial verdict on the
final point (queries rejected by the detector return a benign default).
Counting the attack's internal success flags instead would credit successes
the defender never served.
² *Steady-state* rate. exp9's 90.70% for HSJ vs TASCP is the *transient*
during-attack success; re-querying the final adversarial point under TASCP's
stateful rotation flips it back to benign for 75.7% of samples — TASCP's
verdicts are **non-stationary** (the same input classifies differently across
queries), which is itself a deployment defect: IDS decisions are irreversible.

## Key Findings

1. **The blind spots are disjoint, exactly as `doc/RELATED_WORK.md` argues —
   per client identity.** The density detector flags what TASCP's momentum
   trigger cannot see (NES: TASCP rotation fired on 0.00% of queries,
   detector flagged 97.8%). **Exp11 scope check:** this complementarity holds
   per session; an adversary rotating across ≥ 5 client identities splits
   queries into windows that never reach k_min (per-session flag rate 97.8%
   → 1.2% on UNSW), so detection must be paired with network-level identity
   correlation to be a security boundary rather than a session-level filter.
2. **Detection, not perturbation, is the effective stateful mechanism** on
   this stream: reject policy reduces both attack classes to 0.00% verified
   ASR at 0.00% benign FPR.
3. **TASCP adds nothing measurable on top of detection** (hybrid ≈ detection
   only) — consistent with exp9: its trigger either never fires (NES) or
   injects non-stationary verdicts (HSJ).

## Honest Caveats

- The detector is an L2-density **proxy**, not Blacklight's perceptual-hash
  machinery; absolute detection rates should not be quoted as Blacklight's.
- Benign calibration models a session as i.i.d. held-out benign flows; real
  deployments have correlated/replayed traffic (which raises detector FPR —
  the well-known per-client whitelisting problem).
- UNSW-NB15 only; the CICIDS2017 hybrid (where TASCP *amplifies* HSJ to
  87.40%, exp9) is the natural next replication.

## Outputs

- `exp10_summary.md` — narrative summary
- `results/logs/exp10_hybrid/metrics.json` — full metrics
- `results/tables/exp10_hybrid_tradeoff.csv` — paper table
