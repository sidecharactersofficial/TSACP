# Evaluation Guide — Three Core Claims

The experiment suite (exp1–exp9 + cross-architecture) reads as a lab notebook
if presented in run order. It reads as an argument if grouped into three core
claims. This is the structure to use in the paper's evaluation section; every
experiment appears exactly once, in exactly one claim.

```
Claim I   The mechanism behaves as specified        (does the math hold?)
Claim II  It provides no security boundary          (does it defend?)
Claim III It is deployable and still pointless      (could we even ship it?)
```

---

## Claim I — The mechanism behaves as specified

*The trajectory machinery (momentum tracking, penetration integral, Givens
rotation) does what the formal results say it does.*

| Experiment | Validates | Result |
|------------|-----------|--------|
| exp1 | Momentum separation (Assumptions 1–2): benign vs adversarial trajectories | Benign momentum ≈ 0; adversarial ≫ 0 |
| exp2 | Gradient correlation decay (Lemma 2): cos∠(g, g′) = cos(Δθ) | Validated on UNSW-NB15 and CICIDS2017 |
| exp3 | Query budget (Proposition 1): threshold-avoidant adversaries bounded by n* | Measured n* matches prediction |

**Purpose in the argument:** establishes that the *implementation* is faithful
to the *theory*. Any failure found later is therefore a failure of the design,
not of the code. This is what separates "we couldn't make it work" from "it
cannot work."

## Claim II — It provides no security boundary

*Under both threat models (`doc/THREAT_MODEL.md`), the mechanism fails —
catastrophically in B2, inertly-to-mildly in B1 — for identified mechanistic
reasons.*

| Experiment | Threat model | Result |
|------------|--------------|--------|
| exp4 | B2 | TASCP underperforms AT, RS, and even static DACM |
| exp5 | B2 (adaptive BPDA) | **ASR 53.60% vs 10.50% undefended** — defense actively harmful |
| exp8 | B2 (adaptive ablations) | Basis coupling isolated as the leak (fixing it collapses effect to 10.30%, p = 0.836) |
| exp_transformer | B2, cross-architecture | Pattern replicates on Transformer classifier |
| exp9 (NES) | B1 (score-based) | Rotation fires on **0.00%** of queries (antithetic momentum ≤ 0); ASR identical with/without TASCP — **replicates cross-dataset** (UNSW 2.70% vs 2.60%; CICIDS2017 22.30% vs 22.30%) as Corollary 3 predicts |
| exp9 (HSJ) | B1 (decision-based) | UNSW: rotation fires (90.55%) but only taxes the attacker (−6.4pp, still 90.70%). **CICIDS2017: amplification — 15.05% → 87.40% (5.8×)** |
| exp9 (domain-constrained) | B1 | UNSW: NES ASR −70% relative (2.70% → 0.80%). CICIDS2017: negligible (−0.8pp) — the constrained fields are not where CICIDS's attack surface lives |
| exp10 (hybrid detection) | B1 | Per-session density detector: 0.00% benign FPR, 91–98% of attack queries flagged, reject policy → **0.00% verified ASR** for both attack classes; TASCP adds nothing on top |
| exp11 (detector robustness) | B1 | Adaptive evaders vs the detector: multi-identity rotation cuts per-session flag rate to 1.2% (UNSW) / 45.5% (CICIDS); L2-density ≫ quantize-then-hash at safe FPR (hash best NES 5.4% flag on CICIDS, benign FPR 84–88% at detection points); ROC NES 96.8% / HSJ 89.8–91.5% TPR @ 0.1% FPR on both datasets |
| exp12 (CICIDS disambiguation) | B1 | Controls (undefended, Δθ=0) both 15.05% vs TASCP 87.40%; basis-fixed 25.85%; random-plane placebo 33.75% — amplification requires the trajectory-resolved plane, not just a large angle |
| exp13 (white-box multiseed) | B2 | 5 seeds, pooled Wilson CIs: UNSW adaptive 53.0% [52.0, 54.0] vs undefended 10.7% [10.1, 11.3], basis-fixed 10.5% [9.9, 11.1] ≈ undefended; CICIDS adaptive 45.6% [44.7, 46.6] vs 40.8% [39.8, 41.7] — all disjoint |
| exp14 (stationarity) | B1/B2 | ν (verdict-flip rate of a fixed probe) grows with γ, control ν ≡ 0 on both datasets at every γ; sin(Δθ/2) scaling holds only at small Δθ (fit error 31% UNSW, 71% CICIDS). CICIDS multiseed (Opt-B): HSJ/TASCP 88.2% [87.3, 88.9] vs HSJ/undef 15.3% [14.4, 16.2] disjoint; NES/undef vs NES/TASCP overlap (20.8% vs 20.8%) |
| exp15 (offline + scope) | B1/B2 | Offline single-flow: TASCP verdict **bit-identical** to DACM/undefended — no stateful protection against the offline threat model. Cor-3 scope over ALL queries: all six non-antithetic attackers fire the trigger 2.5–65% (SPSA 64.9%, NES-iid 20.9%, Square 20.7%, Boundary 41.0%) but only Square/Boundary get real ASR (46.7/75.8%) — gradient-estimator attackers that fire the trigger still fail (ASR ≤ 1.25%) |
| exp16 (reference math) | I | Lemma 3: autograd ∇ε_p vs central FD — median rel err 1.4e-5, p90 1.0e-4 (9,216 real PGD comparisons). Lemma 2 ρ: measured 0.198 vs isotropic 0.216 (0.92×). Prop 3: measured NES cosine ≈ 0 at all q — the √(q/(q+d+1)) scaffold overpredicts. Cor-3 M_t: E[M_t] = −0.70, P(M>0) = 1.6% (theory −0.894) |
| exp17 (hyperparameter sweep) | B2 | 27 points (γ×τ×W): adaptive < undefended on 6/27 (all τ∈{0,0.01}, γ=10; best 9.8% → 1.4%); basis-fixed within 2pp of undefended on 26/27 — the gain survives tuning but the data-dependent basis carries it everywhere |
| exp18 (Rust bench) | III | Fast-loop hot path: 53.3 ns/query (criterion median, 7.08 ms HTTP figure is transport overhead ~130,000×) |

**Purpose in the argument:** the two B1 results are *not* redundant with the
B2 results — they show failure under two disjoint mechanisms (trigger
blindness vs gradient leakage). The domain-constrained conditions (exp9, P2)
additionally show that most of the residual black-box attack surface is
infeasible data (2.70% → 0.80% ASR from domain constraints alone, undefended).

## Claim III — It is deployable and still pointless

*The system meets its operational non-security targets — and that does not
rescue it.*

| Experiment | Validates | Result |
|------------|-----------|--------|
| exp6 | False positives on benign traffic | 0.00% FPR; rotation fires on 1.21% of active steps |
| exp7 | Fast-loop latency | 7.08 ms mean (p99: 20.79 ms) vs <1 ms target |

**Purpose in the argument:** closes the last two escape hatches. "Maybe it
would break in production" — no: it is accurate on benign traffic. "Maybe it
was mis-tuned" — no: it runs at production latency. The failure is
*structural* (Claims I–II), not operational (Claim III).

---

## Reading Order for Reviewers

| Reviewer objection | Point to |
|--------------------|----------|
| "Only white-box tested" | Claim II, exp9 (both black-box classes, both datasets) |
| "Stateful defenses already exist (Blacklight)" | `doc/RELATED_WORK.md` — detection vs perturbation delta; exp10 for the empirical complementarity |
| "L∞-style perturbations are unrealistic for tabular data" | Claim II, exp9 domain-constrained conditions (+ their dataset dependence — an honest negative) |
| "Maybe the implementation is wrong" | Claim I (faithful to theory) + Claim III (operational health) |
| "20-step PGD budget is arbitrary" | exp5 budget-corrected CICIDS2017 runs (budget=100); exp9 query sweep |
| "Single-seed results?" | exp9 multi-seed (`results/tables/exp9_multiseed.csv`), exp13 5-seed white-box + exp14 CICIDS 3-seed (Wilson 95% CIs, disjoint where claimed) |
| "Is the measurement math even right?" | exp16 Lemma 3 (autograd vs FD on real trajectories, 1.4e-5 median) + exp18 Rust bench |
| "Detection recommendation overstated?" | exp11: per-identity only; identity rotation evades; quantize-then-hash fails FPR — see exp10 scope note |
| "Third dataset?" | exp24: NF-ToN-IoT-v2 downloaded/processed/trained with the identical protocol (`data/ton_iot/PROVENANCE.md`); all core audits replicated (`doc/THEOREMS.md`, Third Dataset section) |
| "Is the phenomenon TASCP-specific?" | exp19 transformation family (RQ1) + exp20 coupling sweep (RQ2) |
| "Is the antithetic-NES result just the momentum trigger?" | exp21 trigger × attack-family matrix (RQ3): no trigger covers all families |
| "Does it depend on differentiability / architecture?" | exp22 split: coupling differentiable-only; trigger + stationarity universal (RQ5) |
| "Why should we trust these audits?" | exp23: they separate TASCP (unsafe perturbation) from detector/rate-limiter (monotone-safe) (RQ6) |
| "Are you overclaiming ('all moving-target defenses fail')?" | No — the supported claim is three pre-deployment audits for trajectory-conditioned perturbation; detectors pass the direction-of-flip audit (`doc/THEOREMS.md`, claim-discipline note) |

## Experiment-to-Claim Map (one line each)

```
Claim I   : exp1, exp2, exp3
Claim I   : exp1, exp2, exp3 (+exp16 reference-math validation, exp18 hot-path bench)
Claim II  : exp4, exp5, exp8, exp9 (+CICIDS2017), exp10–exp15, exp17, exp_transformer
Claim III : exp6, exp7, exp10 benign control, exp18
RQ1       : exp19 (transformation family)
RQ2       : exp20 (coupling audit, 6 state functions × 5 transformations)
RQ3       : exp21 (trigger × attack-family matrix)
RQ4       : exp19 stationarity replay + exp23 nu_allow
RQ5       : exp22 (MLP/Transformer/XGB/RF split) + exp24 (NF-ToN-IoT-v2)
RQ6       : exp23 (defense-class audit)
d = 78    : CICIDS2017 parquet feature count incl. duplicate `Fwd Header Length.1` (d = 43 for UNSW-NB15, d = 41 for NF-ToN-IoT-v2)
Theorems  : doc/THEOREMS.md (Thm 1 + decomposition, Lemma 2/3, Prop 2/3/4/5, Cor 3 + scope, RQ1–RQ6 results)
```

---

## RQ Structure for the Paper's Evaluation Section (Reviewer A response)

For the restructured paper, the six research questions replace the linear
experiment chain; full results and honest caveats in `doc/THEOREMS.md`
(sections RQ1–RQ6, claim-discipline note):

```
RQ1 — Is the phenomenon specific to TASCP?            exp19
RQ2 — What mechanism makes trajectory state exploitable?   exp20
RQ3 — Does the problem survive different attacker query processes?   exp21
RQ4 — Does stateful perturbation violate the IDS operational contract?   exp19/exp23
RQ5 — Does it generalize beyond UNSW/CICIDS (data + architectures)?   exp24/exp22
RQ6 — Can the proposed audits distinguish different stateful defenses?   exp23
```
