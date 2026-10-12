# TASCP: Trajectory-Aware Spiral Constraint Projection

**A stateful, moving-target defense against adversarial examples for deep neural network intrusion detection — and a case study in why gradient-masking defenses fail.**

---

## The Problem

Static constraint projection defenses (like DACM) are structurally broken: a white-box adversary with unlimited offline compute can study the fixed boundary and route around it. Every query-independent defense leaks the same exploitable geometry. The field needed a **stateful** approach — one with memory of the attacker's trajectory.

## The Approach

TASCP tracks the **spatial momentum** of a session's query trajectory and applies a momentum-conditioned Givens rotation to the constraint manifold before each classification. The rotation lives in the 2D subspace spanned by the current step direction and its orthogonal complement — directly perturbing the feature dimensions the attacker is actively exploiting.

The system is built as a **two-loop architecture**:
- **Fast loop** (Rust): in-memory, per-request defense decision using a sliding window of recent queries
- **Slow loop** (Python): async pattern analysis over the persisted event log, recalibrating defense parameters

## Key Theoretical Results

| Result | Statement |
| :--- | :--- |
| **Lemma 2** (Gradient Correlation Decay) | A Givens rotation by $\Delta\theta$ reduces cosine similarity between original and rotated gradients to exactly $\cos(\Delta\theta)$ within the rotation subspace |
| **Proposition 1** (Query Complexity) | An adversary staying under the rotation threshold $\gamma$ is bounded to $n^* = \lfloor \arccos(\tau) / \lambda\gamma \rfloor$ consecutive aligned steps before gradient degradation forces a restart |
| **Theorem 1** (White-Box Adaptive Impossibility) | Deterministic, piecewise-differentiable stateful defenses provide no first-order security boundary; data-dependent state leaks analytic gradients and can strictly *degrade* robustness |
| **Proposition 2** (Black-Box Query Complexity) | Replicating an n-step white-box trajectory with score-based queries costs $\Omega(d \cdot n / \delta^2)$ |
| **Corollary 3** (Antithetic Blindness) | Antithetic NES queries induce $E[M_t] = -2/\sqrt{5} < 0$, so momentum-clipped triggers never activate |

Full statements and proof sketches: `doc/THEOREMS.md`.

## Empirical Findings

The honest picture — TASCP does not work as a defense.

```
Attack Success Rate (2,000 samples, max 20 queries):

  Undefended          10.50%  (baseline)
  DACM (static)       10.50%  (no improvement — stateless bound)
  TASCP (naive)       22.95%  (rotation acts as adversarial perturbation!)
  TASCP (adaptive)    53.60%  (fully bypassed via straight-through BPDA)
```

### Root Cause

Isolated through rigorous ablation:
1. **Basis Coupling** — rotation basis vectors are data-dependent; the attacker gets perfect analytic gradients through them
2. **Gradient Masking** — the rotation magnitude creates a gradient penalty that *masks* the vulnerability rather than removing it
3. With basis coupling removed, TASCP's protective effect drops to statistical zero (10.30% ASR vs 10.50% baseline, $p=0.836$)

### Black-Box Query Attacks (exp9)

Under the operational black-box threat model (Threat Model B1, `doc/THREAT_MODEL.md`):

```
Attack Success Rate (2,000 samples):

  NES (score-based, 250 queries)      Undefended 2.70%  |  TASCP 2.60%   ← rotation never fires (0.00%)
  HSJ (decision-based, 1000 queries)  Undefended 97.10% |  TASCP 90.70%  ← rotation fires (90.55%), only taxes attacker
  NES + domain constraints            Undefended 0.80%  |  TASCP 0.80%   ← feasible attack surface shrinks 70%
```

**Why the rotation never fires against NES:** antithetic query pairs alternate
direction (x + σu, then x − σu), giving consecutive trajectory steps a
systematically negative momentum (E[M_t] = −2/√5 ≈ −0.894). The penetration
integral clips negative momentum to zero, so Δθ = 0 for every query — TASCP is
structurally blind to exactly the query pattern the canonical score-based
attack must emit (Corollary 3, `doc/THEOREMS.md`). This replicates exactly on
CICIDS2017 (rotation 0.00%, ASR 22.30% with or without TASCP) — the mechanism
is dataset-independent, as the corollary predicts.

**CICIDS2017 inversion — the defense amplifies decision-based attacks.** On
CICIDS2017 the persistent-query HSJ attack triggers the rotation (87.35% of
final queries) and succeeds **5.8× more often under TASCP than without it
(87.40% vs 15.05%)**. The same rotation-as-adversarial-perturbation mechanism
isolated in the white-box ablations (exp5/exp8) is reachable through label-only
feedback. Domain constraints, which cut UNSW-NB15's NES attack surface by 70%,
bind weakly on CICIDS2017 (−0.8pp): the constrained fields (flags, protocol)
are not where that dataset's realizable attack surface lives.

**Exp10 — detection closes what perturbation cannot.** A per-session
near-duplicate density detector (Blacklight-style mechanism proxy; τ/k
calibrated on a held-out benign control at **0.00% FPR**) flags 91–98% of
attack queries, and a reject policy reduces verified attack success to **0.00%**
for both NES and HSJ. TASCP adds nothing measurable on top. Worse, TASCP's
verdicts are **non-stationary**: HSJ's transient 90.70% success under TASCP
collapses to 24.30% when the final points are re-queried — the same input
classifies differently across queries, an operational defect for any IDS whose
actions are irreversible. The design conclusion: stateful NIDS defenses should
detect, not perturb (`experiments/exp10_hybrid_detection_sim/`).

### Latency

```
  Target:  <1ms per request
  Actual:   7.08ms mean  (p50: 6.69ms, p95: 12.09ms, p99: 20.79ms)
```

The fast-loop prototype misses its sub-millisecond target by nearly an order of magnitude under concurrent load.

### False Positives (Benign Traffic)

```
  Rotation trigger rate:  1.21% of active steps
  False positives:        0.00%
```

Benign momentum is near zero — the mechanism adds no accuracy overhead on normal traffic, but this doesn't rescue it.

## Threat Model and Related Work

The evaluation uses a **bifurcated threat model** (`doc/THREAT_MODEL.md`):

- **B1 — Black-box, operational:** query access only (scores for NES, labels
  for HopSkipJump). What a deployed NIDS actually faces. → exp9.
- **B2 — White-box, worst-case bound:** full gradients through the defense
  (BPDA). Cryptographic worst-case analysis. → exp1–exp5, exp8, exp_transformer.

The existing 20-step PGD experiments are retained as the B2 worst-case
analysis; exp9 establishes the B1 operational picture.

**Related work:** stateful defenses against query-based black-box attacks are
established — most prominently **Blacklight** (Li et al., USENIX Security
2022), a stateful *detector* built on probabilistic perceptual-hash
fingerprints of consecutive queries. TASCP is a stateful *perturbation*
mechanism — a different actuation contract with different failure modes.
The two families have disjoint blind spots (Blacklight is maximally sensitive
to the near-duplicate antithetic pairs TASCP is blind to, and vice versa);
see `doc/RELATED_WORK.md` for the full positioning.

```
tascp-proto/        Two-loop prototype (Rust fast-loop, Python slow-loop)
experiments/        Modular empirical validation (exp1–exp17)
src/tascp_eval/     Python evaluation framework, baseline/black-box attacks, reference math
configs/            Hydra configuration for hyperparameters, datasets, domain constraints
data/               CICIDS-2017, UNSW-NB15 ingestion points
tests/              Drift-detection tests against the prototype math
results/            Generated LaTeX tables and summaries
```

## The Real Takeaway

TASCP is a **cautionary result**: dynamic constraint manifolds that depend deterministically on the attacker's trajectory do not mask gradients. The Givens rotation inadvertently perturbs samples *toward* adversarial regions, and a fully adaptive attacker can backpropagate through the rotation to achieve higher success rates with fewer queries.

The theoretical bounds (Lemma 2, Proposition 1) hold mathematically, but the defense fails in practice — a reminder that provable gradient degradation does not imply empirical robustness, and that gradient-masking defenses are structurally vulnerable to BPDA-style attacks. Under black-box query attacks the failure is quieter but total: the momentum trigger never fires against antithetic score-based adversaries (and even when non-antithetic estimators do fire it — exp15 — they gain nothing; only decision-based refinement attacks convert the trigger into a target), and decision-based adversaries absorb its friction on UNSW while being *amplified* 5.8× on CICIDS2017 (exp12: the amplification is rotation-caused and localized to the trajectory-resolved basis). The offline attacker needs none of this: a single fresh-session flow is classified bit-identically under TASCP and the undefended model (exp15). The recommended alternative — per-session density detection (exp10) — covers the per-identity blind spot but is itself evaded by client-identity rotation (exp11): detection is the better mechanism, not a security boundary.

## Experiments

| Experiment | What It Validates | Status |
| :--- | :--- | :--- |
| exp1 | Momentum separation (Assumptions 1–2) | Complete |
| exp2 | Gradient correlation decay (Lemma 2) | Complete |
| exp3 | Query budget (Proposition 1) | Complete |
| exp4 | Defense comparison (AT, RS, DACM, TASCP) | Complete |
| exp5 | Adaptive BPDA attack | **Complete — defense broken** |
| exp6 | False-positive rate on benign traffic | Complete |
| exp7 | Fast-loop latency benchmark | Complete |
| exp8 | Adaptive ablations (basis coupling isolation) | Complete |
| exp9 | Black-box query attacks (NES, HopSkipJump) + domain constraints — UNSW-NB15 **and CICIDS2017**, 3-seed CIs | **Complete — defense inert (NES, both datasets) / friction-only (HSJ, UNSW) / amplifying (HSJ, CICIDS2017: 15.05% → 87.40%)** |
| exp10 | Hybrid detection simulation (Blacklight-style density detector + TASCP) | **Complete — 0.00% verified ASR for both attacks at 0.00% benign FPR; TASCP adds nothing** |
| exp11 | Detector robustness under adaptive evaders | **Complete — identity rotation evades the per-session aggregator (flag 97.8% → 1.2%/45.5%); L2-density ≫ quantize-then-hash at safe FPR** |
| exp12 | CICIDS2017 amplification disambiguation | **Complete — amplification is rotation-caused: controls 15.05%, TASCP 87.40%, basis-fixed 25.85%, random-plane placebo 33.75%** |
| exp13 | White-box multiseed replication (5 seeds, both datasets) | **Complete — adaptive 53.0% vs 10.7% undefended (UNSW), disjoint CIs; basis-fixed CI ≈ undefended** |
| exp14 | Stationarity (ν vs Δθ, γ-sweep) + CICIDS black-box multiseed | **Complete — ν ≡ 0 for controls at every γ; sin(Δθ/2) small-angle only; HSJ/TASCP 88.2% vs 15.3% (disjoint 3-seed CIs)** |
| exp15 | Offline attacker + Corollary-3 scope | **Complete — single-flow TASCP verdict bit-identical to undefended; non-antithetic attacks fire trigger but only Square/Boundary gain ASR** |
| exp16 | Reference-math validation (Lemma 3 FD-vs-autograd, ρ, M_t, NES cosine) | **Complete — median FD rel err 1.4e-5 on 9,216 real comparisons; E[M_t] = −0.70 vs theory −0.894** |
| exp17 | Hyperparameter sweep (γ×τ×W, 27 points) | **Complete — adaptive wins 6/27 points; basis-fixed ≈ baseline on 26/27** |
| exp18 | Rust fast-loop microbenchmark (criterion) | **Complete — 53.3 ns/query in-process; the 7.08 ms figure is transport overhead (~130,000×)** |
| — | Cross-architecture (Transformer) replication | Complete |

**Dataset table:** CICIDS2017 d = 78 (parquet includes duplicate `Fwd Header
Length.1`); UNSW-NB15 d = 43. Only these two datasets are available in
`data/raw`; a third dataset was requested but does not exist locally
(out of scope, see `doc/THEOREMS.md`).

## Getting Started

```bash
# Install the evaluation framework
pip install -e src/tascp_eval

# Run an experiment (e.g., the adaptive attack stress test)
python experiments/exp5_adaptive_bpda_attack/run.py

# Run the full validation suite
make test
```
