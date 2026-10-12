# TASCP Reproducibility Artifact

**A stateful moving-target defense for DNN-based intrusion detection — and a rigorous empirical autopsy of why gradient-masking fails.**

Everything needed to reproduce every table and figure of the paper from the
shipped checkpoints, scripts, and archived result files.

## Artifact Navigation

| If you are reviewing... | Start here |
|---|---|
| Claim-to-evidence mapping | `CLAIM_MAP.md` |
| Reviewer entry point | `REVIEWER_GUIDE.md` |
| Bounded verification checklist | `REVIEW_CHECKLIST.md` |
| Fast verification (no re-runs) | `VERIFY.md` |
| Full reproduction | `REPRODUCE.md` |
| Reproducibility guarantees (R1–R4) | `REPRODUCIBILITY_LEVELS.md` |
| Expected outputs per experiment | `EXPECTED_OUTPUTS.md` |
| Known limitations | `LIMITATIONS.md` |
| Methodological history | `PROVENANCE.md` |
| Machine-readable experiment manifests | `experiment_manifests/` |
| Experiment-ID → claim → output inventory | `doc/OPEN_SCIENCE.md` |
| Checkpoint/dataset integrity | `manifest/`, `verification/` |
| Environment, lockfile, Docker, hardware | `environment/` |
| Dataset sources | `dataset_links.md` |
| Threat model (B1/B2) | `doc/THREAT_MODEL.md` |
| Formal results | `doc/THEOREMS.md` |
| Citation/versioning | `CITATION_TO_ARTIFACT.md` |

**Honest status:** the headline white-box table (exp5), the exp8 ablations, and
the exp6 false-positive results are now **machine-traceable** (`results/logs/`,
reproducibility level **R3**: seeded scripts + archived records with config,
checkpoint and dataset SHA-256s; re-run reproduces the archived numbers
exactly). The exp7 latency numbers remain prose-only (they require the Rust
fast-loop server, level **R4**). All remaining gaps are tracked in
`REPRODUCIBILITY_DISCREPANCY_REPORT.md`.

## What This Is

This repository is the complete empirical validation suite for TASCP. It evaluates whether a **momentum-conditioned Givens rotation** of the constraint manifold can prevent iterative white-box adversaries from crafting adversarial examples against deep network intrusion detectors.

**Spoiler:** it can't. TASCP degrades robustness — doubling attack success rate against naive attackers and yielding 53.60% ASR against adaptive ones.

---

## Key Results

```
Attack Success Rate (max 20 queries, 2,000 samples):

  Undefended          10.50%
  DACM (static)       10.50%       ← stateless bound confirmed
  TASCP (naive)       22.95%       ← rotation acts as adversarial perturbation
  TASCP (adaptive)    53.60%       ← fully bypassed via straight-through BPDA
```

**Root cause (UNSW-NB15):** The rotation basis vectors are data-dependent, giving the attacker perfect analytic gradients. Remove the basis coupling and TASCP's effect drops to zero (10.30% ASR, $p=0.836$ vs baseline).

**Dataset dependence:** The gradient-masking signature is **UNSW-NB15-specific**. On CICIDS2017 (corrected budget=100), all conditions cluster within 6pp ASR — the mechanism's signature does not replicate.

**Architecture dependence:** TASCP requires gradient access. Against XGBoost via surrogate gradients, the mechanism pattern does not replicate (undefended ASR = 0.85%).

---

## Documentation

| Doc | Contents |
| :--- | :--- |
| `doc/ARCHITECTURE.md` | Two-loop design, evaluation framework, defense/attack/config structure |
| `doc/TESTING.md` | Test file inventory, drift-detection tests, how to run |
| `doc/TRAINING.md` | Classifier training (MLP, XGBoost), preprocessing, weight files |
| `doc/TASCP_WHITEPAPER_README.md` | Full whitepaper-style write-up with all results |
| `doc/THREAT_MODEL.md` | **Bifurcated threat model** — B1 black-box (operational) vs B2 white-box (worst-case bound) |
| `doc/RELATED_WORK.md` | **Related-work overhaul** — positioning vs Blacklight (Li et al., USENIX Security 2022) and baselines |
| `doc/EVALUATION_GUIDE.md` | **Evaluation restructure** — the experiment suite as three core claims |
| `doc/THEOREMS.md` | **Formal results extension** — white-box impossibility, black-box query complexity, antithetic blindness |
| `doc/adr/ADR-001..004.md` | Architectural Decision Records — stateful evaluation, multi-seed re-eval, independent math validation, canonical-vs-fresh outputs |

## Architecture

A **two-loop system** separating the per-request defense decision from lag-tolerant pattern analysis:

```
[ Incoming Request ]
        ↓
[ Fast Loop (Rust) ]   ← per-session VecDeque window, computes M_t, ε_p, Δθ, R(Δθ)
        ↓                              ↑
[ Rotated Constraint Projection ]      │
        ↓                               │
[ DNN Classifier ]              [ Slow Loop (Python) ] — async, polls event log,
        ↓                        pattern analysis, writes updated params
[ Classification Output ]
```

---

## Repository Structure

```
README.md, CLAIM_MAP.md, VERIFY.md, ...   Reviewer-facing artifact documentation
manifest/            SHA-256 pins for checkpoints and preprocessed datasets
verification/        Integrity checks and fresh-vs-archive comparison
experiment_manifests/ One manifest per experiment (E-EXP01..E-EXP24 + E-EXP-CROSS-ARCH-TRANSFORMER)
doc/                 Whitepaper, architecture, threat model, theorems, evaluation guide,
                     related work, training, testing, and ADR-001..004 (doc/adr/)
tascp-proto/         Two-loop prototype (Rust fast-loop, Python slow-loop)
experiments/         exp1–exp24 (exp1–exp4 placeholders = not reproducible) + exp_transformer
src/tascp_eval/      Evaluation framework, attacks, defenses, models, reference math
configs/             Dataset/attack configs for hyperparameters and domain constraints
data/                CICIDS-2017, UNSW-NB15, NF-ToN-IoT-v2 ingestion
environment/         Dockerfile, pinned lockfile, hardware/environment notes
.github/workflows/   CI: integrity + traceability + tests + exp5/exp8 discrepancy check
tests/               Independent math oracles + invariant/drift tests (67 tests)
results/             Archived JSON/CSV tables, logs and summaries
scripts/             Automation for experiment orchestration
```

> exp18 ships a runnable driver (`experiments/exp18_rust_bench/run.py`) over the
> Rust criterion bench in `tascp-proto/fast-loop/benches/hot_path.rs`; exp24
> ships `experiments/exp24_ton_iot_transfer/run.py` (delegates to exp19–exp23 on
> `ton_iot`). exp1–exp4 are explicit placeholders (no source/archive) and exit
> code 2; see `doc/OPEN_SCIENCE.md` and `LIMITATIONS.md`.

## Experiments

Grouped into three core claims (full mapping in `doc/EVALUATION_GUIDE.md`):

**Claim I — the mechanism behaves as specified:**

| # | Experiment | Key Finding |
| :--- | :--- | :--- |
| 1 | Momentum Separation | Benign momentum ≈ 0; adversarial momentum ≫ 0 (separation confirmed) |
| 2 | Gradient Correlation Decay | Lemma 2 validated — $\cos\angle(g_t, g_t') = \cos(\Delta\theta)$ |
| 3 | Query Budget | Proposition 1 validated — threshold-avoidant adversaries bounded |

> **Not reproducible in this artifact:** exp1–exp4 ship no implementation and
> no archived output (the runner files are placeholders that exit code 2).
> See `doc/OPEN_SCIENCE.md`. The reproducible component of exp4 (DACM / TASCP /
> undefended) is covered by exp5.

**Claim II — it provides no security boundary:**

| # | Experiment | Threat Model | Key Finding |
| :--- | :--- | :--- | :--- |
| 4 | Defense Comparison | B2 | TASCP underperforms AT, RS, and static DACM |
| 5 | Adaptive BPDA Attack | B2 | **ASR 53.60% — defense broken** |
| 8 | Adaptive Ablations | B2 | Basis coupling isolated as the gradient leak |
| — | Cross-architecture (Transformer) | B2 | Pattern replicates on Transformer classifier |
| 9 | **Black-Box Query Attacks** | **B1** | **NES: rotation never fires (0.00% on both datasets — Corollary 3 replicates cross-dataset). HSJ: friction-only on UNSW (97.10% → 90.70%), amplification on CICIDS2017 (15.05% → 87.40%). Domain constraints: −70% ASR on UNSW, negligible on CICIDS.** |
| 10 | **Hybrid Detection Sim** | **B1** | **Blacklight-style density detector: 0.00% FPR, flags 91–98% of attack queries, reject policy → 0.00% verified ASR. TASCP adds nothing on top; its verdicts are non-stationary (HSJ 90.7% transient → 24.3% steady-state). Scope note (exp11): per-session only — identity rotation evades the per-session aggregator.** |
| 11 | **Detector Robustness** | **B1** | **Adaptive evaders vs the detector: query jitter (flag 97.8% → 78.1%), multi-identity rotation (→ 1.2–45.5%) — detection covers the blind spot per identity, not against identity rotation. L2-density ≫ quantize-then-hash at safe FPR. ROC: NES 96.8% / HSJ 89.8–91.5% TPR @ 0.1% FPR (both datasets).** |
| 12 | **CICIDS Disambiguation** | **B1** | **HSJ amplification is real and rotation-caused: controls (undefended, Δθ=0) both 15.05%; TASCP 87.40%; basis-fixed 25.85%; random-plane placebo only 33.75%. Angle detachment irrelevant for decision-based attacks.** |
| 13 | **White-Box Multiseed** | **B2** | **5-seed replication: Full Adaptive 53.0% [52.0, 54.0] vs Undefended 10.7% [10.1, 11.3] (UNSW); +4.9pp on CICIDS. Basis-fixed CI overlaps undefended — the leak localizes to the data-dependent basis across seeds.** |
| 14 | **Stationarity + CICIDS Multiseed** | **B1/B2** | **Verdict-flip rate ν grows monotonically with γ (UNSW), saturates non-monotonically on CICIDS; sin(Δθ/2) scaling is small-angle only (fit error 31–71%). Controls ν ≡ 0. CICIDS 3-seed: HSJ/TASCP 88.2% [87.3, 88.9] vs 15.3% [14.4, 16.2] — disjoint CIs; NES/undefended overlap.** |
| 15 | **Offline Attacker + Cor-3 Scope** | **B1/B2** | **Offline single-flow submissions: TASCP verdict bit-identical to DACM/undefended (max diff 0.0). Scope: all 6 non-antithetic attackers fire the trigger 2.5–65% of queries; only Square/Boundary get real ASR (46.7/75.8%) — Corollary 3's scope narrowed, honest.** |
| 16 | **Reference-Math Validation** | **I** | **Autograd ∇ε_p vs central FD: median rel err 1.4e-5 over 9,216 real comparisons (Lemma 3). E[ρ] = 0.198 vs isotropic 0.216. E[M_t] = −0.70, P(M>0) = 1.6% vs theory −2/√5. NES cosine ≈ 0 at all q (isotropic scaffold overpredicts).** |
| 17 | **Hyperparameter Sweep** | **B2** | **27-point grid (γ×τ×W): adaptive beats undefended on 6/27 points (best γ=10, τ=0: 9.8% → 1.4%); basis-fixed within 2pp of baseline on 26/27 — the effect survives tuning but vanishes without the data-dependent basis.** |

**RQ1–RQ6 — reviewer-response experiments (session 5; mapping in `doc/THEOREMS.md`):**

| RQ | Experiment | Key Finding |
| :--- | :--- | :--- |
| RQ1 | exp19 Transformation Family (rotation/scaling/translation/shear/noise) | The channel is NOT rotation-specific: rotation +10.4pp, translation +42.4pp, scaling +4.8pp adaptive-vs-fixed-state (UNSW); shear SUPPRESSES (−48.8pp); noise inert — variation, not uniform failure |
| RQ2 | exp20 Coupling Audit (6 state functions × 5 transformations × 2 datasets) | C = 0 exactly identifies the closed channel; coupling MAGNITUDE does not predict exploitability (Spearman ±0.27, p > 0.14) — geometry of the transform sets the failure mode |
| RQ3 | exp21 Trigger × Attack-Family Matrix (5 × 8) | No trigger fires on >50% of sessions for every attack family; trajectory statistics are attack-family dependent — a stateful trigger must be audited against the attacker's query-generation statistics |
| RQ4 | exp19/exp23 Stationarity | Verdict non-stationarity is family-wide (5/5 variants on TON-IoT, up to 37%); toward-allow flips (the exploitable direction) are TASCP-only: 36.2% UNSW / 18.5% TON-IoT vs 0% for detectors |
| RQ5 | exp22 Architecture Split + exp24 Third Dataset (NF-ToN-IoT-v2) | Coupling is differentiability-dependent (XGB/RF: not measurable); trigger blindness + stationarity violations are NOT (all four architectures). Mechanism transfers to a different schema/class distribution/flow process |
| RQ6 | exp23 Defense Audit (TASCP vs Blacklight-style detector vs rate limiter) | The three audits separate the defense classes: perturbation fails all three; detection/rejection is monotone-safe (nu_allow = 0) at a real benign-traffic cost. NOT "all moving-target defenses fail" |

**Claim III — deployable and still pointless:**

| # | Experiment | Key Finding |
| :--- | :--- | :--- |
| 6 | False Positive Rate | 0.00% FPR on benign traffic |
| 7 | Latency Benchmark | 7.08ms mean (target: <1ms) |
| 10 | Hybrid Detection (benign control) | 0.00% FPR on held-out benign sessions |
| 18 | Rust hot-path microbenchmark (criterion) | in-process compute 53.3 ns/query — the 7.08 ms HTTP figure is ~130,000× transport overhead; the fast-loop math itself is nanoseconds |

---

## The Takeaway

TASCP's theoretical bounds (Lemma 2, Proposition 1) are mathematically sound, but the defense fails empirically because:

1. **Basis coupling** leaks exploitable gradients
2. The rotation **pushes samples toward** misclassification
3. Adaptive attackers backpropagate through the full mechanism
4. Against black-box query attacks the trigger is structurally blind: antithetic
   NES queries have systematically negative momentum (E[M] = −2/√5), so the
   rotation never fires — on both datasets (exp9, Corollary 3 in `doc/THEOREMS.md`;
   exp16 validates the momentum math: E[M_t] = −0.70, P(M>0) = 1.6%; exp15
   delimits the corollary's scope — non-antithetic attackers fire the trigger
   but gain nothing, and only decision-based refinement attacks bypass it)
5. On CICIDS2017 the failure inverts into amplification: label-only HSJ attacks
   succeed 5.8× more often under TASCP (15.05% → 87.40%) because persistent
   queries trigger the rotation and its displacement dominates the verdict
   (exp12 pinpoints the mechanism: controls at 15.05%, basis-fixed 25.85%,
   random-plane placebo only 33.75% — the trajectory-resolved plane is what
   the amplification rides on)
6. Detection-based statefulness helps where perturbation-based statefulness
   cannot: a per-session density detector flags 91–98% of attack queries at
   0.00% benign FPR, and reject policy yields 0.00% verified ASR (exp10) —
   but only per client identity: exp11 shows identity rotation collapses the
   per-session flag rate (97.8% → 1.2%), so "detect, not perturb" must be
   paired with network-level identity correlation (exp10/exp11 scope notes)
7. The offline attacker defeats both outright: a single fresh-session flow is
   classified bit-identically under TASCP, DACM, and the undefended model
   (exp15, max abs diff 0.0) — statefulness provides zero protection against
   the classic transfer/offline threat model

This is a **cautionary result**: provable gradient degradation ≠ empirical robustness. Gradient-masking defenses remain structurally vulnerable to BPDA-style attacks.

## Quick Start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
make verify        # checkpoint + dataset SHA-256 integrity
make smoke         # environment + integrity + traceability + tests (67 tests)
make reproduce-adaptive     # exp5 white-box adaptive attack
```

See `REPRODUCE.md` for full reproduction, `VERIFY.md` for fast verification,
and `CLAIM_MAP.md` for the claim-to-evidence navigation.
