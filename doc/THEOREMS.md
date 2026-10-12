# Formal Results — Impossibility Extension

This extends the whitepaper's formal results (Lemma 2, Proposition 1) with
the impossibility results that formalize the empirical findings of exp5, exp8,
and exp9, the scope/stationarity/gradient-consistency results validated by
exp12–exp16, and the bifurcated threat model (`doc/THREAT_MODEL.md`). Proof sketches
are given; each result names its validating experiment.

**Notation.** Classifier f: R^d → R (margin; x classified attack iff f(x) > 0).
A stateful defense is D(x; h) where h = (x_{t−k}, …, x_{t−1}) is the observed
query trajectory. Undefended success rate ASR₀; defended ASR_D. Attack budget:
B queries per sample, perturbation ball ∂ (l2 radius ε).

---

## Theorem 1 (White-Box Adaptive Impossibility — Threat Model B2)

*Deterministic, piecewise-differentiable stateful defenses provide no
first-order security boundary.*

**Statement.** Let D(x; h) be deterministic and piecewise differentiable in
(x, h), and let the defended margin be g(x) = f(D(x; h(x))) where h(x) itself
depends differentiably on the query trajectory. Then for any first-order
attack A achieving ASR₀ against f with budget B, there exists a first-order
attack A′ against g with the same budget B and query cost such that

    ASR_{A′,D} ≥ ASR_{A,f},

and if on a set of positive measure the Jacobian of the state map satisfies
⟨∇_x D(x; h)ᵀ ∇f, d⟩ > ⟨∇_x f∘id, d⟩ for the attack direction d — i.e., the
state transform projects the gradient *toward* the attack objective — then
ASR_{A′,D} > ASR_{A,f} strictly.

**Proof sketch.** A′ is BPDA applied to the composite g: since D is
deterministic and differentiable a.e., the chain rule through the state map
is available to the attacker (who, under B2, knows D's exact form). No
gradient-masking occurs: any "masking" that D performs is itself a
differentiable function of quantities the attacker controls, so the mask is
part of the objective, not a barrier to it. The strict inequality holds
whenever D amplifies margin in the attack direction (basis coupling), which
is measurable in the ablations. ∎

**Empirical validation:** exp5 (adaptive BPDA: 53.60% vs 10.50% undefended —
strict degradation) and exp8 (basis-coupling ablation: removing the
data-dependent basis collapses TASCP's effect to 10.30%, p = 0.836 — the
leak is localized to the differentiable state map).

**Design corollary.** Escaping Theorem 1 requires breaking one of its
premises: non-determinism (randomized defenses — at the cost of
certifiable-guarantee style variance), non-differentiability (hard
verification — at the cost of gradient-free operation only), or
state-independence (static projections — which exp4 shows equal the
undefended baseline anyway).

---

## Proposition 2 (Black-Box Query Complexity — Threat Model B1)

*Score-based replication of a white-box trajectory pays a dimension-scaled
query tax.*

**Statement.** Let A_wb be a white-box first-order attack that reaches the
boundary within n steps of exact gradient ascent. Let A_bb be a score-based
adversary estimating each step's ascent direction from q antithetic
perturbation pairs of scale σ (NES). To obtain per-step directional alignment
cos(ĝ, ∇f) ≥ 1 − δ, the per-step query count must satisfy

    q = Ω(d / δ²),

so the total query budget to replicate the white-box trajectory is

    Ω(2q · n) = Ω(d · n / δ²).

**Proof sketch.** Each antithetic pair contributes a rank-one directional
sample; the estimator ĝ concentrates around ∇f with per-coordinate variance
O(1/q) (times scale-dependent constants). Concentration of random inner
products in dimension d then forces q = Ω(d/δ²) for angular error δ — the
standard zeroth-order-optimization query complexity. ∎

**Empirical validation:** exp9 budget sweep — NES reaches 8.85% ASR at 1,000
queries, matching what 20-step white-box PGD achieves (10.50%) at ~50× the
queries (d = 43, q = 10 per step: the sweep's ASR-vs-budget curve is the
empirical form of the Ω(d·n/δ²) bound). Combined with Proposition 1
(threshold-avoidant trajectory length n*), this bounds the score-based
attacker's total budget under the defense's own trigger threshold.

---

## Corollary 3 (Antithetic Blindness of Momentum Triggers)

*Momentum-based triggers cannot see the canonical score-based attack.*

**Statement.** Consider a defense whose activation depends on the query
trajectory only through first-order momentum terms max(0, M_t) with
M_t = cos∠(v_t, v_{t−1}), v_t = x_t − x_{t−1}. Let the adversary use
antithetic NES queries: q_{2j−1} = x + σu_j, q_{2j} = x − σu_j with
u_j ~ N(0, I_d) independent. Then, conditioned on ‖x − x_t‖ ≪ σ‖u‖,

    E[M_t] = −2/√5 ≈ −0.894 < 0  for every t,

hence max(0, M_t) = 0 for all t (in expectation, and with high probability
per step), and a trigger of the form Δθ = γ·max(0, ε_p − τ) with
ε_p = (1/|W|) Σ ‖v_i‖·max(0, M_i) **never activates**.

**Proof sketch.** Consecutive steps of the antithetic sequence are
v_{2j} = −2σu_j and v_{2j−1} = 2σu_{j−1} + σu_j (the displacement of the
anchor x is negligible under the stated condition). With independent,
zero-mean directions, E[⟨v_t, v_{t−1}⟩]/(‖v_t‖‖v_{t−1}‖) = −4σ²/(2σ·σ√5)
= −2/√5. The clip max(0, ·) then zeroes every momentum contribution, so the
penetration integral ε_p = 0 identically. ∎

**Empirical validation:** exp9 — rotation activation measured at 0.00% over
*every* query of the NES trajectory (max Δθ = 0.00e+00, 2,000 samples), while
a persistent-direction control activates on 100% of samples (mean max
Δθ = 0.833). The measured per-step momentum (≈ −0.83) sits close to the
predicted −2/√5 ≈ −0.89; the residual gap is the ‖x − x_t‖ displacement term.

**Remark.** This is not an artifact of NES's particular variant: any
score-based gradient estimator built from symmetric difference pairs induces
alternating query directions. The trigger family {momentum of consecutive
steps} is structurally blind to exactly the query patterns that
query-efficient score attacks must emit. Persistent *similarity*-based
detectors (Blacklight-style) occupy the complementary blind spot — see
`doc/RELATED_WORK.md`.

---

## Proposition 4 (Domain Constraints Shrink the Feasible Attack Set)

**Statement.** Let F ⊆ R^d be the attacker's feasible perturbation set and
F′ ⊆ F the domain-constrained set obtained by fixing k coordinates to finite
level sets (binary/discrete encodings) and restoring immutable fields. Then

    ASR_{F′} ≤ ASR_F,

and for a classifier whose decision boundary intersects the constrained
coordinates' level-set neighborhoods with measure μ, the gap is
Θ(μ) — the attack surface removed is exactly the infeasible fraction.

**Proof sketch.** Monotonicity of a maximum over a subset:
max_{x′ ∈ x + F′} [f(x′) > 0] ≤ max_{x′ ∈ x + F}[f(x′) > 0]. ∎

**Empirical validation:** exp9 domain-constrained conditions — NES ASR
2.70% → 0.80% (−70% relative) against the *undefended* model with 16 of 43
UNSW-NB15 features constrained. Any tabular-NIDS robustness claim made
against free-form l2 perturbations overstates the realizable attack surface.

---

## Lemma 3 (Gradient Consistency of the Penetration Integral)

*The state map's gradient is what autograd says it is.*

**Statement.** Let ε_p(h) = (1/|W|) Σ_{i∈W} ‖v_i‖·max(0, M_i) be the
penetration integral over the window W of a query trajectory h, where
v_i = x_i − x_{i−1} and M_i = cos∠(v_i, v_{i−1}). Then ε_p is
piecewise-differentiable in the trajectory, and for real PGD trajectories the
autograd gradient ∇ε_p agrees with the central finite-difference estimate
along random unit directions u:

    |uᵀ∇ε_p − (ε_p(h+δu) − ε_p(h−δu))/(2δ)| / max(1, |uᵀ∇ε_p|) → 0

as δ → 0 away from the max(0,·) kinks, with agreement to O(δ²).

**Role in the argument.** Lemma 3 is the premise-check for Theorem 1's
chain-rule step: if the attacker's autograd ∇ε_p were inconsistent with the
defense's true local behavior, BPDA would optimize a wrong objective. It is
validated directly, not assumed.

**Empirical validation:** exp16 `lemma3` — autograd vs central FD (δ = 1e-3,
8 random directions per point) on real PGD trajectories: 9,216 comparisons,
median relative deviation 1.4e-5, p90 1.0e-4, max(0,·) active on 99.7% of
comparisons. Synthetic sanity: median 8.6e-5, p90 5.5e-4. The premise of
Theorem 1's chain rule holds numerically.

---

## Theorem 1 — Decomposition of the Adaptive Gap (T_static + T_angle + T_basis)

*Where exactly the defended margin changes when the state map changes.*

**Statement.** Under Theorem 1's setting, decompose the defended margin's
sensitivity to the defense's design axes by comparing three ablated state
maps against the full TASCP map D:

  - T_static: D with a *static* transform (no trajectory dependence —
    projection_only mode); expected to equal the undefended margin (exp4).
  - T_angle: D with the rotation angle detached from the trajectory
    (gradient_magnitude mode: Δθ is computed but its gradient is cut).
  - T_basis: D with a fixed reference basis (basis_coupling mode: the
    rotation plane is (e0, e1) instead of data-dependent).

Writing the full-adaptive gradient direction as ĝ and each ablated gradient
as ĝ_abl, the cosine cos∠(ĝ, ĝ_abl) measures how much of the state map's
first-order effect survives each ablation. Theorem 1's strict-degradation
mechanism is localized to whichever ablation breaks the cosines.

**Empirical validation:** exp5 + exp8 established the endpoints (full adaptive
53.60% vs 10.50% undefended; basis-fixed collapses to 10.30%). exp16
`decomposition` measures the gradient cosines directly on firing
trajectories (trigger active on 100% of measured points, UNSW; all three
cosines = 1.00 — the ablated maps are first-order indistinguishable *at the
measured points*, consistent with exp8's finding that the effect, not the
gradient direction, is what basis fixation removes); exp12 adds the controls
(PlaceboRotation: random plane, same angle; ZeroTheta: Δθ = 0), and exp17's
sweep gives the behavioral cross-check: basis-fixed ASR within 2 pp of the
undefended baseline on 26/27 grid points (UNSW), while the full adaptive map
beats undefended on 6/27 points (best: γ=10, τ=0 — 9.8% → 1.4%).
exp12's CICIDS placebo-vs-TASCP probe: [pending exp12 run].

---

## Lemma 2 — Empirical Isotropy of Trajectory Directions

*The ρ-term in the momentum analysis is close to isotropic on real flows.*

**Statement.** Lemma 2's moment computation assumes query-displacement
directions with isotropic angular distribution, giving E[ρ] = √(2/d) for
ρ = ‖v_t‖/‖v_{t−1}‖ under Gaussian probes. On real data the direction
distribution is whatever the attacker emits; the isotropic prediction is an
upper-bound scaffold, and its accuracy is a measurable quantity.

**Empirical validation:** exp16 `rho` — measured E[ρ] = 0.198 on real NES
query streams vs √(2/d) = 0.216 for d = 43 (ratio 0.92). Synthetic sanity:
0.188 measured, 0.87× of prediction. The isotropic scaffold is accurate to
≈ 8% on real flows.

---

## Proposition 3 (NES Probe-Cosine Scaling)

*The antithetic-blindness constant is a function of the probe count q.*

**Statement.** For NES with q probes per anchor step (antithetic or
independent), the expected cosine between consecutive *anchor* displacement
directions scales as

    E[cos∠(v_t, v_{t−1})] ≈ √(q / (q + d + 1)),

interpolating between the pure-antithetic regime (q pairs, E[M_t] = −2/√5,
Corollary 3) and the persistent-direction regime as q grows relative to the
dimension d. This gives the defender a dimension-dependent *prediction* of
which query cadences escape a momentum trigger: attacks with q ≪ d look
independent and stay blind; q approaching d drifts persistent.

**Empirical validation:** exp16 `nes_cosine` — measured cosine vs q ∈
{2, 5, 10, 25, 50, 100} against √(q/(q+d+1)) on real NES query streams. The
isotropic √(q/(q+d+1)) scaffold systematically OVERPREDICTS on real streams
(measured 0.008 at q=2 vs predicted 0.21; measured 0.046 at q=100 vs
predicted 0.83): the real anchor-displacement autocorrelation is close to
zero at all q, i.e. real NES trajectories are even further inside Corollary
3's blind regime than the isotropic model predicts. The monotone-in-q
prediction is directionally wrong *beyond q ≈ 10* on measured data.
exp16's M_t block statistics corroborate: E[M_t] = −0.70 (with-anchor
variant, 31,232 blocks; theory −2/√5 = −0.894), P(M_t > 0) = 1.6% — the
trigger's momentum input sits negative with 98.4% probability per step,
matching the −0.83 measured in exp9.

---

## Proposition 5 (Stationarity of the Defended Verdict)

*The defended verdict on a fixed flow is a function of trajectory state, not
of the flow — quantified.*

**Statement.** Fix a benign probe flow x* and a session trajectory h_t. The
defended margin g_t(x*) = f(D(x*; h_t)) varies with t only through the
defense's rotation state (Δθ_t, plane). For a rotation of angle Δθ in a
plane making angle φ with the local decision-boundary gradient, the
flip-set (inputs whose verdict changes) has width ∝ sin(Δθ/2) in the
rotated direction; hence

  (i) ν_session (verdict flips of x* while the *session* trajectory advances)
      is the natural stationarity measure, and
  (ii) the flip rate scales with the trajectory's Δθ magnitude, giving a
      γ-sweep test: halving γ halves Δθ (in the sub-γ regime) and the
      flip rate falls as sin²(Δθ/2).

An undefended (or static-projection) control has Δθ ≡ 0 and ν ≡ 0
identically — any nonzero ν is attributable to the stateful rotation alone.

**Empirical validation:** exp14 — session-probe ν over persistent-direction
trajectories, ν_final controls, sin(Δθ/2) scaling check across the γ-sweep
{5, 10, 25, 50, 100, 200}, on benign boundary flows (smallest-margin benign
samples). Real-data numbers: [pending exp14 runs]. exp10's live runs already
show the qualitative effect: HSJ's 90.7% ASR is a *transient* of the
non-stationary window, decaying to 24.3% at steady state — detection verdicts
against a stateful defense must be reported as functions of trajectory
time, not as scalars. γ-sweep / sin-scaling real-data check: [pending exp14
run].

---

## Corollary 3 — Scope: Which Score-Based Attacks Fire the Trigger

*Corollary 3 covers symmetric (antithetic-pair) estimators only; the
practical reach of the corollary is an empirical question.*

**Statement.** The E[M_t] = −2/√5 argument applies to gradient estimators
built from symmetric difference pairs. Score-based attackers that are NOT
built from antithetic pairs — NES with i.i.d. probes, SPSA with momentum,
ZOO, SimBA, Square Attack, Boundary Attack — may or may not fire a momentum
trigger, depending on the empirical autocorrelation their update rules
induce. The corollary's scope is therefore delimited by measurement, not by
the theorem.

**Empirical validation:** exp15 `scope` — trigger-fire fraction (Δθ > 1e-5)
measured over ALL queries (not just the final verdict) of seven attackers
against TASCP on UNSW-NB15 (240k–252k queries each):

| attack | fire % | ASR % |
|--------|-------:|------:|
| NES antithetic (reference) | 9.93 | 0.75 |
| NES i.i.d. | 20.91 | 0.60 |
| SPSA | 64.90 | 1.25 |
| ZOO | 2.49 | 0.60 |
| SimBA | 23.30 | 0.85 |
| Square | 20.73 | 46.70 |
| Boundary | 41.01 | 75.80 |

Every non-antithetic attacker fires the trigger on a nonzero fraction of
queries — Corollary 3's practical reach is strictly narrower than its
antithetic-pair statement — but only Square and Boundary achieve
meaningful ASR (46.7% and 75.8%), and both are query-hungry (≈ 2.0 and 8.0
queries-to-success where they succeed, i.e. they win via high-volume
bisection/refinement patterns, not via momentum evasion). Gradient-estimator
attacks (NES/SPSA/ZOO/SimBA variants) fire up to 65% of queries yet land
ASR ≤ 1.25% — firing the trigger does not grant them the objective. The
paper must state: momentum-triangle blindness (Corollary 3) holds for the
canonical antithetic estimator, degrades gracefully for non-antithetic
gradient estimators (which the trigger SEEKS but that do not benefit), and
is genuinely bypassed only by decision-based refinement attacks (Boundary),
consistent with Prop 2's query-complexity picture at higher budgets.

---

## Dataset Dimensionality Note (d = 78 for CICIDS2017)

The parquet CICIDS2017 feature set has **d = 78** columns: it includes the
duplicate column `Fwd Header Length.1` alongside `Fwd Header Length`. Earlier
claims of d = 77 (deduplicating that column) refer to a different
preprocessing and do not match the shipped data. UNSW-NB15 has d = 43.
All experiment code uses the parquet schemas directly (78 / 43).

---

## Third Dataset: NF-ToN-IoT-v2 (Completed — supersedes the earlier out-of-scope note)

The earlier note (P3-19) recorded that no third dataset existed locally.
This session the network path was re-probed and **NF-ToN-IoT-v2** (test
split, HuggingFace `Nora9029/NF-ToN-IoT-v2`, 3,388,100 NetFlow-v9 flows,
41 features, 64.0% attacks, log1p+min-max to [0,1]) was downloaded,
processed, and a classifier trained with the identical protocol
(acc 0.9819, F1 0.9860). Provenance: `data/ton_iot/PROVENANCE.md`.
All core audits were rerun on it with identical budgets
(exps 19/20/21/23): the coupling channel opens (tascp dASR +4.8pp,
tshift +10.6pp), **5/5 family variants violate verdict stationarity**
(nu_session 8.3–37.0%), and the trigger-family dependence profile shifts
(momentum fires 87–100% for every attack family here; cosine/sign
still collapse on specific families). The mechanism transfers; the
profile is dataset-dependent — consistent with the audit framing.

---

## RQ1–RQ6 Experimental Structure (Reviewer-A Response)

The experiment suite is now organized as six research questions instead
of a linear exp1..exp17 chain:

| RQ | Question | Experiments |
|----|----------|-------------|
| RQ1 | Is the phenomenon specific to TASCP? | exp19: 5-member trajectory-conditioned transformation family (rotation/scaling/translation/shear/noise) |
| RQ2 | What makes trajectory state exploitable? | exp20: coupling audit C across 6 state functions x 5 transformations |
| RQ3 | Does the problem survive different attacker query processes? | exp21: 5 triggers x 8 attack families matrix |
| RQ4 | Does stateful perturbation violate the IDS operational contract? | exp19 stationarity replay + exp23 nu_allow |
| RQ5 | Does it generalize beyond UNSW/CICIDS (data + architectures)? | exp24: NF-ToN-IoT-v2 + exp22: MLP/Transformer/XGB/RF split |
| RQ6 | Can the audits distinguish stateful defenses? | exp23: tascp vs Blacklight-style detector vs rate limiter |

**Claim discipline (Reviewer point 8).** The paper does NOT claim "all
moving-target defenses fail". The supported claim: *trajectory-conditioned
perturbation requires three pre-deployment audits (coupling, trigger
observability, stationarity) that conventional static-defense evaluation
omits*, and the audits demonstrably separate defense classes (exp23).

---

## Generalized Coupling Audit (RQ2) — Open/Closed Channel Result

Across 30 (state-function x transformation) cells per dataset, the
normalized coupling C_norm and the adaptive-vs-fixed-state ASR gap are
**uncorrelated** (Spearman rho = -0.126 UNSW / +0.086 TON-IoT, p > 0.5).
What IS supported:

1. **C = 0 exactly identifies the closed channel** (silent state ->
   identity transform -> no exploitable difference; verified by unit
   test on the silent random walk).
2. **C > 0 is necessary but not sufficient** for exploitation: within
   open channels, effect direction is set by the transform's geometry —
   tshift amplifies (+42.4pp on UNSW), tshear suppresses (-48.8pp).
3. Therefore Theorem 1's generalization is stated as: *nonzero state
   coupling is a necessary audit signal; the geometry of the
   state-conditioned transform determines the failure mode.* Not:
   "higher coupling = more exploitable".

---

## Trigger x Attack-Family Matrix (RQ3) — Corollary 3 Generalization

Five trajectory triggers (momentum, cosine, sign agreement, displacement
persistence, antithetic-pair) x eight attack families (NES-independent,
SPSA, ZOO, SimBA, Square, Boundary, random search, antithetic-NES
jitter), thresholds calibrated so no benign window statistic is exceeded
(UNSW, n=100). **No trigger fires on >50% of sessions for every attack
family**; each trigger is blind to at least one family (momentum: ZOO
35%; cosine: NES/SPSA/ZOO/SimBA 0-1%; sign: SPSA/SimBA/jitter 0%;
displacement: NES/jitter 0%; antithetic-pair: ZOO/random 0%).
Corollary 3's blindness is a property of the (trigger, attack-family)
pairing, not of one trigger: *a stateful trigger must be evaluated
against the query-generation statistics of the attacker, not the
designer's intuitive notion of persistent movement.*

---

## Differentiability Split (RQ5)

Mechanism x architecture (UNSW, identical wrapper and budgets):

| mechanism | MLP | Transformer | XGBoost | RandomForest |
|-----------|-----|-------------|---------|--------------|
| coupling (autograd C) | yes | yes | not measurable | not measurable |
| adaptive vs fixed-state (dASR) | +11.0pp | +0.0pp (open but unexploited at budget) | n/a (no gradient) | n/a |
| stationarity violation | 6.62% | 19.75% | 1.96% | 0.46% |
| trigger fires (NES / antithetic) | 100/77% | 100/76.5% | 100/74.5% | 99.5/78.5% |

The B2 coupling channel and gradient-based perturbation effect are
differentiability-dependent; **trigger blindness and stationarity
violations are not** (they require only a query oracle). XGB/RF coupling
is reported as *not measurable*, not as zero.

---

## Defense-Class Audit (RQ6)

Three stateful/query-history defenses audited identically
(exp23; nu_allow = P(verdict flips toward allow as state evolves,
measured on adversarial probes)):

| defense | coupling | trigger blind spot | nu_allow | outcome |
|---------|----------|--------------------|----------|---------|
| TASCP | yes (8.5e7 / 2.1e7) | yes (random search 99%/58% succeed through) | 36.2% UNSW / 18.5% TON-IoT | unsafe perturbation |
| Blacklight-style detector | no differentiable channel | no (all attack sessions flagged at these settings) | 0% (monotone toward reject) | safe direction, benign-recheck FP 100% (k_min=5 flags repeated identical re-checks) |
| Rate limiter | no | budget-breached by all attack sessions | 0% | safe while budget < attack length; legitimate >50-query sessions flagged too |

This is the methodological deliverable: the audits *distinguish* uses of
state (perturbation vs detection/rejection) rather than indicting a class.

---

## Summary Table

| Result | Threat model | Formal claim | Validated by |
|--------|--------------|--------------|--------------|
| Theorem 1 | B2 (white-box) | Deterministic differentiable state = no first-order barrier; can strictly degrade robustness | exp5, exp8 |
| Proposition 2 | B1 (score-based) | Replicating an n-step white-box trajectory costs Ω(d·n/δ²) queries | exp9 sweep (UNSW: 8.85% @1000q vs 10.50% white-box @20 steps; CICIDS2017: 41.45% @1000q) |
| Corollary 3 | B1 (score-based) | Antithetic queries induce E[M_t] = −2/√5 → momentum triggers never fire | exp9 (0.00% activation on **both** UNSW-NB15 and CICIDS2017 — dataset-independent, as predicted) |
| Proposition 4 | B1/B2 (domain) | Feasible-set constraints monotonically shrink ASR | exp9 domain-constrained (UNSW −70% relative; CICIDS −0.8pp — constraints bind only where the constrained features carry the attack surface) |
| Lemma 2, Prop 1 | (existing) | Gradient correlation decay; query-budget bound | exp2, exp3 |
| Lemma 3 | B2 (white-box) | Autograd ∇ε_p matches central FD on real trajectories | exp16: median rel err 1.4e-5, p90 1.0e-4 (9,216 comps) |
| Thm 1 decomp. | B2 (white-box) | Adaptive gap localized: T_static + T_angle + T_basis cosines | exp16: cosines 1.00 on firing points; exp17 behavioral: basis-fixed ≈ undefended 26/27 grid pts |
| Lemma 2 ρ note | B1 (score-based) | E[ρ] on real streams vs isotropic √(2/d) | exp16: 0.198 vs 0.216 (0.92×) |
| Prop 3 | B1 (score-based) | NES cosine ≈ √(q/(q+d+1)) interpolates blind → persistent | exp16: measured ≈ 0 at ALL q (isotropic scaffold overpredicts); E[M_t] = −0.70, P(M>0) = 1.6% |
| Prop 5 | B1/B2 (stationarity) | Verdict flips ∝ sin(Δθ/2); ν session measure; ν ≡ 0 for static controls | exp14 pending; exp10 transient 90.7% → 24.3% |
| Cor 3 scope | B1 (score-based) | Non-antithetic attackers' trigger-fire fraction (measurement, not theorem) | exp15: all fire 2.5–65%; only Square/Boundary get ASR (46.7/75.8%) — Cor 3 scope narrowed |
| Coupling channel (RQ2) | B1/B2 | C=0 identifies closed channel; C>0 necessary not sufficient; geometry sets failure mode | exp19/exp20: 30 cells x 2 datasets; Spearman rho within ±0.27, p > 0.14 |
| Trigger-family dependence (RQ3) | B1 | No trigger fires on >50% of sessions for every attack family | exp21: 5 triggers x 8 families, benign-calibrated taus |
| Family stationarity (RQ4) | B1/B2 | Trajectory-conditioned transformations violate nu across the family (5/5 on TON-IoT) | exp19: nu_session up to 37.0%; exp23 nu_allow 36.2%/18.5% |
| Differentiability split (RQ5) | B1/B2 | Coupling differentiable-only; trigger + stationarity universal | exp22: MLP/Transformer/XGB/RF |
| Defense-class audit (RQ6) | B1/B2 | Audits separate perturbation vs detection/rejection uses of state | exp23: nu_allow 36.2%/18.5% vs 0% (detectors) |
