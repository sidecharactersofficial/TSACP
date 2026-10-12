# Related Work — Positioning TASCP

This document is the reference for the related-work section of the paper. It
exists because the most dangerous novelty critique is: *"stateful defenses
against query-based attacks already exist — see Blacklight."* The answer is a
precise delta, not a bigger-claims hedge.

## The Anchor: Blacklight (Li et al., USENIX Security 2022)

> Huiying Li, Shawn Shan, Emily Wenger, Jiayun Zhang, Haitao Zheng, Ben Y.
> Zhao. **"Blacklight: Scalable Defense for Neural Networks against
> Query-Based Black-Box Attacks."** 31st USENIX Security Symposium (USENIX
> Security 22), 2022, pp. 2117–2134. ([arXiv:2006.14042](https://arxiv.org/abs/2006.14042))

**What Blacklight does.** Its design insight is that query-based black-box
attacks are iterative optimizers, so their queries are *highly similar in the
input space*. Blacklight fingerprints each incoming query with probabilistic
perceptual hashing and flags an attack when a query's fingerprint matches the
per-user query history — detecting all eight state-of-the-art attacks they
evaluate, often after only a handful of queries, and blocking completion by
rejecting detected queries. It is robust to several countermeasures, including
an optimized black-box attack designed to defeat similarity detection, and
generalizes beyond images (e.g., text classification).

**Blacklight vs. TASCP — the actual deltas:**

| Axis | Blacklight | TASCP |
|------|-----------|-------|
| Defense *type* | Stateful **detection** — match query fingerprints, then reject/ban | Stateful **perturbation** — rotate the constraint manifold before classification |
| Mechanism | Input-space similarity of consecutive queries | Trajectory momentum → Givens rotation of features |
| Threat model | Black-box query attacks only (B1) | Claims white-box + black-box; we show it fails B2 (exp5) and is inert vs NES in B1 (exp9) |
| Domain | Vision images (perceptual hashing); generalizes to text | Tabular NIDS features (no perceptual-hash analogue; requires feature-space trajectory) |
| Failure mode | Fingerprint-evasive query sequences (large per-query jumps) | Antithetic NES queries: persistent *similarity*, zero *momentum* (exp9) |
| Output on detection | Reject query / ban client | None — silently perturbs the input |

**What is left that is novel here (and what is not):**

1. *Not novel:* "stateful defense that watches the query stream." Blacklight
   established this (2022), and both systems exploit the same asymmetry: the
   defender sees every query; the attacker cannot see the defense's state.
2. *Novel (and testable):* acting on the query stream by **perturbing the
   input geometry** rather than by **detecting and rejecting**. Detection
   (Blacklight) is a fail-closed access-control decision; perturbation (TASCP)
   attempts to degrade the attacker's optimization *without* refusing service.
   These are different deployment contracts, and the perturbation contract
   fails in an instructive way (exp5, exp9).
3. *Novel:* the **mechanism-level autopsy**. We show *why* trajectory-rotation
   fails in both threat models: in B2 because the rotation is
   data-dependent and differentiable (basis coupling → analytic gradients,
   exp5); in B1 because the NES trigger condition is momentum-based and
   antithetic query pairs have systematically non-positive momentum (exp9).
   Blacklight's paper does not contain this failure analysis — it is in a
   different defense family.
4. *Complementarity, honestly stated:* the two mechanisms have *disjoint*
   blind spots. Similarity detection (Blacklight) is maximally sensitive to
   antithetic pairs (near-duplicate queries) — the exact queries TASCP is
   blind to. Momentum rotation (TASCP) fires on persistent directions —
   query patterns Blacklight-style similarity must tolerate to avoid false
   positives on legitimate retries. A deployed system would sensibly layer
   both; neither subsumes the other.

## Other Stateful / Query-Stream Defenses

- **Stateful detection follow-ups.** A growing line of work detects black-box
  attacks from query-stream statistics (similarity, entropy, provenance of
  queries). All are *detectors* in the Blacklight family; TASCP remains in the
  (much less explored) perturbation family. When citing these, the same
  detection-vs-perturbation delta applies.

## Hardening Baselines (compared empirically in exp4)

- **Adversarial training** (Madry et al., 2018): robust optimization against
  white-box PGD. The strongest standard B2 baseline; TASCP underperforms it.
- **Randomized smoothing** (Cohen et al., 2019): certified radii via noise.
  A certified defense — a different guarantee class than TASCP's heuristic
  bound; included in exp4 as an upper-anchor on achievable robustness.
- **DACM (static constraint projection)**: query-independent projection —
  the stateless baseline TASCP was designed to beat. exp4 shows it equals the
  undefended ASR (10.50%), which is itself a finding about static projections.

## Black-Box Attack Algorithms (used in exp9)

- **NES / score-based** (Ilyas et al., ICML 2019): gradient estimation from
  antithetic Gaussian perturbations; the canonical query-efficient score
  attack. Its antithetic structure is what defeats TASCP's trigger (exp9).
- **HopSkipJumpAttack** (Chen, Jordan, Wainwright, IEEE S&P 2020):
  decision-based boundary attack; label-only access. Its persistent boundary
  walk triggers TASCP's rotation — and shrugs it off (90.70% ASR).
- **Practical black-box attacks** (Papernot et al., 2017): substitution-model
  transfer — the original demonstration that black-box attackers need no
  internals; motivates the B1 threat model.

## The Gradient-Masking Critique Line

- **Obfuscated gradients** (Athalye, Carlini, Huang, 2018): defenses that
  break gradient descent without providing robustness; the BPDA recipe.
  exp5 is a TASCP-specific instance of this critique.
- **Adaptive-evaluation norm** (Carlini et al.): evaluate defenses against
  attackers aware of the defense. exp5/exp8/exp9 implement exactly this norm,
  per threat model (see `doc/THREAT_MODEL.md`).

## One-Paragraph Placement (for the paper)

> Stateful defenses that exploit the defender's view of the query stream are
> established — most prominently Blacklight [Li et al., USENIX Security 2022],
> which detects query-based black-box attacks by fingerprint-matching
> consecutive queries. TASCP differs in its actuation: instead of detecting
> and rejecting, it perturbs the input geometry (a momentum-conditioned Givens
> rotation) while continuing to serve queries. Our contribution is not the
> stateful idea but a mechanism-level evaluation of the perturbation contract
> in both threat models: against an adaptive white-box attacker the rotation
> is a differentiable, data-dependent transform that *lowers* robustness
> (53.60% vs 10.50% ASR), and against score-based black-box attackers the
> momentum trigger is structurally blind to antithetic query pairs (0.00%
> rotation activation), leaving the defense inert exactly where its threat
> model should apply. The two failure modes are disjoint from Blacklight's
> known blind spots, indicating that detection- and perturbation-based
> stateful defenses are complementary rather than competing designs.
