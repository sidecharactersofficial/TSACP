# Bifurcated Threat Model

TASCP is evaluated under two deliberately separated threat models. Conflating
them is how defense papers end up over-claiming; separating them is how the
existing 20-step white-box PGD experiments (exp1–exp8) stay valid *and* how
operational relevance is established (exp9).

---

## Threat Model B1 — Operational Deployment (Black-Box)

**Motivation:** how attacks actually arrive at a deployed NIDS: a remote
client probes a service it does not own, through the query interface only.

| Capability | Attacker has | Attacker lacks |
|------------|--------------|----------------|
| Model access | query API: submit x, observe output | weights, architecture internals, gradients |
| Output | scores/logits (NES), or label only (HopSkipJump) | — |
| Adaptivity | adapts to *outputs* across queries | cannot backpropagate through the defense |
| Offline compute | arbitrary (can pre-train surrogates on other data) | cannot inspect the defense's state |

**Defense-side assumption:** the defense may be stateful — it observes every
query in a session (this is the operating regime of both TASCP and
Blacklight-style stateful detectors).

**Experiments:** exp9 (NES, HopSkipJump), exp3 (query budget).
**Interpretation:** operational relevance. Results here speak to what a
deployed system would actually face.

## Threat Model B2 — Worst-Case Theoretical Bound (White-Box)

**Motivation:** the cryptographic convention of evaluating against the
strongest possible adversary. If a defense survives this, it survives
anything weaker; if it fails, the *mechanism* — not the deployment — is
disqualified as a security boundary.

| Capability | Attacker has | Notes |
|------------|--------------|-------|
| Model access | full weights, architecture, gradients | — |
| Defense awareness | knows the defense exists and its exact form; backpropagates through it (BPDA) | the "adaptive" attacks of exp5/exp8 |
| Offline compute | arbitrary | — |
| Queries | every evaluation reveals exact gradients | strictly stronger than B1 |

**Experiments:** exp1–exp2 (mechanism validation), exp4 (defense comparison),
exp5 (adaptive BPDA), exp8 (adaptive ablations), exp_transformer (cross-architecture).
**Interpretation:** worst-case bound. Failure here (ASR 53.60% vs 10.50%
undefended) proves the mechanism provides no security *boundary* — it does not
by itself describe a realistic deployment attack, which is why B1 exists.

---

## Why Bifurcate

1. **It protects the existing results.** The 20-step PGD white-box experiments
   are not thrown away — they are the B2 worst-case analysis. Reviewer
   objections of the form "you only tested white-box" are answered by exp9
   (B1), not by discarding exp5.
2. **It prevents over-claiming in both directions.** A defense that fails B2
   may still be operationally useful (if it holds in B1); a defense that holds
   in B1 says nothing about B2. Claims must be scoped to a threat model.
3. **It matches the attack economics.** B1 attacks pay a large query tax
   (~50x: 1,000 NES queries to match 20-step PGD, exp9). B2 attacks pay none.
   A defense that only works by taxing queries (rate limiting, query caps) is
   a B1 mitigation, not a B2 defense — the bifurcation makes this visible.
4. **It isolates what statefulness can and cannot buy.** Statefulness can
   detect or tax persistent-query patterns (B1). Against an adaptive B2
   attacker, deterministic statefulness leaks gradients (Theorem 1,
   `doc/THEOREMS.md`) and buys nothing.

## Summary Table

| | B1 (Black-Box, Operational) | B2 (White-Box, Worst-Case) |
|---|---|---|
| Attacker knowledge | query interface only | full model + defense internals |
| Gradient access | none | exact, through the defense |
| Adaptive to defense | only via output feedback | fully (BPDA) |
| Query economics | expensive (~50x white-box) | free |
| Defense may assume | stateful observation of queries | nothing (attacker sees everything) |
| Experiments | exp9, exp3 | exp1–exp2, exp4–exp5, exp8, exp_transformer |
| What failure means | not deployable against real attackers | no security boundary; mechanism disqualified |

## Claim Scoping Rule

Every empirical claim in this repository is prefixed with its threat model.
"TASCP fails" means: *under B2, TASCP actively degrades robustness
(53.60% vs 10.50% ASR); under B1, TASCP provides no benefit — its rotation
never triggers against NES and merely taxes HSJ (-6.4pp ASR, still 90.70%)*
(exp9). Neither statement alone is the paper's conclusion; together they are.
