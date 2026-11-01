# Formal Results & Machine-Checked Mathematical Foundations

This document details the formal mathematical results established in the paper:
**"Singularity: The Illusion of Dynamic Security"** (*IEEE Symposium on Security and Privacy*).

**Every theorem, lemma, proposition, and corollary throughout the paper has been formally machine-checked and verified in Lean 4 with 0 `sorry`** (located in `leanproofs/ArchimedesProofs/Basic.lean`).

---

## Complete Theorem & Proposition Inventory

### 1. Proposition 1: Observability Gap (§3.3)
*Single-query offline attacks bypass trajectory-conditioned mechanisms entirely.*

**Statement.** Adopt the convention $M_t := 0$ when $|W| < 2$. If the attacker's session contains a single query, then penetration $\epsilon_p = 0$, phase shift $\Delta\theta = 0$ for both schedules (evaluated and prototype; $\tau \ge 0$), $R(0) = I$, and $D(x; \emptyset) = \Pi_{\mathcal C}(x)$, the static DACM projection. Hence $\mathrm{ASR}_{\mathrm{TASCP}} = \mathrm{ASR}_{\mathrm{DACM}}$ for every attacker that submits only a final adversarial flow.
- **Empirical Validation:** Exp 15 on UNSW-NB15 (max absolute logit difference 0.0, 100% verdict agreement, identical 10.50% ASR).
- **Lean 4 Proof:** `prop1_schedule_evaluated_zero`, `prop1_schedule_prototype_zero`, `prop1_rotation_identity_at_zero`, `prop1_observability_gap`.

---

### 2. Lemma 1: Structure of Givens Rotation and Projected Transformation (§4.3, App. A.1)
*Givens transformation is orthogonal, acts as a planar rotation, and preserves projection equivariance.*

**Statement.**
1. $R_t$ is orthogonal ($R_t^\top R_t = I$, $\|R_t y\|_2 = \|y\|_2$), rotates $e_t$ toward $e_t^\perp$ ($R_t e_t = \cos\Delta\theta\,e_t + \sin\Delta\theta\,e_t^\perp$, $R_t e_t^\perp = \cos\Delta\theta\,e_t^\perp - \sin\Delta\theta\,e_t$), and fixes the orthogonal complement of the plane ($R_t w = w$ for $w \perp \mathrm{span}\{e_t, e_t^\perp\}$).
2. For any closed set $\mathcal C$ with nearest-point projection $\Pi_{\mathcal C}$, $\Pi_{R_t\mathcal C}(x) = R_t \Pi_{\mathcal C}(R_t^\top x)$. Hence TASCP's output is $D(x; h) = R_t \Pi_{\mathcal C}(R_t^\top x)$, which equals the static projection $\Pi_{\mathcal C}(x)$ when $\Delta\theta = 0$.
- **Lean 4 Proof:** `lem1_givens_in_plane_action`, `lem1_givens_orthogonal_action`, `lem1_givens_orthogonality_algebra`, `lem1_projection_equivariance`.

---

### 3. Lemma 2: Gradient Correlation Decay (§4.4, App. A.2)
*High-dimensional isotropic gradients decorrelate only weakly under planar rotation.*

**Statement.** For $g' = R_t g$ and $\rho = \|P_t g\|^2 / \|g\|^2$:
$$\|g'\| = \|g\|, \qquad \cos\angle(g, g') = 1 - (1 - \cos\Delta\theta)\rho$$
In particular, restricted to the plane ($\rho = 1$), $\cos\angle(g, g') = \cos\Delta\theta$. For isotropic $g \sim \mathrm{Unif}(\mathbb S^{d-1})$, $\mathbb E[\rho] = 2/d$.
- **Empirical Validation:** Exp 2 and Exp 16 (simulated mean $0.0466$ vs $2/d = 0.0465$ at $d=43$; real probe ratio $0.198$ vs $\sqrt{2/d} = 0.216$; full-space correlation at $\Delta\theta = 0.5$ rad is $0.994$).
- **Lean 4 Proof:** `lem2_gradient_correlation_algebra`, `lem2_in_plane_reduction`, `lem2_isotropic_d43_expectation`.

---

### 4. Lemma 3: Angle-Channel Gradient (§4.4, App. A.3)
*The spatial momentum derivative features an exact radical cancellation, leaving a rank-one gradient pointing along the previous step.*

**Statement.** If $v_t, v_{t-1} \ne 0$ and momentum $M_t > 0$:
$$\nabla_{x_t} \|v_t\| = e_t, \qquad \nabla_{x_t} M_t = \frac{(I - e_t e_t^\top) e_{t-1}}{\|v_t\|}$$
By product rule, $M_t e_t + e_{t-1} - M_t e_t = e_{t-1}$. Hence:
$$\nabla_{x_t} \epsilon_p = \frac{e_{t-1}}{|W|}, \qquad \nabla_{x_t} \Delta\theta = \Delta\theta'(\epsilon_p) \frac{e_{t-1}}{|W|}$$
and $0$ for $M_t < 0$.
- **Empirical Validation:** Exp 16 (9,216 comparisons on real PGD trajectories; autograd and central differences agree to a median relative deviation of $1.4 \times 10^{-5}$).
- **Lean 4 Proof:** `lem3_momentum_product_rule_cancellation`, `lem3_clipped_gradient_zero`, `lem3_angle_gradient_rank_one`.

---

### 5. Proposition 2: One-Step Angle Bound Below the Threshold (§4.4, App. A.4)
*Single-step gradient decorrelation is bounded for threshold-avoidant adversaries.*

**Statement.** For the evaluated schedule, if $\epsilon_p \le \tau + \bar\theta/\gamma$ with $\bar\theta \in [0, \pi/2]$, then $\Delta\theta \le \bar\theta$ and $\cos\angle(g, g') = \cos\Delta\theta \ge \cos\bar\theta$; the clamp gives $\bar\theta = \pi/2$ unconditionally. For the prototype schedule, $\epsilon_p \le \gamma \implies \Delta\theta \le \lambda\gamma$ and $\cos\angle(g, g') \ge \cos(\lambda\gamma)$ whenever $\lambda\gamma \le \pi$.
- **Empirical Validation:** Exp 3 and Exp 17.
- **Lean 4 Proof:** `prop2_angle_bound_evaluated`, `prop2_angle_bound_prototype`, `prop2_cosine_monotonicity`.

---

### 6. Theorem 1: No Obfuscation Barrier (B2) (§4.5, App. A.4)
*Deterministic trajectory-conditioned defenses provide no gradient obfuscation; the full gradient is reconstructible in a single backward pass.*

**Statement.** $g = f \circ D$ is differentiable almost everywhere off a null set, with gradient:
$$\nabla g^\top = \nabla f(D)^\top \left(T_{\mathrm{static}} + T_{\mathrm{angle}} + T_{\mathrm{basis}}\right)$$
where $T_{\mathrm{static}} = R_t J_t R_t^\top$, $T_{\mathrm{angle}} = a_t b_t^\top$ with $b_t = \Delta\theta'(\epsilon_p) \frac{e_{t-1}}{|W|}$, and $T_{\mathrm{basis}} = \Psi_t \frac{I - e_t e_t^\top}{\|v_t\|}$. Operator norm bound: $\|T_{\mathrm{basis}}\|_{\mathrm{op}} \le \frac{\|\Psi_t\|_{\mathrm{op}}}{\|v_t\|}$. Fixing the basis yields $T_{\mathrm{basis}} = 0$; detaching the angle yields $T_{\mathrm{angle}} = 0$.
- **Empirical Validation:** Exp 5 (adaptive BPDA: 53.60% vs 10.50% undefended on UNSW-NB15) and Exp 8 (basis-fixed: 10.30%, angle-detached: 61.60%).
- **Lean 4 Proof:** `thm1_differential_decomposition`, `thm1_ablation_basis_zero`, `thm1_ablation_angle_zero`, `thm1_basis_norm_scaling`.

---

### 7. Proposition 3: Query Tax of Gradient Estimation (§4.6, App. A.5)
*Score-based zeroth-order gradient estimators pay a dimension-scaled query tax.*

**Statement.** For $\hat g = \sum_{i=1}^q \langle\nabla f, u_i\rangle u_i$ with i.i.d. $u_i \sim \mathcal N(0, I_d)$:
$$\mathbb E\langle \hat g, \nabla f\rangle = q \|\nabla f\|^2, \qquad \mathbb E\|\hat g\|^2 = q(d + q + 1) \|\nabla f\|^2$$
$$\cos\angle(\hat g, \nabla f) \approx \sqrt{\frac{q}{q + d + 1}}$$
Target alignment $\ge 1 - \delta$ per step requires $q \ge \frac{(d+1)(1-\delta)^2}{\delta(2-\delta)} \approx \frac{d}{2\delta}$ queries, yielding total query cost $\Theta(dn/\delta)$ for an $n$-step trajectory.
- **Empirical Validation:** Exp 9 and Exp 16 (NES at 1,000 queries reaches 8.85% on UNSW-NB15 vs 10.50% for 20-step PGD, a ~50× query tax).
- **Lean 4 Proof:** `prop3_first_moment_expectation`, `prop3_second_moment_expansion`, `prop3_ratio_of_expectations`, `prop3_delta_denominator_identity`.

---

### 8. Corollary 1: Antithetic Blindness (B1) (§4.6, App. A.6)
*Antithetic NES probe sequences induce negative momentum and exponentially blind momentum triggers.*

**Statement.** For NES queries $q_{2j-1} = x + \sigma u_j$, $q_{2j} = x - \sigma u_j$ with $a = u_j, b = u_{j-1}$:
$$v_{2j} = -2\sigma a, \quad v_{2j-1} = \sigma(a + b) \implies M_{2j} = -\frac{\|a\|^2 + \langle a, b\rangle}{\|a\| \|a + b\|} \xrightarrow{d \to \infty} -\frac{1}{\sqrt 2} \approx -0.7071$$
$M_t > 0$ requires $\cos\angle(a, b) < -\|a\|/\|b\|$, which has probability at most $2e^{-d/8} + e^{-d/16}$. At $d=43$, $\Pr[M_t > 0] \le 0.0773$.
- **Empirical Validation:** Exp 9 (NES rotation firing rate is 0.00% across both UNSW-NB15 and CICIDS2017) and Exp 16 (simulated mean $-0.703$).
- **Lean 4 Proof:** `cor1_nes_steps_algebra`, `cor1_nes_inner_product_algebra`, `cor1_positivity_condition`, `cor1_nes_asymptotic_limit`, `cor1_d43_exponential_tail_bound`.

---

### 9. Proposition 4: Constraints Shrink the Feasible Set (§4.7, App. A.7)
*Domain-constrained optimization monotonically upper-bounds constrained attack success.*

**Statement.** For feasible sets $\mathcal F' \subseteq \mathcal F$, the supremum attack success over $\mathcal F'$ cannot exceed that over $\mathcal F$:
$$\sup_{x \in \mathcal F'} \mathrm{ASR}(x) \le \sup_{x \in \mathcal F} \mathrm{ASR}(x)$$
- **Empirical Validation:** Exp 9 (domain constraints drop ASR by ~70% on UNSW-NB15).
- **Lean 4 Proof:** `prop4_feasible_set_shrinkage`, `prop4_heuristic_lower_bound`.

---

### 10. Proposition 5: Rejection-Only Mechanisms are Directionally Safe (§4.8, App. A.8)
*Stateful detectors that only reject are provably directionally safe ($\nu_{\mathrm{allow}} = 0$).*

**Statement.** If $\mathrm{verdict}(x; h) = \text{allow} \iff [f(x) \le 0] \wedge [\neg\mathrm{flag}(h)]$, then $\nu_{\mathrm{allow}}(S) = 0$ identically for every flow set $S$. If $\mathrm{flag}$ is monotone in $h$, decisions can only transition from allow to block (no resurrection of blocked queries).
- **Empirical Validation:** Exp 23 (Blacklight-style density detector and query-budget rate limiter exhibit $\nu_{\mathrm{allow}} = 0.0\%$ on both UNSW-NB15 and NF-ToN-IoT-v2, whereas TASCP exhibits $\nu_{\mathrm{allow}} = 36.2\%$ and $18.5\%$).
- **Lean 4 Proof:** `prop5_rejection_only_directional_safety`, `prop5_monotone_flag_no_resurrection`.

---

### 11. Proposition 6: Where Verdicts Can Flip (§4.8, App. A.9)
*Verdict inversions under stateful perturbation are confined to an explicit Lipschitz tube around the static decision boundary.*

**Statement.** With $B = 2|\sin(\Delta\theta/2)| \left(\|P_t \Pi_{\mathcal C}(R_t^\top x)\| + L_\Pi \|P_t x\|\right)$:
$$\|D(x; h) - \Pi_{\mathcal C}(x)\| \le B$$
and TASCP's verdict can differ from the static verdict only if $|f(\Pi_{\mathcal C}(x))| \le L_f B$.
- **Empirical Validation:** Exp 10 and Exp 14 (boundary-walking HSJ flips from 90.70% to 24.30% upon re-query; static control exhibits $\nu = 0$).
- **Lean 4 Proof:** `prop6_rotation_displacement_norm_algebra`, `prop6_triangle_inequality_tube_bound`, `prop6_boundary_flip_necessary_condition`.

---

## Machine-Checked Formal Verification Status

All 11 theoretical foundations above are machine-checked in Lean 4:

```bash
cd leanproofs
lake build
./.lake/build/bin/archimedes_proofs
python3 verify_math.py
```
All targets build cleanly with **0 `sorry` assumptions**.
