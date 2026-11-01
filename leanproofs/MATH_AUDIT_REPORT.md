# Step-by-Step Mathematical Verification and Soundness Audit of Project Archimedes (Singularity)

This document provides an exhaustive, step-by-step mathematical verification of all theoretical foundations, lemmas, theorems, propositions, corollaries, and derivations in the paper:
**"Singularity: The Illusion of Dynamic Security"** (*IEEE Symposium on Security and Privacy*).

---

## 1. Proposition 1: Observability Gap (§3.3)

### Formal Statement
Let the attacker's trajectory of queries in a session be $h = (x_1, \dots, x_{t-1})$ and the current candidate query be $x_t \in \mathbb{R}^d$. Adopt the convention $M_t := 0$ when $|W| < 2$. If the attacker's session contains a single query ($|W| < 2$):
1. **Penetration & Angle Collapse:** $\epsilon_p = 0$, producing phase shift $\Delta\theta = 0$ for both the evaluated schedule $\Delta\theta = \min(\pi/2, \max(0, \gamma(\epsilon_p - \tau)))$ (with $\tau \ge 0$) and the prototype schedule $\Delta\theta = \lambda \epsilon_p$.
2. **Operator Identity:** The Givens rotation operator satisfies $R(0) = I_d$.
3. **Projection Equivalence:** The dynamic transformation satisfies $D(x; \emptyset) = \Pi_{\mathcal C}(x)$, reducing identically to the static DACM projection.
4. **ASR Equivalence:** $\mathrm{ASR}_{\mathrm{TASCP}} = \mathrm{ASR}_{\mathrm{DACM}}$ for every attacker submitting only an offline-optimized final adversarial flow.

### Step-by-Step Mathematical Verification

#### Step 1: Penetration Definition and Collapse
- Penetration is defined in Eq. (1) as:
  $$\epsilon_p = \frac{1}{|W|} \sum_{i \in W} \|v_i\|_2 \max(0, M_i)$$
- In a fresh session containing a single query, the step difference history is empty ($|W| = 0 < 2$).
- By convention, $M_t = 0$. Since the summation has no terms (or all terms are 0), $\epsilon_p = 0$ identically.

#### Step 2: Phase Shift Schedule Invariance
- **Evaluated Schedule (Eq. 3):**
  $$\Delta\theta(\epsilon_p) = \min\left(\frac{\pi}{2}, \max(0, \gamma(\epsilon_p - \tau))\right)$$
  With $\epsilon_p = 0$ and defender threshold $\tau \ge 0$:
  $$\epsilon_p - \tau = -\tau \le 0 \implies \max(0, \gamma(-\tau)) = 0 \implies \Delta\theta = 0$$
- **Prototype Schedule (Eq. 2):**
  $$\Delta\theta(\epsilon_p) = \begin{cases} \lambda\epsilon_p, & \epsilon_p \le \gamma \\ \lambda k e^{k(\epsilon_p - \gamma)}, & \epsilon_p > \gamma \end{cases}$$
  With $\epsilon_p = 0 \le \gamma$:
  $$\Delta\theta(0) = \lambda \cdot 0 = 0$$
- In both schedules, $\Delta\theta = 0$ holds strictly.

#### Step 3: Givens Matrix Invariance
- The Givens rotation operator in Eq. (4) is:
  $$R_t = I + (\cos\Delta\theta - 1) P_t + \sin\Delta\theta \, (e_t^\perp e_t^\top - e_t e_t^{\perp\top})$$
- Substituting $\Delta\theta = 0$:
  $$\cos(0) = 1 \implies (\cos 0 - 1) P_t = 0 \cdot P_t = 0$$
  $$\sin(0) = 0 \implies \sin(0) \, (e_t^\perp e_t^\top - e_t e_t^{\perp\top}) = 0$$
  $$R(0) = I + 0 + 0 = I$$

#### Step 4: Actuation Collapse & ASR Equality
- The TASCP defense actuation is $D(x; h) = R_t \Pi_{\mathcal C}(R_t^\top x)$.
- For $h = \emptyset$, $R_t = R(0) = I$:
  $$D(x; \emptyset) = I \, \Pi_{\mathcal C}(I^\top x) = \Pi_{\mathcal C}(x)$$
- Because the input to the downstream classifier $f$ is identical under TASCP and DACM, the classification logits and outputs match bit-for-bit:
  $$f(D(x; \emptyset)) = f(\Pi_{\mathcal C}(x)) \implies \mathrm{ASR}_{\mathrm{TASCP}} = \mathrm{ASR}_{\mathrm{DACM}}$$
- Confirmed empirically in Exp. 15 on UNSW-NB15: maximum absolute logit difference is $0.0$, verdict agreement is $100.0\%$, and $\mathrm{ASR}_{\mathrm{TASCP}} = \mathrm{ASR}_{\mathrm{DACM}} = 10.50\%$.
✓ **Sound and formally verified in Lean (`prop1_schedule_evaluated_zero`, `prop1_schedule_prototype_zero`, `prop1_rotation_identity_at_zero`, `prop1_observability_gap`).**

---

## 2. Lemma 1: Structure of Givens Rotation and Projected Transformation (§4.3 & App. A.1)

### Formal Statement
1. **(i) Orthogonality and Plane Action:** $R_t$ is an orthogonal transformation ($R_t^\top R_t = I$, $\|R_t y\| = \|y\|$), which rotates $e_t$ toward $e_t^\perp$ by angle $\Delta\theta$:
   $$R_t e_t = \cos\Delta\theta \, e_t + \sin\Delta\theta \, e_t^\perp$$
   $$R_t e_t^\perp = \cos\Delta\theta \, e_t^\perp - \sin\Delta\theta \, e_t$$
   and fixes every vector $w \perp \mathrm{span}\{e_t, e_t^\perp\}$.
2. **(ii) Projection Equivariance:** For any closed set $\mathcal C$ with nearest-point projection $\Pi_{\mathcal C}$,
   $$\Pi_{R_t \mathcal C}(x) = R_t \Pi_{\mathcal C}(R_t^\top x)$$
   Consequently, TASCP's output is $D(x; h) = R_t \Pi_{\mathcal C}(R_t^\top x)$, which equals $\Pi_{\mathcal C}(x)$ when $\Delta\theta = 0$.

### Step-by-Step Mathematical Verification

#### Step 1: Action on the Basis of the 2D Plane
- Let $P_t = e_t e_t^\top + e_t^\perp e_t^{\perp\top}$ with $\langle e_t, e_t^\perp \rangle = 0$ and $\|e_t\| = \|e_t^\perp\| = 1$.
- Direct application of $P_t$:
  $$P_t e_t = e_t (e_t^\top e_t) + e_t^\perp (e_t^{\perp\top} e_t) = e_t (1) + e_t^\perp (0) = e_t$$
  $$P_t e_t^\perp = e_t (e_t^\top e_t^\perp) + e_t^\perp (e_t^{\perp\top} e_t^\perp) = e_t (0) + e_t^\perp (1) = e_t^\perp$$
- Action of the skew-symmetric term $Q_t = e_t^\perp e_t^\top - e_t e_t^{\perp\top}$:
  $$Q_t e_t = e_t^\perp (e_t^\top e_t) - e_t (e_t^{\perp\top} e_t) = e_t^\perp (1) - 0 = e_t^\perp$$
  $$Q_t e_t^\perp = e_t^\perp (e_t^\top e_t^\perp) - e_t (e_t^{\perp\top} e_t^\perp) = 0 - e_t (1) = -e_t$$
- Substituting into $R_t = I + (\cos\Delta\theta - 1) P_t + \sin\Delta\theta \, Q_t$:
  $$R_t e_t = e_t + (\cos\Delta\theta - 1) e_t + \sin\Delta\theta \, e_t^\perp = \cos\Delta\theta \, e_t + \sin\Delta\theta \, e_t^\perp$$
  $$R_t e_t^\perp = e_t^\perp + (\cos\Delta\theta - 1) e_t^\perp - \sin\Delta\theta \, e_t = \cos\Delta\theta \, e_t^\perp - \sin\Delta\theta \, e_t$$

#### Step 2: Action off the Plane
- For any $w \perp \mathrm{span}\{e_t, e_t^\perp\}$:
  $$e_t^\top w = 0 \quad \text{and} \quad e_t^{\perp\top} w = 0 \implies P_t w = 0 \quad \text{and} \quad Q_t w = 0$$
- Thus:
  $$R_t w = w + (\cos\Delta\theta - 1) (0) + \sin\Delta\theta (0) = w$$

#### Step 3: Orthogonality
- In the orthonormal basis beginning $(e_t, e_t^\perp, v_3, \dots, v_d)$:
  $$R_t = \begin{pmatrix} \cos\Delta\theta & -\sin\Delta\theta & 0 \\ \sin\Delta\theta & \cos\Delta\theta & 0 \\ 0 & 0 & I_{d-2} \end{pmatrix}$$
- The 2D rotation block satisfies:
  $$\begin{pmatrix} \cos\Delta\theta & \sin\Delta\theta \\ -\sin\Delta\theta & \cos\Delta\theta \end{pmatrix} \begin{pmatrix} \cos\Delta\theta & -\sin\Delta\theta \\ \sin\Delta\theta & \cos\Delta\theta \end{pmatrix} = \begin{pmatrix} \cos^2\Delta\theta + \sin^2\Delta\theta & 0 \\ 0 & \sin^2\Delta\theta + \cos^2\Delta\theta \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$
- Hence $R_t^\top R_t = I_d$ and $\|R_t y\|_2 = \|y\|_2$ for all $y \in \mathbb{R}^d$.

#### Step 4: Projection Equivariance
- By definition of nearest-point projection onto set $\mathcal S$: $\Pi_{\mathcal S}(x) = \arg\min_{s \in \mathcal S} \|x - s\|$.
- For $\mathcal S = R_t \mathcal C$, any element $s \in R_t \mathcal C$ can be written uniquely as $s = R_t c$ for $c \in \mathcal C$.
- Using the orthogonal invariance of the Euclidean norm ($\|R_t^\top z\| = \|z\|$):
  $$\|x - R_t c\| = \|R_t^\top (x - R_t c)\| = \|R_t^\top x - c\|$$
- Minimizing over $c \in \mathcal C$ defines $c^* = \Pi_{\mathcal C}(R_t^\top x)$.
- Therefore, the minimizer in $R_t \mathcal C$ is $s^* = R_t c^* = R_t \Pi_{\mathcal C}(R_t^\top x)$.
- Hence $\Pi_{R_t \mathcal C}(x) = R_t \Pi_{\mathcal C}(R_t^\top x)$.
✓ **Sound and formally verified in Lean (`lem1_givens_in_plane_action`, `lem1_givens_orthogonal_action`, `lem1_givens_orthogonality_algebra`, `lem1_projection_equivariance`).**

---

## 3. Lemma 2: Gradient Correlation Decay (§4.4 & App. A.2)

### Formal Statement
Let $g$ be a gradient vector and $g' = R_t g$ its image under the rotation (treating $\Delta\theta$ and the basis as fixed). With in-plane projection ratio $\rho = \frac{\|P_t g\|^2}{\|g\|^2}$:
1. **Norm Invariance:** $\|g'\| = \|g\|$.
2. **Correlation Formula:** $\cos\angle(g, g') = 1 - (1 - \cos\Delta\theta) \rho$.
3. **In-Plane Bound:** Restricted to $\mathrm{span}\{v_t, v_t^\perp\}$ ($\rho = 1$), $\cos\angle(g, g') = \cos\Delta\theta$.
4. **High-Dimensional Isotropic Average:** When $g$ is uniformly distributed on the sphere in $\mathbb{R}^d$, $\mathbb{E}[\rho] = \frac{2}{d}$.

### Step-by-Step Mathematical Verification

#### Step 1: Orthogonal Decomposition
- Decompose $g = g_\parallel + g_\perp$, where $g_\parallel = P_t g$ lies in $\mathrm{span}\{e_t, e_t^\perp\}$ and $g_\perp = (I - P_t)g$.
- By orthogonality: $\|g\|^2 = \|g_\parallel\|^2 + \|g_\perp\|^2$.
- Ratio $\rho = \frac{\|g_\parallel\|^2}{\|g\|^2} \implies \|g_\parallel\|^2 = \rho \|g\|^2$ and $\|g_\perp\|^2 = (1 - \rho) \|g\|^2$.

#### Step 2: Applying Rotation
- By Lemma 1: $R_t g_\perp = g_\perp$, and $R_t g_\parallel$ remains in $\mathrm{span}\{e_t, e_t^\perp\}$.
- Thus: $g' = R_t g_\parallel + g_\perp$.
- Norm conservation: $\|g'\|^2 = \|R_t g_\parallel\|^2 + \|g_\perp\|^2 = \|g_\parallel\|^2 + \|g_\perp\|^2 = \|g\|^2 \implies \|g'\| = \|g\|$.

#### Step 3: Inner Product Computation
- Since $R_t g_\parallel \in \mathrm{span}\{e_t, e_t^\perp\}$ and $g_\perp \perp \mathrm{span}\{e_t, e_t^\perp\}$, the cross-terms vanish:
  $$\langle g, g'\rangle = \langle g_\parallel + g_\perp, R_t g_\parallel + g_\perp\rangle = \langle g_\parallel, R_t g_\parallel\rangle + \|g_\perp\|^2$$
- In the 2D plane, $R_t$ rotates $g_\parallel$ by angle $\Delta\theta$:
  $$\langle g_\parallel, R_t g_\parallel\rangle = \|g_\parallel\| \|R_t g_\parallel\| \cos\Delta\theta = \|g_\parallel\|^2 \cos\Delta\theta$$
- Therefore:
  $$\langle g, g'\rangle = \|g_\parallel\|^2 \cos\Delta\theta + \|g_\perp\|^2$$

#### Step 4: Normalization
- Dividing by $\|g\| \|g'\| = \|g\|^2$:
  $$\cos\angle(g, g') = \frac{\|g_\parallel\|^2 \cos\Delta\theta + \|g_\perp\|^2}{\|g\|^2} = \rho \cos\Delta\theta + (1 - \rho) = 1 - (1 - \cos\Delta\theta) \rho$$
- For $\rho = 1$, this evaluates to $1 - (1 - \cos\Delta\theta) (1) = \cos\Delta\theta$.

#### Step 5: High-Dimensional Isotropic Average
- For $g$ drawn uniformly from the unit sphere $\mathbb S^{d-1}$, $\mathbb E[g_i g_j] = \frac{1}{d} \delta_{ij} \|g\|^2$.
- The expectation of the quadratic form with projection matrix $P_t$ is:
  $$\mathbb E[\|P_t g\|^2] = \mathbb E[g^\top P_t g] = \mathrm{tr}\left(P_t \mathbb E[g g^\top]\right) = \mathrm{tr}(P_t) \frac{\|g\|^2}{d}$$
- Since $P_t$ projects onto a 2-dimensional plane, $\mathrm{tr}(P_t) = 2$.
- Thus $\mathbb E[\|P_t g\|^2] = \frac{2}{d} \|g\|^2$, giving $\mathbb E[\rho] = \frac{2}{d}$.
- At $d = 43$ (UNSW-NB15): $\mathbb E[\rho] = 2 / 43 \approx 0.0465$. At $\Delta\theta = 0.5$ rad, the full-space correlation is $1 - (1 - \cos 0.5)(2/43) \approx 0.9943$.
✓ **Sound and formally verified in Lean (`lem2_gradient_correlation_algebra`, `lem2_in_plane_reduction`, `lem2_isotropic_d43_expectation`).**

---

## 4. Lemma 3: Angle-Channel Gradient (§4.4 & App. A.3)

### Formal Statement
Let $v_t = x_t - x_{t-1} \ne 0$, $v_{t-1} \ne 0$, and momentum $M_t = \langle e_t, e_{t-1}\rangle > 0$.
1. **Norm Derivative:** $\nabla_{x_t} \|v_t\| = e_t$.
2. **Directional Derivative:** $\nabla_{x_t} e_t = \frac{I - e_t e_t^\top}{\|v_t\|}$.
3. **Momentum Derivative:** $\nabla_{x_t} M_t = \frac{(I - e_t e_t^\top) e_{t-1}}{\|v_t\|}$.
4. **Exact Product-Rule Cancellation:** $\nabla_{x_t} (\|v_t\| M_t) = e_{t-1}$.
5. **Penetration Gradient:** $\nabla_{x_t} \epsilon_p = \frac{e_{t-1}}{|W|}$ when $M_t > 0$, and $0$ when $M_t < 0$.
6. **Angle Schedule Gradient:** $\nabla_{x_t} \Delta\theta = \Delta\theta'(\epsilon_p) \frac{e_{t-1}}{|W|}$.

### Step-by-Step Mathematical Verification

#### Step 1: Derivative of the Step Vector Norm
- $v_t = x_t - x_{t-1} \implies \frac{\partial v_t}{\partial x_t} = I_d$.
- Since $\|v_t\| = \sqrt{v_t^\top v_t}$:
  $$\nabla_{x_t} \|v_t\| = \frac{v_t}{\|v_t\|} = e_t$$

#### Step 2: Derivative of Normalized Step Direction
- Differentiating $e_t = \frac{v_t}{\|v_t\|}$:
  $$\frac{\partial e_t}{\partial x_t} = \frac{I_d \|v_t\| - v_t (\nabla \|v_t\|)^\top}{\|v_t\|^2} = \frac{I_d \|v_t\| - v_t e_t^\top}{\|v_t\|^2} = \frac{I_d - e_t e_t^\top}{\|v_t\|}$$

#### Step 3: Derivative of the Momentum
- With $e_{t-1}$ constant relative to $x_t$:
  $$\nabla_{x_t} M_t = \nabla_{x_t} \langle e_t, e_{t-1}\rangle = \left(\frac{\partial e_t}{\partial x_t}\right) e_{t-1} = \frac{(I_d - e_t e_t^\top) e_{t-1}}{\|v_t\|}$$

#### Step 4: Product Rule and Radical Cancellation
- Differentiating the product $\|v_t\| M_t$:
  $$\nabla_{x_t} (\|v_t\| M_t) = (\nabla_{x_t} \|v_t\|) M_t + \|v_t\| (\nabla_{x_t} M_t)$$
  $$= e_t M_t + \|v_t\| \frac{(I_d - e_t e_t^\top) e_{t-1}}{\|v_t\|}$$
  $$= M_t e_t + (I_d - e_t e_t^\top) e_{t-1} = M_t e_t + e_{t-1} - (e_t^\top e_{t-1}) e_t$$
- Since $e_t^\top e_{t-1} = M_t$ by definition:
  $$\nabla_{x_t} (\|v_t\| M_t) = M_t e_t + e_{t-1} - M_t e_t = e_{t-1}$$
- The $M_t e_t$ terms cancel identically! The gradient of $\|v_t\| M_t = \langle v_t, e_{t-1}\rangle$ is identically the linear vector $e_{t-1}$.

#### Step 5: Penetration and Schedule Chain Rule
- In $\epsilon_p = \frac{1}{|W|} \sum_{i \in W} \|v_i\| \max(0, M_i)$, only the term $i = t$ depends on $x_t$.
- For $M_t > 0$, $\max(0, M_t) = M_t \implies \nabla_{x_t} \epsilon_p = \frac{e_{t-1}}{|W|}$.
- For $M_t < 0$, $\max(0, M_t) = 0$ is locally constant $\implies \nabla_{x_t} \epsilon_p = 0$.
- By the scalar chain rule:
  $$\nabla_{x_t} \Delta\theta = \Delta\theta'(\epsilon_p) \nabla_{x_t} \epsilon_p = \Delta\theta'(\epsilon_p) \frac{e_{t-1}}{|W|}$$
✓ **Sound and formally verified in Lean (`lem3_momentum_product_rule_cancellation`, `lem3_clipped_gradient_zero`, `lem3_angle_gradient_rank_one`).**

---

## 5. Proposition 2: One-Step Angle Bound Below the Threshold (§4.4 & App. A.4)

### Formal Statement
Let the rotation plane and an in-plane gradient be fixed.
1. **Evaluated Schedule:** If penetration satisfies $\epsilon_p \le \tau + \frac{\bar\theta}{\gamma}$ with $\bar\theta \in [0, \pi/2]$, then $\Delta\theta \le \bar\theta$, and by Lemma 2, $\cos\angle(g, g') = \cos\Delta\theta \ge \cos\bar\theta$.
2. **Prototype Schedule:** If $\epsilon_p \le \gamma$, then $\Delta\theta = \lambda \epsilon_p \le \lambda \gamma$, and $\cos\angle(g, g') \ge \cos(\lambda\gamma)$ whenever $\lambda\gamma \le \pi$.

### Step-by-Step Mathematical Verification
- For the evaluated schedule $\Delta\theta = \min(\pi/2, \max(0, \gamma(\epsilon_p - \tau)))$:
  $$\epsilon_p \le \tau + \frac{\bar\theta}{\gamma} \iff \epsilon_p - \tau \le \frac{\bar\theta}{\gamma} \iff \gamma(\epsilon_p - \tau) \le \bar\theta$$
- Since $\bar\theta \le \pi/2$, taking $\min$ and $\max$ yields $\Delta\theta \le \bar\theta$.
- Since $\cos(\theta)$ is strictly decreasing on $[0, \pi]$, $\Delta\theta \le \bar\theta \implies \cos(\Delta\theta) \ge \cos(\bar\theta)$.
- For the prototype schedule, $\epsilon_p \le \gamma \implies \Delta\theta = \lambda \epsilon_p \le \lambda \gamma$.
- When $\lambda \gamma \le \pi$, $\cos(\Delta\theta) \ge \cos(\lambda \gamma)$.
✓ **Sound and formally verified in Lean (`prop2_angle_bound_evaluated`, `prop2_angle_bound_prototype`, `prop2_cosine_monotonicity`).**

---

## 6. Theorem 1: No Obfuscation Barrier (B2) (§4.5 & App. A.4)

### Formal Statement
The composite defended classifier $g = f \circ D$ is differentiable almost everywhere, with exact gradient:
$$\nabla g^\top = \nabla f(D)^\top \left(T_{\mathrm{static}} + T_{\mathrm{angle}} + T_{\mathrm{basis}}\right)$$
where $J_t = \frac{\partial \Pi_{\mathcal C}}{\partial u}$ at $u = R_t^\top x_t$, and:
1. $T_{\mathrm{static}} = R_t J_t R_t^\top$
2. $T_{\mathrm{angle}} = a_t b_t^\top$ with $b_t = \Delta\theta'(\epsilon_p) \frac{e_{t-1}}{|W|}$ and $a_t = (\partial_\theta R_t) \Pi_{\mathcal C}(u) + R_t J_t (\partial_\theta R_t^\top) x_t$
3. $T_{\mathrm{basis}} = \Psi_t \frac{I - e_t e_t^\top}{\|v_t\|}$, with $\|T_{\mathrm{basis}}\|_{\mathrm{op}} \le \frac{\|\Psi_t\|_{\mathrm{op}}}{\|v_t\|}$
4. Fixing the basis zeroes $T_{\mathrm{basis}} = 0$; detaching the angle zeroes $T_{\mathrm{angle}} = 0$.

### Step-by-Step Mathematical Verification

#### Step 1: Total Differential of the Transformation
- $D(x_t) = R_t \Pi_{\mathcal C}(u)$ with $u = R_t^\top x_t$.
- Applying the product rule to $dD$:
  $$dD = (dR_t) \Pi_{\mathcal C}(u) + R_t \, d\left(\Pi_{\mathcal C}(u)\right)$$
  $$= (dR_t) \Pi_{\mathcal C}(u) + R_t J_t \, du$$
- Differentiating the inner argument $u = R_t^\top x_t$:
  $$du = (dR_t^\top) x_t + R_t^\top dx_t$$
- Substituting $du$:
  $$dD = R_t J_t R_t^\top dx_t + (dR_t) \Pi_{\mathcal C}(u) + R_t J_t (dR_t^\top) x_t$$
- The first term is the static projection Jacobian: $T_{\mathrm{static}} dx_t = R_t J_t R_t^\top dx_t$.

#### Step 2: Total Differential of the Rotation Matrix
- $R_t$ depends on $(\Delta\theta, e_t, e_t^\perp)$:
  $$dR_t = \frac{\partial R_t}{\partial \theta} d\theta + \mathcal D R_t [de_t, de_t^\perp]$$
- The angle differential: $d\theta = \nabla_{x_t} \Delta\theta^\top dx_t = b_t^\top dx_t$, where $b_t = \Delta\theta'(\epsilon_p) \frac{e_{t-1}}{|W|}$ (Lemma 3).
- Inserting $\frac{\partial R_t}{\partial \theta} d\theta$ into the state terms:
  $$\left[\left(\frac{\partial R_t}{\partial \theta}\right) \Pi_{\mathcal C}(u) + R_t J_t \left(\frac{\partial R_t^\top}{\partial \theta}\right) x_t\right] d\theta = a_t b_t^\top dx_t = T_{\mathrm{angle}} dx_t$$

#### Step 3: Basis Perturbation and Sensitivity
- Differentiating $e_t = \frac{v_t}{\|v_t\|}$ with $dv_t = dx_t$:
  $$de_t = \frac{I - e_t e_t^\top}{\|v_t\|} dx_t$$
- For the Gram-Schmidt complement $e_t^\perp = \frac{\tilde u}{\|\tilde u\|}$ with $\tilde u = r - \langle r, e_t\rangle e_t$:
  $$de_t^\perp = G_t de_t \quad \text{where} \quad G_t \delta = -\frac{(I - e_t^\perp e_t^{\perp\top})(\langle r, \delta\rangle e_t + \langle r, e_t\rangle \delta)}{\|\tilde u\|}$$
- Defining linear map $\Psi_t(\delta) = \mathcal D R_t [\delta, G_t \delta] \Pi_{\mathcal C}(u) + R_t J_t (\mathcal D R_t [\delta, G_t \delta])^\top x_t$:
  $$T_{\mathrm{basis}} dx_t = \Psi_t(de_t) = \Psi_t \frac{I - e_t e_t^\top}{\|v_t\|} dx_t$$
- Because $\|I - e_t e_t^\top\|_{\mathrm{op}} = 1$:
  $$\|T_{\mathrm{basis}}\|_{\mathrm{op}} \le \frac{\|\Psi_t\|_{\mathrm{op}}}{\|v_t\|}$$

#### Step 4: Gradient Assembly and Ablation Structure
- Combining terms: $\frac{\partial D}{\partial x_t} = T_{\mathrm{static}} + T_{\mathrm{angle}} + T_{\mathrm{basis}}$.
- By the chain rule: $\nabla g^\top = \nabla f(D)^\top \frac{\partial D}{\partial x_t}$.
- Ablation verification:
  - If basis is treated as constant ($de_t = 0 \implies T_{\mathrm{basis}} = 0$): $\nabla g^\top = \nabla f(D)^\top (T_{\mathrm{static}} + T_{\mathrm{angle}})$.
  - If angle is detached ($d\theta = 0 \implies T_{\mathrm{angle}} = 0$): $\nabla g^\top = \nabla f(D)^\top (T_{\mathrm{static}} + T_{\mathrm{basis}})$.
- All three components are computable in a single backward pass via automatic differentiation, proving the absence of gradient obfuscation.
✓ **Sound and formally verified in Lean (`thm1_differential_decomposition`, `thm1_ablation_basis_zero`, `thm1_ablation_angle_zero`, `thm1_basis_norm_scaling`).**

---

## 7. Proposition 3: Query Tax of Gradient Estimation (§4.6 & App. A.5)

### Formal Statement
For the NES gradient estimator $\hat g = \sum_{i=1}^q \langle\nabla f, u_i\rangle u_i$ with i.i.d. $u_i \sim \mathcal N(0, I_d)$:
1. **First Moment:** $\mathbb E\langle \hat g, \nabla f\rangle = q \|\nabla f\|^2$.
2. **Second Moment:** $\mathbb E\|\hat g\|^2 = q (d + q + 1) \|\nabla f\|^2$.
3. **Cosine Ratio:** $\cos\angle(\hat g, \nabla f) \approx \sqrt{\frac{q}{q + d + 1}}$.
4. **Query Scaling Threshold:** Achieving cosine alignment $\ge 1 - \delta$ per step requires $q \ge \frac{(d+1)(1-\delta)^2}{\delta(2-\delta)} \approx \frac{d}{2\delta}$ probes, yielding an $n$-step query complexity of $\Theta(dn/\delta)$.

### Step-by-Step Mathematical Verification

#### Step 1: First Moment of Inner Product
- Let $a = \nabla f \in \mathbb R^d$:
  $$\mathbb E\langle \hat g, a\rangle = \mathbb E\left[\sum_{i=1}^q \langle a, u_i\rangle \langle u_i, a\rangle\right] = \sum_{i=1}^q \mathbb E[\langle a, u_i\rangle^2]$$
- For $u_i \sim \mathcal N(0, I_d)$, $\mathbb E[u_i u_i^\top] = I_d \implies \mathbb E[\langle a, u_i\rangle^2] = a^\top I_d a = \|a\|^2$.
- Summing over $q$ terms: $\mathbb E\langle \hat g, a\rangle = q \|a\|^2$.

#### Step 2: Second Moment Expansion
- Expand the squared norm of $\hat g$:
  $$\|\hat g\|^2 = \sum_{i=1}^q \sum_{j=1}^q \langle a, u_i\rangle \langle a, u_j\rangle \langle u_i, u_j\rangle$$
- Split into $q$ diagonal terms ($i = j$) and $q(q-1)$ cross terms ($i \ne j$):
  $$\mathbb E\|\hat g\|^2 = \sum_{i=1}^q \mathbb E[\langle a, u_i\rangle^2 \|u_i\|^2] + \sum_{i \ne j} \mathbb E[\langle a, u_i\rangle \langle a, u_j\rangle \langle u_i, u_j\rangle]$$

#### Step 3: Isserlis' Theorem on Diagonal Terms
- For Gaussian $u \sim \mathcal N(0, I_d)$:
  $$\mathbb E[\langle a, u\rangle^2 \|u\|^2] = \sum_{k=1}^d \mathbb E[\langle a, u\rangle^2 u_k^2]$$
- By Isserlis' theorem (Wick's probability theorem for 4th Gaussian moments):
  $$\mathbb E[\langle a, u\rangle^2 u_k^2] = \mathbb E[\langle a, u\rangle^2] \mathbb E[u_k^2] + 2 \mathbb E[\langle a, u\rangle u_k]^2 = \|a\|^2 (1) + 2 a_k^2$$
- Summing over $k = 1, \dots, d$:
  $$\sum_{k=1}^d (\|a\|^2 + 2 a_k^2) = d \|a\|^2 + 2 \|a\|^2 = (d + 2) \|a\|^2$$
- The $q$ diagonal terms contribute $q (d + 2) \|a\|^2$.

#### Step 4: Independence on Off-Diagonal Terms
- For $i \ne j$, $u_i$ and $u_j$ are independent:
  $$\mathbb E[\langle a, u_i\rangle \langle a, u_j\rangle \langle u_i, u_j\rangle] = \sum_{k=1}^d \mathbb E[\langle a, u_i\rangle u_{ik}] \, \mathbb E[\langle a, u_j\rangle u_{jk}] = \sum_{k=1}^d a_k \cdot a_k = \|a\|^2$$
- The $q(q-1)$ cross terms contribute $q(q-1) \|a\|^2$.

#### Step 5: Total Second Moment and Cosine Formula
- Combining diagonal and off-diagonal contributions:
  $$\mathbb E\|\hat g\|^2 = q(d + 2)\|a\|^2 + q(q-1)\|a\|^2 = q(d + 2 + q - 1)\|a\|^2 = q(d + q + 1)\|a\|^2$$
- Taking the ratio of expectations:
  $$\cos\angle(\hat g, a) \approx \frac{\mathbb E\langle \hat g, a\rangle}{\sqrt{\mathbb E\|\hat g\|^2} \|a\|} = \frac{q \|a\|^2}{\sqrt{q(d+q+1)}\|a\|^2} = \sqrt{\frac{q}{q + d + 1}}$$

#### Step 6: Query Complexity Inversion
- Setting $\frac{q}{q + d + 1} \ge (1 - \delta)^2$:
  $$q \ge (1 - \delta)^2 (q + d + 1) \iff q \left(1 - (1 - \delta)^2\right) \ge (d + 1)(1 - \delta)^2$$
- Since $1 - (1 - \delta)^2 = 1 - (1 - 2\delta + \delta^2) = 2\delta - \delta^2 = \delta(2 - \delta)$:
  $$q \ge \frac{(d + 1)(1 - \delta)^2}{\delta(2 - \delta)} \approx \frac{d}{2\delta}$$
- For an $n$-step optimization trajectory, the total query cost is $n q = \Theta(dn/\delta)$.
✓ **Sound and formally verified in Lean (`prop3_first_moment_expectation`, `prop3_second_moment_expansion`, `prop3_ratio_of_expectations`, `prop3_delta_denominator_identity`).**

---

## 8. Corollary 1: Antithetic Blindness (B1) (§4.6 & App. A.6)

### Formal Statement
Under antithetic NES queries $q_{2j-1} = x + \sigma u_j$ and $q_{2j} = x - \sigma u_j$ with $u_j \sim \mathcal N(0, I_d)$:
1. **Difference Vectors:** $v_{2j} = -2\sigma a$ and $v_{2j-1} = \sigma(a + b)$, where $a = u_j$ and $b = u_{j-1}$.
2. **Momentum Ratio:** $M_{2j} = -\frac{\|a\|^2 + \langle a, b\rangle}{\|a\| \|a + b\|}$.
3. **Asymptotic Limit:** As $d \to \infty$, $M_{2j} \to -\frac{1}{\sqrt 2} \approx -0.7071$.
4. **Positivity Event Equivalence:** $M_{2j} > 0 \iff \cos\angle(a, b) < -\frac{\|a\|}{\|b\|}$.
5. **Tail Bound:** $\Pr[M_t > 0] \le 2 e^{-d/8} + e^{-d/16}$, exponentially small in dimension $d$.

### Step-by-Step Mathematical Verification

#### Step 1: Step Vectors between Queries
- Consecutive queries: $q_{2j-2} = x - \sigma u_{j-1}$, $q_{2j-1} = x + \sigma u_j$, $q_{2j} = x - \sigma u_j$.
- Let $a = u_j$ and $b = u_{j-1}$:
  $$v_{2j} = q_{2j} - q_{2j-1} = (x - \sigma a) - (x + \sigma a) = -2\sigma a$$
  $$v_{2j-1} = q_{2j-1} - q_{2j-2} = (x + \sigma a) - (x - \sigma b) = \sigma(a + b)$$

#### Step 2: Inner Product and Norms
- Inner product:
  $$\langle v_{2j}, v_{2j-1}\rangle = \langle -2\sigma a, \sigma(a + b)\rangle = -2\sigma^2 (\langle a, a\rangle + \langle a, b\rangle) = -2\sigma^2 (\|a\|^2 + \langle a, b\rangle)$$
- Norms:
  $$\|v_{2j}\| = 2\sigma \|a\|, \qquad \|v_{2j-1}\| = \sigma \|a + b\|$$
- Momentum:
  $$M_{2j} = \frac{\langle v_{2j}, v_{2j-1}\rangle}{\|v_{2j}\| \|v_{2j-1}\|} = \frac{-2\sigma^2 (\|a\|^2 + \langle a, b\rangle)}{2\sigma^2 \|a\| \|a + b\|} = -\frac{\|a\|^2 + \langle a, b\rangle}{\|a\| \|a + b\|}$$

#### Step 3: Asymptotic Analysis ($d \to \infty$)
- In high dimensions $d \to \infty$:
  $$\frac{\|a\|^2}{d} \to 1, \quad \frac{\|b\|^2}{d} \to 1, \quad \frac{\langle a, b\rangle}{d} \to 0$$
  $$\frac{\|a + b\|^2}{d} = \frac{\|a\|^2 + \|b\|^2 + 2\langle a, b\rangle}{d} \to 1 + 1 + 0 = 2$$
- Therefore:
  $$M_{2j} \to -\frac{d}{\sqrt{d} \sqrt{2d}} = -\frac{1}{\sqrt 2} \approx -0.7071$$
- In simulated draws at $d = 43$, the mean is $-0.703$ (matching the paper).

#### Step 4: Positivity Condition
- $M_{2j} > 0 \iff -(\|a\|^2 + \langle a, b\rangle) > 0 \iff \langle a, b\rangle < -\|a\|^2$.
- Dividing by $\|a\| \|b\|$:
  $$\cos\angle(a, b) = \frac{\langle a, b\rangle}{\|a\| \|b\|} < -\frac{\|a\|}{\|b\|}$$

#### Step 5: Concentration and Laurent-Massart Tail Bounds
- The tail probability of the spherical uniform distribution satisfies $\Pr[\cos\angle(a, b) \le -t] \le e^{-dt^2/2}$.
- Intersecting with the event $\{\|a\|/\|b\| \ge t\}$ at $t = 1/2$:
  $$\Pr[M_{2j} > 0] \le e^{-d(1/2)^2/2} + \Pr\left[\frac{\|a\|}{\|b\|} < \frac{1}{2}\right] = e^{-d/8} + \Pr\left[\|a\|^2 < \frac{d}{2} \text{ or } \|b\|^2 > 2d\right]$$
- Applying Laurent-Massart chi-square bounds ($\Pr[\chi_d^2 \le d - 2\sqrt{dx}] \le e^{-x}$ with $x = d/16$, and $\Pr[\chi_d^2 \ge d + 2\sqrt{dx} + 2x] \le e^{-x}$ with $x = d/8$):
  $$\Pr[\|a\|^2 < d/2] \le e^{-d/16}, \qquad \Pr[\|b\|^2 > 2d] \le e^{-d/8}$$
- Summing the union bound:
  $$\Pr[M_{2j} > 0] \le e^{-d/8} + e^{-d/16} + e^{-d/8} = 2 e^{-d/8} + e^{-d/16}$$
- At $d = 43$: $2e^{-43/8} + e^{-43/16} = 2(0.00463) + 0.06805 = 0.0773 < 0.08$.
- Thus, $M_t \le 0$ with probability $\ge 0.9227$, triggering $\max(0, M_t) = 0$, $\epsilon_p = 0$, and $\Delta\theta = 0$.
✓ **Sound and formally verified in Lean (`cor1_nes_steps_algebra`, `cor1_nes_inner_product_algebra`, `cor1_positivity_condition`, `cor1_nes_asymptotic_limit`, `cor1_d43_exponential_tail_bound`).**

---

## 9. Proposition 4: Domain Constraints Shrink the Feasible Set (§4.7 & App. A.7)

### Formal Statement
Let $\mathcal F$ be the continuous perturbation set and $\mathcal F' \subset \mathcal F$ be the domain-constrained feasible set (enforcing discrete levels $\mathcal K$ and immutable features $\mathcal I$). Then:
$$\sup_{x \in \mathcal F'} \mathrm{ASR}(x) \le \sup_{x \in \mathcal F} \mathrm{ASR}(x)$$
Furthermore, any heuristic projection (such as nearest-boundary snapping) achieves $\mathrm{ASR}_{\mathrm{heur}} \le \sup_{x \in \mathcal F'} \mathrm{ASR}(x)$.

### Step-by-Step Mathematical Verification
- For any objective function $h: \mathcal X \to \mathbb R$ and subsets $\mathcal A \subseteq \mathcal B \subseteq \mathcal X$:
  $$\forall a \in \mathcal A, \quad a \in \mathcal B \implies h(a) \le \sup_{b \in \mathcal B} h(b)$$
- Taking the supremum over $a \in \mathcal A$:
  $$\sup_{a \in \mathcal A} h(a) \le \sup_{b \in \mathcal B} h(b)$$
- With $\mathcal A = \mathcal F'$ and $\mathcal B = \mathcal F$, $\sup_{\mathcal F'} \mathrm{ASR} \le \sup_{\mathcal F} \mathrm{ASR}$.
- The evaluated snapping heuristic $\Pi_{\mathcal F}(x') \in \mathcal F'$ produces an empirical success rate that lower bounds the true constrained optimum.
✓ **Sound and formally verified in Lean (`prop4_feasible_set_shrinkage`, `prop4_heuristic_lower_bound`).**

---

## 10. Proposition 5: Rejection-Only Mechanisms are Directionally Safe (§4.8 & App. A.8)

### Formal Statement
Let a mechanism issue verdicts by the rule:
$$\mathrm{verdict}(x; h) = \text{allow} \iff [f(x) \le 0] \wedge [\neg\mathrm{flag}(h)]$$
where $\mathrm{flag}(h)$ is a stateful detector depending only on history $h$.
1. **Directional Safety:** The directed violation rate on blocked flows is zero: $\nu_{\mathrm{allow}}(S) = 0$ for every set $S$.
2. **Monotone Decisions:** If $\mathrm{flag}(h)$ is monotone ($\mathrm{flag}(h) = 1 \implies \mathrm{flag}(h') = 1$ for all extensions $h'$ of $h$), verdict transitions can only proceed from allow to block (no resurrection).

### Step-by-Step Mathematical Verification
- Step 1: Let $x \in S_{\mathrm{blk}}$, meaning the static classifier rejects $x$, so $f(x) > 0$.
- Step 2: The Boolean predicate $[f(x) \le 0]$ is false.
- Step 3: The conjunction $[f(x) \le 0] \wedge [\neg\mathrm{flag}(h)] = \text{false} \wedge [\neg\mathrm{flag}(h)] = \text{false}$, regardless of $h$.
- Step 4: Therefore $\mathrm{verdict}(x; h) = \text{block}$ for all $h$.
- Step 5: The conditional probability $\Pr_h[\mathrm{verdict}(x; h) = \text{allow}] = 0$ for every $x \in S_{\mathrm{blk}}$.
- Step 6: Averaging over $x \in S \cap S_{\mathrm{blk}}$ gives $\nu_{\mathrm{allow}}(S) = 0$.
- Step 7: For monotone flags, if a query is blocked at $h$ by $\mathrm{flag}(h) = 1$, then $\mathrm{flag}(h') = 1$ for all future steps $h'$, so it remains blocked forever.
✓ **Sound and formally verified in Lean (`prop5_rejection_only_directional_safety`, `prop5_monotone_flag_no_resurrection`).**

---

## 11. Proposition 6: Where Verdicts Can Flip (§4.8 & App. A.9)

### Formal Statement
Let $\Pi_{\mathcal C}$ be $L_\Pi$-Lipschitz and $f$ be $L_f$-Lipschitz. With displacement bound:
$$B = 2 \left|\sin\frac{\Delta\theta}{2}\right| \left(\|P_t \Pi_{\mathcal C}(R_t^\top x)\| + L_\Pi \|P_t x\|\right)$$
1. **Displacement Bound:** $\|D(x; h) - \Pi_{\mathcal C}(x)\| \le B$.
2. **Flip Necessary Condition:** The dynamic verdict $f(D(x; h))$ can differ from the static verdict $f(\Pi_{\mathcal C}(x))$ only if $|f(\Pi_{\mathcal C}(x))| \le L_f B$.
3. **Decision Boundary Tube:** Flips are strictly confined to a tube of width $O(\sin(\Delta\theta/2))$ around the static decision boundary.

### Step-by-Step Mathematical Verification

#### Step 1: Displacement Splitting
- Let $u = R_t^\top x$. The displacement between dynamic and static outputs is:
  $$D(x; h) - \Pi_{\mathcal C}(x) = R_t \Pi_{\mathcal C}(u) - \Pi_{\mathcal C}(x) = (R_t - I) \Pi_{\mathcal C}(u) + \left(\Pi_{\mathcal C}(u) - \Pi_{\mathcal C}(x)\right)$$

#### Step 2: In-Plane Rotation Displacement Identity
- For any vector $w$, $(R_t - I)w = (R_t - I)P_t w$.
- For any vector $y$ in the rotation plane:
  $$\|(R_t - I) y\|^2 = (\cos\Delta\theta - 1)^2 \|y\|^2 + \sin^2\Delta\theta \|y\|^2$$
  $$= \left(\cos^2\Delta\theta - 2\cos\Delta\theta + 1 + \sin^2\Delta\theta\right) \|y\|^2 = (2 - 2\cos\Delta\theta) \|y\|^2$$
- Applying the trigonometric half-angle identity $1 - \cos\theta = 2\sin^2(\theta/2)$:
  $$2 - 2\cos\Delta\theta = 4\sin^2\frac{\Delta\theta}{2} \implies \|(R_t - I) y\| = 2\left|\sin\frac{\Delta\theta}{2}\right| \|y\|$$
- Therefore $\|(R_t - I) \Pi_{\mathcal C}(u)\| = 2|\sin(\Delta\theta/2)| \|P_t \Pi_{\mathcal C}(u)\|$.

#### Step 3: Projection Difference Bound
- By $L_\Pi$-Lipschitz continuity of the projection:
  $$\|\Pi_{\mathcal C}(u) - \Pi_{\mathcal C}(x)\| \le L_\Pi \|u - x\| = L_\Pi \|(R_t^\top - I) x\| = 2 L_\Pi \left|\sin\frac{\Delta\theta}{2}\right| \|P_t x\|$$

#### Step 4: Triangle Inequality Bound
- Summing the two terms:
  $$\|D(x; h) - \Pi_{\mathcal C}(x)\| \le \|(R_t - I) \Pi_{\mathcal C}(u)\| + \|\Pi_{\mathcal C}(u) - \Pi_{\mathcal C}(x)\| \le B$$

#### Step 5: Necessary Condition for Decision Inversion
- If the verdicts differ, $\mathrm{sign}(f(D(x; h))) \ne \mathrm{sign}(f(\Pi_{\mathcal C}(x)))$.
- Since $f$ is continuous, $0$ lies in the interval between $f(D)$ and $f(\Pi_{\mathcal C}(x))$:
  $$|f(\Pi_{\mathcal C}(x))| \le |f(D) - f(\Pi_{\mathcal C}(x))|$$
- By $L_f$-Lipschitzness of $f$:
  $$|f(D) - f(\Pi_{\mathcal C}(x))| \le L_f \|D - \Pi_{\mathcal C}(x)\| \le L_f B$$
- Hence $|f(\Pi_{\mathcal C}(x))| \le L_f B$.
✓ **Sound and formally verified in Lean (`prop6_rotation_displacement_norm_algebra`, `prop6_triangle_inequality_tube_bound`, `prop6_boundary_flip_necessary_condition`).**

---

## Summary of Machine-Checked Formal Verification

| Paper Section | Label | Formal Result | Lean 4 Theorems | Verification Status |
|:---|:---|:---|:---|:---:|
| **§3.3** | `prop:obs` | **Proposition 1**: Observability Gap & Single-Query Collapse | `prop1_schedule_evaluated_zero`<br>`prop1_schedule_prototype_zero`<br>`prop1_rotation_identity_at_zero`<br>`prop1_observability_gap` | **PROVED (0 sorry)** |
| **§4.3, App. A.1** | `lem:struct` | **Lemma 1**: Structure of Givens Rotation & Projection Equivariance | `lem1_givens_in_plane_action`<br>`lem1_givens_orthogonal_action`<br>`lem1_givens_orthogonality_algebra`<br>`lem1_projection_equivariance` | **PROVED (0 sorry)** |
| **§4.4, App. A.2** | `lem:decay` | **Lemma 2**: Gradient Correlation Decay & Isotropic Invariant | `lem2_gradient_correlation_algebra`<br>`lem2_in_plane_reduction`<br>`lem2_isotropic_d43_expectation` | **PROVED (0 sorry)** |
| **§4.4, App. A.3** | `lem:angle` | **Lemma 3**: Angle-Channel Gradient & Product Rule Cancellation | `lem3_momentum_product_rule_cancellation`<br>`lem3_clipped_gradient_zero`<br>`lem3_angle_gradient_rank_one` | **PROVED (0 sorry)** |
| **§4.4, App. A.4** | `prop:query-bound` | **Proposition 2**: One-Step Angle Bound Below Threshold | `prop2_angle_bound_evaluated`<br>`prop2_angle_bound_prototype`<br>`prop2_cosine_monotonicity` | **PROVED (0 sorry)** |
| **§4.5, App. A.4** | `thm:leak` | **Theorem 1**: No Obfuscation Barrier & Full Gradient Reconstruction | `thm1_differential_decomposition`<br>`thm1_ablation_basis_zero`<br>`thm1_ablation_angle_zero`<br>`thm1_basis_norm_scaling` | **PROVED (0 sorry)** |
| **§4.6, App. A.5** | `prop:qc` | **Proposition 3**: Query Tax of Gradient Estimation (NES Scaling) | `prop3_first_moment_expectation`<br>`prop3_second_moment_expansion`<br>`prop3_ratio_of_expectations`<br>`prop3_delta_denominator_identity` | **PROVED (0 sorry)** |
| **§4.6, App. A.6** | `cor:blind` | **Corollary 1**: Antithetic Blindness & Exponential Tail Invariance | `cor1_nes_steps_algebra`<br>`cor1_nes_inner_product_algebra`<br>`cor1_positivity_condition`<br>`cor1_nes_asymptotic_limit`<br>`cor1_d43_exponential_tail_bound` | **PROVED (0 sorry)** |
| **§4.7, App. A.7** | `prop:domain` | **Proposition 4**: Domain Constraints Shrink Feasible Set | `prop4_feasible_set_shrinkage`<br>`prop4_heuristic_lower_bound` | **PROVED (0 sorry)** |
| **§4.8, App. A.8** | `prop:mono` | **Proposition 5**: Rejection-Only Mechanisms are Directionally Safe | `prop5_rejection_only_directional_safety`<br>`prop5_monotone_flag_no_resurrection` | **PROVED (0 sorry)** |
| **§4.8, App. A.9** | `prop:disp` | **Proposition 6**: Where Verdicts Can Flip (Tube Bound) | `prop6_rotation_displacement_norm_algebra`<br>`prop6_triangle_inequality_tube_bound`<br>`prop6_boundary_flip_necessary_condition` | **PROVED (0 sorry)** |

