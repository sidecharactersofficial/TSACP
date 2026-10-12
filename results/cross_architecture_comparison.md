# Cross-Architecture Comparison: MLP vs XGBoost (UNSW-NB15)

## Methodology

**XGBoost** was trained on the same UNSW-NB15 train/test split (80/20,
stratified) as the primary NIDS MLP. It has **no native gradient** — attack
gradients are obtained from a **differentiable surrogate neural network**
(128×128 MLP, trained to mimic XGBoost's logit output via MSE on 200k
subsamples).

The surrogate's prediction agreement with XGBoost on the evaluation set is
**87.4%** (100% on benign samples, indicating disagreement concentrates on
attack-class samples).

**Attack procedure** (BPDA via surrogate):
1. Gradient computation flows through the surrogate (wrapped in TASCPWrapper
   for rotation conditions).
2. TASCP's rotation operates on the real input space; only gradient
   computation goes through the surrogate.
3. **Success is evaluated against the real XGBoost model's output**, not the
   surrogate's.
4. Query budget = 20 (matched to the primary MLP paper table).

This follows standard black-box attack methodology (Papernot et al., 2017).

---

## Classifier Quality

| Metric       | MLP       | XGBoost   |
|--------------|-----------|-----------|
| Accuracy     | 99.05%    | **99.44%** |
| F1 Score     | 0.9631    | **0.9776** |
| Confusion Matrix | —    | [[442577, 1176], [1690, 62567]] |

XGBoost is slightly more accurate than the MLP on UNSW-NB15.

---

## 5-Condition Attack Results (Budget=20)

| Condition                     | MLP ASR   | XGBoost ASR (surrogate-gradient) | MLP MeanQ | XGBoost MeanQ |
|-------------------------------|-----------|----------------------------------|-----------|---------------|
| Undefended                    | **10.50%**| **0.85%**                        | 10.7      | 5.4           |
| TASCP (Naive)                 | 22.95%    | **34.75%**                       | 10.2      | 11.9          |
| TASCP (Full Adaptive)         | 53.60%    | **1.55%**                        | 4.3       | 3.7           |
| Ablation 1: Grad Mag Detached | 61.60%    | **1.75%**                        | 5.6       | 5.1           |
| Ablation 2: Basis Uncoupled   | 10.30%    | **0.60%**                        | 14.6      | 5.0           |
| Ablation 3: Proj Only / Rounded | **95.70%** | **36.05%**                    | 6.1       | 9.0           |
| **Spread (max−min)**          | **85.40pp**| **35.45pp**                     | —         | —             |

---

## Key Findings

### 1. Surrogate-gradient attack has very low transfer success
The undefended baseline is only **0.85%** (vs 10.50% for MLP). The surrogate
(87.4% label agreement) does not provide gradient directions that reliably
cross XGBoost's decision boundary. This is a well-known limitation of
black-box surrogate attacks — the attack is fundamentally weaker against a
non-differentiable target.

### 2. The MLP pattern does NOT replicate on XGBoost
- **Adaptive vs naive flipped:** On MLP, adaptive (53.60%) >> naive (22.95%).
  On XGBoost, naive (34.75%) >> adaptive (1.55%). The adaptive gradient path
  through the surrogate does not help — possibly because the surrogate's
  decision surface is misaligned with XGBoost's, making the "adaptive"
  gradient direction counterproductive.
- **Ablation 3 remains highest but not dominant:** On MLP, angle-rounding
  explodes to 95.70% (clear gradient-masking signature). On XGBoost, it
  reaches 36.05% — the highest ASR but only 1.3pp above naive (34.75%). The
  dramatic gradient-masking signature is absent.
- **Basis-fixed is lowest (0.60%):** Statistically indistinguishable from
  undefended (0.85%, p=0.3514).

### 3. Z-test: Basis vs Undefended

| Architecture | Basis ASR | Undefended ASR | Z      | p-value | Significant? |
|-------------|-----------|----------------|--------|---------|-------------|
| MLP         | 10.30%    | 10.50%         | −0.207 | 0.836   | No          |
| XGBoost     | 0.60%     | 0.85%          | −0.932 | 0.351   | No          |

Both architectures agree: basis-fixed is indistinguishable from undefended.

---

## Verification: Is the TASCP-Naive 34.75% a real finding?

A detailed diagnostic was run (200 benign samples, trajectory tracking at every
PGD iteration, per-sample L2 verification and cross-referencing with XGBoost
predictions). Results:

1. **L2 budget respected in all inspected successes.** Every successful sample
   stayed within epsilon=0.1 (observed range 0.01–0.10).

2. **Rotation is genuinely applied.** `||x_adv − x_rot||` at success ranges
   0.03–0.48, confirming the defended (rotated) input is what reaches XGBoost.
   No tensor-mismatch bug.

3. **XGBoost genuinely misclassifies the rotated input.** At the success
   iteration, XGBoost outputs class 1 on the exact rotated tensor. No
   indexing or logging error.

4. **Successes do NOT cluster in surrogate/XGBoost initial disagreement.**
   Surrogate and XGBoost agree 100% on the 200 benign samples used for the
   diagnostic. The 87.4% overall agreement figure reflects disagreement on
   attack-class samples only. All successes originate from the agreement
   region — the ASR is not an artifact of pre-existing label mismatch.

**Mechanism explanation:** TASCP-Naive applies a rotation that is
*parameterized by the surrogate's state but detached from the gradient graph*.
The PGD step moves `x_adv` along the surrogate's gradient direction; the
rotation then displaces the point to `x_rot`. Because the rotation is
detached, it does not track the surrogate's decision surface, and can push
`x_rot` into regions where the surrogate and XGBoost differ — even though
they agree at the origin. This is a genuine (but architecture-specific)
interaction between TASCP's rotation and the surrogate/target mismatch.
**The 34.75% ASR is real, not a bug.**

## Interpretation

**Does the basis-coupling mechanism replicate against XGBoost?**

No — but for a different reason than the CICIDS2017 case.

- On **CICIDS2017**: the mechanism doesn't replicate because the decision
  boundary compresses all conditions. The mechanism's signal exists but is
  swamped by high baseline vulnerability.
- On **XGBoost**: the mechanism can't be evaluated at the same fidelity
  because gradient-based attacks have very low transfer through the surrogate.
  The 0.85% undefended ASR indicates the attack is barely working at all.

**Does TASCP defend XGBoost?** The question is ill-posed without gradient
access. TASCP's mechanism (rotation in the span of the attacker's step)
requires gradient-based attacks to defend against. For non-differentiable
models, TASCP is not applicable in its current form.

**Paper implication:** The paper should clearly scope its claims to
**differentiable architectures** where gradient-based attacks are feasible.
The mechanism's dataset dependence (UNSW-NB15 vs CICIDS2017) is a genuine
finding; its architecture dependence follows from the fundamental requirement
of gradient access.
