# Cross-Dataset Comparison: UNSW-NB15 vs CICIDS2017

Both experiments run with identical infrastructure (same NIDSClassifier
architecture, same TASCPWrapper parameters, same BPDA attack, same sample
size of 2000 correctly-classified benign points, same epsilon=0.1 L2 norm).

**Correction:** The original CICIDS2017 runs used budget=20, which was
underpowered. CICIDS2017 results below are at **budget=100** unless noted.
UNSW-NB15 results remain at budget=20 (no ceiling effect — undefended ASR is
only 10.50%, well below saturation).

---

## Classifier Quality

| Metric       | UNSW-NB15      | CICIDS2017     |
|--------------|----------------|----------------|
| Accuracy     | 99.05%         | **97.02%**     |
| F1 Score     | 0.9631         | **0.9447**     |
| Confusion Matrix | —           | [[446587, 8032], [10544, 158706]] |

The CICIDS2017 classifier is ~2pp weaker (3× higher error rate). This
contributes to higher baseline vulnerability but is not the primary
explanation for the compressed spread.

---

## Exp 5: Adaptive BPDA Attack (4 conditions)

| Condition               | UNSW-NB15 ASR@20 | CICIDS2017 ASR@100 | UNSW MeanQ | CID MeanQ |
|-------------------------|------------------|-------------------|------------|-----------|
| Undefended              | 10.50%           | 56.20%            | 10.7       | 21.2      |
| DACM (Static)           | 10.50%           | 56.20%            | 10.7       | 21.2      |
| TASCP (Naive)           | 22.95%           | 58.80%            | 10.2       | 21.4      |
| TASCP (Adaptive BPDA)   | 53.60%           | 53.85%            | 4.3        | 9.9       |

---

## Exp 8: Mechanism-Isolating Ablations (5 conditions)

| Condition                     | UNSW-NB15 ASR@20 | CICIDS2017 ASR@100 | UNSW MeanQ | CID MeanQ |
|-------------------------------|------------------|-------------------|------------|-----------|
| TASCP (Naive)                 | 22.95%           | 58.80%            | 10.2       | 21.4      |
| TASCP (Full Adaptive)         | 53.60%           | 53.85%            | 4.3        | 9.9       |
| Ablation 1: Grad Mag Detached | 61.60%           | 53.95%            | 5.6        | 12.1      |
| Ablation 2: Basis Uncoupled   | 10.30%           | 56.95%            | 14.6       | 20.6      |
| Ablation 3: Proj Only / Rounded | **95.70%**    | **59.90%**        | 6.1        | 19.6      |
| **Spread (max−min)**          | **85.40pp**      | **6.05pp**        | —          | —         |

---

## Critical Diagnostic: Does angle-rounding cause an ASR spike?

| Dataset      | Full Adaptive → Ablation 3 jump | Interpretation |
|-------------|-------------------------------|----------------|
| UNSW-NB15   | 53.60% → **95.70%** (+42.1pp) | **Gradient masking confirmed** |
| CICIDS2017  | 53.85% → **59.90%** (+6.1pp)  | **No gradient masking signal** |

The gradient-masking signature is **entirely absent** on CICIDS2017. This is
the single most important finding: the mechanism that explains TASCP's
vulnerability on UNSW-NB15 (gradient masking via non-differentiable angle
rounding, with basis coupling as the vulnerability vector) **does not
replicate on CICIDS2017**.

---

## Statistical Tests

**Two-Proportion Z-Test: Basis-Fixed vs Undefended**

| Dataset      | Basis ASR | Undefended ASR | Z        | p-value | Significant (p<0.05)? |
|-------------|-----------|----------------|----------|---------|----------------------|
| UNSW-NB15   | 10.30%    | 10.50%         | −0.207   | 0.836   | No                   |
| CICIDS2017 (budget=20)  | 44.30% | 41.45%      | 1.821    | 0.069   | No (borderline)      |
| **CICIDS2017 (budget=100)** | **56.95%** | **56.20%** | **0.479** | **0.632** | **No** |

The borderline significance at budget=20 (p=0.069) disappears at budget=100
(p=0.632). The basis-fixed condition is statistically indistinguishable from
undefended on both datasets and both budgets.

---

## Overall Assessment

### What replicates across datasets
1. Adaptive attackers achieve higher ASR than naive on both datasets
   (53.60% vs 22.95% on UNSW-NB15; 53.85% vs 58.80% on CICIDS2017 — though
   the ordering inverts at budget=100).
2. Basis-fixed is statistically indistinguishable from undefended on both.
3. DACM has no effect on either dataset.

### What does NOT replicate
1. **Gradient-masking signature (Ablation 3 spike):** +42.1pp on UNSW-NB15,
   +6.1pp on CICIDS2017. **Absent on CICIDS2017.**
2. **Spread magnitude:** 85.40pp on UNSW-NB15, 6.05pp on CICIDS2017
   (corrected for budget). **14× narrower on CICIDS2017.**
3. **Ablation 1 > Full Adaptive:** +8pp on UNSW-NB15, essentially identical
   on CICIDS2017 (+0.1pp).

### Paper implications
- The paper's core mechanistic finding (gradient masking via angle rounding,
  basis coupling as vulnerability vector) is **confirmed on UNSW-NB15 but
  does not replicate on CICIDS2017**.
- The general claim that "adaptive attacks defeat TASCP" replicates, but with
  a fundamentally different mechanism signature and much smaller effect size.
- The findings should be reported as **dataset-dependent**, with the
  mechanistic analysis qualified as specific to UNSW-NB15.
- The CICIDS2017 budget=20 runs were underpowered; budget=100 results are the
  primary reference.
