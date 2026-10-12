# CICIDS2017 Exp8 Ablations at Budget=100

All five ablation conditions re-run on CICIDS2017 with query budget=100 (2000
correctly-classified benign samples, same random seed 42, same epsilon=0.1 L2).

---

## Complete 5-Condition Table: CICIDS2017 @ Budget=100

| Condition                     | ASR@100 | MeanQ@100 | Mean L2 |
|-------------------------------|---------|-----------|---------|
| TASCP (Naive)                 | 58.80%  | 21.4      | 0.0614  |
| TASCP (Full Adaptive)         | 53.85%  | 9.9       | 0.0479  |
| Ablation 1: Grad Mag Detached | 53.95%  | 12.1      | 0.0518  |
| Ablation 2: Basis Uncoupled   | 56.95%  | 20.6      | 0.0614  |
| Ablation 3: Proj Only / Rounded | 59.90% | 19.6     | 0.0606  |

Undefended (budget=100): **56.20%**

---

## Critical Question: Does angle-rounding cause an ASR spike?

**No.** Ablation 3 (projection-only / rounded angle) produces **59.90% ASR**,
the highest of the five conditions but barely above naive (58.80%). This is a
**+3.7pp spread** across the five conditions — essentially the same compressed
band as at budget=20.

Compare with UNSW-NB15 at budget=20:

| Condition                     | UNSW-NB15 ASR@20 | CICIDS2017 ASR@100 |
|-------------------------------|------------------|-------------------|
| TASCP (Naive)                 | 22.95%           | 58.80%            |
| TASCP (Full Adaptive)         | 53.60%           | 53.85%            |
| Ablation 1: Grad Mag Detached | 61.60%           | 53.95%            |
| Ablation 2: Basis Uncoupled   | 10.30%           | 56.95%            |
| Ablation 3: Proj Only / Rounded | **95.70%**    | **59.90%**        |
| **Spread (max−min)**          | **85.40pp**      | **6.05pp**        |

The gradient-masking signature (Ablation 3 >> all others) is **absent** on
CICIDS2017. On UNSW-NB15, rounding the angle causes ASR to explode from 53.60%
to 95.70% — the paper's clearest evidence of gradient masking. On CICIDS2017,
the same ablation produces 59.90%, only 6pp above full adaptive (53.85%). The
mechanism is **UNSW-NB15-specific**.

---

## Spread Comparison at Matched Budget=100

| Dataset      | Spread (5 cond) | Low→High Range  |
|--------------|-----------------|-----------------|
| UNSW-NB15@20 | 85.40pp         | 10.30% → 95.70% |
| CICIDS2017@100 | **6.05pp**    | 53.85% → 59.90% |

Even after removing the budget ceiling (20→100), CICIDS2017's spread is
**14× narrower** than UNSW-NB15's.

---

## Two-Proportion Z-Test: Basis-Fixed vs Undefended @ Budget=100

| Basis ASR | Undefended ASR | Z-statistic | p-value | Significant (p<0.05)? |
|-----------|----------------|-------------|---------|----------------------|
| 56.95%    | 56.20%         | 0.4785      | 0.6323  | **No**               |

The basis-fixed condition remains statistically indistinguishable from
undefended at budget=100, consistent with the budget=20 result (p=0.0686 at
budget=20, now p=0.6323 at budget=100).

---

## Conclusion

The CICIDS2017 decision boundary produces fundamentally different behavior:
1. All five conditions cluster within a 6pp band — no condition dramatically
   outperforms or underperforms the others.
2. The gradient-masking signature (angle-rounding spike) is entirely absent.
3. Removing basis coupling does not reduce ASR to undefended level.
4. The 20-step budget was underpowered (ASR values increased 7–16pp when raised
   to 100), but this does not change the clustering pattern.
5. **The paper's core mechanistic finding (gradient masking via angle rounding,
   basis coupling as vulnerability vector) is confirmed on UNSW-NB15 but does
   not replicate on CICIDS2017.**
