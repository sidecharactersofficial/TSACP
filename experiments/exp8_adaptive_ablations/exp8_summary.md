# Experiment 8 Summary: Adaptive Attacker Ablations

**Machine-readable archive:** `results/logs/exp8_adaptive_ablations/metrics_unsw_nb15.json`
(table: `results/tables/exp8_ablations_unsw_nb15.csv`). Every number below is
copied from that archive, which records the seed (42), config + SHA-256,
checkpoint SHA-256, dataset SHA-256, and per-condition numerators/denominators.

This experiment isolates the mechanism that the adaptive attacker (from Exp 5) is exploiting to achieve a 53.60% Attack Success Rate (ASR) against TASCP. Three targeted ablations were implemented to narrow down the exact vulnerability path.

### Experimental Setup
- **Dataset**: UNSW-NB15 (processed)
- **Traffic Scope**: 2000 correctly classified benign samples
- **Budget**: $L_\infty \le 0.05$, Max Iterations: 20
- **Model**: Trained NIDS Classifier

### Results Grid

| Condition | ASR | Mean Queries | Mean L2 Perturbation |
| :--- | :--- | :--- | :--- |
| **TASCP (Naive Baseline)** | 22.95% | 10.2 | 0.0995 |
| **TASCP (Full Adaptive Baseline)** | 53.60% | 4.3 | 0.0582 |
| **Ablation 1 (Grad Mag Detached)** | 61.60% | 5.6 | 0.0646 |
| **Ablation 2 (Basis Uncoupled)** | **10.30%** | 14.6 | 0.0994 |
| **Ablation 3 (Projection-Only / Rounded Angle)** | 95.70% | 6.1 | 0.0907 |

### 1. Statistical Significance of Ablation 2 (Fixed Basis)

The ASR dropped to 10.30% under Ablation 2, while the undefended static baseline is 10.50% ASR. 
We computed a proportions z-test to check if this difference is significant:
- **Z-statistic**: 0.207
- **p-value**: ~0.836

Since $p \gg 0.05$, the difference is **not statistically significant**. The confidence intervals completely overlap.
**Conclusion**: "Fixed-basis TASCP" does not successfully defend against the attacker. Removing the data-dependent basis coupling neutralizes both the vulnerability *and* the defense's effect. The defense becomes statistically equivalent to having no defense at all (a no-op).

### 2. The 95.70% ASR of Ablation 3 (Projection-Only)

Why did ASR jump from 53.60% to 95.70% when we applied a step-function (`round`) to $\Delta\theta$?
By tracing the $\Delta\theta$ values across iterations, we found the angle simply jumped discontinuously between 0 and 0.2 radians. However, zeroing out the gradient of $\Delta\theta$ had a profound effect on the optimizer:
In standard TASCP, $\Delta\theta$ is proportional to the attacker's trajectory momentum. When the attacker steps toward the decision boundary (increasing momentum), $\Delta\theta$ increases, which rotates the manifold and *pushes the sample away* from the boundary. Thus, the gradient of the loss with respect to $\Delta\theta$ acts as a penalty, effectively telling the optimizer "stepping here will trigger a rotation that pushes you back." 
This is classic **gradient masking**.

By zeroing the gradient of $\Delta\theta$ in Ablation 3, we removed this "fear" from the optimizer. The attacker aggressively optimized against the pure, unmasked gradient leaking through the rotation basis vectors ($u, v_{t\_perp}$), completely ignoring the consequences of triggering the rotation, and achieved 95.70% ASR.

### Final Conclusion
TASCP is fundamentally broken in a way that cannot be patched by a simple fix:
1. If we keep the data-dependent basis, the adaptive attacker exploits it (53.60% - 95.70% ASR).
2. If we remove the data-dependent basis, the defense's protective effect drops to zero (10.30% ASR, same as undefended).

The geometric property that makes TASCP work mathematically (Lemma 2 relies on rotating in the span of the attacker's step) is the exact same property that leaks the vulnerability. They are inseparable.
