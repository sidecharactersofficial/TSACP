# Experiment 6 Summary: False-Positive Rate on Benign Traffic

**Machine-readable archive:** `results/logs/exp6_false_positive_rate/metrics_unsw_nb15.json`.
Every number below is copied from that archive, which records the seed/config,
checkpoint SHA-256, dataset SHA-256, the session-selection rule
(`max_sessions = 500`, `min_session_len = 5`, `groupby(sort=True)` order), and
`n_evaluated`.

This experiment validates the paper's claims regarding the behavior of TASCP on non-adversarial traffic. Specifically, it tests **Assumption 1** (Benign momentum is near zero) and measures the False-Positive Rate (FPR) induced by the defense.

### Experimental Setup
- **Dataset**: UNSW-NB15 (processed)
- **Traffic Scope**: 500 contiguous benign sessions of length $\ge 5$ packets.
- **Model**: Trained NIDS Classifier
- **Parameters**: `window_size` = 5, `max_theta` = $\pi/2$, `gamma` = 50.0

### Results Summary

| Metric | Result |
| :--- | :--- |
| **Total Benign Packets Evaluated** | 21,509 |
| **Active TASCP Steps** ($t \ge 2$) | 20,509 |
| **Rotation Trigger Frequency** ($\Delta\theta_t > 0$) | 248 (1.21%) |
| **Undefended False Positives** | 0 (0.00%) |
| **TASCP False Positives** | 0 (0.00%) |

### Key Findings
1. **Assumption 1 is Validated**: In normal, non-adversarial traffic, consecutive packets do not exhibit the persistent, aligned spatial momentum characteristic of iterative gradient optimization. The rotation condition ($\epsilon_p > \tau$) was triggered in only **1.21%** of active steps.
2. **Zero Accuracy Degradation**: Even when rotation is triggered on benign traffic, it did not push any benign samples across the attack decision boundary. Both the undefended model and the TASCP-defended model maintained a flawless **0.00% False-Positive Rate** on this evaluation subset.
3. **No Defense Overhead on Benign Flows**: Because spatial momentum on benign traffic remains negligible, TASCP functions as a near-transparent pass-through filter under normal operating conditions. The rotation penalty is almost exclusively reserved for adversarially structured query trajectories.
