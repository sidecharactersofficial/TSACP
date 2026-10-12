# Experiment 8: Adaptive Attacker Ablations

Isolates the vulnerability path that the adaptive attacker exploits against TASCP
via three targeted ablations:

1. **Gradient Magnitude Detached** — angle gradient detached; tests if the
   vulnerability is in the angle magnitude vs. rotation basis
2. **Basis Coupling Removed** — fixed, data-independent rotation basis;
   tests if data-dependent basis is the vulnerability vector
3. **Projection-Only / Rounded Angle** — angle rounded to discrete steps;
   tests for gradient masking (expected: ASR explodes if masking exists)

## Scripts

| Script | Dataset | Budget | Description |
|--------|---------|--------|-------------|
| `run.py` | UNSW-NB15 | 20 | Primary ablation experiment |
| `run_cicids2017.py` | CICIDS2017 | 20 | Cross-dataset replication |
| `run_cicids2017_budget100.py` | CICIDS2017 | 100 | Budget-corrected re-run |

## Key Findings

- **UNSW-NB15:** Gradient masking confirmed — Ablation 3 spikes to 95.70%.
  Basis coupling is the vulnerability vector (Ablation 2 drops to undefended,
  p=0.836).
- **CICIDS2017 (budget=100):** All conditions cluster within 6pp (53.85%–59.90%).
  Gradient-masking signature entirely absent. Mechanism is UNSW-NB15-specific.
