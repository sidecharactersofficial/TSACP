# Experiment 5: Adaptive BPDA Attack

The critical stress test against adaptive attackers.

## Scripts

| Script | Target | Budget | Description |
|--------|--------|--------|-------------|
| `run.py` | UNSW-NB15 (MLP) | 20 | Primary experiment: 4 conditions (undefended, DACM, naive, adaptive) |
| `run_cicids2017.py` | CICIDS2017 (MLP) | 20 | Cross-dataset replication of original exp5 |
| `run_cicids2017_budget100.py` | CICIDS2017 (MLP) | 100 | Budget-corrected re-run (20 steps was underpowered for CICIDS2017) |
| `run_xgboost.py` | UNSW-NB15 (XGBoost) | 20 | Cross-architecture replication via surrogate-gradient attack |
| `diagnose_xgboost_naive.py` | UNSW-NB15 (XGBoost) | 20 | Trajectory-level diagnostic of TASCP-Naive XGBoost results |

## Key Findings

- **UNSW-NB15 (MLP):** Adaptive (53.60%) > naive (22.95%) > undefended (10.50%). Gradient masking confirmed: Ablation 3 spikes to 95.70%.
- **CICIDS2017 (MLP, budget=100):** All conditions cluster 53.85%–59.90% (6pp spread). Gradient-masking signature absent.
- **XGBoost (via surrogate):** Weak transfer (undefended ASR = 0.85%). Mechanism pattern does not replicate.
