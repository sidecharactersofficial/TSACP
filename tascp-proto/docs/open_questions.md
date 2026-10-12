# Open Questions

1. **BPDA Resistance**: Untested in this prototype. Requires empirical validation in the main experiments repository using the BPDA-aware adaptive attacker.
2. **Benign Momentum Assumption**: Assumption 1 (benign trajectories exhibit small momentum variance) is theoretically grounded but remains empirically untested until Exp 1 is fully executed on real UNSW-NB15/CICIDS2017 datasets.
3. **Classifier Misclassification Safety**: If the slow loop misclassifies a benign session as adversarial, we need empirical data to prove the fast-loop fallback is genuinely "safe" (low false positive disruption).
4. **Rotation Subspace Choice**: The Givens rotation acts in `span{v_t, v_t_perp}`. Alternative spanning vectors might offer better robustness bounds; investigating this is out of scope for the current prototype but theoretically relevant.
