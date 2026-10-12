# TASCP Experiments

This repository contains the empirical evaluation of TASCP, closely matching the claims from the TASCP paper. 

**IMPORTANT NOTE**: The mathematical implementations here (e.g. windowed momentum $\epsilon_p$, rotation scaling, and Givens generation) correspond to the corrected math deployed in the `tascp-proto/fast-loop` repository. The evaluation natively calls into implementations that mirror the prototype, meaning all empirical results directly validate the actual operational prototype.

## Claims Validated

- **Experiment 1 (Assumption 2: Momentum Separation)**: Validates that gradient-based attacks (Standard PGD, MIM, and Decaying-Step PGD) consistently converge to trajectory structures with extremely high momentum alignment ($M_t \approx 1.0$), easily separable from the quasi-random momentum signatures of standard network traffic (AUC > 0.90).
- **Experiment 2 (Lemma 2: Gradient Correlation Decay)**: Verifies the core geometric premise of the defense. By measuring the gradient calculated through a fixed rotated constraint surface vs. an unrotated surface, this experiment confirms that the unrotated and rotated gradients' correlation decays mathematically according to $\cos(\Delta\theta)$. (Note: Evaluation requires detaching the gradient graph from the rotation generation parameters to simulate a fixed barrier as required by Lemma 2).

## Next Steps

- **Experiment 5 (Adaptive BPDA Attack)**: Test TASCP's robustness against an adaptive attacker attempting to differentiate straight-through the rotation parameters natively.
- **Cross-Dataset Validation**: Integrate and evaluate performance on the CICIDS2017 dataset to verify broad applicability.
