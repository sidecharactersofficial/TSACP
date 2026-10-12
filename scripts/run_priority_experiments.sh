#!/bin/bash
# Runs exp1, exp2, exp5, exp9 (+CICIDS2017), exp10 (white-box mechanism + adaptive + black-box + hybrid)
echo "Running priority experiments..."
python -m experiments.exp1_momentum_separation.run
python -m experiments.exp2_gradient_correlation.run
python -m experiments.exp5_adaptive_bpda_attack.run
python -m experiments.exp9_blackbox_attack.run
python -m experiments.exp9_blackbox_attack.run_cicids2017
python -m experiments.exp10_hybrid_detection_sim.run
