#!/bin/bash
# Runs the full experiment suite (exp1–exp9), grouped by core claim.
# See doc/EVALUATION_GUIDE.md for the claim structure.
set -e
echo "=== Claim I: mechanism behaves as specified ==="
python -m experiments.exp1_momentum_separation.run
python -m experiments.exp2_gradient_correlation.run
python -m experiments.exp3_query_budget.run

echo "=== Claim II: no security boundary ==="
python -m experiments.exp4_defense_comparison.run
python -m experiments.exp5_adaptive_bpda_attack.run
python -m experiments.exp8_adaptive_ablations.run
python -m experiments.exp9_blackbox_attack.run
python -m experiments.exp9_blackbox_attack.run_cicids2017
python -m experiments.exp9_blackbox_attack.run_multiseed
python -m experiments.exp10_hybrid_detection_sim.run

echo "=== Claim III: deployable and still pointless ==="
python -m experiments.exp6_false_positive_rate.run
python -m experiments.exp7_latency_benchmark.run
# (exp10's benign control doubles as a Claim-III check; see doc/EVALUATION_GUIDE.md)

echo "All experiments complete."
