import os
import textwrap

structure = {
    "tascp-experiments/README.md": """# TASCP Experiments Repository

This repository contains the empirical validation suite for TASCP.

**CRITICAL COMMITMENT**: The mathematical reference implementation here (`src/tascp_eval/reference_math.py`) MUST stay strictly in sync with the live Rust prototype in `tascp-proto/fast-loop`. 
The current evaluations are verified against the prototype's `windowed-average` epsilon_p formulation and `single-step` momentum (Section IV-B revision).

## PRIORITY ORDER
1. **exp1 and exp2 first** — cheapest, and validate the two things everything else depends on.
2. **exp5 next, before exp3/exp4/exp6/exp7** — determines the paper's ceiling.
3. **exp3, exp4, exp6, exp7 after**, in any order.
""",
    "tascp-experiments/pyproject.toml": """[tool.poetry]
name = "tascp-experiments"
version = "0.1.0"
description = "Empirical validation suite for TASCP"
authors = ["Anonymous Authors"]  # No co-authors — double-blind review artifact

[tool.poetry.dependencies]
python = "^3.11"
torch = "*"
numpy = "*"
scikit-learn = "*"
pandas = "*"
matplotlib = "*"
foolbox = "*"
hydra-core = "*"

[tool.poetry.group.dev.dependencies]
pytest = "*"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
""",
    "tascp-experiments/configs/base.yaml": """# TASCP Base Configuration
# MUST match the current defaults of the tascp-proto fast-loop!
defense:
  gamma: 0.5
  lambda_coeff: 1.0
  k: 2.0
  tau: 1.0
  window_size_w: 10
""",
    "tascp-experiments/configs/dataset/unsw_nb15.yaml": """dataset: "unsw_nb15"
# TODO: add dataset-specific configs
""",
    "tascp-experiments/configs/dataset/cicids2017.yaml": """dataset: "cicids2017"
# TODO: add dataset-specific configs
""",
    "tascp-experiments/configs/attack/pgd.yaml": """attack: "pgd"
# TODO: add pgd hyperparameters
""",
    "tascp-experiments/configs/attack/bim.yaml": """attack: "bim"
# TODO: add bim hyperparameters
""",
    "tascp-experiments/configs/attack/bpda.yaml": """attack: "bpda_adaptive"
# TODO: add bpda hyperparameters
""",
    "tascp-experiments/data/raw/.gitignore": "*\n!.gitignore\n",
    "tascp-experiments/data/processed/.gitignore": "*\n!.gitignore\n",
    "tascp-experiments/data/scripts/download_unsw_nb15.py": "# TODO: script to download UNSW-NB15\n",
    "tascp-experiments/data/scripts/download_cicids2017.py": "# TODO: script to download CICIDS-2017\n",
    "tascp-experiments/data/scripts/preprocess.py": "# TODO: generic preprocessing routines\n",
    "tascp-experiments/src/tascp_eval/__init__.py": "",
    "tascp-experiments/src/tascp_eval/reference_math.py": """# Python re-implementation of momentum/penetration/rotation.
# EXPLICITLY matching the corrected windowed epsilon_p formula and single-step momentum.
import numpy as np

def compute_momentum(vel_t, vel_prev):
    '''Single step M_t (Eq 4-5)'''
    norm_t = np.linalg.norm(vel_t)
    norm_prev = np.linalg.norm(vel_prev)
    if norm_t > 1e-9 and norm_prev > 1e-9:
        return np.dot(vel_t, vel_prev) / (norm_t * norm_prev)
    return 0.0

def compute_penetration(window):
    '''
    window: list of tuples (v_i, m_i)
    epsilon_p = (1 / |W|) * sum_{i in W} ||v_i||_2 * max(0, M_i)
    '''
    if not window:
        return 0.0
    
    total = 0.0
    for v_i, m_i in window:
        total += np.linalg.norm(v_i) * max(0.0, m_i)
    return total / len(window)

# TODO: implement apply_rotation reference logic here matching rust's apply_rotation.
""",
    "tascp-experiments/src/tascp_eval/models/nids_classifier.py": "# TODO: define NIDS PyTorch models\n",
    "tascp-experiments/src/tascp_eval/models/train.py": "# TODO: training loops\n",
    "tascp-experiments/src/tascp_eval/baselines/adversarial_training.py": "# TODO: adversarial training baseline\n",
    "tascp-experiments/src/tascp_eval/baselines/randomized_smoothing.py": "# TODO: randomized smoothing baseline\n",
    "tascp-experiments/src/tascp_eval/baselines/dacm.py": "# TODO: DACM baseline\n",
    "tascp-experiments/src/tascp_eval/attacks/pgd.py": "# TODO: standard PGD attack\n",
    "tascp-experiments/src/tascp_eval/attacks/bim.py": "# TODO: standard BIM attack\n",
    "tascp-experiments/src/tascp_eval/attacks/bpda_adaptive.py": """# TODO: Adaptive BPDA Attack
# Explicitly models R(delta_theta) in its gradient estimate
""",
    "tascp-experiments/experiments/exp1_momentum_separation/run.py": """# Validates Assumptions 1 & 2.
# Report: M_t distributions benign vs adversarial, AUC of a threshold classifier on M_t alone.
# Do NOT pre-commit to an AUC target — report what's measured.
print("Running Exp 1")
""",
    "tascp-experiments/experiments/exp2_gradient_correlation/run.py": """# Validates Lemma 2.
# Report: measured cos(angle(g_t,g_t')) vs predicted cos(delta_theta).
# Swept across a range of delta_theta, end-to-end through a real classifier.
print("Running Exp 2")
""",
    "tascp-experiments/experiments/exp3_query_budget/run.py": """# Validates Proposition 1.
# Report: measured n* vs theoretical prediction.
print("Running Exp 3")
""",
    "tascp-experiments/experiments/exp4_defense_comparison/run.py": """# Attack success rate: AT vs RS vs DACM vs TASCP.
print("Running Exp 4")
""",
    "tascp-experiments/experiments/exp5_adaptive_bpda_attack/run.py": """# THE CRITICAL EXPERIMENT.
# Report honestly regardless of outcome.
print("Running Exp 5")
""",
    "tascp-experiments/experiments/exp6_false_positive_rate/run.py": """# Rotation frequency / accuracy degradation on benign traffic.
# MUST specifically test sustained-mild-correlation benign sessions.
print("Running Exp 6")
""",
    "tascp-experiments/experiments/exp7_latency_benchmark/run.py": """# Calls into the actual Rust fast-loop (via HTTP to the /predict endpoint)
# under realistic concurrent session load.
print("Running Exp 7")
""",
    "tascp-experiments/results/figures/.gitignore": "*\n!.gitignore\n",
    "tascp-experiments/results/tables/README.md": "# Generated LaTeX/CSV tables matching paper's Section IX subsection structure exactly\n",
    "tascp-experiments/results/logs/.gitignore": "*\n!.gitignore\n",
    "tascp-experiments/tests/test_reference_math_matches_prototype.py": """import numpy as np
from tascp_eval.reference_math import compute_penetration

def test_penetration_converged_pgd():
    # Matches Rust test: 5 items, ||v_i||=1.0, M_i=1.0
    v_i = np.zeros(10)
    v_i[0] = 1.0
    window = [(v_i, 1.0) for _ in range(5)]
    assert abs(compute_penetration(window) - 1.0) < 1e-9

def test_penetration_benign_traffic():
    # Matches Rust test: 5 items, ||v_i||=1.0, M_i=0.0
    v_i = np.zeros(10)
    v_i[0] = 1.0
    window = [(v_i, 0.0) for _ in range(5)]
    assert abs(compute_penetration(window) - 0.0) < 1e-9

def test_penetration_window_size_invariance():
    # Matches Rust test: ||v_i||=2.0, M_i=0.5, sizes 5 vs 50
    v_i = np.zeros(10)
    v_i[0] = 2.0
    
    def create_window(size):
        return [(v_i, 0.5) for _ in range(size)]
        
    ep_small = compute_penetration(create_window(5))
    ep_large = compute_penetration(create_window(50))
    
    assert abs(ep_small - ep_large) < 1e-9
    assert abs(ep_small - 1.0) < 1e-9
""",
    "tascp-experiments/scripts/run_priority_experiments.sh": """#!/bin/bash
# Runs exp1, exp2, exp5 only
echo "Running priority experiments..."
python -m experiments.exp1_momentum_separation.run
python -m experiments.exp2_gradient_correlation.run
python -m experiments.exp5_adaptive_bpda_attack.run
""",
    "tascp-experiments/scripts/generate_paper_tables.py": "# TODO: script to compile results into tables\n",
}

for filepath, content in structure.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)

os.chmod("tascp-experiments/scripts/run_priority_experiments.sh", 0o755)

print("tascp-experiments directory scaffolded successfully.")
