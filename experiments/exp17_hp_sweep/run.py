"""Exp 17 (P3-16): hyperparameter sweep — gamma x tau x window.

Shows TASCP's failure is not a tuning accident: across the grid
gamma in {10, 50, 100}, tau in {0.0, 0.01, 0.05}, window in {3, 5, 10},
the adaptive BPDA ASR stays above the undefended baseline and the basis-fixed
ablation stays at the baseline. Conditions per grid point (UNSW-NB15,
white-box 20-step PGD budget, 500-sample draws to keep the 27-point grid
affordable):
  - TASCP Full Adaptive (BPDA)
  - TASCP Basis fixed
"""

import csv
import json
import math
import os
import sys

import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
for _p in (
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "experiments"),
    os.path.join(PROJECT_ROOT, "experiments", "exp9_blackbox_attack"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp9_common import load_data
from src.tascp_eval.defenses.tascp import TASCPWrapper
from src.tascp_eval.attacks.pgd import load_config
from src.tascp_eval.attacks.bpda_adaptive import stateful_pgd_attack

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

GAMMAS = [10.0, 50.0, 100.0]
TAUS = [0.0, 0.01, 0.05]
WINDOWS = [3, 5, 10]


def run_one(base_model, X, y, config, gamma, tau, window, mode):
    w = TASCPWrapper(base_model, window_size=window, max_theta=math.pi / 2, tau=tau, gamma=gamma)
    w.adaptive = True
    if mode == "basis_fixed":
        w.ablation_mode = "basis_coupling"
    torch.manual_seed(42)
    x_adv, success, queries = stateful_pgd_attack(w, X, y, config, adaptive=True)
    return {"asr": success.float().mean().item(), "k": int(success.sum().item()), "n": int(X.size(0))}


def run(dataset="unsw_nb15", n_samples=500, seed=42):
    config = load_config()
    base_model, feature_cols, spec, levels, X_sample = load_data(dataset, n_samples, seed)
    y = torch.zeros(X_sample.size(0), 1)
    print(f"\n=== EXP 17: hyperparameter sweep ({dataset}, {n_samples} samples) ===")

    # undefended reference
    torch.manual_seed(42)
    _, success_u, _ = stateful_pgd_attack(base_model, X_sample, y, config, adaptive=False)
    asr_undef = success_u.float().mean().item()
    print(f"Undefended reference ASR: {asr_undef*100:.2f}%")

    results = {}
    n = 0
    for gamma in GAMMAS:
        for tau in TAUS:
            for window in WINDOWS:
                key = f"g={gamma},tau={tau},w={window}"
                adaptive = run_one(base_model, X_sample, y, config, gamma, tau, window, "adaptive")
                basis = run_one(base_model, X_sample, y, config, gamma, tau, window, "basis_fixed")
                results[key] = {"adaptive": adaptive, "basis_fixed": basis, "undefended": asr_undef}
                print(
                    f"  {key:26s} | adaptive {adaptive['asr']*100:6.2f}% | basis-fixed {basis['asr']*100:6.2f}% | undef {asr_undef*100:6.2f}%"
                )

    # aggregate: fraction of grid where adaptive > undefended
    n_grid = len(results)
    n_above = sum(1 for r in results.values() if r["adaptive"]["asr"] > r["undefended"])
    n_basis_at_base = sum(1 for r in results.values() if abs(r["basis_fixed"]["asr"] - r["undefended"]) < 0.02)
    summary = {
        "grid_points": n_grid,
        "adaptive_above_undefended": n_above,
        "basis_fixed_within_2pp_of_undefended": n_basis_at_base,
        "undefended_asr": asr_undef,
    }
    print(f"\nGrid: adaptive beats undefended on {n_above}/{n_grid} points; basis-fixed within 2pp of baseline on {n_basis_at_base}/{n_grid}")

    out_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp17_hp_sweep")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"metrics_{dataset}.json"), "w") as f:
        json.dump({"summary": summary, "grid": results}, f, indent=2)

    with open(os.path.join(PROJECT_ROOT, "results", "tables", f"exp17_hp_sweep_{dataset}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gamma", "tau", "window", "adaptive_asr", "basis_fixed_asr", "undefended_asr"])
        for key, r in results.items():
            g, t, win = key.split(",")
            w.writerow([g[2:], t[4:], win[2:], f"{r['adaptive']['asr']:.4f}", f"{r['basis_fixed']['asr']:.4f}", f"{r['undefended']:.4f}"])
    return results


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "unsw_nb15"
    run(dataset=ds)
