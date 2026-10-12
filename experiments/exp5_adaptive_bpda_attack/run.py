"""Exp 5: white-box adaptive (BPDA) attack against TASCP.

Streams the dataset (row-group-wise, like exp9) to bound memory, and persists a
machine-readable, per-number-traceable record for every condition to

    results/logs/exp5_adaptive_bpda_attack/metrics_<dataset>.json
    results/tables/exp5_summary_<dataset>.csv

Each record carries the seed, config (and config hash), checkpoint hash,
dataset hash, n_evaluated, n_successful, ASR, and environment.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "experiments", "exp9_blackbox_attack")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp9_common import DATASET_ROOTS, load_data  # noqa: E402
from src.tascp_eval.defenses.tascp import TASCPWrapper  # noqa: E402
from src.tascp_eval.baselines.dacm import DACMWrapper  # noqa: E402
from src.tascp_eval.attacks.bpda_adaptive import stateful_pgd_attack  # noqa: E402
from src.tascp_eval.attacks.pgd import load_config  # noqa: E402
from src.tascp_eval import traceability as T  # noqa: E402

ATTACKER = "stateful_pgd_bpda"


def build_conditions(base_model, gamma=50.0, window_size=5, tau=0.0):
    naive = TASCPWrapper(base_model, window_size=window_size, max_theta=math.pi / 2, tau=tau, gamma=gamma)
    naive.adaptive = False
    adaptive = TASCPWrapper(base_model, window_size=window_size, max_theta=math.pi / 2, tau=tau, gamma=gamma)
    adaptive.adaptive = True
    return [
        ("Undefended", base_model, False),
        ("DACM (Static)", DACMWrapper(base_model), False),
        ("TASCP (Naive)", naive, False),
        ("TASCP (Adaptive BPDA)", adaptive, True),
    ]


def run_exp5(dataset="unsw_nb15", n_samples=2000, seed=42):
    config = load_config("pgd")
    print(f"exp5 | dataset={dataset} n={n_samples} seed={seed} config={config}")
    base_model, feature_cols, spec, levels, X_sample = load_data(dataset, n_samples=n_samples, seed=seed)

    n = len(X_sample)
    y_target = torch.zeros(n, 1, dtype=torch.float32)
    checkpoint = os.path.join(PROJECT_ROOT, "src", "tascp_eval", "models", "weights", f"{dataset}_nids.pth")
    dataset_path = DATASET_ROOTS[dataset]

    records = {}
    for name, model, adaptive in build_conditions(base_model):
        print(f"\n--- {name} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        x_adv, success, queries = stateful_pgd_attack(model, X_sample, y_target, config, adaptive=adaptive)
        k = int(success.sum().item())
        asr = k / n
        mean_q = float(queries[success].mean().item()) if success.any() else float("nan")
        mean_l2 = float((x_adv - X_sample).norm(p=2, dim=1).mean().item())
        print(f"ASR {asr * 100:.2f}% | n_successful {k}/{n} | mean Q {mean_q:.2f} | mean L2 {mean_l2:.4f}")

        records[name] = T.make_record(
            "exp5", dataset, name, ATTACKER, seed, config, n, k, asr,
            metrics={"mean_queries_to_success": mean_q, "mean_l2_distance": mean_l2},
            checkpoint_path=checkpoint, dataset_path=dataset_path,
        )

    out_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp5_adaptive_bpda_attack")
    T.write_results(out_dir, f"metrics_{dataset}.json", records)

    tables = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(tables, exist_ok=True)
    with open(os.path.join(tables, f"exp5_summary_{dataset}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n_evaluated", "n_successful", "asr", "mean_queries_to_success", "mean_l2_distance", "config_sha256"])
        for name, r in records.items():
            m = r["metrics"]
            w.writerow([name, r["n_evaluated"], r["n_successful"], f"{r['asr']:.4f}",
                        f"{m['mean_queries_to_success']:.2f}", f"{m['mean_l2_distance']:.4f}", r["config_sha256"]])

    print(f"\nWrote {out_dir}/metrics_{dataset}.json and results/tables/exp5_summary_{dataset}.csv")
    return records


def main():
    ap = argparse.ArgumentParser(description="exp5 adaptive BPDA attack")
    ap.add_argument("dataset", nargs="?", default="unsw_nb15")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_exp5(dataset=args.dataset, n_samples=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
