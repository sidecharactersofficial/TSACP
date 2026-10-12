"""Exp 8: targeted adaptive-attacker ablations.

Isolates the mechanism the adaptive attacker exploits by comparing the full
adaptive map against three ablations (gradient-magnitude detach, basis
uncoupling, projection-only). Streams data and persists traceable records to

    results/logs/exp8_adaptive_ablations/metrics_<dataset>.json
    results/tables/exp8_ablations_<dataset>.csv
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
from src.tascp_eval.attacks.bpda_adaptive import stateful_pgd_attack  # noqa: E402
from src.tascp_eval.attacks.pgd import load_config  # noqa: E402
from src.tascp_eval import traceability as T  # noqa: E402

ATTACKER = "stateful_pgd_bpda"
UNDEFENDED_REFERENCE_ASR = 0.1050


def _tascp(base_model, adaptive, ablation=None):
    m = TASCPWrapper(base_model, window_size=5, max_theta=math.pi / 2, tau=0.0, gamma=50.0)
    m.adaptive = adaptive
    m.ablation_mode = ablation
    return m


def build_conditions(base_model):
    return [
        ("TASCP (Naive)", _tascp(base_model, False), False),
        ("TASCP (Full Adaptive)", _tascp(base_model, True), True),
        ("TASCP (Ablation 1: Grad Mag)", _tascp(base_model, True, "gradient_magnitude"), True),
        ("TASCP (Ablation 2: Basis)", _tascp(base_model, True, "basis_coupling"), True),
        ("TASCP (Ablation 3: Proj Only)", _tascp(base_model, True, "projection_only"), True),
    ]


def _two_proportion_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    # two-sided normal p-value
    pval = math.erfc(abs(z) / math.sqrt(2))
    return z, pval


def run_exp8(dataset="unsw_nb15", n_samples=2000, seed=42):
    config = load_config("pgd")
    print(f"exp8 | dataset={dataset} n={n_samples} seed={seed} config={config}")
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
            "exp8", dataset, name, ATTACKER, seed, config, n, k, asr,
            metrics={"mean_queries_to_success": mean_q, "mean_l2_distance": mean_l2},
            checkpoint_path=checkpoint, dataset_path=dataset_path,
        )

    # Significance of Ablation 2 (basis) vs the undefended baseline (10.50%).
    basis = records["TASCP (Ablation 2: Basis)"]
    k_ref = int(round(UNDEFENDED_REFERENCE_ASR * n))
    z, pval = _two_proportion_z(basis["n_successful"], n, k_ref, n)
    stats = {
        "undefended_reference_asr": UNDEFENDED_REFERENCE_ASR,
        "ablation2_basis_asr": basis["asr"],
        "z_statistic": z,
        "p_value": pval,
        "significant_at_0.05": bool(pval < 0.05),
    }
    print(f"\nAblation 2 (basis) vs undefended: z={z:.3f}, p={pval:.3f}, significant={stats['significant_at_0.05']}")

    out_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp8_adaptive_ablations")
    T.write_results(out_dir, f"metrics_{dataset}.json", {"conditions": records, "significance": stats})

    tables = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(tables, exist_ok=True)
    with open(os.path.join(tables, f"exp8_ablations_{dataset}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n_evaluated", "n_successful", "asr", "mean_queries_to_success", "mean_l2_distance", "config_sha256"])
        for name, r in records.items():
            m = r["metrics"]
            w.writerow([name, r["n_evaluated"], r["n_successful"], f"{r['asr']:.4f}",
                        f"{m['mean_queries_to_success']:.2f}", f"{m['mean_l2_distance']:.4f}", r["config_sha256"]])

    print(f"\nWrote {out_dir}/metrics_{dataset}.json and results/tables/exp8_ablations_{dataset}.csv")
    return {"conditions": records, "significance": stats}


def main():
    ap = argparse.ArgumentParser(description="exp8 adaptive ablations")
    ap.add_argument("dataset", nargs="?", default="unsw_nb15")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_exp8(dataset=args.dataset, n_samples=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
