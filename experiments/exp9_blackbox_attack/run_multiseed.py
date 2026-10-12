import csv
import json
import math
import os
import sys

import torch
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp9_common import (
    PROJECT_ROOT,
    load_yaml,
    load_data,
    make_tascp,
    run_condition,
)
from src.tascp_eval.attacks.nes import nes_attack
from src.tascp_eval.attacks.hopskipjump import hopskipjump_attack

SEEDS = [42, 43, 44]


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def run_multiseed(dataset="unsw_nb15", n_samples=2000):
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    cfg = load_yaml("configs/attack/blackbox.yaml")
    nes_cfg = dict(cfg["nes"])
    hsj_cfg = dict(cfg["hopskipjump"])

    # Primary paired conditions (domain-constrained variants reported once in
    # the seed-42 run; here we quantify seed variance of the headline pairs).
    per_seed = {s: {} for s in SEEDS}
    for seed in SEEDS:
        base_model, feature_cols, spec, levels, X_sample = load_data(dataset, n_samples, seed)
        print(f"\n=== EXP 9 multi-seed: {dataset}, seed {seed} ===")
        per_seed[seed]["NES / Undefended"] = run_condition(
            f"NES vs Undefended", base_model, X_sample, nes_attack, nes_cfg, seed=seed
        )
        per_seed[seed]["NES / TASCP (Naive)"] = run_condition(
            f"NES vs TASCP (Naive)", make_tascp(base_model), X_sample, nes_attack, nes_cfg, seed=seed
        )
        per_seed[seed]["HSJ / Undefended"] = run_condition(
            f"HSJ vs Undefended", base_model, X_sample, hopskipjump_attack, hsj_cfg, seed=seed
        )
        per_seed[seed]["HSJ / TASCP (Naive)"] = run_condition(
            f"HSJ vs TASCP (Naive)", make_tascp(base_model), X_sample, hopskipjump_attack, hsj_cfg, seed=seed
        )
        del base_model, X_sample

    # Aggregate: successes are per-sample booleans pooled across seeds.
    n = n_samples
    summary = {}
    for cond in per_seed[SEEDS[0]]:
        asrs = [per_seed[s][cond]["asr"] for s in SEEDS]
        k_pool = round(sum(asrs) * n)  # per-seed counts from reported rates
        lo, hi = wilson_ci(k_pool, n * len(SEEDS))
        summary[cond] = {
            "per_seed_asr": asrs,
            "pooled_asr": sum(asrs) / len(SEEDS),
            "wilson95": [lo, hi],
            "per_seed_rotation": [
                per_seed[s][cond].get("rotation_triggered_frac") for s in SEEDS
            ],
        }

    print(f"\n=== EXP 9 MULTI-SEED SUMMARY ({dataset}, n={n} x {len(SEEDS)} seeds) ===")
    for cond, s in summary.items():
        per_seed_str = ", ".join(f"{a * 100:.2f}%" for a in s["per_seed_asr"])
        print(
            f"{cond:28s} | pooled {s['pooled_asr'] * 100:6.2f}% "
            f"(Wilson95 [{s['wilson95'][0] * 100:.2f}%, {s['wilson95'][1] * 100:.2f}%]) "
            f"| seeds: {per_seed_str}"
        )

    out_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp9_blackbox")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "multiseed.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(PROJECT_ROOT, "results", "tables", "exp9_multiseed.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "pooled_asr", "wilson95_lo", "wilson95_hi"] + [f"seed_{s}" for s in SEEDS])
        for cond, s in summary.items():
            w.writerow(
                [cond, f"{s['pooled_asr']:.4f}", f"{s['wilson95'][0]:.4f}", f"{s['wilson95'][1]:.4f}"]
                + [f"{a:.4f}" for a in s["per_seed_asr"]]
            )


if __name__ == "__main__":
    run_multiseed()
