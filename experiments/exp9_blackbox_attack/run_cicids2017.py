import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp9_common import (
    PROJECT_ROOT,
    load_yaml,
    load_data,
    make_projector,
    make_tascp,
    run_condition,
    run_nes_budget_sweep,
    print_summary,
    save_results,
)
from src.tascp_eval.attacks.nes import nes_attack
from src.tascp_eval.attacks.hopskipjump import hopskipjump_attack


def run_exp9_cicids2017(n_samples=2000, seed=42, chunk_size=500):
    """Cross-dataset replication (Threat Model B1) on CICIDS2017.

    Memory-constrained: 3.1M x 78 float32 rows are processed in 500-sample chunks
    (chunking bounds the attack loop's working set, not the dataset load).
    """
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    cfg = load_yaml("configs/attack/blackbox.yaml")
    nes_cfg = dict(cfg["nes"])
    hsj_cfg = dict(cfg["hopskipjump"])
    budget_sweep = nes_cfg.pop("budget_sweep", [])

    base_model, feature_cols, spec, levels, X_sample = load_data("cicids2017", n_samples, seed)
    projector = make_projector(feature_cols, spec, levels)
    n_constrained = len(spec.get("binary", [])) + len(spec.get("discrete", [])) + len(spec.get("immutable", []))
    print(f"\nDomain constraints active on {n_constrained} of {len(feature_cols)} features")

    results = {}
    print(f"\n=== EXP 9: Black-box query attacks on cicids2017 ({len(X_sample)} samples, seed {seed}) ===")

    results["NES / Undefended"] = run_condition(
        f"NES vs Undefended (budget={nes_cfg['max_queries']})", base_model, X_sample, nes_attack, nes_cfg, seed=seed, chunk_size=chunk_size
    )
    results["NES / TASCP (Naive)"] = run_condition(
        f"NES vs TASCP (Naive) (budget={nes_cfg['max_queries']})", make_tascp(base_model), X_sample, nes_attack, nes_cfg, seed=seed, chunk_size=chunk_size
    )
    results["NES / TASCP (Naive) + Domain-Constrained"] = run_condition(
        f"NES vs TASCP (Naive) + domain constraints (budget={nes_cfg['max_queries']})",
        make_tascp(base_model), X_sample, nes_attack, nes_cfg, projector=projector, seed=seed, chunk_size=chunk_size,
    )
    results["NES / Undefended + Domain-Constrained"] = run_condition(
        f"NES vs Undefended + domain constraints (budget={nes_cfg['max_queries']})",
        base_model, X_sample, nes_attack, nes_cfg, projector=projector, seed=seed, chunk_size=chunk_size,
    )
    results["HSJ / Undefended"] = run_condition(
        f"HSJ vs Undefended (budget={hsj_cfg['max_queries']})", base_model, X_sample, hopskipjump_attack, hsj_cfg, seed=seed, chunk_size=chunk_size
    )
    results["HSJ / TASCP (Naive)"] = run_condition(
        f"HSJ vs TASCP (Naive) (budget={hsj_cfg['max_queries']})", make_tascp(base_model), X_sample, hopskipjump_attack, hsj_cfg, seed=seed, chunk_size=chunk_size
    )

    results["nes_budget_sweep"] = run_nes_budget_sweep(
        base_model, base_model, X_sample, nes_cfg, budget_sweep, seed=seed, chunk_size=chunk_size
    )

    log_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp9_blackbox_cicids2017")
    save_results(results, log_dir, "exp9_blackbox_asr_cicids2017.csv")
    print_summary(results, f"EXP 9 (cicids2017)")


if __name__ == "__main__":
    run_exp9_cicids2017()
