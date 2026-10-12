"""Exp 15: P1-9 offline attacker + P1-8 Corollary-3 trigger scope.

Part A (offline attacker):
  - Optimize a perturbation against the UNDEFENDED replica with white-box PGD,
    submit ONE flow as a fresh session.
  - Assert TASCP's output is bit-identical to DACM's (and to the undefended
    model's), since delta_theta = 0 on the first query of a session.
  - Surrogate-mismatch variant: optimize against a different replica (a
    separately initialized MLP surrogate), submit to the deployed MLP.

Part B (Corollary 3 scope):
  - Run the non-antithetic score/decision-based attacks (SPSA, independent-
    Gaussian NES, ZOO, SimBA, Square, Boundary) against naive TASCP and log
    the trigger-activation rate (delta_theta > 1e-5) over query streams,
    alongside NES (antithetic, expected 0.00%).
"""

import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
for _p in (
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "experiments"),
    os.path.join(PROJECT_ROOT, "experiments", "exp9_blackbox_attack"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp9_common import load_data, make_tascp, load_yaml, run_condition
from src.tascp_eval.attacks.offline_attacker import (
    offline_attack_bit_identical,
    offline_attack_surrogate_mismatch,
)
from src.tascp_eval.attacks.pgd import load_config
from src.tascp_eval.attacks.stateful import init_stateful_history, observe_query
from src.tascp_eval.attacks.trigger_scope import TRIGGER_SCOPE_ATTACKS
from src.tascp_eval.attacks.nes import nes_attack

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

TRIGGER_THRESH = 1e-5


# ---------------------------------------------------------------------------
# Part A: offline attacker
# ---------------------------------------------------------------------------
def trigger_fire_stats(model, attack_fn, X, cfg, seed=42, projector=None):
    """Run an attack while counting trigger firings over ALL queries."""
    fires = {"q": 0, "f": 0}
    orig_fwd = model.forward

    def fwd(x_t, *a, **k):
        out = orig_fwd(x_t, *a, **k)
        if hasattr(model, "last_delta_theta"):
            fires["q"] += x_t.size(0)
            fires["f"] += int((model.last_delta_theta.detach().squeeze() > TRIGGER_THRESH).sum().item())
        return out

    model.forward = fwd
    try:
        torch.manual_seed(seed)
        result = run_condition("scope", model, X, attack_fn, cfg, seed=seed, projector=projector)
    finally:
        model.forward = orig_fwd
    result["trigger_fire_frac"] = fires["f"] / max(fires["q"], 1)
    return result


def run_offline_part(base_model, X_sample, config, dataset):
    print(f"\n=== EXP 15 part A: offline attacker ({dataset}) ===")
    results = {}
    torch.manual_seed(42)
    y = torch.zeros(X_sample.size(0), 1)
    from src.tascp_eval.attacks.bpda_adaptive import stateful_pgd_attack

    x_star, _, _ = stateful_pgd_attack(base_model, X_sample, y, config, adaptive=False)
    a = offline_attack_bit_identical(base_model, x_star)
    results["whitebox_offline"] = {
        "bit_identical_dacm": a["tascp_bit_identical_dacm"],
        "bit_identical_undefended": a["tascp_bit_identical_undefended"],
        "max_abs_diff_tascp_dacm": a["max_abs_diff_tascp_dacm"],
        "verdict_agreement": a["verdict_agreement"],
        "submitted_asr": float(((a["undefended_logits"] > 0).squeeze(-1)).float().mean().item()),
    }
    print(
        f"  white-box offline: bit-identical TASCP==DACM: {a['tascp_bit_identical_dacm']} "
        f"(max|diff|={a['max_abs_diff_tascp_dacm']:.2e}), verdict agreement {a['verdict_agreement']}"
    )

    # surrogate mismatch: independently initialized replica
    torch.manual_seed(123)
    surrogate = nn.Sequential(
        nn.Linear(X_sample.size(1), 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
    ).eval()
    b, _ = offline_attack_surrogate_mismatch(surrogate, base_model, X_sample, config, seed=42)
    results["surrogate_mismatch"] = b["surrogate_mismatch"]
    sm = results["surrogate_mismatch"]
    print(
        f"  surrogate mismatch: transfer ASR undefended {sm['transfer_success_undefended']*100:.2f}% "
        f"| DACM {sm['transfer_success_dacm']*100:.2f}% | TASCP fresh session {sm['transfer_success_tascp_fresh_session']*100:.2f}%"
    )
    return results


# ---------------------------------------------------------------------------
# Part B: Corollary 3 trigger scope
# ---------------------------------------------------------------------------
def run_scope_part(base_model, X_sample, nes_cfg, dataset):
    print(f"\n=== EXP 15 part B: Corollary-3 trigger scope ({dataset}) ===")
    scope_cfg = {
        "epsilon": nes_cfg.get("epsilon", 0.1),
        "sigma": nes_cfg.get("sigma", 0.05),
        "alpha": nes_cfg.get("alpha", 0.02),
        "q": nes_cfg.get("q", 10),
        "max_queries": min(nes_cfg.get("max_queries", 250), 120),
    }
    results = {}

    # NES antithetic reference condition
    tascp = make_tascp(base_model)
    r = trigger_fire_stats(tascp, nes_attack, X_sample, dict(nes_cfg, max_queries=scope_cfg["max_queries"]))
    results["NES (antithetic)"] = {"asr": r["asr"], "trigger_fire_frac": r["trigger_fire_frac"], "total_queries": r["total_queries"]}
    print(f"  {'NES (antithetic)':22s} | fire {r['trigger_fire_frac']*100:6.2f}% | ASR {r['asr']*100:6.2f}%")

    for name, fn in TRIGGER_SCOPE_ATTACKS.items():
        tascp = make_tascp(base_model)
        r = trigger_fire_stats(tascp, fn, X_sample, scope_cfg)
        results[name] = {"asr": r["asr"], "trigger_fire_frac": r["trigger_fire_frac"], "total_queries": r["total_queries"]}
        print(f"  {name:22s} | fire {r['trigger_fire_frac']*100:6.2f}% | ASR {r['asr']*100:6.2f}%")
    return results


def run(dataset="unsw_nb15", n_samples=2000, seed=42):
    cfg = load_yaml("configs/attack/blackbox.yaml")
    nes_cfg = dict(cfg["nes"])
    config = load_config()

    base_model, feature_cols, spec, levels, X_sample = load_data(dataset, n_samples, seed)
    print(f"\n=== EXP 15: offline attacker + Corollary-3 scope ({dataset}, {len(X_sample)} samples) ===")

    results = {
        "offline": run_offline_part(base_model, X_sample, config, dataset),
        "cor3_scope": run_scope_part(base_model, X_sample, nes_cfg, dataset),
    }

    out_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp15_offline_and_scope")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"metrics_{dataset}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(os.path.join(PROJECT_ROOT, "results", "tables", f"exp15_cor3_scope_{dataset}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attack", "trigger_fire_frac", "asr", "total_queries"])
        for name, r in results["cor3_scope"].items():
            w.writerow([name, f"{r['trigger_fire_frac']:.4f}", f"{r['asr']:.4f}", r["total_queries"]])

    with open(os.path.join(PROJECT_ROOT, "results", "tables", f"exp15_offline_{dataset}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "bit_identical_dacm", "bit_identical_undefended", "max_abs_diff", "submitted_asr"])
        a = results["offline"]["whitebox_offline"]
        w.writerow(["whitebox_offline", a["bit_identical_dacm"], a["bit_identical_undefended"], f"{a['max_abs_diff_tascp_dacm']:.2e}", f"{a['submitted_asr']:.4f}"])
        sm = results["offline"]["surrogate_mismatch"]
        w.writerow(["surrogate_mismatch", sm.get("bit_identical_dacm", ""), "", "", f"{sm['transfer_success_undefended']:.4f}"])

    print("\n=== EXP 15 SUMMARY ===")
    a = results["offline"]["whitebox_offline"]
    print(f"Offline attacker: TASCP==DACM bit-identical: {a['bit_identical_dacm']}; verdict agreement: {a['verdict_agreement']}")
    print("Corollary-3 scope (trigger fire rate per attack):")
    for name, r in results["cor3_scope"].items():
        print(f"  {name:22s} | fire {r['trigger_fire_frac']*100:6.2f}%")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "unsw_nb15"
    run(dataset=ds)
