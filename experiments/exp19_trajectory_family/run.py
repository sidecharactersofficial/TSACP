"""RQ1 + RQ4: is the exploitable trajectory coupling specific to TASCP?

Runs the full transformation family (tascp / tscale / tshift / tshear /
tnoise — see family.py) through the same three-condition protocol used for
TASCP throughout the paper:

  naive attack     straight-through autograd through the deployed wrapper
  adaptive attack  B2: gradients through a *local replica* of the transform
  fixed-state      control: S clamped to 0 (no trajectory coupling)

plus the coupling audit C = ||dT/dh||_F and the stationarity replay
(nu_session / nu_final) for every variant.

CLI: python -m experiments.exp19_trajectory_family.run [dataset]
Writes results/logs/exp19_trajectory_family/metrics_<ds>.json,
results/tables/exp19_family_<ds>.csv and exp19_stationarity_<ds>.csv.
"""

import csv
import json
import math
import os
import sys

import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, EXP_DIR)

from experiments.exp9_blackbox_attack.exp9_common import load_data, load_yaml  # noqa: E402
from src.tascp_eval.attacks.bpda_adaptive import stateful_pgd_attack  # noqa: E402
from src.tascp_eval.attacks.stateful import init_stateful_history, observe_query  # noqa: E402
from family import (FAMILY_VARIANTS, FamilyWrapper, STATE_FN_NAMES,  # noqa: E402
                    stationarity_replay)

PGD_CFG = {"epsilon": 0.3, "alpha": 0.01, "iterations": 20, "norm": "l2"}


class _ReplicaAttack:
    """Threat-model-B2 adaptive attack: the attacker holds a *local replica*
    of the family transform (same variant, same hyperparameters) and takes
    gradients through the replica while the deployed wrapper only sees the
    queries. The replica's history is stepped in lockstep with the deployed
    wrapper's."""

    def __init__(self, base_model, variant, state_fn="momentum"):
        self.replica = FamilyWrapper(base_model, variant=variant, tau=0.0, gamma=50.0,
                                     state_fn=state_fn)
        self.replica.adaptive = True

    def step(self, x_query):
        # keep the replica's history in lockstep with the deployed session
        self.replica.add_to_history(x_query)

    def loss_and_grad(self, x_adv, y):
        import torch.nn as nn

        self.replica.adaptive = True
        x = x_adv.clone().detach().requires_grad_(True)
        out = self.replica(x)
        loss = nn.BCEWithLogitsLoss()(out, y)
        g = torch.autograd.grad(loss, x)[0]
        return g


def adaptive_family_attack(deployed, replica_atk, x, y, config, seed=42):
    """PGD where gradients come from the local replica, queries hit the
    deployed defense. Mirrors exp5's protocol otherwise."""
    epsilon = config.get("epsilon", 0.3)
    alpha_0 = config.get("alpha", 0.05)
    iterations = config.get("iterations", 20)
    device = x.device
    batch = x.size(0)
    torch.manual_seed(seed)

    x_adv = x.clone().detach()
    success = torch.zeros(batch, dtype=torch.bool, device=device)
    q_to_success = torch.full((batch,), float(iterations), device=device)

    init_stateful_history(deployed, x_adv)
    init_stateful_history(replica_atk.replica, x_adv)

    for i in range(iterations):
        g = replica_atk.loss_and_grad(x_adv, y)
        with torch.no_grad():
            gn = torch.clamp(torch.norm(g, p=2, dim=1, keepdim=True), min=1e-8)
            x_adv = x_adv + alpha_0 * (g / gn)
            eta = x_adv - x
            en = torch.clamp(torch.norm(eta, p=2, dim=1, keepdim=True), min=1e-8)
            x_adv = x + eta * torch.min(torch.ones_like(en), epsilon / en)
            x_adv = x_adv.clamp(0.0, 1.0)

            out = deployed(x_adv)
            observe_query(deployed, x_adv)
            replica_atk.step(x_adv)
            preds = (out > 0).squeeze(-1)
            newly = preds & ~success
            if newly.any():
                q_to_success[newly] = i + 1
                success |= newly
    with torch.no_grad():
        out = deployed(x_adv)
        observe_query(deployed, x_adv)
        replica_atk.step(x_adv)
        preds = (out > 0).squeeze(-1)
        newly = preds & ~success
        q_to_success[newly] = iterations
        success |= newly
    return x_adv.detach(), success, q_to_success


def naive_family_attack(wrapper, x, y, config, seed=42):
    """Straight-through autograd through the deployed wrapper (exp5 protocol)."""
    wrapper.adaptive = True
    return stateful_pgd_attack(wrapper, x, y, config, adaptive=True)


def eval_variants(base_model, X, y, cfg, variants=FAMILY_VARIANTS):
    device = X.device
    results = {}
    for variant in variants:
        print(f"\n--- family variant: {variant} ---")

        def fresh(freeze=None, state_fn="momentum"):
            w = FamilyWrapper(base_model, variant=variant, tau=0.0, gamma=50.0,
                              state_fn=state_fn)
            w.adaptive = False
            w.freeze_state = freeze
            return w

        # naive: straight-through through the deployed wrapper
        w = fresh()
        _, s_n, q_n = naive_family_attack(w, X, y, cfg)
        naive_asr = float(s_n.float().mean().item())

        # fixed-state control: S clamped to 0 -> no trajectory coupling
        w_ctl = fresh(freeze=0.0)
        _, s_c, q_c = naive_family_attack(w_ctl, X, y, cfg)
        control_asr = float(s_c.float().mean().item())

        # adaptive: gradients through a local replica (B2)
        w = fresh()
        rep = _ReplicaAttack(base_model, variant)
        _, s_a, q_a = adaptive_family_attack(w, rep, X, y, cfg)
        adaptive_asr = float(s_a.float().mean().item())

        # coupling audit on an ACTIVE trajectory: a persistent-direction
        # walk (positive window momentum) is the regime where the state
        # channel is open; on a random walk the trigger is silent and C = 0
        # by construction (delta_theta = 0 -> identity transform).
        w = fresh()
        init_stateful_history(w, X)
        cursor = X.clone()
        u_dir = torch.nn.functional.normalize(torch.rand_like(X), dim=1)
        for _ in range(3):
            cursor = (cursor + 0.15 * u_dir).clamp(0.0, 1.0)
            w.add_to_history(cursor)
        C = w.coupling_norm(X)
        C_mean = float(C.mean().item())
        C_op_active = float((w.last_S > w.tau).float().mean().item())  # firing frac

        # stationarity replay
        w = fresh()
        stat = stationarity_replay(w, X, n_steps=12, step_scale=0.3)

        results[variant] = {
            "naive_asr": naive_asr,
            "adaptive_asr": adaptive_asr,
            "fixed_state_asr": control_asr,
            "adaptive_delta": adaptive_asr - control_asr,
            "naive_delta": naive_asr - control_asr,
            "coupling_C": C_mean,
            "coupling_firing_frac": C_op_active,
            "q_to_success_adaptive": float(q_a.mean().item()),
            "q_to_success_naive": float(q_n.mean().item()),
            "nu_session": stat["nu_session"],
            "nu_final": stat["nu_final"],
            "mean_state": stat["mean_state"],
        }
        r = results[variant]
        print(
            f"  naive {r['naive_asr']*100:6.2f}% | adaptive {r['adaptive_asr']*100:6.2f}% | "
            f"fixed-state {r['fixed_state_asr']*100:6.2f}% | C {r['coupling_C']:.4f} | "
            f"nu_session {r['nu_session']*100:5.2f}% | nu_final {r['nu_final']*100:5.2f}%"
        )
    return results


def main(dataset="unsw_nb15", n_samples=500, seed=42):
    torch.manual_seed(seed)
    base_model, feature_cols, spec, levels, X = load_data(dataset, n_samples, seed)
    y = torch.zeros(X.size(0), 1)
    results = eval_variants(base_model, X, y, PGD_CFG)

    out_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp19_trajectory_family")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"metrics_{dataset}.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(PROJECT_ROOT, "results", "tables", f"exp19_family_{dataset}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "naive_asr", "adaptive_asr", "fixed_state_asr",
                    "adaptive_delta", "coupling_C", "nu_session", "nu_final"])
        for v, r in results.items():
            w.writerow([v, f"{r['naive_asr']:.4f}", f"{r['adaptive_asr']:.4f}",
                        f"{r['fixed_state_asr']:.4f}", f"{r['adaptive_delta']:.4f}",
                        f"{r['coupling_C']:.4f}", f"{r['nu_session']:.4f}", f"{r['nu_final']:.4f}"])

    print("\n=== EXP 19 SUMMARY ===")
    n = len(results)
    coupled = sum(1 for r in results.values() if r["adaptive_delta"] > 0.02)
    stationary_bad = sum(1 for r in results.values() if r["nu_session"] > 0.01)
    print(f"variants with adaptive_delta > 2pp over fixed-state: {coupled}/{n}")
    print(f"variants with nu_session > 1%: {stationary_bad}/{n}")
    for v, r in results.items():
        print(f"  {v:8s} | dASR {r['adaptive_delta']*100:+6.2f}pp | C {r['coupling_C']:.3f} | nu {r['nu_session']*100:5.2f}%")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "unsw_nb15"
    main(ds)
