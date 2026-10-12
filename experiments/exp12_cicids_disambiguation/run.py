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

from exp9_common import load_data, make_tascp, load_yaml
from src.tascp_eval.attacks.hopskipjump import hopskipjump_attack
from src.tascp_eval.attacks.stateful import init_stateful_history, observe_query

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

TRIGGER_THRESH = 1e-5


# ---------------------------------------------------------------------------
# Placebo / detached HSJ variants for the amplification disambiguation (P1-6)
# ---------------------------------------------------------------------------
class PlaceboRotation(torch.nn.Module):
    """Random-rotation-same-angle placebo: applies R(delta_theta) in a RANDOM
    plane (u ~ uniform on the sphere, v_perp ~ uniform orthogonal), instead of
    the data-dependent Givens plane spanned by the query trajectory. Same
    marginal rotation magnitude, no trajectory geometry. If this placebo
    reproduces TASCP's HSJ amplification, the effect is generic adversarial
    displacement; if not, the effect is specific to the trajectory geometry."""

    def __init__(self, base_model, window_size=5, max_theta=math.pi / 2, tau=0.0, gamma=50.0, seed=42):
        super().__init__()
        self.base_model = base_model
        self.window_size = window_size
        self.max_theta = max_theta
        self.tau = tau
        self.gamma = gamma
        self.history = []
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)
        self.last_delta_theta = torch.zeros(1)

    def reset_history(self, batch_size, device):
        self.history = []

    def add_to_history(self, x):
        self.history.append(x.detach().clone())
        if len(self.history) > self.window_size + 1:
            self.history.pop(0)

    def forward(self, x_t):
        if len(self.history) < 2:
            self.last_delta_theta = torch.zeros(x_t.size(0), 1)
            return self.base_model(x_t)

        v_t = x_t - self.history[-1]
        v_prev = self.history[-1] - self.history[-2]
        norm_t = v_t.norm(p=2, dim=1, keepdim=True)
        norm_prev = v_prev.norm(p=2, dim=1, keepdim=True)
        valid = (norm_t > 1e-9) & (norm_prev > 1e-9)
        m_t = torch.where(
            valid,
            (v_t * v_prev).sum(1, keepdim=True) / torch.clamp(norm_t * norm_prev, min=1e-9),
            torch.zeros_like(norm_t),
        )

        eps_p = torch.zeros_like(norm_t)
        eps_p = eps_p + norm_prev  # conservative: m_prev clamped at 0 contribution
        eps_p = eps_p + norm_t * torch.clamp(m_t, min=0.0)
        eps_p = eps_p / 2.0

        delta_theta = self.gamma * torch.clamp(eps_p - self.tau, min=0.0)
        delta_theta = torch.clamp(delta_theta, max=self.max_theta).detach()
        self.last_delta_theta = delta_theta

        # RANDOM plane, same angle
        b, d = x_t.shape
        u = torch.randn(b, d, generator=self.gen)
        u = u / u.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
        w = torch.randn(b, d, generator=self.gen)
        w = w - (w * u).sum(1, keepdim=True) * u
        v_perp = w / w.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)

        comp_u = (x_t * u).sum(1, keepdim=True)
        comp_perp = (x_t * v_perp).sum(1, keepdim=True)
        ortho = x_t - comp_u * u - comp_perp * v_perp
        c, s = torch.cos(delta_theta), torch.sin(delta_theta)
        x_rot = ortho + (comp_u * c - comp_perp * s) * u + (comp_u * s + comp_perp * c) * v_perp
        return self.base_model(x_rot)


class ZeroThetaWrapper(torch.nn.Module):
    """Delta-theta forced to 0: reproduces TASCP's history plumbing but never
    rotates. Should be behaviorally identical to the undefended model (control)."""

    def __init__(self, base_model, window_size=5):
        super().__init__()
        self.base_model = base_model
        self.window_size = window_size
        self.history = []
        self.last_delta_theta = torch.zeros(1)

    def reset_history(self, batch_size, device):
        self.history = []

    def add_to_history(self, x):
        self.history.append(x.detach().clone())
        if len(self.history) > self.window_size + 1:
            self.history.pop(0)

    def forward(self, x_t):
        self.last_delta_theta = torch.zeros(x_t.size(0), 1)
        return self.base_model(x_t)


def make_angle_detached(base_model):
    """TASCP with the rotation angle's gradient path severed (delta_theta
    detached) but the basis still data-dependent: isolates the T_angle term
    of the Theorem-1 decomposition."""
    w = make_tascp(base_model)
    w.adaptive = True
    w.ablation_mode = "gradient_magnitude"  # detaches delta_theta in forward()
    return w


def make_basis_fixed(base_model):
    """TASCP with the data-dependent basis replaced by fixed e_0/e_1 vectors:
    isolates the T_basis term."""
    w = make_tascp(base_model)
    w.adaptive = True
    w.ablation_mode = "basis_coupling"
    return w


def run_condition_rot_stats(name, model, X_sample, attack_fn, attack_cfg, seed=42, chunk_size=500):
    """exp9_common.run_condition + fraction of ALL queries that fired the trigger.
    Trigger-fire fraction is measured by wrapping forward to observe delta_theta
    on every query, not just the final one."""
    from exp9_common import run_condition

    fire_counts = {"queries": 0, "fires": 0}
    orig_forward = model.forward

    def counting_forward(x_t, *a, **k):
        out = orig_forward(x_t, *a, **k)
        if hasattr(model, "last_delta_theta"):
            fire_counts["queries"] += x_t.size(0)
            fire_counts["fires"] += int((model.last_delta_theta.detach().squeeze() > TRIGGER_THRESH).sum().item())
        return out

    model.forward = counting_forward
    try:
        result = run_condition(name, model, X_sample, attack_fn, attack_cfg, seed=seed, chunk_size=chunk_size)
    finally:
        model.forward = orig_forward
    result["trigger_fire_frac_all_queries"] = fire_counts["fires"] / max(fire_counts["queries"], 1)
    print(f"  trigger fired on {result['trigger_fire_frac_all_queries']*100:.2f}% of all queries")
    return result


def run_exp12(dataset="cicids2017", n_samples=2000, seed=42, chunk_size=500):
    """P1-6: HSJ amplification disambiguation on CICIDS2017 (where TASCP
    amplifies HSJ 15.05% -> 87.40%). Four variants:
      - TASCP Naive (reference amplification)
      - Basis fixed (ablation_mode='basis_coupling')
      - Angle detached (delta_theta detached -> 'gradient_magnitude' mode)
      - Random-rotation-same-angle placebo
      - Delta-theta=0 control
    """
    cfg = load_yaml("configs/attack/blackbox.yaml")
    hsj_cfg = dict(cfg["hopskipjump"])

    base_model, feature_cols, spec, levels, X_sample = load_data(dataset, n_samples, seed)
    print(f"\n=== EXP 12: HSJ amplification disambiguation ({dataset}, {len(X_sample)} samples) ===")

    conditions = [
        ("HSJ / Undefended (control)", base_model),
        ("HSJ / Delta-theta=0 (control)", ZeroThetaWrapper(base_model, window_size=5)),
        ("HSJ / TASCP (Naive)", make_tascp(base_model)),
        ("HSJ / Basis fixed", make_basis_fixed(base_model)),
        ("HSJ / Angle detached", make_angle_detached(base_model)),
        ("HSJ / Random-rotation placebo", PlaceboRotation(base_model, window_size=5, max_theta=math.pi / 2, tau=0.0, gamma=50.0, seed=seed)),
    ]
    results = {}
    for name, model in conditions:
        results[name] = run_condition_rot_stats(name, model, X_sample, hopskipjump_attack, hsj_cfg, seed=seed, chunk_size=chunk_size)

    log_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp12_cicids_disambiguation")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(os.path.join(PROJECT_ROOT, "results", "tables", "exp12_hsj_disambiguation_cicids2017.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "asr", "mean_queries_to_success", "trigger_fire_frac"])
        for name, r in results.items():
            q = r["mean_queries_to_success"]
            w.writerow([name, f"{r['asr']:.4f}", f"{q:.2f}" if q == q else "nan", f"{r['trigger_fire_frac_all_queries']:.4f}"])

    print("\n=== EXP 12 SUMMARY (HSJ on cicids2017) ===")
    for name, r in results.items():
        print(f"{name:38s} | ASR {r['asr']*100:6.2f}% | fire {r['trigger_fire_frac_all_queries']*100:5.1f}%")
    return results


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "cicids2017"
    run_exp12(dataset=ds)
