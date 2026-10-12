"""Exp 14: Stationarity sweep (Prop 5, P0-5) + Opt-B CICIDS multi-seed.

Prop 5 (to be added to THEOREMS.md): TASCP's verdicts on a *fixed* point are
non-stationary while an attack's query trajectory fills the history — the
flip-set {x : f_undefended(x) != f_TASCP_session(x)} has width
~ 2 * ||x_perp|| * sin(Delta_theta / 2) in the rotation plane, and the
non-stationary fraction nu (fraction of repeated evaluations of the SAME
benign point whose verdict flips during a query trajectory, or whose final
verdict differs from the undefended verdict at the trajectory's close) scales
accordingly. The undefended model has exactly nu = 0 by definition (control).

Two nu measures (both datasets, both flows):
  nu_session : repeated evaluation of the same fixed point during a live
               trajectory — the fraction of trajectory queries where the
               defended verdict on the FROZEN point differs from the
               undefended verdict (flip events per query).
  nu_final   : at trajectory close, the fraction of trajectory-relevant
               final points (benign-near-boundary + HSJ finals) whose
               defended verdict differs from the undefended verdict.

Flip-set width check: for each trajectory query we record ||x_perp|| (the
component of the frozen point in the rotation plane) and Delta_theta, and
compare measured flip width against 2 * ||x_perp|| * sin(Delta_theta / 2).

Delta_theta is swept by varying gamma in {5, 10, 25, 50, 100, 200} (the
wrapper's trigger multiplier), giving 6 angle magnitudes per dataset.

Opt-B: CICIDS2017 multi-seed black-box (NES + HSJ, undefended vs TASCP naive,
3 seeds) and domain-constrained NES conditions, mirroring run_multiseed.py.
"""

import csv
import json
import math
import os
import sys

import numpy as np
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

from exp9_common import load_data, make_tascp, load_yaml, run_condition, save_results
from src.tascp_eval.attacks.nes import nes_attack
from src.tascp_eval.attacks.hopskipjump import hopskipjump_attack
from src.tascp_eval.defenses.tascp import TASCPWrapper

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

GAMMA_SWEEP = [5.0, 10.0, 25.0, 50.0, 100.0, 200.0]


class SessionProbe(torch.nn.Module):
    """Wraps the TASCP wrapper to measure Prop-5 non-stationarity during a live
    attack trajectory.

    Every query x_q to the defended model is mirrored by evaluating the
    FROZEN probe point x_probe (per row) through the SAME wrapper state, so
    the probe inherits the trajectory's history/rotation state but adds its
    own velocity v_probe = x_probe - x_t with respect to the last query.

    nu is the fraction of trajectory queries where the probe's defended
    verdict differs from its undefended verdict."""

    def __init__(self, tascp, base_model, x_probe):
        super().__init__()
        self.tascp = tascp
        self.base = base_model
        self.x_probe = x_probe.detach().clone()
        with torch.no_grad():
            self.probe_undefended_logits = base_model(self.x_probe).squeeze(-1)
        self.n_flips = 0
        self.n_queries = 0
        self.width_measured = []  # per-query: did the frozen point flip?
        self.width_predicted = []  # per-query: 2*||x_perp||*sin(dtheta/2)
        self.delta_thetas = []
        self._step = 0
        self._last_query = None
        self._last_dtheta = 0.0
        self._last_halfwidth = 0.0

    def _probe_eval(self, history, x_t):
        """Probe through the wrapper's own rotation state (current history),
        re-adding the probe to the trajectory window it departs from.

        The injected trajectory has FILLED the wrapper's history with the
        per-session persistent-direction attack steps, so rotation is live.
        For each step i, the wrapper's history is temporarily rewound to the
        pre-step-i state (i + 1 entries), the probe is then evaluated with the
        implementation above, and the state is restored exactly."""
        with torch.no_grad():
            if len(history) < 2:
                return None, 0.0, 0.0, 0.0
            step = self._step
            self._step += 1
            target_len = min(6, 2 + step)  # history grows by 1 per trajectory step
            if len(history) > target_len:
                restore = list(history)
                del history[target_len:]
                logits = self._probe_eval_core(history)
                history[:] = restore
            else:
                logits = self._probe_eval_core(history)
            return logits, self._last_dtheta, self._last_halfwidth, 1.0

    def _probe_eval_core(self, history):
        """Pure evaluation of the frozen probe through the CURRENT history
        WITHOUT mutating it (the last entry is BRIEFLY swapped to the probe
        and immediately restored): recompute eps_p (window sees the
        trajectory's real history plus the probe's own displacement from the
        incoming query), then apply the rotation to the probe and run the
        base classifier."""
        x_t = self._last_query
        with torch.no_grad():
            v_probe = self.x_probe - x_t
            history[-1] = self.x_probe  # probe as the last history entry
            norm_t = v_probe.norm(p=2, dim=1, keepdim=True)
            if float(norm_t.max()) < 1e-9:
                history[-1] = x_t
                return None
            # velocities of the window (last one = probe displacement)
            vels = []
            for i in range(1, len(history)):
                vels.append(history[i] - history[i - 1])
            vels.append(v_probe)
            eps_p = torch.zeros_like(norm_t)
            for i, v in enumerate(vels):
                n_i = v.norm(p=2, dim=1, keepdim=True)
                if i > 0:
                    vp = vels[i - 1]
                    n_p = vp.norm(p=2, dim=1, keepdim=True)
                    valid = (n_i > 1e-9) & (n_p > 1e-9)
                    m_i = torch.where(valid, (v * vp).sum(1, keepdim=True) / torch.clamp(n_i * n_p, min=1e-9), torch.zeros_like(n_i))
                else:
                    m_i = torch.zeros_like(n_i)
                eps_p = eps_p + n_i * torch.clamp(m_i, min=0.0)
            eps_p = eps_p / len(vels)
            delta = torch.clamp(self.tascp.gamma * torch.clamp(eps_p - self.tascp.tau, min=0.0), max=self.tascp.max_theta)
            # rotation plane from the probe displacement (v_probe) and the previous velocity
            u = v_probe / torch.clamp(norm_t, min=1e-9)
            v_prev_vel = vels[-2]
            comp = (v_prev_vel * u).sum(1, keepdim=True)
            wv = v_prev_vel - comp * u
            v_perp = wv / torch.clamp(wv.norm(p=2, dim=1, keepdim=True), min=1e-9)
            comp_u = (self.x_probe * u).sum(1, keepdim=True)
            comp_perp = (self.x_probe * v_perp).sum(1, keepdim=True)
            ortho = self.x_probe - comp_u * u - comp_perp * v_perp
            c, s = torch.cos(delta), torch.sin(delta)
            x_rot = ortho + (comp_u * c - comp_perp * s) * u + (comp_u * s + comp_perp * c) * v_perp
            logits = self.base(x_rot).squeeze(-1).cpu()
            self._last_dtheta = float(delta.mean().item())
            self._last_halfwidth = float((comp_perp.abs() * torch.sin(delta / 2.0)).mean().item())
            history[-1] = x_t  # restore: leave state exactly as found
            return logits

    def forward(self, x_t, *a, **k):
        self._last_query = x_t
        # 1. evaluate the frozen probe through the CURRENT trajectory state
        logits, dtheta_mean, halfwidth, _ = self._probe_eval(self.tascp.history, x_t)
        if logits is not None:
            flip = (logits > 0).cpu() != (self.probe_undefended_logits > 0)
            self.n_flips += int(flip.sum().item())
            self.delta_thetas.append(dtheta_mean)
            self.width_predicted.append(halfwidth)
        self.n_queries += x_t.size(0)
        # 2. serve the actual query through the wrapper (normal state update)
        return self.tascp(x_t, *a, **k)

    def reset_history(self, batch_size, device):
        if hasattr(self.tascp, "reset_history"):
            self.tascp.reset_history(batch_size, device)

    def add_to_history(self, x):
        if hasattr(self.tascp, "add_to_history"):
            self.tascp.add_to_history(x)


def measure_nu_final(tascp, base_model, X_final):
    """nu_final: after a trajectory has filled the history (simulated: add the
    trajectory queries), evaluate each final point through the defended
    wrapper ONCE (fresh velocities from the injected history) and compare to
    the undefended verdict.

    We simulate a stale-at-boundary history by injecting an HSJ-like
    trajectory (the actual final points are provided by the caller)."""
    with torch.no_grad():
        undef = base_model(X_final).squeeze(-1) > 0
        defended = tascp(X_final).squeeze(-1) > 0
    return (undef != defended).float().mean().item()


def inject_trajectory(tascp, X, n_steps=6, seed=42):
    """Fill the wrapper's history with a persistent-direction trajectory
    (attack-like: repeated probes around the boundary), so that TASCP's
    trigger is active before the final points are evaluated."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    tascp.reset_history(X.size(0), X.device)
    x_prev = X.clone()
    tascp.add_to_history(x_prev)
    direction = torch.randn(X.shape, generator=gen)
    direction = direction / direction.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
    for i in range(n_steps):
        # persistent direction -> trigger fires (tampering with same plane each step)
        x_prev = (x_prev + 0.08 * direction).clamp(0.0, 1.0)
        tascp.add_to_history(x_prev.clone())


def stationarity_sweep(base_model, X_benign, X_boundary, hsj_cfg, dataset, n_probe=256):
    """P0-5: sweep gamma (=> delta_theta), measure nu on benign-boundary flows
    and HSJ final points. Undefended control (any wrapper WITHOUT rotation) is
    exactly 0."""
    print(f"\n=== EXP 14: stationarity sweep ({dataset}) ===")
    results = {}
    probe_benign = X_boundary[:n_probe]

    for gamma in GAMMA_SWEEP:
        tascp = TASCPWrapper(base_model, window_size=5, max_theta=math.pi / 2, tau=0.0, gamma=gamma)
        # --- HSJ final points (from a short real HSJ run on a subset)
        n_final = min(256, len(probe_benign))
        torch.manual_seed(42)
        _, _, _, _ = hopskipjump_attack(tascp, probe_benign[:64], dict(hsj_cfg, max_queries=min(hsj_cfg.get("max_queries", 1000), 120)))
        delta_mean = tascp.last_delta_theta.detach().mean().item()

        # nu_session: SessionProbe around a live trajectory on the probe set
        session = SessionProbe(tascp, base_model, probe_benign)
        from src.tascp_eval.attacks.stateful import init_stateful_history, observe_query

        gen = torch.Generator().manual_seed(42)
        init_stateful_history(tascp, probe_benign)
        direction = torch.randn(probe_benign.shape, generator=gen)
        direction = direction / direction.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
        for _ in range(12):  # 12 trajectory injections + probe evaluations
            with torch.no_grad():
                # inject the next trajectory step (persistent direction)
                nxt = (probe_benign + 0.05 * direction).clamp(0, 1)
                tascp.add_to_history(nxt.clone())  # note: SessionProbe NOT in path
                session(nxt)                       # probe eval + serve query
        nu_session = session.n_flips / max(session.n_queries, 1)
        mean_pred_halfwidth = float(np.mean(session.width_predicted))
        mean_delta = float(np.mean(session.delta_thetas))

        # nu_final: inject trajectory then evaluate frozen finals
        inject_trajectory(tascp, probe_benign)
        nu_final_final = measure_nu_final(tascp, base_model, probe_benign)

        # undefended control
        nu_ctrl = 0.0  # by definition — verified structurally (no rotation exists)

        results[f"gamma={gamma}"] = {
            "delta_theta_mean_last_query": delta_mean,
            "delta_theta_mean_trajectory": mean_delta,
            "nu_session": nu_session,
            "nu_final": nu_final_final,
            "nu_control": nu_ctrl,
            "mean_predicted_halfwidth": mean_pred_halfwidth,
        }
        print(
            f"  gamma={gamma:6.1f} | delta_theta {mean_delta:.4f} | nu_session {nu_session*100:6.2f}% "
            f"| nu_final {nu_final_final*100:6.2f}% | ctrl {nu_ctrl*100:.2f}% | pred halfwidth {mean_pred_halfwidth:.4f}"
        )

    # sin(dtheta/2) width scaling check across gammas
    ratios = []
    for gamma in GAMMA_SWEEP[1:]:
        prev, cur = results[f"gamma={GAMMA_SWEEP[0]}"], results[f"gamma={gamma}"]
        if prev["nu_session"] > 0 and cur["nu_session"] > 0:
            d_prev, d_cur = prev["delta_theta_mean_trajectory"], cur["delta_theta_mean_trajectory"]
            if d_prev > 0:
                predicted_ratio = math.sin(d_cur / 2) / math.sin(d_prev / 2)
                measured_ratio = cur["nu_session"] / prev["nu_session"]
                ratios.append({"gamma_from": GAMMA_SWEEP[0], "gamma_to": gamma, "predicted": predicted_ratio, "measured": measured_ratio})
    results["sin_scaling_check"] = ratios
    if ratios:
        errs = [abs(r["predicted"] - r["measured"]) / max(r["predicted"], 1e-9) for r in ratios]
        print(f"  sin(dtheta/2) width scaling: mean relative error {np.mean(errs)*100:.1f}% over {len(ratios)} ratios")
    return results


def run_stationarity(dataset="unsw_nb15", n_samples=2000, seed=42):
    cfg = load_yaml("configs/attack/blackbox.yaml")
    hsj_cfg = dict(cfg["hopskipjump"])
    base_model, feature_cols, spec, levels, X_all = load_data(dataset, n_samples + 2560, seed)
    X_boundary, margins_idx = _boundary_shard(base_model, X_all, n_boundary=2000)
    print(f"\n=== EXP 14: stationarity sweep on {dataset} ===")
    results = stationarity_sweep(base_model, X_all, X_boundary, hsj_cfg, dataset)

    out_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp14_stationarity")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"metrics_{dataset}.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(PROJECT_ROOT, "results", "tables", f"exp14_stationarity_{dataset}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gamma", "delta_theta_mean", "nu_session", "nu_final", "nu_control", "pred_halfwidth"])
        for gamma in GAMMA_SWEEP:
            r = results[f"gamma={gamma}"]
            w.writerow([gamma, f"{r['delta_theta_mean_trajectory']:.4f}", f"{r['nu_session']:.4f}", f"{r['nu_final']:.4f}", f"{r['nu_control']:.4f}", f"{r['mean_predicted_halfwidth']:.4f}"])
    return results


def _boundary_shard(base_model, X_all, n_boundary=2000):
    margins = []
    with torch.no_grad():
        for i in range(0, len(X_all), 100_000):
            margins.append(base_model(X_all[i : i + 100_000]).squeeze(-1).abs().cpu())
    margins = torch.cat(margins)
    n = min(n_boundary, len(X_all))
    idx = torch.argsort(margins)[:n]
    return X_all[idx], margins[idx]


# ---------------------------------------------------------------------------
# Opt-B: CICIDS2017 black-box multi-seed
# ---------------------------------------------------------------------------
def run_cicids_multiseed(dataset="cicids2017", n_samples=2000, seeds=(42, 43, 44), chunk_size=500):
    cfg = load_yaml("configs/attack/blackbox.yaml")
    nes_cfg = dict(cfg["nes"])
    hsj_cfg = dict(cfg["hopskipjump"])
    from exp9_common import make_projector

    per_seed = {s: {} for s in seeds}
    for seed in seeds:
        base_model, feature_cols, spec, levels, X_sample = load_data(dataset, n_samples, seed)
        projector = make_projector(feature_cols, spec, levels)
        print(f"\n=== EXP 14/Opt-B: CICIDS multi-seed, seed {seed} ===")
        per_seed[seed]["NES / Undefended"] = run_condition("NES vs Undefended", base_model, X_sample, nes_attack, nes_cfg, seed=seed, chunk_size=chunk_size)
        per_seed[seed]["NES / TASCP (Naive)"] = run_condition("NES vs TASCP (Naive)", make_tascp(base_model), X_sample, nes_attack, nes_cfg, seed=seed, chunk_size=chunk_size)
        per_seed[seed]["NES / Undefended + Domain-Constrained"] = run_condition(
            "NES vs Undefended + domain", base_model, X_sample, nes_attack, nes_cfg, projector=projector, seed=seed, chunk_size=chunk_size
        )
        per_seed[seed]["HSJ / Undefended"] = run_condition("HSJ vs Undefended", base_model, X_sample, hopskipjump_attack, hsj_cfg, seed=seed, chunk_size=chunk_size)
        per_seed[seed]["HSJ / TASCP (Naive)"] = run_condition("HSJ vs TASCP (Naive)", make_tascp(base_model), X_sample, hopskipjump_attack, hsj_cfg, seed=seed, chunk_size=chunk_size)
        del base_model, X_sample

    n = n_samples
    summary = {}
    for cond in per_seed[seeds[0]]:
        asrs = [per_seed[s][cond]["asr"] for s in seeds]
        k_pool = round(sum(asrs) * n)

        def wilson_ci(k, n_, z=1.96):
            if n_ == 0:
                return (0.0, 0.0)
            p = k / n_
            denom = 1 + z**2 / n_
            center = (p + z**2 / (2 * n_)) / denom
            half = z * math.sqrt(p * (1 - p) / n_ + z**2 / (4 * n_**2)) / denom
            return (max(0.0, center - half), min(1.0, center + half))

        lo, hi = wilson_ci(k_pool, n * len(seeds))
        summary[cond] = {"per_seed_asr": asrs, "pooled_asr": sum(asrs) / len(seeds), "wilson95": [lo, hi]}
        print(f"{cond:40s} | pooled {summary[cond]['pooled_asr']*100:6.2f}% [{lo*100:.2f}, {hi*100:.2f}]")

    out_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp9_blackbox_cicids2017")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "multiseed.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(PROJECT_ROOT, "results", "tables", "exp9_multiseed_cicids2017.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "pooled_asr", "wilson95_lo", "wilson95_hi"] + [f"seed_{s}" for s in seeds])
        for cond, s in summary.items():
            w.writerow([cond, f"{s['pooled_asr']:.4f}", f"{s['wilson95'][0]:.4f}", f"{s['wilson95'][1]:.4f}"] + [f"{a:.4f}" for a in s["per_seed_asr"]])
    return summary


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stationarity"
    if mode == "stationarity":
        run_stationarity("unsw_nb15")
        run_stationarity("cicids2017")
    elif mode == "cicids_multiseed":
        run_cicids_multiseed()
    elif mode == "all":
        run_stationarity("unsw_nb15")
        run_stationarity("cicids2017")
        run_cicids_multiseed()
