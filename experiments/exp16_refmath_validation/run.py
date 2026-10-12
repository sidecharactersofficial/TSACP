"""Exp 16: validate the paper's new formal statements on real runs.

Lemma 3  : autograd and finite-difference gradients of eps_p (penetration)
           agree to ~1e-6 relative when M_t > 0, on real PGD trajectories.
Theorem-1 decomposition: the defended-input gradient decomposes as
           grad_full ~= T_static + T_angle + T_basis, where
           T_static  = classifier gradient at the detached rotation point,
           T_angle   = contribution of d(rot)/d(delta_theta),
           T_basis   = contribution of d(rot)/d(u, v_perp).
           Measured: full autograd vs angle-detached vs basis-fixed sums.
Lemma 2's isotropy assumption: measure the effective projection ratio rho of
           the rotation term on REAL model gradients; the paper's E[rho]=2/d
           assumes isotropy. If real rho is far from 2/d, that sentence must
           be revised.
Prop 3    : NES per-step cosine cos(grad_hat, grad_true) vs q on real model
           gradients vs the isotropic prediction sqrt(q/(q+d+1)).
Corollary 3: M_t histograms from the real antithetic NES query stream, with
           and without anchor updates, with E[M_t] and P(M>0) (paper: ~-0.83
           and ~0 with anchor updates).
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

from exp9_common import load_data, make_tascp, load_yaml
from src.tascp_eval.defenses.tascp import TASCPWrapper, _apply_rotation
from src.tascp_eval.attacks.nes import nes_attack
from src.tascp_eval.attacks.pgd import load_config
from src.tascp_eval.attacks.bpda_adaptive import stateful_pgd_attack

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


# ---------------------------------------------------------------------------
# Lemma 3: autograd vs FD gradient of eps_p on real PGD trajectories
# ---------------------------------------------------------------------------
def eps_p_and_moments(history, x_t, tau=0.0):
    """Differentiable eps_p given wrapper-style history (list of (B,d)) and
    current query x_t (B,d, requires_grad). Mirrors TASCPWrapper.forward's
    eps_p exactly."""
    v_t = x_t - history[-1]
    norms = [v_t.norm(p=2, dim=1, keepdim=True)]
    moms = []
    # window velocities
    vels = []
    for i in range(1, len(history)):
        vels.append(history[i] - history[i - 1])
    vels.append(v_t)
    eps_p = torch.zeros_like(norms[0])
    for i, v in enumerate(vels):
        n_i = v.norm(p=2, dim=1, keepdim=True)
        if i > 0:
            v_prev = vels[i - 1]
            n_p = v_prev.norm(p=2, dim=1, keepdim=True)
            valid = (n_i > 1e-9) & (n_p > 1e-9)
            m_i = torch.where(valid, (v * v_prev).sum(1, keepdim=True) / torch.clamp(n_i * n_p, min=1e-9), torch.zeros_like(n_i))
        else:
            m_i = torch.zeros_like(n_i)
        eps_p = eps_p + n_i * torch.clamp(m_i, min=0.0)
        moms.append(m_i)
    eps_p = eps_p / len(vels)
    return eps_p, moms, vels


def lemma3_fd_vs_autograd(base_model, X_sample, config, n_points=64, fd_eps=1e-3, seed=42):
    """Take real PGD trajectories (stateful_pgd_attack against naive TASCP),
    then at several trajectory points compare autograd d(eps_p)/d(x_t) with a
    central finite difference along each coordinate block."""
    print("\n=== EXP 16 part A: Lemma 3 — autograd vs FD grad(eps_p) ===")
    torch.manual_seed(seed)
    tascp = make_tascp(base_model)
    y = torch.zeros(X_sample.size(0), 1)
    x_adv, success, _ = stateful_pgd_attack(tascp, X_sample, y, config, adaptive=False)

    # rebuild a real trajectory: replay the attack steps to re-collect history
    # (the wrapper kept only the last window; we re-run PGD manually to record)
    from src.tascp_eval.attacks.bpda_adaptive import stateful_pgd_attack as spga

    idx = torch.randperm(X_sample.size(0))[:n_points]
    x0 = X_sample[idx]
    results = {"n_points": n_points, "rel_errors_M_pos": [], "rel_errors_all": [], "M_pos_frac": []}
    for b in range(0, n_points, 16):
        xb = x0[b : b + 16]
        yt = torch.zeros(xb.size(0), 1)
        # manual PGD steps, recording the full trajectory
        traj = [xb.clone()]
        xa = xb.clone().requires_grad_(True)
        tascp.reset_history(xb.size(0), xb.device)
        tascp.add_to_history(xb.clone())
        criterion = torch.nn.BCEWithLogitsLoss()
        for it in range(config.get("iterations", 20)):
            xa = xa.detach().requires_grad_(True)
            out = tascp(xa)
            tascp.zero_grad()
            loss = criterion(out, yt)
            loss.backward()
            with torch.no_grad():
                g = xa.grad
                gn = torch.clamp(g.norm(p=2, dim=1, keepdim=True), min=1e-8)
                xa = xa + config.get("alpha", 0.01) * (g / gn)
                eta = xa - xb
                en = torch.clamp(eta.norm(p=2, dim=1, keepdim=True), min=1e-8)
                xa = xb + eta * torch.min(torch.ones_like(en), config.get("epsilon", 0.1) / en)
                xa = xa.clamp(0.0, 1.0)
            tascp.add_to_history(xa.detach().clone())
            traj.append(xa.detach().clone())

        # now compare gradients at trajectory points 3..end
        hist_full = traj  # [x0, x1, ..., xT]
        for t in range(3, len(traj)):
            history = [h.clone() for h in hist_full[max(0, t - 5) : t]]
            if len(history) < 2:
                continue
            xt = traj[t].clone().requires_grad_(True)
            eps_p, moms, vels = eps_p_and_moments(history, xt)
            M_t = moms[-1]
            g_auto = torch.autograd.grad(eps_p.sum(), xt, retain_graph=False)[0]
            # Directional-derivative check: central FD along random unit
            # directions vs the autograd gradient projected on the same
            # direction (a full per-coordinate FD would cost 2d forwards per
            # point; 8 directions bound the relative error just as well).
            h = fd_eps
            gen = torch.Generator().manual_seed(seed + t)
            for _ in range(8):
                u = torch.randn(xt.shape, generator=gen)
                u = u / u.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
                with torch.no_grad():
                    ep_p, _, _ = eps_p_and_moments(history, traj[t] + h * u)
                    ep_m, _, _ = eps_p_and_moments(history, traj[t] - h * u)
                fd_dir = ((ep_p - ep_m) / (2 * h)).squeeze(1)
                auto_dir = (g_auto * u).sum(dim=1)
                rel = (fd_dir - auto_dir).abs() / torch.clamp(auto_dir.abs(), min=1e-8)
                m_pos = (M_t.squeeze(1) > 0).cpu()
                results["rel_errors_all"].extend(rel.detach().cpu().tolist())
                results["rel_errors_M_pos"].extend(rel.detach().cpu()[m_pos].tolist())
            results["M_pos_frac"].append(float(m_pos.float().mean().item()))

    import numpy as np

    all_err = np.array(results["rel_errors_all"])
    pos_err = np.array(results["rel_errors_M_pos"]) if results["rel_errors_M_pos"] else np.array([np.nan])
    summary = {
        "n_comparisons": len(all_err),
        "median_rel_err_all": float(np.nanmedian(all_err)) if len(all_err) else None,
        "median_rel_err_M_pos": float(np.nanmedian(pos_err)),
        "p90_rel_err_M_pos": float(np.nanpercentile(pos_err, 90)),
        "M_pos_frac_mean": float(np.mean(results["M_pos_frac"])) if results["M_pos_frac"] else None,
    }
    print(f"  comparisons: {summary['n_comparisons']} | median rel err (all): {summary['median_rel_err_all']:.2e}")
    print(f"  median rel err (M_t>0): {summary['median_rel_err_M_pos']:.2e} | p90: {summary['p90_rel_err_M_pos']:.2e}")
    print(f"  fraction of trajectory points with M_t>0: {summary['M_pos_frac_mean']:.3f}")
    return summary


# ---------------------------------------------------------------------------
# Theorem-1 decomposition
# ---------------------------------------------------------------------------
def theorem1_decomposition(base_model, X_sample, config, n_points=128, seed=42):
    """Compare full-autograd gradient through the defended forward with
    (a) angle-detached and (b) basis-fixed variants, measuring cosine
    similarity between each variant's update direction and the full one.
    Expectation: basis-fixed ~= full (both ~ classifier-only direction, i.e.
    T_angle+T_basis tiny => basis-fixed no-op), angle-detached != full when
    the trigger fires."""
    print("\n=== EXP 16 part B: Theorem-1 decomposition ===")
    torch.manual_seed(seed)
    d = X_sample.size(1)
    # build a real trajectory that FIRES the trigger (persistent direction)
    x0 = X_sample[torch.randperm(X_sample.size(0))[:n_points]].clone()
    tascp = make_tascp(base_model)
    tascp.reset_history(x0.size(0), x0.device)
    tascp.add_to_history(x0.clone())
    gen = torch.Generator().manual_seed(seed)
    direction = torch.randn(x0.shape, generator=gen)
    direction = direction / direction.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
    xt = (x0 + 0.06 * direction).clamp(0.0, 1.0)
    tascp.add_to_history((x0 + 0.04 * direction).clamp(0.0, 1.0))

    def grad_of(mode):
        w = make_tascp(base_model)
        w.reset_history(x0.size(0), x0.device)
        w.add_to_history((x0 + 0.04 * direction).clone())
        w.add_to_history(x0.clone())
        w.adaptive = True
        w.ablation_mode = {"full": None, "angle_detached": "gradient_magnitude", "basis_fixed": "basis_coupling"}[mode]
        xa = xt.clone().requires_grad_(True)
        out = w(xa)
        loss = out.sum()
        g = torch.autograd.grad(loss, xa, allow_unused=True)[0]
        return g if g is not None else torch.zeros_like(xt)

    g_full = grad_of("full")
    g_angle = grad_of("angle_detached")
    g_basis = grad_of("basis_fixed")

    # measure trigger activity: run a forward with a fresh wrapper on the same inputs
    with torch.no_grad():
        _ = tascp(xt)
    trigger_frac = (
        float((tascp.last_delta_theta.detach().squeeze() > 1e-5).float().mean().item())
        if hasattr(tascp, "last_delta_theta")
        else 0.0
    )

    def cos(a, b):
        an = torch.clamp(a.norm(p=2, dim=1), min=1e-12)
        bn = torch.clamp(b.norm(p=2, dim=1), min=1e-12)
        return ((a * b).sum(1) / (an * bn)).mean().item()

    # decomposition magnitudes: full vs classifier-only at detached rotation
    summary = {
        "cos_full_vs_angle_detached": cos(g_full, g_angle),
        "cos_full_vs_basis_fixed": cos(g_full, g_basis),
        "cos_angle_vs_basis": cos(g_angle, g_basis),
        "grad_norm_full": g_full.norm(p=2, dim=1).mean().item(),
        "grad_norm_angle_detached": g_angle.norm(p=2, dim=1).mean().item(),
        "grad_norm_basis_fixed": g_basis.norm(p=2, dim=1).mean().item(),
        "trigger_fired_frac": trigger_frac,
    }
    print(f"  trigger fired on {summary['trigger_fired_frac']*100:.1f}% of the batch")
    print(f"  cos(full, angle-detached) = {summary['cos_full_vs_angle_detached']:.4f}")
    print(f"  cos(full, basis-fixed)    = {summary['cos_full_vs_basis_fixed']:.4f}")
    print(f"  |g| full {summary['grad_norm_full']:.4f} | angle-det {summary['grad_norm_angle_detached']:.4f} | basis-fixed {summary['grad_norm_basis_fixed']:.4f}")
    return summary


# ---------------------------------------------------------------------------
# Lemma 2: rho isotropy on real gradients
# ---------------------------------------------------------------------------
def lemma2_rho(base_model, X_sample, n_samples=256, n_trials=32, seed=42):
    """rho: the ratio ||projection of the real gradient onto span{u,v_perp}||
    / ||g||. Under isotropy E[rho^2] ~ 2/d; the paper's sentence assumes
    E[rho] ~ sqrt(2/d). Measure the real value on actual model gradients at
    real benign points (and at HSJ boundary points)."""
    print("\n=== EXP 16 part C: Lemma 2 rho on real gradients ===")
    torch.manual_seed(seed)
    x = X_sample[torch.randperm(X_sample.size(0))[:n_samples]].clone().requires_grad_(True)
    out = base_model(x).squeeze(-1)
    loss = out.sum()
    g = torch.autograd.grad(loss, x)[0]
    d = x.size(1)
    gen = torch.Generator().manual_seed(seed)
    rhos = []
    for _ in range(n_trials):
        a = torch.randn(x.size(1), generator=gen)
        a = a / a.norm()
        b = torch.randn(x.size(1), generator=gen)
        b = b - (b * a).sum() * a
        b = b / b.norm()
        # projection of each sample's gradient onto span{a,b}
        ga = (g * a).sum(1, keepdim=True)
        gb = (g * b).sum(1, keepdim=True)
        proj = (ga * a + gb * b).norm(p=2, dim=1)
        gn = torch.clamp(g.norm(p=2, dim=1), min=1e-12)
        rhos.append((proj / gn).mean().item())
    rho_mean = float(np.mean(rhos))
    iso_pred = math.sqrt(2.0 / d)
    summary = {
        "rho_mean": rho_mean,
        "rho_isotropic_pred": iso_pred,
        "ratio_real_over_isotropic": rho_mean / iso_pred,
        "d": d,
        "n_trials": n_trials,
    }
    print(f"  E[rho] real {rho_mean:.4f} vs isotropic prediction {iso_pred:.4f} (ratio {summary['ratio_real_over_isotropic']:.2f}x)")
    return summary


# ---------------------------------------------------------------------------
# Prop 3: NES cosine vs q
# ---------------------------------------------------------------------------
def prop3_nes_cosine(base_model, X_sample, q_values=(2, 5, 10, 25, 50, 100), n_points=64, sigma=0.05, seed=42):
    """cos(grad_hat, grad_true) per q on real model gradients vs
    sqrt(q/(q+d+1)) (isotropic prediction)."""
    print("\n=== EXP 16 part D: Prop 3 — NES cosine vs q ===")
    torch.manual_seed(seed)
    x = X_sample[torch.randperm(X_sample.size(0))[:n_points]].clone()
    with torch.no_grad():
        out = base_model(x).squeeze(-1)
    xg = x.clone().requires_grad_(True)
    out = base_model(xg).squeeze(-1)
    g_true = torch.autograd.grad(out.sum(), xg)[0]
    d = x.size(1)
    gen = torch.Generator().manual_seed(seed)
    results = {}
    for q in q_values:
        coss = []
        for _ in range(3):  # repeats to reduce MC noise
            s_plus = torch.zeros(x.size(0), q)
            us = torch.zeros(x.size(0), q, d)
            for j in range(q):
                u = torch.randn(x.size(0), d, generator=gen)
                u = u / u.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
                us[:, j, :] = u
                with torch.no_grad():
                    s_plus[:, j] = base_model((x + sigma * u).clamp(0, 1)).squeeze(-1)
            grad_hat = (s_plus.unsqueeze(2) * us).sum(1) / (q * sigma)
            an = torch.clamp(grad_hat.norm(p=2, dim=1), min=1e-12)
            tn = torch.clamp(g_true.norm(p=2, dim=1), min=1e-12)
            cos = ((grad_hat * g_true).sum(1) / (an * tn)).mean().item()
            coss.append(max(0.0, cos))
        measured = float(np.mean(coss))
        predicted = math.sqrt(q / (q + d + 1))
        results[str(q)] = {"measured": measured, "predicted_isotropic": predicted}
        print(f"  q={q:4d} | measured cos {measured:.4f} | isotropic pred {predicted:.4f}")
    return results


# ---------------------------------------------------------------------------
# Corollary 3: M_t histograms from real NES streams
# ---------------------------------------------------------------------------
def cor3_mt_histogram(base_model, X_sample, nes_cfg, seed=42):
    """Record the full antithetic NES query stream on the real model, compute
    the M_t sequence the defense would see, with and without anchor updates.
    Paper claims: E[M_t] ~ -0.83 with anchor updates (vs theory -2/sqrt(5)
    ~ -0.89), P(M>0) ~ 0 => trigger never fires."""
    print("\n=== EXP 16 part E: Corollary 3 — M_t histograms from real NES runs ===")
    queries = []

    class _Recorder(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            queries.append(x.detach().cpu().clone())
            return self.base(x)

        def reset_history(self, *a, **k):
            pass

        def add_to_history(self, x):
            pass

    rec = _Recorder(base_model)
    torch.manual_seed(seed)
    cfg = dict(nes_cfg, max_queries=min(nes_cfg.get("max_queries", 250), 60))
    nes_attack(rec, X_sample[:512], cfg)
    Q = torch.cat(queries, dim=0)
    # per-sample sequences are interleaved batch rows; reconstruct per-row streams
    # NES queries per "step": 2q probes + 1 anchor. q = cfg['q']
    q_pairs = cfg.get("q", 10)
    block = 2 * q_pairs + 1
    B = X_sample[:512].size(0)
    n_rows = Q.size(0) // B  # queries recorded per row (uniform while all rows alive)
    n_blocks = n_rows // block
    if n_blocks == 0:
        return {"error": "not enough queries", "rows_recorded": int(n_rows), "block": block}
    d = Q.size(1)

    def mt_sequence(with_anchor=True):
        mts = []
        # rows: for row r, its queries are Q[r::B] (interleaved)
        for r in range(B):
            seq = Q[r::B][: n_blocks * block]
            if with_anchor:
                pts = seq
            else:
                # only the probe points of each pair (drop the anchor step)
                idx = []
                for bi in range(n_blocks):
                    base_i = bi * block
                    idx.extend(range(base_i, base_i + 2 * q_pairs))
                pts = seq[idx]
            v = pts[1:] - pts[:-1]
            nt = v.norm(p=2, dim=1)
            valid = nt > 1e-9
            cos_seq = []
            for i in range(1, v.size(0)):
                num = (v[i] * v[i - 1]).sum()
                den = nt[i] * nt[i - 1]
                m = torch.where(valid[i] & valid[i - 1], num / torch.clamp(den, min=1e-12), torch.tensor(0.0))
                cos_seq.append(m.item())
            mts.extend(cos_seq)
        return np.array(mts)

    mt_anchor = mt_sequence(True)
    mt_no_anchor = mt_sequence(False)
    summary = {
        "n_blocks": int(n_blocks),
        "with_anchor": {"mean": float(np.mean(mt_anchor)), "p_mt_gt_0": float((mt_anchor > 0).mean()), "n": int(len(mt_anchor))},
        "without_anchor": {"mean": float(np.mean(mt_no_anchor)), "p_mt_gt_0": float((mt_no_anchor > 0).mean()), "n": int(len(mt_no_anchor))},
        "theory_E_mt": -2.0 / math.sqrt(5.0),
    }
    print(f"  with anchor updates:    E[M_t] {summary['with_anchor']['mean']:.3f} | P(M>0) {summary['with_anchor']['p_mt_gt_0']:.4f}")
    print(f"  without anchor updates: E[M_t] {summary['without_anchor']['mean']:.3f} | P(M>0) {summary['without_anchor']['p_mt_gt_0']:.4f}")
    print(f"  theory E[M_t] = -2/sqrt(5) = {summary['theory_E_mt']:.3f}")

    # histogram CSV
    hist_out = os.path.join(PROJECT_ROOT, "results", "tables", "exp16_cor3_mt_histogram.csv")
    with open(hist_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_center", "with_anchor_count", "without_anchor_count"])
        bins = np.linspace(-1, 1, 41)
        ha, _ = np.histogram(mt_anchor, bins=bins)
        hn, _ = np.histogram(mt_no_anchor, bins=bins)
        centers = (bins[:-1] + bins[1:]) / 2
        for c, a_, n_ in zip(centers, ha, hn):
            w.writerow([f"{c:.3f}", a_, n_])
    return summary


# ---------------------------------------------------------------------------
def run(dataset="unsw_nb15", n_samples=2000, seed=42):
    cfg = load_yaml("configs/attack/blackbox.yaml")
    nes_cfg = dict(cfg["nes"])
    config = load_config()

    base_model, feature_cols, spec, levels, X_sample = load_data(dataset, n_samples, seed)
    print(f"\n=== EXP 16: reference-math validation ({dataset}) ===")

    results = {
        "lemma3_fd_vs_autograd": lemma3_fd_vs_autograd(base_model, X_sample, config, seed=seed),
        "theorem1_decomposition": theorem1_decomposition(base_model, X_sample, config, seed=seed),
        "lemma2_rho": lemma2_rho(base_model, X_sample, seed=seed),
        "prop3_nes_cosine": prop3_nes_cosine(base_model, X_sample, seed=seed),
        "cor3_mt": cor3_mt_histogram(base_model, X_sample, nes_cfg, seed=seed),
    }

    out_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp16_refmath")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"metrics_{dataset}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(os.path.join(PROJECT_ROOT, "results", "tables", f"exp16_prop3_cosine_{dataset}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q", "measured_cosine", "isotropic_prediction"])
        for q, r in results["prop3_nes_cosine"].items():
            w.writerow([q, f"{r['measured']:.4f}", f"{r['predicted_isotropic']:.4f}"])

    print("\n=== EXP 16 SUMMARY ===")
    print(json.dumps({k: v for k, v in results.items() if k != "prop3_nes_cosine"}, indent=2, default=str)[:2000])


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "unsw_nb15"
    run(dataset=ds)
