"""RQ2: coupling magnitude vs adaptive attack effect, across state functions.

Generalizes the coupling audit from TASCP's single state rule (directional
momentum) to six trajectory state functions x five transformations:

  state fns: momentum (TASCP's eps_p), cosine, disp_mag, running_mean_dir,
             ema, curvature            (see family.STATE_FNS)
  transforms: tascp, tscale, tshift, tshear, tnoise  (see family.py)

Per (state_fn, transform) cell we measure
  C_raw   ||dT/dh||_F  (per-sample mean, on an active trajectory)
  C_norm  C_raw / median_i ||T(x_i) - x_i||_2   (scale-free relative coupling;
          median-normalized, same-batch geometry)
  adaptive ASR (B2 replica attack) vs fixed-state control ASR
  nu_session (stationarity replay)

and reports Spearman/Pearson correlation of log10 C against the adaptive
ASR increase over the fixed-state baseline. This is the data behind the
RQ2 scatter figure: higher coupling -> larger exploitable effect.

CLI: python -m experiments.exp20_coupling_audit.run [dataset]
Writes results/tables/exp20_coupling_scatter_<ds>.csv and
results/logs/exp20_coupling_audit/metrics_<ds>.json.
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
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "exp19_trajectory_family"))

from experiments.exp9_blackbox_attack.exp9_common import load_data  # noqa: E402
from experiments.exp19_trajectory_family.run import (  # noqa: E402
    PGD_CFG,
    _ReplicaAttack,
    adaptive_family_attack,
    naive_family_attack,
)
from family import (FAMILY_VARIANTS, STATE_FN_NAMES, FamilyWrapper,  # noqa: E402
                    stationarity_replay)

try:
    from scipy import stats as scipy_stats
except ImportError:  # scipy is available in this env, but be safe
    scipy_stats = None


def spearman(a, b):
    """Hand-rolled Spearman (average ranks) as fallback if scipy is absent."""
    ra = _ranks(a)
    rb = _ranks(b)
    return _pearson(ra, rb)


def _ranks(x):
    idx = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and x[idx[j + 1]] == x[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


def _pearson(a, b):
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return cov / max(va * vb, 1e-12)


def eval_cell(base_model, X, y, cfg, variant, state_fn):
    """One (state_fn, transform) cell: fixed-state control, adaptive (B2),
    naive, coupling (raw + normalized), stationarity."""

    def fresh(freeze=None):
        w = FamilyWrapper(base_model, variant=variant, tau=0.0, gamma=50.0,
                          state_fn=state_fn)
        w.adaptive = False
        w.freeze_state = freeze
        return w

    w_ctl = fresh(freeze=0.0)
    _, s_c, _ = naive_family_attack(w_ctl, X, y, cfg)
    control_asr = float(s_c.float().mean().item())

    w = fresh()
    _, s_n, _ = naive_family_attack(w, X, y, cfg)
    naive_asr = float(s_n.float().mean().item())

    w = fresh()
    rep = _ReplicaAttack(base_model, variant, state_fn=state_fn)
    _, s_a, _ = adaptive_family_attack(w, rep, X, y, cfg)
    adaptive_asr = float(s_a.float().mean().item())

    # coupling audit on an active persistent-direction trajectory
    w = fresh()
    w.adaptive = False
    w.reset_history(X.size(0), X.device)
    w.add_to_history(X.clone())
    cursor = X.clone()
    u_dir = torch.nn.functional.normalize(torch.rand_like(X), dim=1)
    for _ in range(3):
        cursor = (cursor + 0.15 * u_dir).clamp(0.0, 1.0)
        w.add_to_history(cursor)
    C_raw = float(w.coupling_norm(X).mean().item())
    C_norm = float(w.coupling_norm(X, normalized=True).mean().item())
    firing = float((w.last_S > w.tau).float().mean().item())

    w = fresh()
    stat = stationarity_replay(w, X, n_steps=12, step_scale=0.3)

    return {
        "state_fn": state_fn,
        "variant": variant,
        "naive_asr": naive_asr,
        "adaptive_asr": adaptive_asr,
        "fixed_state_asr": control_asr,
        "adaptive_delta": adaptive_asr - control_asr,
        "naive_delta": naive_asr - control_asr,
        "C_raw": C_raw,
        "C_norm": C_norm,
        "coupling_firing_frac": firing,
        "nu_session": stat["nu_session"],
        "nu_final": stat["nu_final"],
        "mean_state": stat["mean_state"],
    }


def main(dataset="unsw_nb15", n_samples=500, seed=42):
    torch.manual_seed(seed)
    base_model, feature_cols, spec, levels, X = load_data(dataset, n_samples, seed)
    y = torch.zeros(X.size(0), 1)

    cells = []
    for state_fn in STATE_FN_NAMES:
        for variant in FAMILY_VARIANTS:
            print(f"\n=== cell state_fn={state_fn} x transform={variant} ===")
            r = eval_cell(base_model, X, y, PGD_CFG, variant, state_fn)
            cells.append(r)
            print(f"  C_raw {r['C_raw']:.4g} | C_norm {r['C_norm']:.4f} | "
                  f"adaptive {r['adaptive_asr']*100:6.2f}% | fixed {r['fixed_state_asr']*100:6.2f}% | "
                  f"dASR {r['adaptive_delta']*100:+6.2f}pp | nu {r['nu_session']*100:5.2f}%")

    # ---- correlation analysis: log10 C vs adaptive delta -------------------
    valid = [c for c in cells if c["C_norm"] > 1e-9 and c["coupling_firing_frac"] > 0.0]
    log_cn = [math.log10(c["C_norm"]) for c in valid]
    log_cr = [math.log10(c["C_raw"]) for c in valid if c["C_raw"] > 0]
    deltas = [c["adaptive_delta"] for c in valid]

    corr = {}
    if len(valid) >= 3:
        if scipy_stats is not None:
            sp_cn, p_cn = scipy_stats.spearmanr(log_cn, deltas)
            pe_cn, pe_p = scipy_stats.pearsonr(log_cn, deltas)
            corr["spearman_logCnorm_vs_dASR"] = {"rho": float(sp_cn), "p": float(p_cn)}
            corr["pearson_logCnorm_vs_dASR"] = {"r": float(pe_cn), "p": float(pe_p)}
        else:
            corr["spearman_logCnorm_vs_dASR"] = {"rho": float(spearman(log_cn, deltas))}
            corr["pearson_logCnorm_vs_dASR"] = {"r": float(_pearson(log_cn, deltas))}

    out = {
        "dataset": dataset,
        "n_samples": X.size(0),
        "n_cells": len(cells),
        "n_cells_with_open_channel": len(valid),
        "correlations": corr,
        "cells": cells,
    }
    log_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp20_coupling_audit")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"metrics_{dataset}.json"), "w") as f:
        json.dump(out, f, indent=2)

    table_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(table_dir, exist_ok=True)
    cols = ["state_fn", "variant", "C_raw", "C_norm", "coupling_firing_frac",
            "naive_asr", "adaptive_asr", "fixed_state_asr", "adaptive_delta",
            "nu_session", "nu_final"]
    with open(os.path.join(table_dir, f"exp20_coupling_scatter_{dataset}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in cells:
            w.writerow([c[k] if not isinstance(c[k], float) else f"{c[k]:.6g}" for k in cols])

    print("\n=== EXP 20 SUMMARY (RQ2 coupling vs adaptive effect) ===")
    print(f"cells: {len(cells)} | open-channel cells: {len(valid)}")
    for k, v in corr.items():
        print(f"  {k}: {v}")
    ordered = sorted(cells, key=lambda c: c["C_norm"] if c["C_norm"] > 0 else float("inf"))
    print("\ncells ordered by normalized coupling C_norm:")
    for c in ordered:
        print(f"  {c['state_fn']:17s} x {c['variant']:7s} | C_norm {c['C_norm']:9.4f} | "
              f"dASR {c['adaptive_delta']*100:+6.2f}pp | nu {c['nu_session']*100:5.2f}%")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "unsw_nb15"
    main(ds)
