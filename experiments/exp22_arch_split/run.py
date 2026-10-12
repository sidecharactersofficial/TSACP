"""RQ5: does the phenomenon depend on differentiability?

Runs the three audits (coupling, trigger observability, stationarity) and
the stateful-perturbation effect across FOUR architectures with different
differentiability status:

  mlp            NIDSClassifier           differentiable (checkpoint)
  transformer    TransformerClassifier    differentiable (checkpoint)
  xgboost        trained XGBClassifier    non-differentiable (checkpoint)
  random_forest  sklearn RF               non-differentiable (trained here,
                 honestly labeled as trained-on-the-fly; no RF checkpoint exists)

Mechanism x architecture matrix (the reviewer's table):
  coupling           C_norm via autograd through the TASCP family wrapper
                     -> only meaningful for differentiable bases; reported
                     as null (n/a) for XGB/RF, NOT zero.
  perturbation       naive straight-through PGD vs fixed-state control
                     -> differentiable bases only; null for XGB/RF.
  trigger            fire rates of the momentum and antithetic-pair triggers
                     under black-box attacks (NES independent, antithetic
                     NES jitter, Square) -> measurable for ALL architectures.
  stationarity       nu_session / nu_final under state replay -> ALL
                     architectures (forward-only).

Expected shape of the result: the B2 coupling channel and the gradient-based
perturbation effect are differentiability-dependent, while trigger blindness
and stationarity violations are NOT (they only require a query oracle).

CLI: python -m experiments.exp22_arch_split.run [dataset]
Writes results/tables/exp22_arch_split_<ds>.csv and
results/logs/exp22_arch_split/metrics_<ds>.json.
"""

import csv
import json
import os
import sys

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
for _p in (os.path.join(PROJECT_ROOT, "experiments", "exp9_blackbox_attack"),
           os.path.join(PROJECT_ROOT, "experiments", "exp11_detector_robustness"),
           os.path.join(PROJECT_ROOT, "experiments", "exp19_trajectory_family"),
           os.path.join(PROJECT_ROOT, "experiments", "exp21_trigger_matrix")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.exp9_blackbox_attack.exp9_common import load_data, DATASET_ROOTS  # noqa: E402
from experiments.exp19_trajectory_family.family import FamilyWrapper, stationarity_replay  # noqa: E402
from experiments.exp19_trajectory_family.run import PGD_CFG, _ReplicaAttack, adaptive_family_attack, naive_family_attack  # noqa: E402
from experiments.exp21_trigger_matrix.run import (ATTACKS, QueryRecorder, TRIGGERS,  # noqa: E402
                                                  benign_sessions, calibrate_taus,
                                                  fire_rate, run_attack)
from src.tascp_eval.attacks.bpda_adaptive import stateful_pgd_attack  # noqa: E402
from src.tascp_eval.models.nids_classifier import NIDSClassifier  # noqa: E402

TRIGGER_ATTACKS = ("nes_independent", "nes_antithetic_jitter", "square")
TRIGGER_CFG = {"epsilon": 0.1, "sigma": 0.05, "alpha": 0.02, "q": 10,
               "max_queries": 100, "square_p": 0.05}


# ---------------------------------------------------------------------------
# Architecture loading
# ---------------------------------------------------------------------------
class SklearnAdapter(torch.nn.Module):
    """Wraps a non-torch classifier (XGBoost / sklearn RF) in the forward(x)
    -> logits contract the family wrapper and attacks expect."""

    def __init__(self, margin_fn):
        super().__init__()
        self.margin_fn = margin_fn

    def forward(self, x):
        m = self.margin_fn(x.detach().cpu().numpy().astype(np.float32))
        return torch.as_tensor(np.asarray(m, dtype=np.float32).reshape(-1, 1),
                               dtype=x.dtype, device=x.device)


def load_xgboost():
    import xgboost as xgb

    path = os.path.join(PROJECT_ROOT, "src", "tascp_eval", "models", "weights",
                        "unsw_nb15_xgboost.json")
    booster = xgb.XGBClassifier()
    booster.load_model(path)

    def margin_fn(x_np):
        return booster.predict(x_np, output_margin=True, validate_features=False)

    return SklearnAdapter(margin_fn), "xgboost"


def load_or_train_random_forest(max_rows=80_000, n_estimators=100, seed=42):
    """No RF checkpoint exists in this repo — train one here on a labeled
    streamed subset of the same processed parquet (honestly labeled as
    trained-on-the-fly in the output)."""
    from sklearn.ensemble import RandomForestClassifier

    path = DATASET_ROOTS["unsw_nb15"]
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    schema_names = [f for f in pf.schema_arrow.names if f != "__index_level_0__"]
    label_col = "label" if "label" in schema_names else "Label"
    feature_cols = [c for c in schema_names if c not in ("session_id", label_col)]

    Xs, ys = [], []
    total = 0
    for rg in range(pf.num_row_groups):
        if total >= max_rows:
            break
        d = pf.read_row_group(rg, columns=feature_cols + [label_col]).to_pandas()
        Xs.append(d[feature_cols].values.astype(np.float32))
        ys.append(d[label_col].values.astype(np.int64))
        total += len(d)
        del d
    X = np.nan_to_num(np.vstack(Xs), nan=0.0)
    y = np.concatenate(ys)
    del Xs, ys
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), min(max_rows, len(X)), replace=False)
    X, y = X[idx], y[idx]
    print(f"  training RF on {len(X)} labeled rows "
          f"(pos rate {y.mean():.3f}) ...")
    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=20,
                                random_state=seed, class_weight="balanced")
    rf.fit(X, y)

    def margin_fn(x_np):
        p = rf.predict_proba(x_np)[:, 1]
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    return SklearnAdapter(margin_fn), "random_forest"


def load_architectures(dataset, base_mlp, n_features):
    """Returns dict name -> (base_model, differentiable). The MLP comes from
    load_data (shared X sampling); the transformer/XGB checkpoints load
    directly; the RF is trained on the fly."""
    archs = {"mlp": (base_mlp, True)}

    from src.tascp_eval.models.transformer_classifier import TransformerClassifier
    tf = TransformerClassifier(input_dim=n_features, d_model=128, num_layers=2)
    tf_path = os.path.join(PROJECT_ROOT, "src", "tascp_eval", "models", "weights",
                           "unsw_nb15_transformer.pth")
    tf.load_state_dict(torch.load(tf_path, map_location="cpu", weights_only=True))
    tf.eval()
    archs["transformer"] = (tf, True)

    xgb_model, _ = load_xgboost()
    archs["xgboost"] = (xgb_model, False)

    rf_model, _ = load_or_train_random_forest()
    archs["random_forest"] = (rf_model, False)
    return archs


# ---------------------------------------------------------------------------
# Mechanism measurements
# ---------------------------------------------------------------------------
def measure_coupling(base_model, X):
    """C_norm on an active persistent-direction trajectory."""
    w = FamilyWrapper(base_model, variant="tascp", tau=0.0, gamma=50.0)
    w.adaptive = False
    w.reset_history(X.size(0), X.device)
    w.add_to_history(X.clone())
    cursor = X.clone()
    u = torch.nn.functional.normalize(torch.rand_like(X), dim=1)
    for _ in range(3):
        cursor = (cursor + 0.15 * u).clamp(0.0, 1.0)
        w.add_to_history(cursor)
    C_norm = float(w.coupling_norm(X, normalized=True).mean().item())
    return C_norm, float((w.last_S > w.tau).float().mean().item())


def measure_perturbation(base_model, X, y, cfg):
    def fresh(freeze=None):
        w = FamilyWrapper(base_model, variant="tascp", tau=0.0, gamma=50.0)
        w.adaptive = False
        w.freeze_state = freeze
        return w

    w_ctl = fresh(freeze=0.0)
    _, s_c, _ = naive_family_attack(w_ctl, X, y, cfg)
    w = fresh()
    _, s_n, _ = naive_family_attack(w, X, y, cfg)
    w = fresh()
    rep = _ReplicaAttack(base_model, "tascp")
    _, s_a, _ = adaptive_family_attack(w, rep, X, y, cfg)
    return {
        "naive_asr": float(s_n.float().mean().item()),
        "adaptive_asr": float(s_a.float().mean().item()),
        "fixed_state_asr": float(s_c.float().mean().item()),
        "adaptive_delta": float(s_a.float().mean().item() - s_c.float().mean().item()),
    }


def measure_triggers(base_model, X, cfg):
    rec = QueryRecorder(base_model)
    benign = benign_sessions(base_model, X, n_repeat=40)
    taus = calibrate_taus(benign)
    out = {}
    for atk in TRIGGER_ATTACKS:
        seq, succ, _, _ = run_attack(rec, ATTACKS[atk], X, None, cfg)
        for trig in ("momentum", "antithetic_pair"):
            out[f"fire_{trig}_{atk}"] = fire_rate(TRIGGERS[trig](seq), taus[trig])
        out[f"asr_{atk}"] = float(succ.float().mean().item())
    out["benign_fp_momentum"] = fire_rate(TRIGGERS["momentum"](benign), taus["momentum"])
    return out


def main(dataset="unsw_nb15", n_samples=200, seed=42):
    torch.manual_seed(seed)
    print("Loading data + MLP (shared benign pool)...")
    base_mlp, feature_cols, spec, levels, X = load_data(dataset, n_samples, seed)
    print("Loading other architectures...")
    archs = load_architectures(dataset, base_mlp, X.size(1))
    y = torch.zeros(X.size(0), 1)
    d = X.size(1)
    cfg = dict(PGD_CFG)

    results = {}
    for name, (base_model, diff) in archs.items():
        print(f"\n=== architecture: {name} (differentiable={diff}) ===")
        r = {"differentiable": diff}
        if diff:
            C, firing = measure_coupling(base_model, X)
            r["coupling_C_norm"] = C
            r["coupling_firing_frac"] = firing
            pert = measure_perturbation(base_model, X, y, cfg)
            r.update(pert)
        else:
            r["coupling_C_norm"] = None
            r["coupling_firing_frac"] = None
            r["naive_asr"] = None
            r["adaptive_asr"] = None
            r["fixed_state_asr"] = None
            r["adaptive_delta"] = None
        # stationarity works for every architecture (forward-only)
        w = FamilyWrapper(base_model, variant="tascp", tau=0.0, gamma=50.0)
        w.adaptive = False
        stat = stationarity_replay(w, X, n_steps=12, step_scale=0.3)
        r["nu_session"] = stat["nu_session"]
        r["nu_final"] = stat["nu_final"]
        # triggers work for every architecture (black-box query oracle)
        r.update(measure_triggers(base_model, X, TRIGGER_CFG))
        results[name] = r
        for k, v in r.items():
            print(f"  {k}: {v if not isinstance(v, float) else round(v, 4)}")

    out_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp22_arch_split")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"metrics_{dataset}.json"), "w") as f:
        json.dump(results, f, indent=2)

    cols = ["differentiable", "coupling_C_norm", "naive_asr", "adaptive_asr",
            "fixed_state_asr", "adaptive_delta", "nu_session", "nu_final",
            "fire_momentum_nes_independent", "fire_momentum_nes_antithetic_jitter",
            "fire_antithetic_pair_nes_independent",
            "fire_antithetic_pair_nes_antithetic_jitter",
            "asr_nes_independent", "asr_nes_antithetic_jitter", "asr_square"]
    table_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(table_dir, exist_ok=True)
    with open(os.path.join(table_dir, f"exp22_arch_split_{dataset}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["architecture"] + cols)
        for name, r in results.items():
            w.writerow([name] + [(f"{r[c]:.6g}" if isinstance(r.get(c), float) else r.get(c, "")) for c in cols])

    print("\n=== EXP 22 SUMMARY (RQ5 differentiability split) ===")
    print(f"{'arch':14s} {'coupl':>8s} {'dASR':>7s} {'nu_s':>7s} {'mom-NES':>8s} {'mom-anti':>8s}")
    for name, r in results.items():
        c = f"{r['coupling_C_norm']:.3f}" if r.get("coupling_C_norm") is not None else "n/a"
        dASR = f"{r['adaptive_delta']*100:+.1f}pp" if r.get("adaptive_delta") is not None else "n/a"
        print(f"{name:14s} {c:>8s} {dASR:>7s} {r['nu_session']*100:6.2f}% "
              f"{r['fire_momentum_nes_independent']*100:7.1f}% "
              f"{r['fire_momentum_nes_antithetic_jitter']*100:7.1f}%")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "unsw_nb15"
    main(ds)
