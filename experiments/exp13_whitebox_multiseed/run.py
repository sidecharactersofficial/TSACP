"""P1-7 part 2 + exp5/exp8 multi-seed (P0-4) experiment driver.

Conditions (per dataset, per seed):
  - Undefended (white-box PGD)
  - TASCP Naive (non-adaptive PGD)
  - TASCP Full Adaptive (BPDA through the state map)
  - TASCP Basis-fixed (ablation_mode='basis_coupling')  <- robust claim
  - XGBoost surrogate path (UNSW only): Undefended + Basis-fixed

Seeds vary BOTH the 2,000-point draw and the attack's torch seed. The trained
model is fixed (one trained checkpoint per dataset is what the paper analyzes;
model-init variance would require retraining, which changes the claim's
target). Data-draw + attack-seed variance is what the "adaptive-vs-naive
inversion" and "basis-fixed no-op" claims rest on.

CICIDS2017 note: uses exp9_common.load_data (pyarrow row-group streaming) —
the full-frame pandas read OOMs in this 7 GB environment.
"""

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
    os.path.join(PROJECT_ROOT, "experiments", "exp12_cicids_disambiguation"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp9_common import load_data
from src.tascp_eval.models.nids_classifier import NIDSClassifier
from src.tascp_eval.defenses.tascp import TASCPWrapper
from src.tascp_eval.attacks.pgd import load_config
from src.tascp_eval.attacks.bpda_adaptive import stateful_pgd_attack

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

SEEDS = [42, 43, 44, 45, 46]


def make_tascp(base_model):
    return TASCPWrapper(base_model, window_size=5, max_theta=math.pi / 2, tau=0.0, gamma=50.0)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def run_condition(name, model, X, y, config, adaptive, seed):
    torch.manual_seed(seed)
    if hasattr(model, "adaptive"):
        model.adaptive = adaptive
    x_adv, success, queries = stateful_pgd_attack(model, X, y, config, adaptive=adaptive)
    asr = success.float().mean().item()
    mq = queries[success].mean().item() if success.any() else float("nan")
    print(f"  {name:34s} | ASR {asr*100:6.2f}% | Q {mq if mq == mq else float('nan'):5.1f}")
    return {"asr": asr, "mean_queries": mq, "n": int(X.size(0)), "k": int(success.sum().item())}


def run_whitebox_multiseed(dataset="unsw_nb15", n_samples=2000, seeds=SEEDS, with_xgb=False):
    from basis_fixed_robustness import reference_vector_robustness

    config = load_config()
    print(f"\n=== EXP 13: white-box multi-seed ({dataset}, {len(seeds)} seeds x {n_samples} samples) ===")
    per_seed = {s: {} for s in seeds}
    for seed in seeds:
        base_model, feature_cols, spec, levels, X_sample = load_data(dataset, n_samples, seed)
        y_target = torch.zeros(X_sample.size(0), 1)
        print(f"\n--- seed {seed} ---")
        per_seed[seed]["Undefended"] = run_condition("Undefended", base_model, X_sample, y_target, config, False, seed)
        naive = make_tascp(base_model)
        per_seed[seed]["TASCP (Naive)"] = run_condition("TASCP (Naive)", naive, X_sample, y_target, config, False, seed)
        adaptive = make_tascp(base_model)
        per_seed[seed]["TASCP (Full Adaptive)"] = run_condition("TASCP (Full Adaptive)", adaptive, X_sample, y_target, config, True, seed)
        basis = make_tascp(base_model)
        basis.adaptive = True
        basis.ablation_mode = "basis_coupling"
        per_seed[seed]["TASCP (Basis fixed)"] = run_condition("TASCP (Basis fixed)", basis, X_sample, y_target, config, True, seed)
        del base_model, X_sample

    # pooled stats
    n = n_samples
    summary = {}
    for cond in per_seed[seeds[0]]:
        asrs = [per_seed[s][cond]["asr"] for s in seeds]
        k_pool = round(sum(asrs) * n)
        lo, hi = wilson_ci(k_pool, n * len(seeds))
        summary[cond] = {
            "per_seed_asr": asrs,
            "pooled_asr": sum(asrs) / len(seeds),
            "wilson95": [lo, hi],
            "per_seed_k": [per_seed[s][cond]["k"] for s in seeds],
        }
        print(
            f"{cond:34s} | pooled {summary[cond]['pooled_asr']*100:6.2f}% "
            f"[{lo*100:.2f}, {hi*100:.2f}] | seeds: {', '.join(f'{a*100:.2f}' for a in asrs)}"
        )

    # basis-fixed reference robustness (P1-7): seed-42 draw, 8 reference pairs
    base_model, _, _, _, X_sample = load_data(dataset, n_samples, 42)
    y_target = torch.zeros(X_sample.size(0), 1)
    ref_results = reference_vector_robustness(base_model, X_sample, y_target, config, n_refs=8, seed=42)
    del base_model, X_sample

    # XGBoost surrogate path (UNSW only)
    xgb_results = None
    if with_xgb:
        xgb_results = run_xgb_basis_fixed(dataset, n_samples, config, seed=42)

    out_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp13_whitebox_multiseed")
    os.makedirs(out_dir, exist_ok=True)
    payload = {"summary": summary, "basis_fixed_reference_robustness": ref_results}
    if xgb_results:
        payload["xgboost"] = xgb_results
    with open(os.path.join(out_dir, f"multiseed_{dataset}.json"), "w") as f:
        json.dump(payload, f, indent=2)

    import csv

    with open(os.path.join(PROJECT_ROOT, "results", "tables", f"exp13_whitebox_multiseed_{dataset}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "pooled_asr", "wilson95_lo", "wilson95_hi"] + [f"seed_{s}" for s in seeds])
        for cond, s in summary.items():
            w.writerow([cond, f"{s['pooled_asr']:.4f}", f"{s['wilson95'][0]:.4f}", f"{s['wilson95'][1]:.4f}"] + [f"{a:.4f}" for a in s["per_seed_asr"]])

    return payload


def run_xgb_basis_fixed(dataset, n_samples, config, seed=42):
    """Basis-fixed ablation on the XGBoost-surrogate path (P1-7 part 2)."""
    import xgboost as xgb

    from src.tascp_eval.attacks.bpda_xgb_surrogate import xgb_stateful_pgd_attack
    from experiments.exp5_adaptive_bpda_attack.run_xgboost import SurrogateMLP, NoOpWrapper

    print("\n--- XGBoost surrogate path (basis-fixed) ---")
    import pandas as pd

    data_path = os.path.join(PROJECT_ROOT, "data", "processed", dataset, f"{dataset}_processed.parquet")
    df = pd.read_parquet(data_path)
    label_col = "label" if "label" in df.columns else "Label"
    feature_cols = [c for c in df.columns if c not in ["session_id", label_col]]
    X = df[feature_cols].values.astype(np.float32)
    np.nan_to_num(X, copy=False)
    y = df[label_col].values
    del df

    model_path = os.path.join(PROJECT_ROOT, "src", "tascp_eval", "models", "weights", "unsw_nb15_xgboost.json")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_path)

    np.random.seed(42)
    subset_idx = np.random.choice(len(X), min(200_000, len(X)), replace=False)
    X_sub, y_sub = X[subset_idx], y[subset_idx]
    y_proba = xgb_model.predict_proba(X_sub)[:, 1]
    y_log_odds = np.log(np.clip(y_proba, 1e-7, 1 - 1e-7) / np.clip(1 - y_proba, 1e-7, 1 - 1e-7))
    X_t = torch.from_numpy(X_sub.copy())
    y_lo_t = torch.from_numpy(y_log_odds.astype(np.float32)).unsqueeze(1)

    surrogate = SurrogateMLP(input_dim=len(feature_cols))
    opt = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
    surrogate.train()
    for ep in range(50):
        opt.zero_grad()
        loss = torch.nn.MSELoss()(surrogate(X_t), y_lo_t)
        loss.backward()
        opt.step()
    surrogate.eval()
    del X_t, y_lo_t, X_sub, y_sub

    benign_mask = y == 0
    X_benign = X[benign_mask]
    del X
    preds = xgb_model.predict(X_benign, validate_features=False)
    X_correct = torch.from_numpy(X_benign[preds == 0].copy())
    n = min(n_samples, len(X_correct))
    np.random.seed(seed)
    X_sample = X_correct[np.random.choice(len(X_correct), n, replace=False)]
    y_target = torch.zeros(n, 1)
    del X_benign, X_correct

    results = {}
    conditions = [
        ("XGB Undefended", NoOpWrapper(surrogate), False),
        ("XGB TASCP (Basis fixed)", make_tascp(surrogate), True),
    ]
    for name, model, adaptive in conditions:
        if "Basis" in name:
            model.adaptive = True
            model.ablation_mode = "basis_coupling"
        x_adv, success, queries = xgb_stateful_pgd_attack(model, xgb_model, surrogate, X_sample, y_target, config, adaptive=adaptive)
        asr = success.float().mean().item()
        results[name] = {"asr": asr, "k": int(success.sum().item()), "n": n}
        print(f"  {name:34s} | ASR {asr*100:6.2f}%")
    return results


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "unsw_nb15"
    run_whitebox_multiseed(dataset=ds, with_xgb=(ds == "unsw_nb15"))
