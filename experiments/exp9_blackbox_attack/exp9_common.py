import csv
import json
import math
import os
import sys

import numpy as np
import torch
import yaml

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tascp_eval.models.nids_classifier import NIDSClassifier
from src.tascp_eval.defenses.tascp import TASCPWrapper
from src.tascp_eval.attacks.nes import nes_attack
from src.tascp_eval.attacks.hopskipjump import hopskipjump_attack
from src.tascp_eval.attacks.domain_constraints import TabularDomainProjector

DATASET_ROOTS = {
    "unsw_nb15": os.path.join(PROJECT_ROOT, "data", "processed", "unsw_nb15", "unsw_nb15_processed.parquet"),
    "cicids2017": os.path.join(PROJECT_ROOT, "data", "processed", "cicids2017", "cicids2017_processed.parquet"),
    "ton_iot": os.path.join(PROJECT_ROOT, "data", "processed", "ton_iot", "ton_iot_processed.parquet"),
}


def load_yaml(relpath):
    with open(os.path.join(PROJECT_ROOT, relpath), "r") as f:
        return yaml.safe_load(f)


def extract_domain_info(df, feature_cols, config_dataset):
    """Return (spec, levels) for the domain projector.

    spec:
      unsw_nb15   -> configs/attack/domain_constraints.yaml
      cicids2017  -> configs/attack/domain_constraints_cicids2017.yaml
                     (curated from a full-data cardinality scan: protocol
                     encoding, TCP flag counts, constant bulk counters)
    levels: {column: sorted list of observed values} for all constrained
    columns, extracted before the dataframe is freed.
    """
    spec_files = {
        "unsw_nb15": "configs/attack/domain_constraints.yaml",
        "cicids2017": "configs/attack/domain_constraints_cicids2017.yaml",
        "ton_iot": "configs/attack/domain_constraints_toniot.yaml",
    }
    spec = load_yaml(spec_files[config_dataset])
    constrained = list(spec.get("binary", [])) + list(spec.get("discrete", [])) + list(spec.get("immutable", []))

    levels = {}
    for c in constrained:
        if c in df.columns:
            levels[c] = sorted(df[c].dropna().unique().tolist())
    return spec, levels


def make_projector(feature_cols, spec, levels):
    return TabularDomainProjector(
        levels=levels,
        feature_cols=feature_cols,
        binary=spec.get("binary", []),
        discrete=spec.get("discrete", []),
        immutable=spec.get("immutable", []),
    )


def load_data(dataset="unsw_nb15", n_samples=2000, seed=42):
    """Load processed parquet + trained classifier.

    Returns (base_model, feature_cols, spec, levels, X_sample). The dataframe
    is referenced only for level extraction and deleted before returning, so
    peak memory stays around the X_benign tensor (CICIDS2017: ~3 GB float32).
    """
    data_path = DATASET_ROOTS[dataset]
    print(f"Loading {dataset} dataset ({os.path.getsize(data_path) // (1024 * 1024)} MB)...")

    all_cols = get_parquet_columns(data_path)
    label_col = "label" if "label" in all_cols else "Label"
    feature_cols = [c for c in all_cols if c not in ("session_id", label_col, "__index_level_0__")]
    n_features = len(feature_cols)
    spec_file = {
        "unsw_nb15": "configs/attack/domain_constraints.yaml",
        "cicids2017": "configs/attack/domain_constraints_cicids2017.yaml",
        "ton_iot": "configs/attack/domain_constraints_toniot.yaml",
    }[dataset]
    spec = load_yaml(spec_file)
    constrained_cols = [
        c for c in spec.get("binary", []) + spec.get("discrete", []) + spec.get("immutable", []) if c in all_cols
    ]

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(data_path)
    read_cols = constrained_cols + feature_cols + [label_col]
    X_benign_np = np.empty((0, n_features), dtype=np.float32)
    level_sets = {c: set() for c in constrained_cols}
    n_benign = 0
    for rg in range(pf.num_row_groups):
        # Stream row groups, keeping only benign rows: peak memory is one
        # row-group frame + the (benign-filtered) float32 matrix, instead of
        # the full 3.1M x 80 frame that OOMs a 7 GB environment.
        tbl = pf.read_row_group(rg, columns=read_cols)
        d = tbl.to_pandas()
        for c in constrained_cols:
            level_sets[c].update(np.unique(d[c].dropna().values).tolist())
        y = d[label_col].values
        xb = d[feature_cols].values.astype(np.float32)
        xb = np.nan_to_num(xb[y == 0], nan=0.0)
        X_benign_np = np.vstack([X_benign_np, xb])
        n_benign += len(xb)
        del tbl, d, xb
    levels = {c: sorted(v) for c, v in level_sets.items()}
    X_benign = X_benign_np
    del X_benign_np
    print(f"Benign samples (streamed): {n_benign}")

    print("Loading trained NIDS classifier...")
    model_path = os.path.join(PROJECT_ROOT, "src", "tascp_eval", "models", "weights", f"{dataset}_nids.pth")
    base_model = NIDSClassifier(input_dim=n_features)
    base_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    base_model.eval()

    X_benign_t = torch.as_tensor(X_benign, dtype=torch.float32)
    del X_benign
    with torch.no_grad():
        preds = []
        for i in range(0, len(X_benign_t), 200_000):
            preds.append((torch.sigmoid(base_model(X_benign_t[i : i + 200_000])) > 0.5).squeeze().numpy().astype(int))
        preds = np.concatenate(preds)

    X_correct = X_benign_t[torch.from_numpy(preds == 0)]
    del X_benign_t, preds
    print(f"Correctly classified benign: {len(X_correct)}")

    n_samples = min(n_samples, len(X_correct))
    np.random.seed(seed)
    indices = np.random.choice(len(X_correct), n_samples, replace=False)
    X_sample = X_correct[torch.from_numpy(indices)].clone()
    del X_correct
    print(f"Sampled {n_samples} points for attack")
    return base_model, feature_cols, spec, levels, X_sample


def get_parquet_columns(path):
    """Read parquet schema without loading data (pyarrow)."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    return [f for f in schema.names if f not in ("__index_level_0__",)]


def pd_read_parquet_compat(path):
    import pandas as pd

    return pd.read_parquet(path)


def pd_read_parquet_columns(path, columns=None):
    """Low-memory path: read only the columns needed (features + label)."""
    import pandas as pd

    if columns is None:
        return pd.read_parquet(path)
    return pd.read_parquet(path, columns=columns)


def make_tascp(model):
    # Same wrapper parameters as exp5 (white-box baseline) for comparability.
    return TASCPWrapper(model, window_size=5, max_theta=math.pi / 2, tau=0.0, gamma=50.0)


def run_condition(
    name,
    model,
    X_sample,
    attack_fn,
    attack_cfg,
    projector=None,
    seed=42,
    chunk_size=1000,
):
    """Process X_sample in fixed-size chunks to bound the attack loop's memory
    (CICIDS rows are 78-dim; per-chunk attacker memory stays well under budget)."""
    print(f"\n--- {name} ---")
    all_success, all_queries, all_x_adv = [], [], []
    per_chunk_rotation, total_q = [], 0
    for i in range(0, len(X_sample), chunk_size):
        x_chunk = X_sample[i : i + chunk_size]
        torch.manual_seed(seed)
        x_adv, success, queries, q_total = attack_fn(model, x_chunk, attack_cfg, projector=projector)
        all_success.append(success.cpu())
        all_queries.append(queries.cpu())
        all_x_adv.append(x_adv.cpu())
        total_q += q_total
        if hasattr(model, "last_delta_theta"):
            with torch.no_grad():
                per_chunk_rotation.append((model.last_delta_theta.detach().squeeze() > 1e-5).float().mean().item())

    success = torch.cat(all_success, dim=0)
    queries = torch.cat(all_queries, dim=0)
    x_adv = torch.cat(all_x_adv, dim=0)

    asr = success.float().mean().item()
    if success.any():
        mean_q = queries[success].mean().item()
        dists = (x_adv[success] - X_sample[success].cpu()).norm(p=2, dim=1)
        mean_dist = dists.mean().item()
    else:
        mean_q = float("nan")
        mean_dist = float("nan")

    result = {
        "asr": asr,
        "mean_queries_to_success": mean_q,
        "mean_l2_distance": mean_dist,
        "total_queries": int(total_q),
    }
    if per_chunk_rotation:
        result["rotation_triggered_frac"] = float(np.mean(per_chunk_rotation))
        print(f"Rotation triggered (final query, mean over chunks): {result['rotation_triggered_frac'] * 100:.2f}%")

    print(
        f"ASR: {asr * 100:.2f}% | Mean Q to success: {mean_q:.1f} | Mean L2: {mean_dist:.4f} | Total Q: {total_q}"
    )
    return result


def run_nes_budget_sweep(model_u, model_t, X_sample, nes_cfg, budget_sweep, projector=None, seed=42, chunk_size=1000):
    sweep = {}
    for budget in budget_sweep:
        b_cfg = dict(nes_cfg, max_queries=budget)
        sweep[budget] = {
            "Undefended": run_condition(
                f"NES budget={budget} vs Undefended", model_u, X_sample, nes_attack, b_cfg, seed=seed, chunk_size=chunk_size
            ),
            "TASCP (Naive)": run_condition(
                f"NES budget={budget} vs TASCP (Naive)",
                make_tascp(model_u),
                X_sample,
                nes_attack,
                b_cfg,
                seed=seed,
                chunk_size=chunk_size,
            ),
        }
    return sweep


def save_results(results, out_dir, table_tag):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "tables"), exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(os.path.join(PROJECT_ROOT, "results", "tables", table_tag), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "asr", "mean_queries_to_success", "mean_l2_distance"])
        for name, r in results.items():
            if name == "nes_budget_sweep":
                continue
            q, d = r["mean_queries_to_success"], r["mean_l2_distance"]
            w.writerow([name, f"{r['asr']:.4f}", f"{q:.2f}" if q == q else "nan", f"{d:.4f}" if d == d else "nan"])


def print_summary(results, label):
    print(f"\n=== {label} SUMMARY (ASR / mean queries / mean L2) ===")
    for name, r in results.items():
        if name == "nes_budget_sweep":
            continue
        q, d = r["mean_queries_to_success"], r["mean_l2_distance"]
        qs = f"{q:.1f}" if q == q else "n/a"
        ds = f"{d:.4f}" if d == d else "n/a"
        print(f"{name:45s} | ASR: {r['asr'] * 100:6.2f}% | Q: {qs:>6s} | L2: {ds}")
    if "nes_budget_sweep" in results:
        print("\nNES budget sweep (ASR):")
        for budget, cond in results["nes_budget_sweep"].items():
            u = cond["Undefended"]["asr"] * 100
            t = cond["TASCP (Naive)"]["asr"] * 100
            print(f"  budget={budget:>5d} | Undefended: {u:6.2f}% | TASCP (Naive): {t:6.2f}%")
