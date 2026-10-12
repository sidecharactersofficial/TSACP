"""Exp 6: operational false-positive rate on sequential benign sessions.

Streams the parquet row-group-wise, assembling up to ``max_sessions``
all-benign sessions of length >= 5, then replays each session through the
undefended model and TASCP. Persists a traceable record to

    results/logs/exp6_false_positive_rate/metrics_<dataset>.json
"""
import argparse
import math
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tascp_eval.models.nids_classifier import NIDSClassifier  # noqa: E402
from src.tascp_eval.defenses.tascp import TASCPWrapper  # noqa: E402
from src.tascp_eval import traceability as T  # noqa: E402

DATASET_PATHS = {
    "unsw_nb15": os.path.join(PROJECT_ROOT, "data", "processed", "unsw_nb15", "unsw_nb15_processed.parquet"),
    "cicids2017": os.path.join(PROJECT_ROOT, "data", "processed", "cicids2017", "cicids2017_processed.parquet"),
}


def collect_benign_sessions(data_path, max_sessions=500, min_len=5):
    """Two-pass, memory-bounded collection matching the original groupby order.

    Pass 1 counts rows and benignness per session_id (two light columns).
    Pass 2 reads features only for the selected session_ids.
    """
    pf = pq.ParquetFile(data_path)
    names = list(pf.schema_arrow.names)
    label_col = "label" if "label" in names else "Label"
    sid_col = "session_id" if "session_id" in names else None
    feature_cols = [c for c in names if c not in (sid_col, label_col, "__index_level_0__")]
    read_cols = [c for c in (sid_col, label_col) if c] + feature_cols

    info = {}
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=[c for c in (sid_col, label_col) if c]).to_pandas()
        for sid, grp in t.groupby(sid_col, sort=False):
            ok = bool((grp[label_col] == 0).all())
            e = info.get(sid)
            if e is None:
                info[sid] = [len(grp), ok]
            else:
                e[0] += len(grp)
                e[1] = e[1] and ok

    ids = sorted(info)  # matches groupby(sort=True)
    targets = [sid for sid in ids if info[sid][1] and info[sid][0] >= min_len][:max_sessions]
    target_set = set(targets)

    buckets = {sid: [] for sid in targets}
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=read_cols).to_pandas()
        sub = t[t[sid_col].isin(target_set)] if sid_col else t
        if len(sub) == 0:
            continue
        for sid, grp in sub.groupby(sid_col, sort=False):
            buckets[sid].append(grp[feature_cols].values.astype(np.float32))

    seqs = [np.vstack(buckets[sid]) for sid in targets]
    return seqs, feature_cols, label_col


def run_exp6(dataset="unsw_nb15", max_sessions=500):
    data_path = DATASET_PATHS[dataset]
    print(f"exp6 | dataset={dataset} max_sessions={max_sessions}")
    seqs, feature_cols, label_col = collect_benign_sessions(data_path, max_sessions=max_sessions)
    print(f"Collected {len(seqs)} all-benign sessions of length >= 5")

    model_path = os.path.join(PROJECT_ROOT, "src", "tascp_eval", "models", "weights", f"{dataset}_nids.pth")
    base_model = NIDSClassifier(input_dim=len(feature_cols))
    base_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    base_model.eval()

    tascp = TASCPWrapper(base_model, window_size=5, max_theta=math.pi / 2, tau=0.0, gamma=50.0)

    undefended_fp = tascp_fp = total_packets = 0
    rotation_steps = rotation_triggered = 0
    for seq in seqs:
        tascp.reset_history(batch_size=1, device=torch.device("cpu"))
        for i in range(len(seq)):
            x_t = torch.tensor(seq[i : i + 1], dtype=torch.float32)
            total_packets += 1
            with torch.no_grad():
                if (base_model(x_t) > 0).item():
                    undefended_fp += 1
                if (tascp(x_t) > 0).item():
                    tascp_fp += 1
                if len(tascp.history) >= 2 and hasattr(tascp, "last_delta_theta"):
                    rotation_steps += 1
                    if tascp.last_delta_theta.item() > 0:
                        rotation_triggered += 1
                tascp.add_to_history(x_t)

    fpr = tascp_fp / total_packets if total_packets else 0.0
    rotation_frac = rotation_triggered / rotation_steps if rotation_steps else 0.0
    metrics = {
        "total_benign_packets": total_packets,
        "undefended_false_positives": undefended_fp,
        "tascp_false_positives": tascp_fp,
        "tascp_fpr": fpr,
        "active_rotation_steps": rotation_steps,
        "rotation_triggered_steps": rotation_triggered,
        "rotation_triggered_frac": rotation_frac,
    }
    print(f"Total benign packets: {total_packets}")
    print(f"Undefended FP: {undefended_fp} ({undefended_fp / total_packets * 100:.2f}%)")
    print(f"TASCP FP:      {tascp_fp} ({fpr * 100:.2f}%)")
    print(f"Rotation triggered: {rotation_triggered}/{rotation_steps} ({rotation_frac * 100:.2f}%)")

    record = T.make_record(
        "exp6", dataset, "TASCP", "benign_traffic", seed=None,
        config={"max_sessions": max_sessions, "min_session_len": 5, "window_size": 5, "gamma": 50.0, "tau": 0.0},
        n_evaluated=total_packets, n_successful=tascp_fp, asr=fpr, metrics=metrics,
        checkpoint_path=model_path, dataset_path=data_path,
    )
    out_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp6_false_positive_rate")
    T.write_results(out_dir, f"metrics_{dataset}.json", record)
    print(f"Wrote {out_dir}/metrics_{dataset}.json")
    return record


def main():
    ap = argparse.ArgumentParser(description="exp6 false-positive rate")
    ap.add_argument("dataset", nargs="?", default="unsw_nb15")
    ap.add_argument("--max-sessions", type=int, default=500)
    args = ap.parse_args()
    run_exp6(dataset=args.dataset, max_sessions=args.max_sessions)


if __name__ == "__main__":
    main()
