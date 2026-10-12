"""Per-number traceability records for reproducible results.

Every archived result should carry enough information to rebuild the chain

    paper table -> aggregation -> raw result -> config -> checkpoint/dataset

The record schema (used by ``experiments/exp*_*/run.py``) is::

    {
      "experiment_id":   "exp5",
      "dataset":         "unsw_nb15",
      "defense":         "TASCP (Adaptive BPDA)",
      "attacker":        "stateful_pgd_bpda",
      "seed":            42,
      "config":          {...},
      "config_sha256":   "...",
      "checkpoint_sha256": "...",
      "dataset_sha256":  "...",
      "n_evaluated":     2000,
      "n_successful":    1072,
      "asr":             0.536,
      "metrics":         {...},
      "git_commit":      "...",
      "environment":     {"python": "...", "torch": "...", ...}
    }
"""

import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def sha256_file(path, chunk_size=1 << 20):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_sha256(config):
    """Canonical (key-sorted) JSON hash of a config dict."""
    return sha256_text(json.dumps(config, sort_keys=True, separators=(",", ":"), default=str))


def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def environment_info():
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ["torch", "numpy", "pandas", "pyarrow", "scikit-learn", "xgboost",
                "scipy", "joblib", "matplotlib", "seaborn"]:
        try:
            info[pkg] = importlib_metadata.version(pkg)
        except Exception:
            info[pkg] = None
    return info


def rel(path):
    try:
        return os.path.relpath(path, ROOT)
    except Exception:
        return path


def make_record(
    experiment_id,
    dataset,
    defense,
    attacker,
    seed,
    config,
    n_evaluated,
    n_successful,
    asr,
    metrics=None,
    checkpoint_path=None,
    dataset_path=None,
):
    record = {
        "experiment_id": experiment_id,
        "dataset": dataset,
        "defense": defense,
        "attacker": attacker,
        "seed": seed,
        "config": config,
        "config_sha256": config_sha256(config),
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path else None,
        "dataset_sha256": sha256_file(dataset_path) if dataset_path else None,
        "n_evaluated": int(n_evaluated),
        "n_successful": int(n_successful),
        "asr": float(asr),
        "metrics": metrics or {},
        "git_commit": git_commit(),
        "environment": environment_info(),
    }
    if checkpoint_path:
        record["checkpoint_path"] = rel(checkpoint_path)
    if dataset_path:
        record["dataset_path"] = rel(dataset_path)
    return record


def write_results(out_dir, filename, records):
    """Write a metrics JSON; `records` may be a list or a dict of records."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    return path
