"""Exp 24: third-dataset transfer to NF-ToN-IoT-v2.

The NF-ToN-IoT-v2 evaluation reuses the same per-experiment runners with
``dataset=ton_iot``; there is no separate numeric implementation. This driver
makes that runnable and discoverable:

    1. (optional) ``--prepare``  -> data/ton_iot/prepare_ton_iot.py
    2. (optional) ``--train``    -> data/ton_iot/train_ton_iot.py
    3. transfer audits          -> exp19, exp20, exp21, exp23 with ton_iot

By default it assumes the processed parquet and the ton_iot checkpoint are
already present and runs only step 3.

Usage:
    python -m experiments.exp24_ton_iot_transfer.run
    python -m experiments.exp24_ton_iot_transfer.run --prepare --train
"""
import argparse
import os
import subprocess
import sys

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed", "ton_iot", "ton_iot_processed.parquet")
CHECKPOINT = os.path.join(PROJECT_ROOT, "src", "tascp_eval", "models", "weights", "ton_iot_nids.pth")
TRANSFER_RUNNERS = [
    "experiments.exp19_trajectory_family.run",
    "experiments.exp20_coupling_audit.run",
    "experiments.exp21_trigger_matrix.run",
    "experiments.exp23_defense_audit.run",
]


def _python(module, *args):
    cmd = [sys.executable, "-m", module, *args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description="exp24 NF-ToN-IoT-v2 transfer")
    ap.add_argument("--prepare", action="store_true", help="run data/ton_iot/prepare_ton_iot.py")
    ap.add_argument("--train", action="store_true", help="run data/ton_iot/train_ton_iot.py")
    args = ap.parse_args()

    if args.prepare:
        _python("data.ton_iot.prepare_ton_iot")
    if args.train:
        _python("data.ton_iot.train_ton_iot")

    if not os.path.exists(PROCESSED):
        sys.exit(f"missing processed dataset: {PROCESSED}\nrun with --prepare")
    if not os.path.exists(CHECKPOINT):
        sys.exit(f"missing checkpoint: {CHECKPOINT}\nrun with --train")

    for runner in TRANSFER_RUNNERS:
        _python(runner, "ton_iot")
    print("\nexp24: transfer audits complete (exp19/20/21/23 on ton_iot)")


if __name__ == "__main__":
    main()
