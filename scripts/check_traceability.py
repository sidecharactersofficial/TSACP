"""Check per-number traceability of archived result files.

Walks ``results/**/*.json`` and verifies that each file either (a) carries the
canonical traceability record written by ``tascp_eval.traceability`` or
(b) is explicitly classified as a legacy aggregate. Result files for the
experiments listed in ``REQUIRED`` must be traceable; their absence or a missing
key is a failure (exit 1), which is what CI uses to flag a regression in
traceability.

Usage:
    python scripts/check_traceability.py            # human summary, exit 1 on failure
    python scripts/check_traceability.py --json     # machine-readable summary
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(PROJECT_ROOT, "results")

REQUIRED_KEYS = ["experiment_id", "dataset", "config_sha256", "checkpoint_sha256", "n_evaluated", "n_successful"]
# Experiments whose archived records must be fully traceable.
REQUIRED = {
    "results/logs/exp5_adaptive_bpda_attack/metrics_unsw_nb15.json",
    "results/logs/exp6_false_positive_rate/metrics_unsw_nb15.json",
    "results/logs/exp8_adaptive_ablations/metrics_unsw_nb15.json",
}


def is_traceable(obj):
    return isinstance(obj, dict) and all(k in obj for k in REQUIRED_KEYS)


def count_records(obj):
    """Count fully-traceable records in a result file.

    Handles three archive shapes produced by the runners:
      * a single traceability record (exp6);
      * a mapping of condition-name -> record (exp5);
      * ``{"conditions": {name: record}, ...}`` (exp8).
    """
    if is_traceable(obj):
        return 1
    n = 0
    if isinstance(obj, dict):
        for key, value in obj.items():
            if is_traceable(value):
                n += 1
            elif isinstance(value, dict):
                n += sum(1 for v in value.values() if is_traceable(v))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    traceable, legacy, missing_required, bad_required = [], [], [], []
    for root, _dirs, files in os.walk(RESULTS):
        for name in files:
            if not name.endswith(".json"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, PROJECT_ROOT)
            try:
                with open(path) as fh:
                    obj = json.load(fh)
            except Exception as exc:  # noqa: BLE001
                legacy.append((rel, f"unreadable: {exc}"))
                continue
            n = count_records(obj)
            if n > 0:
                traceable.append(rel)
            else:
                legacy.append((rel, "legacy aggregate"))

    present = {t for t, _ in legacy} | set(traceable)
    for req in sorted(REQUIRED):
        if req not in present:
            missing_required.append(req)
        elif req in {t for t, _ in legacy}:
            bad_required.append(req)

    summary = {
        "traceable": sorted(traceable),
        "legacy": sorted(t for t, _ in legacy),
        "required_missing": sorted(missing_required),
        "required_not_traceable": sorted(bad_required),
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"traceable result files: {len(traceable)}")
        print(f"legacy aggregate files : {len(legacy)}")
        for req in sorted(missing_required):
            print(f"MISSING required record: {req}")
        for req in sorted(bad_required):
            print(f"NOT TRACEABLE: {req}")

    ok = not missing_required and not bad_required
    if not args.json:
        print("traceability: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
