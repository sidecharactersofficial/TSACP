#!/usr/bin/env python3
"""Verify checkpoint integrity against manifest/checkpoint_sha256.json.

Recomputes the SHA-256 of every shipped checkpoint under
src/tascp_eval/models/weights/ and compares against the manifest.

Usage:
  python verification/verify_manifest.py          # verify + print all hashes
  python verification/verify_manifest.py --quiet  # only report failures
"""

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "manifest", "checkpoint_sha256.json")
WEIGHTS = os.path.join(ROOT, "src", "tascp_eval", "models", "weights")
EXTS = (".pth", ".json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", help="only report failures")
    args = parser.parse_args()

    manifest = json.load(open(MANIFEST))
    expected = manifest["checkpoints"]

    disk = {}
    for dirpath, _, files in os.walk(WEIGHTS):
        for f in sorted(files):
            if f.endswith(EXTS):
                p = os.path.relpath(os.path.join(dirpath, f), ROOT)
                disk[p] = sha256(os.path.join(ROOT, p))

    missing = sorted(set(expected) - set(disk))
    extra = sorted(set(disk) - set(expected))
    mismatched = sorted(p for p in set(expected) & set(disk) if disk[p] != expected[p])

    for p in sorted(disk):
        print(f"{disk[p]}  {p}")

    status = "PASS"
    if missing or extra or mismatched:
        status = "FAIL"

    if not args.quiet or status == "FAIL":
        if missing:
            print(f"MISSING from disk: {missing}")
        if extra:
            print(f"NOT in manifest: {extra}")
        if mismatched:
            print(f"HASH MISMATCH: {mismatched}")

    print(f"verify_manifest: {status} ({len(disk)}/{len(expected)} checkpoints)")
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
