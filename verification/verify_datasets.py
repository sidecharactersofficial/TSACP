#!/usr/bin/env python3
"""Verify dataset integrity against manifest/dataset_sha256.txt.

Recomputes SHA-256 of the three preprocessed parquet files and compares them
against the pinned hashes. Exits non-zero on any mismatch.

Usage:
  python verification/verify_datasets.py
"""

import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "manifest", "dataset_sha256.txt")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    expected = []
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            path, digest = line.split()
            expected.append((path, digest))

    status = "PASS"
    for path, digest in expected:
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            print(f"MISSING: {path}")
            status = "FAIL"
            continue
        actual = sha256(full)
        ok = actual == digest
        status = status if ok else "FAIL"
        print(f"{'OK  ' if ok else 'FAIL'}  {path}")
        if not ok:
            print(f"      expected {digest}")
            print(f"      actual   {actual}")

    print(f"verify_datasets: {status} ({len(expected)} files)")
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
