#!/usr/bin/env python3
"""Compare a fresh result against the shipped archive.

Registered tolerances (pre-declared, see REPRODUCIBILITY_LEVELS.md):

  * pooled ASR / ASR    : absolute tolerance  0.02  (2 percentage points)
  * per-seed ASR        : absolute tolerance  0.03  (3 percentage points)
  * rotation-triggered  : absolute tolerance  0.05  (5 percentage points)
  * mean queries        : relative tolerance  0.25  (25%)  [secondary metric]

Seeded white-box experiments (exp5/exp6/exp8) reproduce their archive exactly;
the tolerance exists so the check also works for the stochastic NES/HSJ
multiseed sweeps, which are unseeded by design (doc/adr/ADR-004).

Usage:
  python verification/compare_results.py ARCHIVE.json FRESH.json
"""

import argparse
import json
import sys

POOLED_ATOL = 0.02
PER_SEED_ATOL = 0.03
ROTATION_ATOL = 0.05
QUERY_RTOL = 0.25


def _compare(name, a, b, atol, failures):
    if a is None or b is None:
        if a != b:
            failures.append(f"{name}: archive={a} fresh={b} (null mismatch)")
        return
    diff = abs(a - b)
    ok = diff <= atol
    print(f"  {'OK  ' if ok else 'FAIL'} {name}: archive={a:.4f} fresh={b:.4f} "
          f"|diff|={diff:.4f} tol={atol}")
    if not ok:
        failures.append(name)


def _compare_rel(name, a, b, rtol, failures):
    if a is None or b is None:
        return
    denom = max(abs(a), 1e-12)
    rel = abs(a - b) / denom
    ok = rel <= rtol
    print(f"  {'OK  ' if ok else 'FAIL'} {name}: archive={a:.4f} fresh={b:.4f} "
          f"reldiff={rel:.4f} tol={rtol}")
    if not ok:
        failures.append(name)


def _conditions(obj):
    """Normalise the several archive shapes into {condition: metrics-dict}."""
    if isinstance(obj, dict):
        if isinstance(obj.get("conditions"), dict):
            return obj["conditions"]
        if isinstance(obj.get("summary"), dict):
            return obj["summary"]
    return obj if isinstance(obj, dict) else {}


def compare(archive, fresh):
    failures = []
    arch_cond = _conditions(archive)
    fresh_cond = _conditions(fresh)

    for cond in sorted(set(arch_cond) & set(fresh_cond)):
        a = arch_cond[cond]
        b = fresh_cond[cond]
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        print(f"[{cond}]")
        if "pooled_asr" in a and "pooled_asr" in b:
            _compare("pooled_asr", a["pooled_asr"], b["pooled_asr"], POOLED_ATOL, failures)
        if "asr" in a and "asr" in b:
            _compare("asr", a["asr"], b["asr"], POOLED_ATOL, failures)
        if "n_successful" in a and "n_successful" in b and a.get("n_evaluated") == b.get("n_evaluated"):
            _compare("n_successful", a["n_successful"], b["n_successful"],
                     POOLED_ATOL * (a.get("n_evaluated") or 1), failures)
        for key, atol in (("per_seed_asr", PER_SEED_ATOL),
                          ("per_seed_rotation", ROTATION_ATOL)):
            av, bv = a.get(key), b.get(key)
            if av is None or bv is None:
                continue
            for i, (x, y) in enumerate(zip(av, bv)):
                _compare(f"{key}[{i}]", x, y, atol, failures)
        am, bm = a.get("metrics", {}), b.get("metrics", {})
        if isinstance(am, dict) and isinstance(bm, dict):
            for key in set(am) & set(bm):
                if "quer" in key and isinstance(am[key], (int, float)) and isinstance(bm[key], (int, float)):
                    _compare_rel(key, am[key], bm[key], QUERY_RTOL, failures)

    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("fresh")
    args = parser.parse_args()

    with open(args.archive) as fh:
        archive = json.load(fh)
    with open(args.fresh) as fh:
        fresh = json.load(fh)
    failures = compare(archive, fresh)

    if failures:
        print(f"\ncompare_results: FAIL ({len(failures)} discrepancies)")
        sys.exit(1)
    print("\ncompare_results: PASS (within pre-registered tolerances)")


if __name__ == "__main__":
    main()
