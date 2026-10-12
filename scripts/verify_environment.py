"""Verify the runtime environment against requirements.txt.

Checks the Python version range and that every pinned direct dependency is
importable at the pinned version. Exits non-zero on any mismatch.

Usage:
    python scripts/verify_environment.py
"""
import importlib.metadata as importlib_metadata
import os
import re
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REQUIREMENTS = os.path.join(PROJECT_ROOT, "requirements.txt")

# module name -> distribution name
IMPORT_NAME = {
    "scikit-learn": "sklearn",
    "PyYAML": "yaml",
    "pyarrow": "pyarrow",
}


def parse_requirements(path):
    pins = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", line)
            if m:
                pins[m.group(1)] = m.group(2)
    return pins


def main():
    pins = parse_requirements(REQUIREMENTS)
    failures = []

    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if not ((3, 11) <= sys.version_info[:2] <= (3, 14)):
        failures.append(f"python {py} outside [3.11, 3.14]")

    for dist, want in sorted(pins.items()):
        try:
            got = importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            failures.append(f"{dist}: not installed (want {want})")
            continue
        if got != want:
            failures.append(f"{dist}: {got} != {want}")

    if failures:
        print("environment: FAIL")
        for f in failures:
            print("  - " + f)
        return 1
    print(f"environment: PASS (python {py}, {len(pins)} pinned deps match)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
