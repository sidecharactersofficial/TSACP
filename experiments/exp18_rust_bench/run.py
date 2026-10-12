"""Exp 18: in-process Rust microbenchmark of the fast-loop hot path.

Drives the Criterion benchmarks in ``tascp-proto/fast-loop/benches/hot_path.rs``
and aggregates ``target/criterion/*/new/estimates.json`` into
``results/logs/exp18_rust_bench/bench_summary.json`` (point estimate, standard
error, 95% CI), the same schema as the shipped archive.

Usage:
    python -m experiments.exp18_rust_bench.run            # run benchmarks
    python -m experiments.exp18_rust_bench.run --no-run   # only aggregate
"""
import argparse
import json
import os
import subprocess
import sys

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
PROTO_DIR = os.path.join(PROJECT_ROOT, "tascp-proto", "fast-loop")
CRITERION = os.path.join(PROTO_DIR, "target", "criterion")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "logs", "exp18_rust_bench")
BENCHES = ["hot_path_single_query", "momentum", "rotation"]


def run_cargo_bench():
    print("Running cargo bench (release) ...")
    subprocess.run(
        ["cargo", "bench", "--bench", "hot_path", "--", "--warm-up-time", "1", "--measurement-time", "3"],
        cwd=PROTO_DIR,
        check=True,
    )


def aggregate():
    summary = {}
    for name in BENCHES:
        est = os.path.join(CRITERION, name, "new", "estimates.json")
        if not os.path.exists(est):
            print(f"WARNING: missing {est}", file=sys.stderr)
            continue
        with open(est) as fh:
            data = json.load(fh)
        summary[name] = {
            "point_estimate": data["median"]["point_estimate"],
            "standard_error": data["median"]["standard_error"],
            "confidence_interval": {
                "confidence_level": data["median"]["confidence_interval"]["confidence_level"],
                "lower_bound": data["median"]["confidence_interval"]["lower_bound"],
                "upper_bound": data["median"]["confidence_interval"]["upper_bound"],
            },
        }
    return summary


def main():
    ap = argparse.ArgumentParser(description="exp18 Rust hot-path microbenchmark")
    ap.add_argument("--no-run", action="store_true", help="skip cargo bench, only aggregate existing criterion output")
    args = ap.parse_args()
    if not args.no_run:
        run_cargo_bench()
    summary = aggregate()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "bench_summary.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    for name, vals in summary.items():
        print(f"{name}: {vals['point_estimate']:.2f} ns")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
