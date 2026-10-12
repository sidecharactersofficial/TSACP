# Exp 18: In-process Rust microbenchmark of the fast-loop hot path

Removes the HTTP-overhead caveat on the paper's 7.08 ms/query latency figure
(exp7 measured the dockerized prototype over HTTP + JSON).

Criterion benchmarks (in-process, release build, N_DIM=10):

- hot_path_single_query: full defense computation per query
  (velocity + momentum + window push/pop + penetration integral + Givens rotation)
- momentum: single-step cosine
- rotation: Givens rotation application

## Result (this machine, 20-core, release)

| benchmark | median |
|---|---|
| hot_path_single_query | ~54.6 ns |
| momentum | ~6.9 ns |
| rotation | ~53.3 ns |

## Reading

The in-process defense compute is ~55 ns/query. The 7.08 ms figure from exp7
is therefore dominated (~130,000x) by HTTP transport + serialization +
container scheduling, not by the defense algorithm. Any claim that TASCP's
per-query cost is prohibitive must be made against the in-process number; the
HTTP figure measures the deployment plumbing. (A production in-process
deployment would still pay feature-collection costs not modeled here.)
