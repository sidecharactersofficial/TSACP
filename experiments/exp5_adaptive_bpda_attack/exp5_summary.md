# Experiment 5 Summary: Adaptive BPDA Attack

**Machine-readable archive:** `results/logs/exp5_adaptive_bpda_attack/metrics_unsw_nb15.json`
(table: `results/tables/exp5_summary_unsw_nb15.csv`). Every number below is
copied from that archive, which also records the seed (42), the attack config
and its SHA-256, the classifier checkpoint SHA-256, the dataset SHA-256,
`n_evaluated`, `n_successful`, and the environment.

This experiment tests TASCP against a white-box PGD attacker, both as a naive
attacker (rotation treated as a fixed barrier) and as a fully adaptive BPDA
attacker (gradients flow through the rotation state via straight-through
estimation).

### Protocol
- Dataset: UNSW-NB15 (processed), 2000 correctly-classified benign flows
  (`n_evaluated = 2000`), seed 42.
- Attack: L2 PGD, `epsilon = 0.1`, `alpha = 0.01`, 20 iterations.
- Wrapper: `window_size = 5`, `theta_max = pi/2`, `tau = 0.0`, `gamma = 50.0`.

### Results (2000 samples, max 20 queries) — from the archive

| Defense | n_successful | ASR | Mean Queries to Success | Mean L2 | Query Factor |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Undefended | 210 | 10.50% | 10.68 | 0.0995 | 1.00x |
| DACM (Static) | 210 | 10.50% | 10.68 | 0.0995 | 1.00x |
| TASCP (Naive) | 459 | 22.95% | 10.18 | 0.0995 | 0.95x |
| TASCP (Adaptive BPDA) | 1072 | 53.60% | 4.26 | 0.0582 | 0.40x |

### Key Findings
1. **TASCP provides no meaningful protection.** The naive attacker already
   raises ASR from 10.50% (undefended) to 22.95%; the static DACM baseline is
   bit-identical to undefended (210/2000), so the increase is attributable to
   the trajectory-dependent rotation, not to wrapping the model.
2. **The adaptive attacker more than doubles it.** Backpropagating through the
   rotation parameters reaches 53.60% ASR (1072/2000) using ~2.5x fewer
   queries (4.26 vs 10.68), i.e. the rotation is not an optimization barrier.
3. **The data-dependent basis is the leak** — see exp8, and
   `results/logs/exp8_adaptive_ablations/metrics_unsw_nb15.json`: fixing the
   basis collapses ASR to 10.30%, statistically indistinguishable from
   undefended (z = -0.207, p = 0.836).

### Reproduce
```bash
python -m experiments.exp5_adaptive_bpda_attack.run unsw_nb15 --n 2000 --seed 42
python -m experiments.exp8_adaptive_ablations.run unsw_nb15 --n 2000 --seed 42
```

> **History.** An earlier revision of this file contained two contradictory
> paragraphs (one claiming the adaptive attacker matched the naive one at
> 53.60%, another claiming a "fixed implementation bug" gave the adaptive
> attacker an advantage). The numbers were hand-typed and not backed by any
> archive. That text is superseded: the archived run above is the single
> source of truth, and the metastasis of the number into the paper should cite
> `results/logs/exp5_adaptive_bpda_attack/metrics_unsw_nb15.json`.
