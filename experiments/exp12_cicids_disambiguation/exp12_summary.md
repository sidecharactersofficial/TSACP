# EXP12 — CICIDS2017 Disambiguation (real data)

Run: `python -m experiments.exp12_cicids_disambiguation.run cicids2017`, exit 0.
Table: `results/tables/exp12_hsj_disambiguation_cicids2017.csv`; metrics:
`results/logs/exp12_cicids_disambiguation/metrics.json`.

## HSJ on CICIDS2017 (2.0M queries per condition)

| condition | ASR | mean Q-to-success | trigger fire (all queries) |
|---|---:|---:|---:|
| Undefended (control) | 15.05% | 56.1 | 0.0% |
| Δθ = 0 wrapper (control) | 15.05% | 56.1 | 0.0% |
| TASCP (Naive) | 87.40% | 62.1 | 99.1% |
| Basis fixed | 25.85% | 58.6 | 97.1% |
| Angle detached | 87.40% | 62.1 | 99.1% |
| Random-rotation placebo (random plane, same angle) | 33.75% | 32.5 | 97.4% |

## Interpretation

1. **The amplification is real, not a wrapper artifact**: both controls
   (undefended, Δθ=0) sit at exactly 15.05% — merely wrapping the model costs
   nothing. TASCP's 87.40% is driven by the rotation itself (5.8× the
   baseline).
2. **Basis fixation removes most of the harm** (87.4% → 25.9%): the
   amplification is carried primarily by the data-dependent rotation plane,
   consistent with exp8/exp13/exp17.
3. **The placebo (random plane, same measured angle distribution) is NOT
   equivalent to TASCP** (33.75% vs 87.40%): a random plane plus a real angle
   only recovers ~40% of the amplification. What matters is the angle applied
   in the *trajectory-resolved* plane — the plane is not incidental, but the
   magnitude/direction of the rotation on decision-relevant directions is
   what the amplification rides on.
4. Angle-detached == naive (87.40%, fire 99.1%): HSJ never used
   straight-through gradients (decision-based), so detaching the angle
   changes nothing for this attacker — the amplification does not require
   attacker-side gradient access through the state map.

Also runs `basis_fixed_robustness.py` variants for reference pairs — see
exp13's reference-robustness block for the extended random-pair sweep on
both datasets.
