# Provenance

This document records the methodological evolution of the TASCP evaluation
suite. Superseded implementations and narrative revisions are retained for
scientific provenance and should not be read as the canonical result.

## Evolution Timeline

```
Initial claim: momentum-conditioned Givens rotation masks gradients
       │
       ▼
First white-box evaluation (exp5)
       │
       ├── Naive attacker: TASCP appears to help
       ├── Implementation bug found in the adaptive path
       ▼
Adaptive BPDA attacker (exp5, bug fixed)
       │
       ├── Adaptive ASR 53.60% vs 10.50% undefended
       ├── Conclusion downgraded: obfuscation, not a barrier
       ▼
Ablations (exp8) + multi-seed (exp13)
       │
       ├── Basis coupling isolated as the gradient leak
       ├── Basis-fixed collapses to baseline (p = 0.836)
       ▼
Black-box audits (exp9–exp12)
       │
       ├── Corollary 3: antithetic NES never fires the trigger
       ├── CICIDS2017 amplification (15.05% → 87.40%) disambiguated
       ▼
Reviewer-response suite (exp19–exp24, session 5)
       │
       ├── Channel is not rotation-specific (exp19)
       ├── Coupling magnitude does not predict exploitability (exp20, null)
       ├── Trigger blindness + stationarity are architecture-universal (exp22)
       └── Third dataset NF-ToN-IoT-v2 transfers the mechanism (exp24)
       ▼
Final position: cautionary result, not a defense
```

## Superseded / corrected components

### exp5 narrative (narrative revision)
`experiments/exp5_adaptive_bpda_attack/exp5_summary.md` contains two
conclusions: an earlier paragraph says TASCP "provides meaningful protection"
against the naive attacker, and a later paragraph says an implementation bug
was fixed and TASCP "degrades robustness". The **later** statement is
canonical; the earlier text is retained as the pre-fix narrative. The headline
values used by the paper are the fixed-run values (adaptive 53.60%).

### exp5 `diagnose_xgboost_naive.py`
Diagnostic only; superseded by `run_xgboost.py` for the XGBoost surrogate arm.

### exp8 `run_cicids2017.py` vs `run_cicids2017_budget100.py`
The original CICIDS budget choice was corrected to budget=100; the corrected
driver is canonical and its summary is
`results/cicids2017_budget100_ablations.md`.

### exp20 coupling audit (honest null)
The hoped-for monotone "coupling magnitude predicts exploitability" figure did
**not** materialize. The canonical claim is the weaker, supported one:
`C = 0` exactly identifies the closed channel; magnitude does not predict
effect (Spearman ±0.27, p > 0.14). See `results/MANIFEST_session5_v5.md`.

### exp9 trigger scope (`exp15_cor3_scope`)
The original Corollary 3 statement over-claimed. exp15 narrows it: all
non-antithetic attackers fire the trigger (2.5–65%) but only decision-based
refinement attacks (Square/Boundary) gain real ASR. The corollary's scope is
now explicitly restricted to antithetic score-based estimators.

### Legacy prototype vs reference math
`tascp-proto/` (Rust fast-loop) and `src/tascp_eval/reference_math.py` (Python)
are kept in sync by drift tests. The Python reference math is canonical for the
paper; the Rust prototype is the source of the exp18 microbenchmark.

## Dataset provenance

- UNSW-NB15, CICIDS2017: `data/scripts/preprocess.py`.
- NF-ToN-IoT-v2: `data/ton_iot/PROVENANCE.md`, `prepare_ton_iot.py`,
  `train_ton_iot.py`.

## Decision Records

See `doc/adr/`:
- ADR-001: Stateful evaluation and query-fed defense history
- ADR-002: Multi-seed statistical re-evaluation with pre-registered tolerances
- ADR-003: Independent numerical validation of the reference math
- ADR-004: Canonical vs fresh reproduction outputs
