# Citation

## Artifact Version

Artifact release: `tascp-artifact v1.0.2`

Artifact commit: `270807a` (see `git log`)

## Associated Paper

**Singularity: The Illusion of Dynamic Security**
IEEE Symposium on Security and Privacy (S&P)

## How to Cite This Artifact

If you reference this artifact, cite the associated paper and link to the
anonymized repository.

## Component Citation

Individual components may be cited separately:

- **Defense under test**: `src/tascp_eval/defenses/tascp.py`
- **Adaptive BPDA attack**: `src/tascp_eval/attacks/bpda_adaptive.py`
- **Black-box attacks**: `src/tascp_eval/attacks/nes.py`, `hopskipjump.py`
- **Reference math**: `src/tascp_eval/reference_math.py`
- **Machine-checked formal verification (Lean 4)**: `leanproofs/ArchimedesProofs/Basic.lean`
- **Mathematical audit report**: `leanproofs/MATH_AUDIT_REPORT.md`
- **Math verification test suite**: `leanproofs/verify_math.py`
- **Evaluation framework**: `src/tascp_eval/`
- **Dataset preparation**: `data/scripts/preprocess.py`, `data/ton_iot/prepare_ton_iot.py`
- **Rust fast loop + microbenchmark**: `tascp-proto/fast-loop/`

## Versioning Policy

This artifact follows the paper's review cycle. Post-review updates and
proof formalizations are tagged under semantic versioning (`v1.0`, `v1.1`, `v1.0.2`).
