# Testing

## Test Files

```
tests/
├── __init__.py
├── test_momentum.py                          # TODO — stub only
├── test_penetration.py                       # TODO — stub only
├── test_rotation.py                          # TODO — stub only
└── test_reference_math_matches_prototype.py  # Implemented — drift detection
```

## Implemented Tests

`test_reference_math_matches_prototype.py` validates that the Python
reference math (`src/tascp_eval/reference_math.py`) matches the Rust
prototype implementation (`tascp-proto/fast-loop/src/`):

| Test | What it checks |
|------|----------------|
| `test_penetration_converged_pgd` | Penetration = 1.0 when all steps aligned (M_i=1) |
| `test_penetration_benign_traffic` | Penetration = 0.0 when no alignment (M_i=0) |
| `test_penetration_window_size_invariance` | Penetration independent of window size |

These are **drift-detection tests**: if the reference math changes, the Rust
prototype must be updated to match.

## Stubbed Tests (TODO)

`test_momentum.py`, `test_rotation.py`, and `test_penetration.py` are
placeholders awaiting unit tests for the individual `src/tascp_eval/*.py`
modules.

## Running Tests

```bash
make test
# or
pytest tests/
```

## Test Data

The drift-detection tests use synthetic data (fixed windows of velocity
vectors), not real traffic. No test fixtures or external datasets are needed.
