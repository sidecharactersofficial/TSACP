# Citation

## Artifact Version

Artifact release: `tascp-artifact v1.0`

Artifact commit: `270807a` (see `git log`)

## Associated Paper

[Redacted for double-blind review]

## How to Cite This Artifact

If you reference this artifact, cite the associated paper and link to the
anonymized repository.

## Component Citation

Individual components may be cited separately:

- **Defense under test**: `src/tascp_eval/defenses/tascp.py`
- **Adaptive BPDA attack**: `src/tascp_eval/attacks/bpda_adaptive.py`
- **Black-box attacks**: `src/tascp_eval/attacks/nes.py`, `hopskipjump.py`
- **Reference math**: `src/tascp_eval/reference_math.py`
- **Evaluation framework**: `src/tascp_eval/`
- **Dataset preparation**: `data/scripts/preprocess.py`, `data/ton_iot/prepare_ton_iot.py`
- **Rust fast loop + microbenchmark**: `tascp-proto/fast-loop/`

## Versioning Policy

This artifact follows the paper's review cycle. Code and data will not change
during the review period. Post-acceptance updates will be released as tags.
