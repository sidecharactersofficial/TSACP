# Hardware and environment

## Reproduction interpreter (this artifact)

The archived results under `results/` and the numbers in the paper were
produced from the following environment.

| Item | Value |
| :--- | :--- |
| OS | Linux (WSL2), kernel `6.18.33.2-microsoft-standard-WSL2`, glibc 2.43 |
| Python | CPython 3.14.6 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB |
| CPU | 20 logical cores |
| RAM | 8.1 GB |
| Torch build | CUDA 12.8 (`torch==2.9.1`) |

> `GPU`/`CPU`/`RAM` above are the host on which the shipped archives were
> regenerated. The original paper runs also used a CUDA 12.x torch build; the
> exact original host is not asserted here beyond that fact. No number in this
> artifact depends on the GPU model.

## Software

Exact versions are pinned in two places:

- `requirements.txt` — direct dependencies (the canonical install manifest).
- `environment/requirements.lock` — full transitive closure, regenerated with
  `python scripts/gen_lock.py`.

Key versions: `torch==2.9.1`, `numpy==2.4.0`, `pandas==2.3.3`,
`pyarrow==24.0.0`, `scipy==1.16.3`, `scikit-learn==1.8.0`, `xgboost==3.3.0`,
`matplotlib==3.10.8`, `seaborn==0.13.2`, `pytest==9.0.2`.

## Container

`environment/Dockerfile` builds the CPU reproduction image and runs
`make smoke` by default. See the file header for the GPU variant.

## Memory notes

The white-box experiments (`exp5`, `exp6`, `exp8`) stream the UNSW-NB15
parquet row-group-wise rather than materialising the full 2.5M-row frame.
Loading the full frame was observed to be OOM-killed (exit 137) on an 8 GB
host. All run scripts are written for the 8 GB envelope.
