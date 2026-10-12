# Dataset Links

Raw datasets are **not** shipped (~2.3 GB). The authoritative inputs for exact
reproduction are the preprocessed parquet tensors pinned in
`manifest/dataset_sha256.txt`.

## Integrity anchors (SHA-256)

Hashes are over the exact tensors consumed by the shipped scripts.

| File | Rows | Cols | SHA-256 |
|------|------|------|---------|
| `data/processed/unsw_nb15/unsw_nb15_processed.parquet` | 2,540,047 | 45 | `080eb491e8240ec2d61f0f43c7fea35fc4dc99b0799931315e7bfe1adf917a43` |
| `data/processed/cicids2017/cicids2017_processed.parquet` | 3,119,345 | 80 | `708b5250f59defe9ee2f3801f9ca84dd10096f4c1a09672c078dcb3a16cf6406` |
| `data/processed/ton_iot/ton_iot_processed.parquet` | 3,388,100 | 42 | `5b6a506f410608c300ad74aff29bc680db540f2d8ac0ca09104203fa00328813` |

Verify with:

```bash
sha256sum -c manifest/dataset_sha256.txt
# or
python3 verification/verify_datasets.py
```

## Sources and preprocessing

### UNSW-NB15

**Script:** `data/scripts/preprocess.py`, download helper `data/scripts/download_unsw_nb15.py`
**Source:** UNSW Canberra (https://research.unsw.edu.au/projects/unsw-nb15-dataset)
**Preprocessing:** drop the 4 IP/Port 5-tuple keys and 2 label columns; 43
features; MinMax scaling to [0, 1]; NaN → 0. The paper's original `d = 49`
count is superseded by the documented 43-feature layout (see
`configs/dataset/unsw_nb15.yaml`).

### CICIDS2017

**Script:** `data/scripts/preprocess.py`, download helper `data/scripts/download_cicids2017.py`
**Source:** University of New Brunswick CIC (https://www.unb.ca/cic/datasets/ids-2017.html)
**Preprocessing:** multi-class → binary; matching feature set; MinMax scaling
to [0, 1]; NaN → 0. 78 effective features.

### NF-ToN-IoT-v2

**Script:** `data/ton_iot/prepare_ton_iot.py`, training `data/ton_iot/train_ton_iot.py`
**Source:** NetFlow ToN-IoT v2 (test split)
**Preprocessing (documented in `data/ton_iot/PROVENANCE.md`):** IPs dropped;
features `log1p` then MinMax normalized; 64% attack rows; classifier accuracy
0.9819. This is the dataset's **test split** (smaller download).

## Manual download fallback

If automatic mirrors are unavailable, download the raw datasets and place them
under `data/raw/<dataset>/`, then run the preprocess/prepare scripts. Public
mirrors occasionally change; from-scratch hashes may differ from the pins above.
For exact reproduction, use the shipped preprocessed parquet files.
