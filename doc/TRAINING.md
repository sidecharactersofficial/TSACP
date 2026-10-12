# Training

## Classifiers

### UNSW-NB15 (MLP)

| | |
|---|---|
| Script | `src/tascp_eval/models/train.py` |
| Architecture | 3-layer MLP: Linear(43,64)→ReLU→Linear(64,64)→ReLU→Linear(64,1) |
| Optimizer | Adam, lr=1e-3 |
| Loss | BCEWithLogitsLoss |
| Epochs | 5 |
| Batch size | 2048 |
| Train/test | 80/20 stratified |
| Input | `data/processed/unsw_nb15/unsw_nb15_processed.parquet` (43 features) |
| Output weights | `src/tascp_eval/models/weights/unsw_nb15_nids.pth` |
| Accuracy | 99.05% |
| F1 Score | 0.9631 |

### CICIDS2017 (MLP)

| | |
|---|---|
| Script | `src/tascp_eval/models/train_cicids2017.py` |
| Architecture | Same 3-layer MLP as UNSW-NB15 |
| Hyperparameters | Identical to UNSW-NB15 |
| Input | `data/processed/cicids2017/cicids2017_processed.parquet` |
| Output weights | `src/tascp_eval/models/weights/cicids2017_nids.pth` |
| Accuracy | 97.02% |
| F1 Score | 0.9447 |

### UNSW-NB15 (XGBoost)

| | |
|---|---|
| Script | `src/tascp_eval/models/xgboost_classifier.py` |
| Model | `XGBClassifier(n_estimators=200, max_depth=6, lr=0.1, subsample=0.8)` |
| Train/test | 80/20 stratified (same split as MLP) |
| Input | Same UNSW-NB15 processed parquet |
| Output weights | `src/tascp_eval/models/weights/unsw_nb15_xgboost.json` |
| Accuracy | 99.44% |
| F1 Score | 0.9776 |

XGBoost also trains a **surrogate MLP** (128×128, MSE loss on XGBoost
log-odds, 50 epochs, 200k subset). The surrogate is used for gradient
computation in black-box attacks; it achieves 87.4% label agreement with
XGBoost overall and 100% on benign samples.

## Dataset Preprocessing

Both datasets are preprocessed by `data/scripts/preprocess.py`:
- Feature selection (43 features for UNSW-NB15, matching set for CICIDS2017)
- MinMax scaling to [0, 1]
- NaN imputation (fill with 0)
- Saved as parquet

```bash
# Preprocess both datasets
python data/scripts/preprocess.py

# Download raw data (if needed)
python data/scripts/download_unsw_nb15.py
python data/scripts/download_cicids2017.py
```

## Weights

| File | Source | Size |
|------|--------|------|
| `unsw_nb15_nids.pth` | `train.py` | 31 KB |
| `cicids2017_nids.pth` | `train_cicids2017.py` | 40 KB |
| `unsw_nb15_xgboost.json` | `xgboost_classifier.py` | 972 KB |

Models are small (<1 MB each) and checked into the repository for
reproducibility.
