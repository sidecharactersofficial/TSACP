"""Train the NIDSClassifier on the processed NF-ToN-IoT-v2 parquet.

Same architecture (NIDSClassifier, 2x64 MLP) and same training protocol as
src/tascp_eval/models/train.py (Adam 1e-3, BCEWithLogitsLoss, batch 2048,
5 epochs, stratified 80/20 split) so the third-dataset comparison is
architecturally identical to UNSW-NB15 / CICIDS2017.

Output: src/tascp_eval/models/weights/ton_iot_nids.pth
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.tascp_eval.models.nids_classifier import NIDSClassifier  # noqa: E402

DATA = os.path.join(PROJECT_ROOT, "data", "processed", "ton_iot", "ton_iot_processed.parquet")
OUT = os.path.join(PROJECT_ROOT, "src", "tascp_eval", "models", "weights", "ton_iot_nids.pth")


def main():
    import pyarrow.parquet as pq

    print(f"loading {DATA}")
    tbl = pq.read_table(DATA)
    df = tbl.to_pandas()
    label_col = "label" if "label" in df.columns else "Label"
    feature_cols = [c for c in df.columns if c not in ("session_id", label_col, "__index_level_0__")]
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    y = df[label_col].values
    print(f"rows={len(X)} features={len(feature_cols)} pos_rate={y.mean():.4f}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)

    model = NIDSClassifier(input_dim=len(feature_cols))
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=2048, shuffle=True)

    for epoch in range(5):
        model.train()
        tot = 0.0
        for bx, by in loader:
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"epoch {epoch+1}/5 loss {tot/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_te_t)) > 0.5).numpy().astype(int).squeeze()
    acc = accuracy_score(y_te, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_te, preds, average="binary")
    print(f"acc={acc:.4f} precision={p:.4f} recall={r:.4f} f1={f1:.4f}")
    print("confusion matrix:")
    print(confusion_matrix(y_te, preds))

    torch.save(model.state_dict(), OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
