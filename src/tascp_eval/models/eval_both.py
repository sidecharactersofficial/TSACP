import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.tascp_eval.models.nids_classifier import NIDSClassifier

# --- UNSW-NB15 ---
print('=== UNSW-NB15 ===')
df = pd.read_parquet('data/processed/unsw_nb15/unsw_nb15_processed.parquet')
label_col = 'label' if 'label' in df.columns else 'Label'
feature_cols = [c for c in df.columns if c not in ['session_id', label_col]]
X = df[feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0)
y = df[label_col].values
del df
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
del X
model = NIDSClassifier(input_dim=len(feature_cols))
model.load_state_dict(torch.load('src/tascp_eval/models/weights/unsw_nb15_nids.pth'))
model.eval()
with torch.no_grad():
    preds = (torch.sigmoid(model(torch.from_numpy(X_test))) > 0.5).numpy().astype(int)
acc = accuracy_score(y_test, preds)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary')
cm = confusion_matrix(y_test, preds)
print(f'Accuracy:  {acc:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall:    {recall:.4f}')
print(f'F1 Score:  {f1:.4f}')
print(f'CM:\n{cm}')

# --- CICIDS2017 ---
print('\n=== CICIDS2017 ===')
df = pd.read_parquet('data/processed/cicids2017/cicids2017_processed.parquet')
label_col = 'label'
feature_cols = [c for c in df.columns if c not in ['session_id', label_col]]
X = df[feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0)
y = df[label_col].values
del df
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
del X
model = NIDSClassifier(input_dim=len(feature_cols))
model.load_state_dict(torch.load('src/tascp_eval/models/weights/cicids2017_nids.pth'))
model.eval()
with torch.no_grad():
    preds = (torch.sigmoid(model(torch.from_numpy(X_test))) > 0.5).numpy().astype(int)
acc = accuracy_score(y_test, preds)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary')
cm = confusion_matrix(y_test, preds)
print(f'Accuracy:  {acc:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall:    {recall:.4f}')
print(f'F1 Score:  {f1:.4f}')
print(f'CM:\n{cm}')
