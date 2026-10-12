import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.tascp_eval.models.nids_classifier import NIDSClassifier

def train():
    print("Loading CICIDS2017 dataset...")
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed", "cicids2017", "cicids2017_processed.parquet"))
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find dataset at {data_path}")
    
    print("Reading parquet (dtype=float32 to save memory)...")
    df = pd.read_parquet(data_path)
    label_col = 'label' if 'label' in df.columns else 'Label'
    feature_cols = [c for c in df.columns if c not in ['session_id', label_col]]
    n_features = len(feature_cols)
    print(f"Features: {n_features}, Total samples: {len(df)}")
    
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    y = df[label_col].values.astype(np.float32).reshape(-1, 1)
    del df
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y.ravel())
    del X
    
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)
    del X_train, X_test, y_train, y_test
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    
    model = NIDSClassifier(input_dim=n_features)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 5
    print(f"Training for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        preds = (torch.sigmoid(test_outputs) > 0.5).numpy().astype(int)
        
    acc = accuracy_score(y_test_t.numpy(), preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_t.numpy(), preds, average='binary')
    cm = confusion_matrix(y_test_t.numpy(), preds)
    
    print("\n--- Evaluation on Test Set ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    if acc < 0.6:
        print("WARNING: Classifier accuracy is poor. The model is near-random.")
    
    os.makedirs(os.path.join(os.path.dirname(__file__), "weights"), exist_ok=True)
    save_path = os.path.join(os.path.dirname(__file__), "weights", "cicids2017_nids.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to weights/cicids2017_nids.pth")

if __name__ == "__main__":
    train()
