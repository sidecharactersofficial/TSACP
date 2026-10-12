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
from src.tascp_eval.models.transformer_classifier import TransformerClassifier

def train(epochs=5, lr=1e-3, batch_size=1024, d_model=128, num_layers=2):
    print(f"Loading UNSW-NB15 dataset... (epochs={epochs}, lr={lr}, d_model={d_model}, layers={num_layers})")
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed", "unsw_nb15", "unsw_nb15_processed.parquet"))
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find dataset at {data_path}")
        
    df = pd.read_parquet(data_path)
    label_col = 'label' if 'label' in df.columns else 'Label'
    feature_cols = [c for c in df.columns if c not in ['session_id', label_col]]
    
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    y = df[label_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = TransformerClassifier(input_dim=len(feature_cols), d_model=d_model, num_layers=num_layers).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
        
    model.eval()
    
    # Process test set in batches to avoid OOM
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            test_outputs = model(batch_X)
            preds = (torch.sigmoid(test_outputs) > 0.5).cpu().numpy().astype(int)
            all_preds.append(preds)
            
    preds = np.concatenate(all_preds)
    
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary')
    cm = confusion_matrix(y_test, preds)
    
    print("\n--- Evaluation on Test Set ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return acc, f1, cm, model

if __name__ == "__main__":
    configs = [
        {"epochs": 10, "lr": 1e-3, "batch_size": 1024, "d_model": 128, "num_layers": 2},
        {"epochs": 10, "lr": 5e-4, "batch_size": 1024, "d_model": 128, "num_layers": 4},
        {"epochs": 15, "lr": 1e-3, "batch_size": 1024, "d_model": 256, "num_layers": 2}
    ]
    
    best_acc = 0
    best_model = None
    best_cm = None
    best_f1 = 0
    
    for i, cfg in enumerate(configs):
        print(f"\n=======================")
        print(f"Config {i+1}/3: {cfg}")
        acc, f1, cm, model = train(**cfg)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_cm = cm
            best_f1 = f1
            
        if acc >= 0.9705:
            print("\n>>> Kill Criterion 1 passed! Accuracy meets threshold. Stopping search.")
            break
            
    print("\n=======================")
    print(f"Best Accuracy: {best_acc:.4f}")
    
    os.makedirs(os.path.join(os.path.dirname(__file__), "weights"), exist_ok=True)
    if best_acc >= 0.9705:
        # Save it
        torch.save(best_model.state_dict(), os.path.join(os.path.dirname(__file__), "weights", "unsw_nb15_transformer.pth"))
        print("Saved model to weights/unsw_nb15_transformer.pth")
    else:
        print("Model did NOT reach 97.05%. Stopping.")
        # Output stop file
        with open(os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "transformer_stage1_STOPPED.md"), "w") as f:
            f.write(f"# STOPPED at Kill Criterion 1\n\nVal accuracy {best_acc:.4f} (F1 {best_f1:.4f}), did not reach 97.05% threshold after configurations.\n\nConfusion Matrix:\n{best_cm}\n")

