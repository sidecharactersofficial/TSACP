import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import xgboost as xgb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

def load_data():
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "unsw_nb15", "unsw_nb15_processed.parquet")
    df = pd.read_parquet(data_path)
    label_col = 'label' if 'label' in df.columns else 'Label'
    feature_cols = [c for c in df.columns if c not in ['session_id', label_col]]
    X = df[feature_cols].values.astype(np.float32)
    np.nan_to_num(X, copy=False)
    y = df[label_col].values
    del df
    return X, y, feature_cols

def train_xgboost(X_train, y_train, X_test, y_test, feature_cols):
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary')
    cm = confusion_matrix(y_test, preds)

    print(f"\n=== XGBoost UNSW-NB15 ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    model_path = os.path.join(PROJECT_ROOT, "src", "tascp_eval", "models", "weights", "unsw_nb15_xgboost.json")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    return model, acc, f1, cm

class SurrogateMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        return self.net(x)

def train_surrogate(xgb_model, X_train, feature_cols, epochs=50):
    print("\nTraining surrogate neural network to mimic XGBoost...")
    y_proba = xgb_model.predict_proba(X_train)[:, 1]
    y_log_odds = np.log(np.clip(y_proba, 1e-7, 1 - 1e-7) / np.clip(1 - y_proba, 1e-7, 1 - 1e-7))
    y_log_odds = y_log_odds.astype(np.float32)

    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_log_odds).unsqueeze(1)

    surrogate = SurrogateMLP(input_dim=len(feature_cols))
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    surrogate.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = surrogate(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    surrogate.eval()
    with torch.no_grad():
        pred_logits = surrogate(X_t)
        pred_proba = torch.sigmoid(pred_logits)
        pred_labels = (pred_proba > 0.5).numpy().astype(int).squeeze()
        xgb_labels = (y_proba > 0.5).astype(int)
        agreement = (pred_labels == xgb_labels).mean()
    print(f"Surrogate-XGBoost label agreement: {agreement:.4f}")
    return surrogate

if __name__ == "__main__":
    X, y, feature_cols = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Free test split memory
    del X, y
    xgb_model, acc, f1, cm = train_xgboost(X_train, y_train, X_test, y_test, feature_cols)
    del X_test, y_test
    surrogate = train_surrogate(xgb_model, X_train, feature_cols, epochs=50)
    del X_train, y_train

    # Verify surrogate
    print("\n=== Verification: Surrogate vs XGBoost on benign subset ===")
    Xb, yb, _ = load_data()
    benign_mask = (yb == 0)
    X_benign = Xb[benign_mask]
    del Xb, yb
    X_benign_t = torch.from_numpy(X_benign[:5000])
    with torch.no_grad():
        surr_logits = surrogate(X_benign_t)
        surr_preds = (torch.sigmoid(surr_logits) > 0.5).numpy().astype(int).squeeze()
    xgb_preds = xgb_model.predict(X_benign[:5000])
    # Check how many XGBoost says 0
    print(f"XGBoost benign predictions (0=benign): {(xgb_preds==0).mean():.4f}")
    print(f"Surrogate benign predictions (0=benign): {(surr_preds==0).mean():.4f}")
    print(f"Agreement on benign sample: {(surr_preds==xgb_preds).mean():.4f}")
