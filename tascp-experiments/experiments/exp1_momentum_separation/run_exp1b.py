import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Paths setup
EXP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", "..", ".."))
TASCP_EXP_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
sys.path.append(TASCP_EXP_ROOT)
sys.path.append(PROJECT_ROOT)

from src.tascp_eval.reference_math import compute_momentum
from src.tascp_eval.models.nids_classifier import NIDSClassifier
from src.tascp_eval.attacks.pgd import pgd_attack, load_config

def run_exp1b():
    print("Loading UNSW-NB15 dataset...")
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "unsw_nb15", "unsw_nb15_processed.parquet")
    df = pd.read_parquet(data_path)
    
    label_col = 'label' if 'label' in df.columns else 'Label'
    feature_cols = [c for c in df.columns if c not in ['session_id', label_col]]
    
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    y = df[label_col].values
    
    # We only want benign traffic
    benign_mask = (y == 0)
    X_benign = X[benign_mask]
    
    print("Loading trained NIDS classifier...")
    model_path = os.path.join(PROJECT_ROOT, "src", "tascp_eval", "models", "weights", "unsw_nb15_nids.pth")
    model = NIDSClassifier(input_dim=len(feature_cols))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Find correctly classified benign points
    X_benign_t = torch.tensor(X_benign, dtype=torch.float32)
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_benign_t)) > 0.5).numpy().astype(int).squeeze()
        
    correct_mask = (preds == 0)
    X_correct = X_benign_t[correct_mask]
    
    # Sample a subset to run PGD (e.g., 2000 points)
    n_samples = min(2000, len(X_correct))
    indices = np.random.choice(len(X_correct), n_samples, replace=False)
    X_sample = X_correct[indices]
    y_target = torch.ones(n_samples, 1, dtype=torch.float32) # Target is attack (1)
    
    print(f"Running PGD variants on {n_samples} benign points...")
    config = load_config()
    
    variants = {
        'Standard PGD': config,
        'MIM (mu=1.0)': {**config, 'mu': 1.0},
        'Decaying PGD (gamma=0.9)': {**config, 'gamma': 0.9}
    }
    
    print("Loading benign M_t from exp1...")
    benign_mt_path = os.path.join(TASCP_EXP_ROOT, "results", "tables", "benign_mt.npy")
    benign_M_t = np.load(benign_mt_path)
    benign_mean = float(np.mean(benign_M_t))
    
    print("\n--- Exp 1b Results ---")
    print(f"Benign (Historical) Mean M_t: {benign_mean:.4f}\n")
    
    for name, v_config in variants.items():
        print(f"--- Running {name} ---")
        trajectories = pgd_attack(model, X_sample, y_target, v_config)
        traj_arr = torch.stack(trajectories, dim=1).numpy()
        
        pgd_M_t = []
        for i in range(n_samples):
            sample_traj = traj_arr[i]
            for t in range(1, len(sample_traj)):
                x_t = sample_traj[t]
                x_prev = sample_traj[t-1]
                m = compute_momentum(x_t, x_prev)
                pgd_M_t.append(m)
                
        pgd_M_t = np.array(pgd_M_t)
        
        pgd_mean = float(np.mean(pgd_M_t))
        pgd_var = float(np.var(pgd_M_t))
        print(f"  {name} Mean M_t: {pgd_mean:.4f} (Var: {pgd_var:.4f})")
        
        all_M_t = np.concatenate([benign_M_t, pgd_M_t])
        all_labels = np.concatenate([np.zeros(len(benign_M_t)), np.ones(len(pgd_M_t))])
        
        auc = roc_auc_score(all_labels, all_M_t)
        print(f"  AUC ({name} vs Benign): {auc:.4f}\n")

if __name__ == "__main__":
    run_exp1b()
