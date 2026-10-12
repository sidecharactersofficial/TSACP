import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
import json

def run():
    print("Loading data...")
    # Adjust path if run fromtascp-experiments/experiments/exp1_momentum_separation
    data_path = "../../../data/processed/unsw_nb15/unsw_nb15_processed.parquet"
    if not os.path.exists(data_path):
        data_path = "../../data/processed/unsw_nb15/unsw_nb15_processed.parquet"
    if not os.path.exists(data_path):
        data_path = "../data/processed/unsw_nb15/unsw_nb15_processed.parquet"
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} rows.")

    label_col = 'label' if 'label' in df.columns else 'Label'
    feature_cols = [c for c in df.columns if c not in ['session_id', label_col]]
    labels = df[label_col].values
    session_ids = df['session_id'].values
    X = df[feature_cols].values

    # Shift by 1 row to get previous step
    print("Computing M_t...")
    X_prev = np.roll(X, 1, axis=0)
    session_prev = np.roll(session_ids, 1)

    # Valid steps: session_id matches previous row (i.e., part of the same session trajectory, |W| > 1)
    valid_mask = (session_ids == session_prev)
    
    # We ignore the first row entirely because it wraps around, but roll with valid_mask handles it (session_prev[0] will be session[-1], highly unlikely to match)
    valid_mask[0] = False

    X_curr_valid = X[valid_mask]
    X_prev_valid = X_prev[valid_mask]
    labels_valid = labels[valid_mask]

    # Compute cosine similarity
    norm_curr = np.linalg.norm(X_curr_valid, axis=1)
    norm_prev = np.linalg.norm(X_prev_valid, axis=1)

    # Avoid division by zero
    nonzero = (norm_curr > 0) & (norm_prev > 0)
    M_t = np.zeros(len(X_curr_valid))
    M_t[nonzero] = np.sum(X_curr_valid[nonzero] * X_prev_valid[nonzero], axis=1) / (norm_curr[nonzero] * norm_prev[nonzero])
    M_t = np.clip(M_t, -1.0, 1.0)

    benign_M_t = M_t[labels_valid == 0]
    attack_M_t = M_t[labels_valid == 1]
    
    print(f"Valid steps: {len(M_t)} (Benign: {len(benign_M_t)}, Attack: {len(benign_M_t)})")

    # Save results directory
    out_dir = "../../results"
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    np.save(os.path.join(out_dir, "tables", "benign_mt.npy"), benign_M_t)

    # Compute Stats
    stats = {
        "benign_mean": float(np.mean(benign_M_t)) if len(benign_M_t) > 0 else 0,
        "benign_var": float(np.var(benign_M_t)) if len(benign_M_t) > 0 else 0,
        "attack_mean": float(np.mean(attack_M_t)) if len(attack_M_t) > 0 else 0,
        "attack_var": float(np.var(attack_M_t)) if len(attack_M_t) > 0 else 0,
    }
    
    print("Stats:")
    print(f"  Benign Mean M_t: {stats['benign_mean']:.4f} (Var: {stats['benign_var']:.4f})")
    print(f"  Attack Mean M_t: {stats['attack_mean']:.4f} (Var: {stats['attack_var']:.4f})")

    # AUC
    auc = roc_auc_score(labels_valid, M_t)
    stats["auc"] = float(auc)
    print(f"  AUC (Thresholding M_t): {auc:.4f}")

    # Save Results
    os.makedirs("../../results/tables", exist_ok=True)
    os.makedirs("../../results/figures", exist_ok=True)
    
    with open("../../results/tables/exp1_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Plot Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(benign_M_t, bins=50, alpha=0.5, density=True, label='Benign', color='blue')
    plt.hist(attack_M_t, bins=50, alpha=0.5, density=True, label='Attack', color='red')
    plt.xlabel('Momentum $M_t$ (Cosine Similarity)')
    plt.ylabel('Density')
    plt.title('Distribution of Momentum $M_t$ (UNSW-NB15)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("../../results/figures/exp1_momentum_distribution.png", dpi=300)
    plt.close()
    
    print("Saved results to tascp-experiments/results/")

if __name__ == '__main__':
    run()
