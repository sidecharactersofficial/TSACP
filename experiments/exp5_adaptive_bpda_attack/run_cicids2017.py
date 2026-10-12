import os
import sys
import numpy as np
import pandas as pd
import torch

EXP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.tascp_eval.models.nids_classifier import NIDSClassifier
from src.tascp_eval.defenses.tascp import TASCPWrapper
from src.tascp_eval.baselines.dacm import DACMWrapper
from src.tascp_eval.attacks.pgd import load_config
from src.tascp_eval.attacks.bpda_adaptive import stateful_pgd_attack

def run_exp5_cicids2017():
    print("Loading CICIDS2017 dataset...")
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "cicids2017", "cicids2017_processed.parquet")
    df = pd.read_parquet(data_path)
    
    label_col = 'label' if 'label' in df.columns else 'Label'
    feature_cols = [c for c in df.columns if c not in ['session_id', label_col]]
    n_features = len(feature_cols)
    
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    y = df[label_col].values
    del df
    
    benign_mask = (y == 0)
    X_benign = X[benign_mask]
    del X
    print(f"Benign samples: {len(X_benign)}")
    
    print("Loading trained CICIDS2017 NIDS classifier...")
    model_path = os.path.join(PROJECT_ROOT, "src", "tascp_eval", "models", "weights", "cicids2017_nids.pth")
    base_model = NIDSClassifier(input_dim=n_features)
    base_model.load_state_dict(torch.load(model_path))
    base_model.eval()
    
    X_benign_t = torch.tensor(X_benign, dtype=torch.float32)
    with torch.no_grad():
        preds = (torch.sigmoid(base_model(X_benign_t)) > 0.5).numpy().astype(int).squeeze()
    
    correct_mask = (preds == 0)
    X_correct = X_benign_t[correct_mask]
    print(f"Correctly classified benign: {len(X_correct)}")
    
    n_samples = min(2000, len(X_correct))
    np.random.seed(42)
    indices = np.random.choice(len(X_correct), n_samples, replace=False)
    X_sample = X_correct[indices]
    y_target = torch.zeros(n_samples, 1, dtype=torch.float32)
    print(f"Sampled {n_samples} points for attack")
    
    config = load_config()
    iterations = config.get('iterations', 20)
    
    conditions = [
        ("Undefended", base_model, False),
        ("DACM (Static)", DACMWrapper(base_model), False),
        ("TASCP (Naive)", TASCPWrapper(base_model, window_size=5, max_theta=np.pi/2, tau=0.0, gamma=50.0), False),
        ("TASCP (Adaptive BPDA)", TASCPWrapper(base_model, window_size=5, max_theta=np.pi/2, tau=0.0, gamma=50.0), True)
    ]
    
    print(f"\nRunning Exp 5: Adaptive BPDA Attack on CICIDS2017 ({n_samples} samples, {iterations} max queries)")
    
    results = {}
    
    for name, model, is_adaptive in conditions:
        print(f"\n--- Testing {name} ---")
        
        _, success_flags, queries = stateful_pgd_attack(model, X_sample, y_target, config, adaptive=is_adaptive)
        
        success_rate = success_flags.float().mean().item()
        
        if success_flags.any():
            mean_queries = queries[success_flags].mean().item()
        else:
            mean_queries = float('nan')
            
        print(f"Attack Success Rate (ASR): {success_rate:.4f}")
        print(f"Mean Queries to Success:   {mean_queries:.2f}")
        
        results[name] = {
            'asr': success_rate,
            'queries': mean_queries
        }
    
    print("\n=== EXP 5 SUMMARY (CICIDS2017) ===")
    undefended_q = results["Undefended"]["queries"]
    
    for name in results:
        asr = results[name]['asr']
        q = results[name]['queries']
        q_factor = q / undefended_q if not np.isnan(q) else float('inf')
        print(f"{name:25s} | ASR: {asr*100:6.2f}% | Mean Q: {q:5.1f} | Q-Factor: {q_factor:.2f}x")

if __name__ == "__main__":
    run_exp5_cicids2017()
