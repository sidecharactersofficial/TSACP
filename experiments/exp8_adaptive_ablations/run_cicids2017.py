import os
import sys
import numpy as np
import pandas as pd
import torch
from math import sqrt

EXP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.tascp_eval.models.nids_classifier import NIDSClassifier
from src.tascp_eval.defenses.tascp import TASCPWrapper
from src.tascp_eval.attacks.pgd import load_config
from src.tascp_eval.attacks.bpda_adaptive import stateful_pgd_attack

def run_exp8_cicids2017():
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
    
    # Instantiate models with same TASCP params as UNSW-NB15 run
    naive = TASCPWrapper(base_model, window_size=5, max_theta=np.pi/2, tau=0.0, gamma=50.0)
    naive.adaptive = False
    
    full_adaptive = TASCPWrapper(base_model, window_size=5, max_theta=np.pi/2, tau=0.0, gamma=50.0)
    full_adaptive.adaptive = True
    
    ablation_grad = TASCPWrapper(base_model, window_size=5, max_theta=np.pi/2, tau=0.0, gamma=50.0)
    ablation_grad.adaptive = True
    ablation_grad.ablation_mode = 'gradient_magnitude'
    
    ablation_basis = TASCPWrapper(base_model, window_size=5, max_theta=np.pi/2, tau=0.0, gamma=50.0)
    ablation_basis.adaptive = True
    ablation_basis.ablation_mode = 'basis_coupling'
    
    ablation_proj = TASCPWrapper(base_model, window_size=5, max_theta=np.pi/2, tau=0.0, gamma=50.0)
    ablation_proj.adaptive = True
    ablation_proj.ablation_mode = 'projection_only'
    
    conditions = [
        ("TASCP (Naive)", naive, False),
        ("TASCP (Full Adaptive)", full_adaptive, True),
        ("TASCP (Ablation 1: Grad Mag)", ablation_grad, True),
        ("TASCP (Ablation 2: Basis)", ablation_basis, True),
        ("TASCP (Ablation 3: Proj Only)", ablation_proj, True)
    ]
    
    print(f"\nRunning Exp 8: Adaptive Ablations on CICIDS2017 ({n_samples} samples, {iterations} max queries)")
    
    results = {}
    
    for name, model, is_adaptive in conditions:
        print(f"\n--- Testing {name} ---")
        
        x_adv, success_flags, queries = stateful_pgd_attack(model, X_sample, y_target, config, adaptive=is_adaptive)
        
        success_rate = success_flags.float().mean().item()
        
        if success_flags.any():
            mean_queries = queries[success_flags].mean().item()
        else:
            mean_queries = float('nan')
        
        mean_l2_dist = torch.norm(x_adv - X_sample, p=2, dim=1).mean().item()
        
        print(f"Attack Success Rate (ASR): {success_rate:.4f}")
        print(f"Mean Queries to Success:   {mean_queries:.2f}")
        print(f"Mean L2 Perturbation:      {mean_l2_dist:.4f}")
        
        if name == "TASCP (Ablation 3: Proj Only)":
            if mean_l2_dist < 1e-5:
                print("WARNING: x_t did not move meaningfully!")
            else:
                print("CONFIRMED: x_t still moved across iterations (L2 dist > 0).")
        
        results[name] = {
            'asr': success_rate,
            'queries': mean_queries,
            'n_success': success_flags.sum().item(),
            'n_total': n_samples,
        }
    
    print("\n=== EXP 8 ABLATIONS SUMMARY (CICIDS2017) ===")
    
    for name in results:
        asr = results[name]['asr']
        q = results[name]['queries']
        print(f"{name:30s} | ASR: {asr*100:6.2f}% | Mean Q: {q:5.1f}")
    
    # Two-proportion z-test: Ablation 2 (Basis) vs Undefended baseline from exp5
    print("\n--- Two-Proportion Z-Test: Basis-Fixed vs Undefended ---")
    # Undefended baseline from exp5 results
    undefended_asr = 0.4145
    undefended_n = 2000
    
    basis_result = results["TASCP (Ablation 2: Basis)"]
    p1 = basis_result['asr']
    n1 = basis_result['n_total']
    p2 = undefended_asr
    n2 = undefended_n
    
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p1 - p2) / se
    # Two-tailed p-value via normal approximation
    from scipy.stats import norm
    p_value = 2 * norm.sf(abs(z))
    
    print(f"Basis ASR: {p1*100:.2f}% (n={n1}), Undefended ASR: {p2*100:.2f}% (n={n2})")
    print(f"Z-statistic: {z:.4f}")
    print(f"p-value: {p_value:.4f}")
    if p_value > 0.05:
        print("Not statistically significant (p > 0.05). Basis-fixed is indistinguishable from undefended.")
    else:
        print("STATISTICALLY SIGNIFICANT (p <= 0.05). Basis-fixed differs from undefended.")

if __name__ == "__main__":
    run_exp8_cicids2017()
