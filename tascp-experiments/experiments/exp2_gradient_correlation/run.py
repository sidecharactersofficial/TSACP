import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.tascp_eval.models.nids_classifier import NIDSClassifier
from src.tascp_eval.attacks.pgd import pgd_attack, load_config
from src.tascp_eval.defenses.tascp import TASCPWrapper

def run_exp2():
    print("Loading UNSW-NB15 dataset...")
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "unsw_nb15", "unsw_nb15_processed.parquet")
    df = pd.read_parquet(data_path)
    
    label_col = 'label' if 'label' in df.columns else 'Label'
    feature_cols = [c for c in df.columns if c not in ['session_id', label_col]]
    
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    y = df[label_col].values
    
    benign_mask = (y == 0)
    X_benign = X[benign_mask]
    
    print("Loading trained NIDS classifier...")
    model_path = os.path.join(PROJECT_ROOT, "src", "tascp_eval", "models", "weights", "unsw_nb15_nids.pth")
    base_model = NIDSClassifier(input_dim=len(feature_cols))
    base_model.load_state_dict(torch.load(model_path))
    base_model.eval()
    
    wrapper = TASCPWrapper(base_model, window_size=5, max_theta=np.pi/2, tau=0.0, gamma=50.0)
    
    X_benign_t = torch.tensor(X_benign, dtype=torch.float32)
    with torch.no_grad():
        preds = (torch.sigmoid(base_model(X_benign_t)) > 0.5).numpy().astype(int).squeeze()
        
    correct_mask = (preds == 0)
    X_correct = X_benign_t[correct_mask]
    
    n_samples = min(1000, len(X_correct))
    indices = np.random.choice(len(X_correct), n_samples, replace=False)
    X_sample = X_correct[indices]
    y_target = torch.ones(n_samples, 1, dtype=torch.float32)
    
    # We use a standard PGD attack config to generate a trajectory.
    # The attack runs against the base model just to gather realistic adversarial trajectory steps.
    # Wait, should the attack run against the TASCPWrapper? 
    # Yes, running against TASCPWrapper generates realistic adversarial steps. 
    # But since we just need a sequence of steps x_t, we can just run it against the base model for simplicity, 
    # and then feed those steps into TASCPWrapper to evaluate the gradients. 
    # But running against the TASCPWrapper is easy, so let's run against the base model.
    config = load_config()
    print(f"Running PGD to generate trajectories over {n_samples} points...")
    trajectories = pgd_attack(base_model, X_sample, y_target, config)
    
    traj_arr = torch.stack(trajectories, dim=1) # (N, iterations+1, D)
    iterations = traj_arr.shape[1] - 1
    
    print("Evaluating gradient correlation...")
    criterion = nn.BCEWithLogitsLoss()
    
    measured_corrs = []
    predicted_corrs = []
    
    # We reset history for each sample
    for i in range(n_samples):
        wrapper.reset_history(batch_size=1, device=X_sample.device)
        sample_traj = traj_arr[i] # (iterations+1, D)
        y_true = y_target[i:i+1]
        
        # We need at least 3 steps to have v_t and v_prev
        for t in range(len(sample_traj)):
            x_t = sample_traj[t:t+1].clone().detach().requires_grad_(True)
            
            # Forward rotated pass
            wrapper.zero_grad()
            out_rot = wrapper(x_t, apply_rotation=True)
            loss_rot = criterion(out_rot, y_true)
            loss_rot.backward()
            
            if len(wrapper.history) >= 2:
                # We have a valid rotation
                g_t_prime = x_t.grad.clone()
                delta_theta = wrapper.last_delta_theta.item()
                
                # Check if rotation was non-zero
                if delta_theta > 1e-9:
                    u = wrapper.last_u
                    v_perp = wrapper.last_v_perp
                    
                    # Unrotated pass
                    x_t.grad.zero_()
                    out_unrot = wrapper(x_t, apply_rotation=False)
                    loss_unrot = criterion(out_unrot, y_true)
                    loss_unrot.backward()
                    g_t = x_t.grad.clone()
                    
                    # Project both onto span{u, v_perp}
                    g_t_u = torch.sum(g_t * u, dim=1).item()
                    g_t_perp = torch.sum(g_t * v_perp, dim=1).item()
                    
                    g_t_prime_u = torch.sum(g_t_prime * u, dim=1).item()
                    g_t_prime_perp = torch.sum(g_t_prime * v_perp, dim=1).item()
                    
                    vec1 = np.array([g_t_u, g_t_perp])
                    vec2 = np.array([g_t_prime_u, g_t_prime_perp])
                    
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    
                    if norm1 > 1e-9 and norm2 > 1e-9:
                        cos_measured = np.dot(vec1, vec2) / (norm1 * norm2)
                        cos_predicted = np.cos(delta_theta)
                        
                        measured_corrs.append(cos_measured)
                        predicted_corrs.append(cos_predicted)
            
            # Step the history
            wrapper.add_to_history(x_t)
            
    measured_corrs = np.array(measured_corrs)
    predicted_corrs = np.array(predicted_corrs)
    
    mse = np.mean((measured_corrs - predicted_corrs)**2)
    
    print("\n--- Exp 2 Results (Lemma 2) ---")
    print(f"Total valid rotation steps evaluated: {len(measured_corrs)}")
    print(f"Mean Squared Error (Measured vs Predicted): {mse:.6f}")
    
    # Bin by delta_theta
    print("\n--- MSE Breakdown by \u0394\u03b8 Magnitude ---")
    dtheta = np.arccos(predicted_corrs)
    bins = [
        ("Small (0 to \u03c0/12)", 0, np.pi/12),
        ("Medium (\u03c0/12 to \u03c0/4)", np.pi/12, np.pi/4),
        ("Large (\u03c0/4 to \u03c0/2)", np.pi/4, np.pi/2)
    ]
    
    for name, low, high in bins:
        mask = (dtheta >= low) & (dtheta < high)
        bin_meas = measured_corrs[mask]
        bin_pred = predicted_corrs[mask]
        if len(bin_meas) > 0:
            bin_mse = np.mean((bin_meas - bin_pred)**2)
            print(f"{name}: MSE = {bin_mse:.6f} (n={len(bin_meas)})")
        else:
            print(f"{name}: No steps in this range")
            
    out_dir = os.path.join(PROJECT_ROOT, "tascp-experiments", "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(predicted_corrs, measured_corrs, alpha=0.5, color='purple', s=10)
    plt.plot([min(predicted_corrs), max(predicted_corrs)], [min(predicted_corrs), max(predicted_corrs)], 
             'k--', lw=2, label='Perfect Agreement')
    plt.xlabel('Predicted Gradient Correlation $\cos(\Delta\\theta)$')
    plt.ylabel('Measured Gradient Correlation $\cos\\angle(g_t, g\'_t)$')
    plt.title('Lemma 2 Validation: Gradient Correlation Decay')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(out_dir, "exp2_gradient_correlation.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved scatter plot to {save_path}")
    
if __name__ == '__main__':
    run_exp2()
