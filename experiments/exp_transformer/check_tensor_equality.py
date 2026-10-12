import os
import sys
import numpy as np
import torch
import pandas as pd
torch.set_num_threads(1)

EXP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.tascp_eval.models.transformer_classifier import TransformerClassifier
from src.tascp_eval.defenses.tascp import TASCPWrapper
from src.tascp_eval.attacks.pgd import load_config

# EXACT MATCH to run.py's attack loop
def transformer_pgd_attack(model, x, y, config, adaptive=False):
    epsilon = config.get('epsilon', 0.1)
    alpha = config.get('alpha', 0.01)
    iterations = config.get('iterations', 20)
    
    if isinstance(model, TASCPWrapper):
        model.reset_history(x.size(0), x.device)
        model.add_to_history(x)
        
    x_adv = x.clone().detach()
    success_flags = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
    queries_to_success = torch.zeros(x.size(0), dtype=torch.float32, device=x.device)
    max_dtheta = torch.zeros(x.size(0), dtype=torch.float32, device=x.device)
    
    for i in range(iterations):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        
        preds = (outputs > 0).squeeze()
        newly_successful = preds & ~success_flags
        if newly_successful.any():
            queries_to_success[newly_successful] = i + 1
            
        if isinstance(model, TASCPWrapper) and hasattr(model, 'last_delta_theta'):
            max_dtheta = torch.maximum(max_dtheta, model.last_delta_theta.squeeze())
            
        loss = outputs.sum()
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            if adaptive and hasattr(model, 'last_u') and hasattr(model, 'last_v_perp'):
                u = model.last_u
                v_perp = model.last_v_perp
                delta_theta = model.last_delta_theta
                
                comp_u = torch.sum(x_adv.grad * u, dim=1, keepdim=True)
                comp_perp = torch.sum(x_adv.grad * v_perp, dim=1, keepdim=True)
                grad_orth = x_adv.grad - (comp_u * u) - (comp_perp * v_perp)
                
                cos_theta = torch.cos(delta_theta)
                sin_theta = torch.sin(delta_theta)
                
                grad_u_new = comp_u * cos_theta + comp_perp * sin_theta
                grad_perp_new = -comp_u * sin_theta + comp_perp * cos_theta
                
                x_adv.grad = grad_orth + grad_u_new * u + grad_perp_new * v_perp
                
            eta = alpha * torch.sign(x_adv.grad)
            if isinstance(model, TASCPWrapper):
                model.add_to_history(x_adv + eta)
                
            x_adv = x_adv + eta
            
            eta_total = x_adv - x
            eta_norm = torch.norm(eta_total, p=float('inf'), dim=1, keepdim=True)
            eta_norm = torch.clamp(eta_norm, min=1e-8)
            factor = torch.min(torch.ones_like(eta_norm), epsilon / eta_norm)
            x_adv = x + eta_total * factor
            
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            
    with torch.no_grad():
        outputs = model(x_adv)
        preds = (outputs > 0).squeeze()
        newly_successful = preds & ~success_flags
        if newly_successful.any():
            queries_to_success[newly_successful] = iterations
            success_flags = success_flags | newly_successful
            
    return x_adv.detach(), success_flags, queries_to_success, max_dtheta

def run_check():
    print("Loading dataset...")
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "unsw_nb15", "unsw_nb15_processed.parquet")
    df = pd.read_parquet(data_path)
    label_col = 'label' if 'label' in df.columns else 'Label'
    feature_cols = [c for c in df.columns if c not in ['session_id', label_col]]
    
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    y = df[label_col].values
    benign_mask = (y == 0)
    X_benign = X[benign_mask]
    
    model_path = os.path.join(PROJECT_ROOT, "src", "tascp_eval", "models", "weights", "unsw_nb15_transformer.pth")
    base_model = TransformerClassifier(input_dim=len(feature_cols), d_model=128, num_layers=2)
    base_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
        
    X_benign_t = torch.tensor(X_benign, dtype=torch.float32)
    n_samples = 2000
    np.random.seed(42)
    indices = np.random.choice(len(X_benign_t), 2500, replace=False)
    candidate_batch = X_benign_t[indices]
    
    with torch.no_grad():
        preds = (torch.sigmoid(base_model(candidate_batch)) > 0.5).numpy().astype(int).squeeze()
        
    correct_mask = (preds == 0)
    X_sample = candidate_batch[correct_mask][:n_samples].clone()
    y_target = torch.zeros(n_samples, 1, dtype=torch.float32)
    
    config = load_config()
    chunk_size = 200
    
    print("\n--- Running Undefended ---")
    u_adv_list, u_flags_list = [], []
    for j in range(0, len(X_sample), chunk_size):
        x_chunk = X_sample[j:j+chunk_size]
        y_chunk = y_target[j:j+chunk_size]
        x_adv, flags, _, _ = transformer_pgd_attack(base_model, x_chunk, y_chunk, config, adaptive=False)
        u_adv_list.append(x_adv)
        u_flags_list.append(flags)
    u_adv = torch.cat(u_adv_list, dim=0)
    u_flags = torch.cat(u_flags_list, dim=0)
    
    print("--- Running TASCP (Ablation 2: Basis) ---")
    basis_model = TASCPWrapper(base_model, window_size=5, max_theta=np.pi/2, tau=0.0, gamma=50.0)
    basis_model.adaptive = True
    basis_model.ablation_mode = 'basis_coupling'
    
    b_adv_list, b_flags_list, max_dtheta_list = [], [], []
    for j in range(0, len(X_sample), chunk_size):
        x_chunk = X_sample[j:j+chunk_size]
        y_chunk = y_target[j:j+chunk_size]
        x_adv, flags, _, max_dt = transformer_pgd_attack(basis_model, x_chunk, y_chunk, config, adaptive=True)
        b_adv_list.append(x_adv)
        b_flags_list.append(flags)
        max_dtheta_list.append(max_dt)
    b_adv = torch.cat(b_adv_list, dim=0)
    b_flags = torch.cat(b_flags_list, dim=0)
    b_max_dt = torch.cat(max_dtheta_list, dim=0)
    
    u_asr = u_flags.float().mean().item()
    b_asr = b_flags.float().mean().item()
    print(f"\nUndefended ASR:  {u_asr*100:.2f}%")
    print(f"Basis-fixed ASR: {b_asr*100:.2f}%")
    
    if abs(u_asr - 0.0105) > 1e-4 or abs(b_asr - 0.0105) > 1e-4:
        print("WARNING: ASR does not match the originally reported 1.05%! Setup differs.")
        
    u_indices = torch.nonzero(u_flags).squeeze().tolist()
    if type(u_indices) != list: u_indices = [u_indices]
    
    top5 = sorted(u_indices)[:5]
    print(f"\nTop 5 successful indices: {top5}")
    
    print("\nBit-for-bit comparison for top 5 successes:")
    for idx in top5:
        l2_diff = torch.norm(u_adv[idx] - b_adv[idx], p=2).item()
        triggered = b_max_dt[idx].item() > 1e-5
        print(f"Sample {idx:4d} | ||x_t_undefended - x_t_basisfixed||_2 = {l2_diff:.6e} | Rotation triggered? {triggered} (max dTheta={b_max_dt[idx].item():.4f})")
        
    print("\nVERDICT:")
    any_diff = any(torch.norm(u_adv[idx] - b_adv[idx], p=2).item() > 1e-7 for idx in top5)
    any_triggered = any(b_max_dt[idx].item() > 1e-5 for idx in top5)
    
    if not any_triggered:
        print("rotation never triggered for these points (expected, low-momentum benign-origin trajectory, not a bug)")
    elif any_triggered and not any_diff:
        print("rotation triggered but tensors are still identical (bug — rotation isn't being applied)")
    else:
        print("tensors differ despite identical binary outcome (coincidental success-set overlap, not a mechanical no-op — safe to report as before)")

if __name__ == '__main__':
    run_check()
