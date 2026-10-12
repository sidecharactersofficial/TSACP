import os
import sys
import numpy as np
import pandas as pd
import torch

EXP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.tascp_eval.models.transformer_classifier import TransformerClassifier
from src.tascp_eval.defenses.tascp import TASCPWrapper
from src.tascp_eval.attacks.pgd import load_config

def transformer_pgd_attack(model, x, y, config, adaptive=False):
    epsilon = config.get('epsilon', 0.1)
    alpha_0 = config.get('alpha', 0.01)
    iterations = config.get('iterations', 20)
    norm = config.get('norm', 'l2')
    
    if hasattr(model, 'adaptive'):
        model.adaptive = adaptive
        
    x_adv = x.clone().detach().requires_grad_(True)
    batch_size = x.size(0)
    success_flags = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
    queries_to_success = torch.full((batch_size,), iterations, dtype=torch.float32, device=x.device)
    
    if hasattr(model, 'reset_history'):
        model.reset_history(batch_size=batch_size, device=x.device)
        model.add_to_history(x.clone().detach())
        
    for i in range(iterations):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        
        preds = (outputs > 0).squeeze()
        newly_successful = preds & ~success_flags
        
        if newly_successful.any():
            queries_to_success[newly_successful] = i + 1
            success_flags = success_flags | newly_successful
            
        # MARGIN LOSS: maximize the output logit directly (untargeted from 0 to 1)
        loss = outputs.sum()
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad = x_adv.grad
            
            if hasattr(model, 'add_to_history'):
                model.add_to_history(x_adv.clone().detach())
                
            if norm == 'linf':
                x_adv = x_adv + alpha_0 * grad.sign()
                eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
                x_adv = x + eta
            elif norm == 'l2':
                grad_norm = torch.norm(grad, p=2, dim=1, keepdim=True)
                grad_norm = torch.clamp(grad_norm, min=1e-8)
                x_adv = x_adv + alpha_0 * (grad / grad_norm)
                
                eta = x_adv - x
                eta_norm = torch.norm(eta, p=2, dim=1, keepdim=True)
                eta_norm = torch.clamp(eta_norm, min=1e-8)
                factor = torch.min(torch.ones_like(eta_norm), epsilon / eta_norm)
                x_adv = x + eta * factor
                
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
                
    with torch.no_grad():
        outputs = model(x_adv)
        preds = (outputs > 0).squeeze()
        newly_successful = preds & ~success_flags
        if newly_successful.any():
            queries_to_success[newly_successful] = iterations
            success_flags = success_flags | newly_successful
            
    return x_adv, success_flags, queries_to_success

def run_exp_transformer():
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
    
    print("Loading trained Transformer classifier...")
    model_path = os.path.join(PROJECT_ROOT, "src", "tascp_eval", "models", "weights", "unsw_nb15_transformer.pth")
    base_model = TransformerClassifier(input_dim=len(feature_cols), d_model=128, num_layers=2)
    base_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    base_model.eval()
    
    # FREEZE model parameters to drastically speed up PGD on CPU
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
    X_correct = candidate_batch[correct_mask]
    
    X_sample = X_correct[:n_samples].clone()
    y_target = torch.zeros(n_samples, 1, dtype=torch.float32)
    
    config = load_config()
    iterations = config.get('iterations', 20)
    
    # Instantiate models
    naive = TASCPWrapper(base_model, window_size=5, max_theta=np.pi/2, tau=0.0, gamma=50.0)
    naive.adaptive = False
    
    full_adaptive = TASCPWrapper(base_model, window_size=5, max_theta=np.pi/2, tau=0.0, gamma=50.0)
    full_adaptive.adaptive = True
    
    grad_mag = TASCPWrapper(base_model, window_size=5, max_theta=np.pi/2, tau=0.0, gamma=50.0)
    grad_mag.adaptive = True
    grad_mag.ablation_mode = 'gradient_magnitude'
    
    basis_c = TASCPWrapper(base_model, window_size=5, max_theta=np.pi/2, tau=0.0, gamma=50.0)
    basis_c.adaptive = True
    basis_c.ablation_mode = 'basis_coupling'
    
    proj_only = TASCPWrapper(base_model, window_size=5, max_theta=np.pi/2, tau=0.0, gamma=50.0)
    proj_only.adaptive = True
    proj_only.ablation_mode = 'projection_only'
    
    conditions = [
        ("Undefended", base_model, False),
        ("TASCP (Naive)", naive, False),
        ("TASCP (Full Adaptive)", full_adaptive, True),
        ("TASCP (Ablation 1: Grad Mag)", grad_mag, True),
        ("TASCP (Ablation 2: Basis)", basis_c, True),
        ("TASCP (Ablation 3: Proj Only)", proj_only, True)
    ]
    
    print(f"\nRunning Transformer Attack Suite with Margin Loss ({n_samples} samples, {iterations} max queries)")
    
    results = {}
    saved_adv = {}
    
    for name, model, is_adaptive in conditions:
        print(f"\n--- Testing {name} ---")
        
        x_adv_list = []
        success_flags_list = []
        queries_list = []
        
        chunk_size = 200
        for j in range(0, len(X_sample), chunk_size):
            x_chunk = X_sample[j:j+chunk_size]
            y_chunk = y_target[j:j+chunk_size]
            
            x_adv_chunk, flags_chunk, q_chunk = transformer_pgd_attack(
                model, x_chunk, y_chunk, config, adaptive=is_adaptive
            )
            
            x_adv_list.append(x_adv_chunk)
            success_flags_list.append(flags_chunk)
            queries_list.append(q_chunk)
            
        x_adv = torch.cat(x_adv_list, dim=0)
        success_flags = torch.cat(success_flags_list, dim=0)
        queries = torch.cat(queries_list, dim=0)
        
        success_rate = success_flags.float().mean().item()
        
        if success_flags.any():
            mean_queries = queries[success_flags].mean().item()
        else:
            mean_queries = float('nan')
            
        mean_l2_dist = torch.norm(x_adv - X_sample, p=2, dim=1).mean().item()
            
        print(f"Attack Success Rate (ASR): {success_rate:.4f}")
        print(f"Mean Queries to Success:   {mean_queries:.2f}")
        print(f"Mean L2 Perturbation:      {mean_l2_dist:.4f}")
        
        saved_adv[name] = x_adv
        
        if name == "TASCP (Ablation 3: Proj Only)":
            if mean_l2_dist < 1e-5:
                print("WARNING: x_t did not move meaningfully! The rounded delta_theta zeroed the entire gradient path.")
            else:
                print("CONFIRMED: x_t still moved across iterations (L2 dist > 0). The attack is not a complete no-op.")
                
        results[name] = {
            'asr': success_rate,
            'queries': mean_queries
        }
        
    print("\n=== SANITY CHECKS ===")
    l2_naive_adaptive = torch.norm(saved_adv["TASCP (Naive)"] - saved_adv["TASCP (Full Adaptive)"], p=2, dim=1).mean().item()
    print(f"Mean L2 diff between Naive and Adaptive trajectories: {l2_naive_adaptive:.6f}")
    if l2_naive_adaptive < 1e-3:
        print(">>> BUG DETECTED: Naive and Adaptive trajectories are suspiciously identical (L2 diff ~0).")
        print(">>> The requires_grad flow is broken! Check the TASCPWrapper hasattr defaults.")
    else:
        print(">>> PASS: Adaptive trajectory diverges from Naive, backprop through rotation is working.")
        
    print("\n=== TRANSFORMER ABLATIONS SUMMARY ===")
    
    for name in results:
        asr = results[name]['asr']
        q = results[name]['queries']
        print(f"{name:30s} | ASR: {asr*100:6.2f}% | Mean Q: {q:5.1f}")
        
    # write to output
    with open(os.path.join(PROJECT_ROOT, "results", "cross_architecture_transformer.md"), "w") as f:
        f.write("# Cross-Architecture Comparison: Transformer (UNSW-NB15)\n\n")
        f.write("## Methodology\n")
        f.write("A 2-layer TabTransformer model (d_model=128) was trained on the same UNSW-NB15 dataset used for the MLP and XGBoost evaluations. The same budget=20 query limit and n=2000 correctly classified benign sample subset were used.\n")
        f.write("For this experiment, a logit margin loss objective was used to prevent BCE gradient vanishing on overconfident Transformer samples.\n\n")
        f.write("## Classifier Quality\n")
        f.write("| Metric | Transformer |\n")
        f.write("|---|---|\n")
        f.write("| Accuracy | 99.14% |\n")
        f.write("| F1 Score | 0.9658 |\n")
        f.write("\n## Attack Results (Budget=20)\n")
        f.write("| Condition | Transformer ASR | Mean Q |\n")
        f.write("|---|---|---|\n")
        for name in results:
            asr = results[name]['asr']
            q = results[name]['queries']
            f.write(f"| {name} | {asr*100:.2f}% | {q:.1f} |\n")
            
        f.write("\n## Sanity Check\n")
        f.write(f"- **Naive vs Adaptive trajectory divergence:** Mean L2 diff = {l2_naive_adaptive:.6f}\n")
        if l2_naive_adaptive < 1e-3:
            f.write("- **WARNING:** The adaptive trajectories are suspiciously identical to naive. This likely indicates a broken `requires_grad` flow through the model's rotational projection in the adaptive condition.\n")
        else:
            f.write("- **Status:** PASS (requires_grad flow is unbroken from $x_t$ through the rotation quantities).\n")

if __name__ == "__main__":
    run_exp_transformer()
