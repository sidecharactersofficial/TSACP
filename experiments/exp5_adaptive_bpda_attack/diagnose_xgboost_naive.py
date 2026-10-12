import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
import xgboost as xgb

EXP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.tascp_eval.defenses.tascp import TASCPWrapper
from src.tascp_eval.attacks.pgd import load_config

class SurrogateMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x): return self.net(x)

def run_diagnostic():
    print("Loading UNSW-NB15...")
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "unsw_nb15", "unsw_nb15_processed.parquet")
    df = pd.read_parquet(data_path)
    label_col = 'label' if 'label' in df.columns else 'Label'
    feature_cols = [c for c in df.columns if c not in ['session_id', label_col]]
    X = df[feature_cols].values.astype(np.float32)
    np.nan_to_num(X, copy=False)
    del df

    # Load XGBoost
    xgb_path = os.path.join(PROJECT_ROOT, "src", "tascp_eval", "models", "weights", "unsw_nb15_xgboost.json")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_path)

    # Train surrogate on subset
    np.random.seed(42)
    subset_idx = np.random.choice(len(X), 200000, replace=False)
    X_sub = X[subset_idx]
    y_proba = xgb_model.predict_proba(X_sub)[:, 1]
    y_log_odds = np.log(np.clip(y_proba, 1e-7, 1-1e-7) / np.clip(1-y_proba, 1e-7, 1-1e-7))
    X_t = torch.from_numpy(X_sub.copy())
    y_lo_t = torch.from_numpy(y_log_odds.astype(np.float32)).unsqueeze(1)
    surrogate = SurrogateMLP(input_dim=len(feature_cols))
    opt = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
    print("Training surrogate...")
    surrogate.train()
    for ep in range(50):
        opt.zero_grad()
        loss = nn.MSELoss()(surrogate(X_t), y_lo_t)
        loss.backward()
        opt.step()
    surrogate.eval()

    # Get correctly-classified benign
    benign_mask = (X_sub.sum(axis=1) >= 0)  # all, we just classify everything
    # Actually, get benign samples from full dataset
    benign_mask_full = (pd.read_parquet(
        os.path.join(PROJECT_ROOT, "data", "processed", "unsw_nb15", "unsw_nb15_processed.parquet"),
        columns=[label_col])[label_col].values == 0)
    # We already have X from the full load above... need to reload
    # Let's just use X_sub since it's available

    # Identify benign samples from full set by loading just labels
    df_labels = pd.read_parquet(
        os.path.join(PROJECT_ROOT, "data", "processed", "unsw_nb15", "unsw_nb15_processed.parquet"),
        columns=[label_col])
    y_full = df_labels.values.squeeze()
    del df_labels
    benign_mask_full = (y_full == 0)
    X_benign = X[benign_mask_full]
    del X

    # Get XGBoost-correct benign samples
    xgb_benign = xgb_model.predict(X_benign, validate_features=False)
    correct_mask = (xgb_benign == 0)
    X_correct = X_benign[correct_mask]
    del X_benign
    print(f"Correctly-classified benign: {len(X_correct)}")

    # Sample 200 for diagnostic (enough to see pattern)
    n_samples = 200
    np.random.seed(42)
    idx = np.random.choice(len(X_correct), n_samples, replace=False)
    X_sample = torch.from_numpy(X_correct[idx].copy())
    y_target = torch.zeros(n_samples, 1, dtype=torch.float32)
    print(f"Sampled {n_samples} for diagnostic")

    config = load_config()

    # Build TASCP-Naive wrapper
    tascp = TASCPWrapper(surrogate, window_size=5, max_theta=np.pi/2, tau=0.0, gamma=50.0)
    tascp.adaptive = False

    # Also create NoOp wrapper for undefended comparison
    class NoOpWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.last_x_rot = None
            self.adaptive = False
        def forward(self, x):
            self.last_x_rot = x
            return self.m(x)
        def reset_history(self, **kwargs): pass
        def add_to_history(self, x): pass
        def zero_grad(self): self.m.zero_grad()

    noop = NoOpWrapper(surrogate)

    # Run attack with detailed tracking for TASCP-Naive
    print("\n=== Running TASCP-Naive attack with trajectory tracking ===")
    # We'll use a modified attack loop that records everything
    epsilon = config.get('epsilon', 0.1)
    alpha_0 = config.get('alpha', 0.01)
    iterations = config.get('iterations', 20)
    norm = 'l2'

    # --- NoOp (undefended) for baseline ---
    print("\nUndefended (NoOp)...")
    x_adv = X_sample.clone().detach().requires_grad_(True)
    batch_size = n_samples
    success_u = torch.zeros(batch_size, dtype=torch.bool)
    queries_u = torch.full((batch_size,), iterations, dtype=torch.float32)
    for i in range(iterations):
        x_adv.requires_grad_(True)
        out = noop(x_adv)
        preds = (out > 0).squeeze()
        new_succ = preds & ~success_u
        if new_succ.any():
            queries_u[new_succ] = i + 1
            success_u = success_u | new_succ
        loss = nn.BCEWithLogitsLoss()(out, y_target)
        noop.zero_grad()
        loss.backward()
        with torch.no_grad():
            grad = x_adv.grad
            noop.add_to_history(x_adv.clone().detach())
            grad_norm = torch.norm(grad, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            x_adv = x_adv + alpha_0 * (grad / grad_norm)
            eta = x_adv - X_sample
            eta_norm = torch.norm(eta, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            factor = torch.min(torch.ones_like(eta_norm), epsilon / eta_norm)
            x_adv = X_sample + eta * factor
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    # Final
    out = noop(x_adv)
    preds = (out > 0).squeeze()
    new_s = preds & ~success_u
    if new_s.any(): queries_u[new_s] = iterations; success_u = success_u | new_s

    # Evaluate undefended against XGBoost
    x_rot_u = noop.last_x_rot.detach()
    xgb_proba_u = xgb_model.predict_proba(x_rot_u.cpu().numpy())[:, 1]
    xgb_success_u = torch.from_numpy((xgb_proba_u > 0.5).astype(np.float32))
    # Count undefended successes
    successful_u = xgb_success_u.squeeze().bool()
    print(f"Undefended ASR (vs XGB): {successful_u.float().mean():.4f} ({successful_u.sum().item()}/{batch_size})")

    # --- TASCP-Naive with trajectory recording ---
    tascp.reset_history(batch_size=batch_size, device='cpu')
    tascp.add_to_history(X_sample.clone().detach())

    x_adv = X_sample.clone().detach().requires_grad_(True)
    success_t = torch.zeros(batch_size, dtype=torch.bool)
    queries_t = torch.full((batch_size,), iterations, dtype=torch.float32)

    # Per-sample trajectory tracking (for successful ones)
    trajs = {i: {'x_adv': [], 'x_rot': [], 'surr_out': [], 'xgb_pred': [],
                  'l2_norm': [], 'iter': []} for i in range(batch_size)}

    for i in range(iterations):
        x_adv.requires_grad_(True)
        out = tascp(x_adv)  # Returns surrogate(x_rot)

        # Record rotated input (set by TASCPWrapper.last_x_rot)
        x_rot = tascp.last_x_rot.detach() if hasattr(tascp, 'last_x_rot') and tascp.last_x_rot is not None else x_adv.detach()

        # XGBoost prediction on rotated input
        xgb_proba = xgb_model.predict_proba(x_rot.cpu().numpy())[:, 1]
        xgb_pred = (xgb_proba > 0.5).astype(np.float32)

        # Check success via XGBoost
        new_succ = torch.from_numpy(xgb_pred).squeeze().bool() & ~success_t
        if new_succ.any():
            queries_t[new_succ] = i + 1
            success_t = success_t | new_succ

        loss = nn.BCEWithLogitsLoss()(out, y_target)
        tascp.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = x_adv.grad
            tascp.add_to_history(x_adv.clone().detach())

            # Record trajectories for all samples
            for j in range(batch_size):
                trajs[j]['x_adv'].append(x_adv[j].clone())
                trajs[j]['x_rot'].append(x_rot[j].clone())
                trajs[j]['surr_out'].append(out[j].item())
                trajs[j]['xgb_pred'].append(xgb_pred[j])
                l2 = torch.norm(x_adv[j] - X_sample[j], p=2).item()
                trajs[j]['l2_norm'].append(l2)
                trajs[j]['iter'].append(i+1)

            # PGD step
            grad_norm = torch.norm(grad, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            x_adv = x_adv + alpha_0 * (grad / grad_norm)
            eta = x_adv - X_sample
            eta_norm = torch.norm(eta, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            factor = torch.min(torch.ones_like(eta_norm), epsilon / eta_norm)
            x_adv = X_sample + eta * factor
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # Final check
    out = tascp(x_adv)
    x_rot = tascp.last_x_rot.detach()
    xgb_proba = xgb_model.predict_proba(x_rot.cpu().numpy())[:, 1]
    xgb_pred = (xgb_proba > 0.5).astype(np.float32)
    new_s = torch.from_numpy(xgb_pred).squeeze().bool() & ~success_t
    if new_s.any(): queries_t[new_s] = iterations; success_t = success_t | new_s
    # Record final
    for j in range(batch_size):
        trajs[j]['x_adv'].append(x_adv[j].clone())
        trajs[j]['x_rot'].append(x_rot[j].clone())
        trajs[j]['surr_out'].append(out[j].item())
        trajs[j]['xgb_pred'].append(xgb_pred[j])
        trajs[j]['l2_norm'].append(torch.norm(x_adv[j] - X_sample[j], p=2).item())
        trajs[j]['iter'].append(iterations)

    asr_t = success_t.float().mean().item()
    print(f"TASCP-Naive ASR (vs XGB): {asr_t:.4f} ({success_t.sum().item()}/{batch_size})")

    # ============================================================
    # DIAGNOSTIC 1: Inspect successful trajectories
    # ============================================================
    successful_idx = torch.where(success_t)[0].tolist()
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC 1: Inspecting {min(15, len(successful_idx))} successful trajectories")
    print(f"{'='*70}")

    # Also compute surrogate-XGBoost agreement on benign samples
    print("\nSurrogate vs XGBoost agreement on these benign samples...")
    with torch.no_grad():
        surr_init = torch.sigmoid(surrogate(X_sample)).squeeze().numpy()
    xgb_init = xgb_model.predict_proba(X_sample.numpy())[:, 1]
    init_agree = ((surr_init > 0.5) == (xgb_init > 0.5)).mean()
    disagree_mask = (surr_init > 0.5) != (xgb_init > 0.5)
    print(f"Initial agreement: {init_agree:.4f}")
    print(f"Samples where surrogate says attack(1) but XGB says benign(0): {disagree_mask.sum().item()}")

    for sidx in successful_idx[:15]:
        t = trajs[sidx]
        orig = X_sample[sidx]
        success_iter = int(queries_t[sidx].item())
        print(f"\n--- Sample {sidx} | Success at iteration {success_iter} ---")
        print(f"  Original surrogate output: {surr_init[sidx]:.4f} (-> {'attack' if surr_init[sidx]>0.5 else 'benign'})")
        print(f"  Original XGBoost output: {xgb_init[sidx]:.4f} (-> {'attack' if xgb_init[sidx]>0.5 else 'benign'})")
        print(f"  Initially in disagreement region? {'YES' if disagree_mask[sidx] else 'NO'}")

        # Show iteration-by-iteration for steps around success
        for step_idx in range(len(t['iter'])):
            it = t['iter'][step_idx]
            l2 = t['l2_norm'][step_idx]
            xgb_p = t['xgb_pred'][step_idx]
            surr_o = t['surr_out'][step_idx]
            surr_p = 'attack' if surr_o > 0 else 'benign'

            # Compute L2 of x_rot vs orig too
            l2_rot = torch.norm(t['x_rot'][step_idx] - orig, p=2).item()

            marker = ' *** SUCCESS ***' if (it == success_iter and xgb_p == 1) else \
                     ' <- FIRST SUCCESS' if (it == success_iter and xgb_p == 1) else ''
            print(f"  Iter {it:2d}: xgb_pred={int(xgb_p)} surr_out={surr_o:+.4f}({surr_p}) "
                  f"L2_adv={l2:.4f} L2_rot={l2_rot:.4f}{marker}")

        # Verify: at success, is L2 <= epsilon?
        success_pos = success_iter  # this is 1-indexed
        if success_pos <= len(t['iter']):
            l2_at_success = t['l2_norm'][success_pos-1]
            l2_rot_at_success = torch.norm(t['x_rot'][success_pos-1] - orig, p=2).item()
            print(f"  [VERIFY] L2 norm at success: adv={l2_at_success:.4f}, rot={l2_rot_at_success:.4f} (eps={epsilon})")
            print(f"  [VERIFY] Within budget? {'YES' if l2_at_success <= epsilon+1e-4 else 'NO'}")
            # XGBoost prediction at that point
            xgb_at_success = t['xgb_pred'][success_pos-1]
            print(f"  [VERIFY] XGBoost prediction at success: {int(xgb_at_success)} (should be 1)")
            # Check x_rot is different from x_adv
            xadv_s = t['x_adv'][success_pos-1]
            xrot_s = t['x_rot'][success_pos-1]
            diff = torch.norm(xadv_s - xrot_s, p=2).item()
            print(f"  [VERIFY] ||x_adv - x_rot|| at success: {diff:.6f} (0 means no rotation applied)")

    # ============================================================
    # DIAGNOSTIC 2: Check if successes cluster in disagreement region
    # ============================================================
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC 2: Do successes cluster in surrogate/XGBoost disagreement region?")
    print(f"{'='*70}")

    total = n_samples
    n_disagree = disagree_mask.sum().item()
    n_agree = total - n_disagree

    # Successes in each region
    succ_in_disagree = success_t[disagree_mask].sum().item()
    succ_in_agree = success_t[~disagree_mask].sum().item()

    print(f"Total samples: {total}")
    print(f"Disagreement region (surrogate != XGB): {n_disagree} ({n_disagree/total*100:.1f}%)")
    print(f"Agreement region (surrogate == XGB):    {n_agree} ({n_agree/total*100:.1f}%)")
    print(f"\nSuccesses in disagreement: {succ_in_disagree}/{n_disagree} ({succ_in_disagree/n_disagree*100:.1f}% of disagree region)")
    print(f"Successes in agreement:     {succ_in_agree}/{n_agree} ({succ_in_agree/n_agree*100:.1f}% of agree region)")
    total_succ = success_t.sum().item()
    frac_from_disagree = succ_in_disagree / total_succ if total_succ > 0 else 0
    print(f"\nOf {total_succ} total successes, {succ_in_disagree} ({frac_from_disagree*100:.1f}%) come from the {n_disagree/total*100:.1f}% disagreement region")

    # Also check undefended
    print(f"\n--- Undefended comparison ---")
    su_disagree = successful_u[disagree_mask].sum().item()
    su_agree = successful_u[~disagree_mask].sum().item()
    print(f"Undefended successes in disagreement: {su_disagree}/{n_disagree} ({su_disagree/n_disagree*100:.1f}%)")
    print(f"Undefended successes in agreement:     {su_agree}/{n_agree} ({su_agree/n_agree*100:.1f}%)")

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Undefended ASR (vs XGB, n={n_samples}): {successful_u.float().mean():.4f}")
    print(f"TASCP-Naive ASR (vs XGB, n={n_samples}): {asr_t:.4f}")
    print(f"Disagreement-dominated? {'YES - successes concentrated in disagreement region' if frac_from_disagree > 0.7 else 'NO - successes distributed across both regions'}")

if __name__ == "__main__":
    run_diagnostic()
