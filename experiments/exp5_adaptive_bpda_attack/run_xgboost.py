import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
import xgboost as xgb

EXP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.tascp_eval.defenses.tascp import TASCPWrapper
from src.tascp_eval.attacks.pgd import load_config
from src.tascp_eval.attacks.bpda_xgb_surrogate import xgb_stateful_pgd_attack

class SurrogateMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x): return self.net(x)

class NoOpWrapper(nn.Module):
    """Wrapper that does nothing — used for Undefended baseline (surrogate only)."""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.last_x_rot = None
        self.adaptive = False
    def forward(self, x):
        self.last_x_rot = x
        return self.base_model(x)
    def reset_history(self, **kwargs): pass
    def add_to_history(self, x): pass
    def zero_grad(self): self.base_model.zero_grad()

def run():
    print("Loading UNSW-NB15 dataset...")
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "unsw_nb15", "unsw_nb15_processed.parquet")
    df = pd.read_parquet(data_path)
    label_col = 'label' if 'label' in df.columns else 'Label'
    feature_cols = [c for c in df.columns if c not in ['session_id', label_col]]

    X = df[feature_cols].values.astype(np.float32)
    np.nan_to_num(X, copy=False)
    y = df[label_col].values
    del df
    print(f"Total samples: {len(X)}")

    # Load XGBoost
    model_path = os.path.join(PROJECT_ROOT, "src", "tascp_eval", "models", "weights", "unsw_nb15_xgboost.json")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_path)

    # Train surrogate on a 200k subset to save memory
    np.random.seed(42)
    subset_idx = np.random.choice(len(X), min(200000, len(X)), replace=False)
    X_sub = X[subset_idx]
    y_sub = y[subset_idx]

    y_proba = xgb_model.predict_proba(X_sub)[:, 1]
    y_log_odds = np.log(np.clip(y_proba, 1e-7, 1-1e-7) / np.clip(1-y_proba, 1e-7, 1-1e-7))

    X_t = torch.from_numpy(X_sub.copy())
    y_lo_t = torch.from_numpy(y_log_odds.astype(np.float32)).unsqueeze(1)

    surrogate = SurrogateMLP(input_dim=len(feature_cols))
    opt = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
    print("Training surrogate (50 epochs, 200k subset)...")
    surrogate.train()
    for ep in range(50):
        opt.zero_grad()
        loss = nn.MSELoss()(surrogate(X_t), y_lo_t)
        loss.backward()
        opt.step()
        if (ep+1) % 10 == 0:
            with torch.no_grad():
                agree = ((torch.sigmoid(surrogate(X_t)) > 0.5).squeeze() == torch.from_numpy((y_proba > 0.5))).float().mean()
            print(f"  Epoch {ep+1}/50  Loss: {loss.item():.6f}  Agreement: {agree:.4f}")

    surrogate.eval()
    del X_t, y_lo_t, X_sub, y_sub

    # Get correctly-classified benign samples (use a separate smaller load)
    print("\nSelecting benign samples...")
    benign_mask = (y == 0)
    X_benign = X[benign_mask]
    del X
    xgb_benign_preds = xgb_model.predict(X_benign, validate_features=False)
    correct_mask = (xgb_benign_preds == 0)
    X_correct = torch.from_numpy(X_benign[correct_mask].copy())
    n_total = len(X_benign)
    del X_benign
    print(f"XGBoost-correctly-classified benign: {len(X_correct)} / {n_total}")

    n_samples = min(2000, len(X_correct))
    np.random.seed(42)
    indices = np.random.choice(len(X_correct), n_samples, replace=False)
    X_sample = X_correct[indices]
    y_target = torch.zeros(n_samples, 1, dtype=torch.float32)
    print(f"Sampled {n_samples} points for attack")

    config = load_config()
    iterations = config.get('iterations', 20)

    def make_tascp():
        return TASCPWrapper(surrogate, window_size=5, max_theta=np.pi/2, tau=0.0, gamma=50.0)

    conditions = [
        ("Undefended", NoOpWrapper(surrogate), False),
        ("TASCP (Naive)", make_tascp(), False),
        ("TASCP (Full Adaptive)", make_tascp(), True),
    ]
    for mode, label in [('gradient_magnitude', 'Grad Mag'),
                         ('basis_coupling', 'Basis'),
                         ('projection_only', 'Proj Only')]:
        m = make_tascp()
        m.adaptive = True
        m.ablation_mode = mode
        conditions.append((f"TASCP (Ablation: {label})", m, True))

    print(f"\nRunning XGBoost surrogate-gradient attack ({n_samples} samples, {iterations} queries)")
    results = {}
    for name, model, is_adaptive in conditions:
        print(f"\n--- {name} ---")
        x_adv, success_flags, queries = xgb_stateful_pgd_attack(
            model, xgb_model, surrogate, X_sample, y_target, config, adaptive=is_adaptive
        )
        asr = success_flags.float().mean().item()
        mq = queries[success_flags].mean().item() if success_flags.any() else float('nan')
        print(f"ASR: {asr:.4f}  |  Mean Q: {mq:.2f}")
        results[name] = {'asr': asr, 'queries': mq,
                         'n_success': success_flags.sum().item(), 'n_total': n_samples}

    print("\n" + "="*70)
    print("XGBoost — UNSW-NB15 — Budget=20 — Surrogate-Gradient Attack")
    print("="*70)
    print(f"{'Condition':30s} | {'ASR':>8s} {'MeanQ':>6s}")
    print("-"*48)
    for name in results:
        r = results[name]
        print(f"{name:30s} | {r['asr']*100:6.2f}% {r['queries']:5.1f}")

    undef = results["Undefended"]['asr']
    print(f"\nUndefended ASR: {undef*100:.2f}%")

    if "TASCP (Ablation: Basis)" in results:
        print("\n--- Two-Proportion Z-Test: Basis-Fixed vs Undefended ---")
        b = results["TASCP (Ablation: Basis)"]
        p1, n1 = b['asr'], b['n_total']
        p2, n2 = undef, 2000
        pp = (p1*n1 + p2*n2) / (n1 + n2)
        se = np.sqrt(pp * (1-pp) * (1/n1 + 1/n2))
        z = (p1 - p2) / se
        from scipy.stats import norm
        pv = 2 * norm.sf(abs(z))
        print(f"Basis ASR: {p1*100:.2f}% (n={n1}), Undefended ASR: {p2*100:.2f}% (n={n2})")
        print(f"Z: {z:.4f}, p: {pv:.4f}, Significant(p<=0.05): {'Yes' if pv<=0.05 else 'No'}")

    print("\n" + "="*60)
    print("MLP vs XGBoost — UNSW-NB15 — Budget=20")
    print("="*60)
    mlp = {
        "Undefended": 0.1050,
        "TASCP (Naive)": 0.2295,
        "TASCP (Full Adaptive)": 0.5360,
        "TASCP (Ablation: Grad Mag)": 0.6160,
        "TASCP (Ablation: Basis)": 0.1030,
        "TASCP (Ablation: Proj Only)": 0.9570,
    }
    name_map = [
        "Undefended", "TASCP (Naive)", "TASCP (Full Adaptive)",
        "TASCP (Ablation: Grad Mag)", "TASCP (Ablation: Basis)", "TASCP (Ablation: Proj Only)",
    ]
    print(f"{'Condition':30s} | {'MLP':>8s} {'XGB':>8s}")
    print("-"*50)
    for n in name_map:
        if n in results and n in mlp:
            print(f"{n:30s} | {mlp[n]*100:6.2f}% {results[n]['asr']*100:6.2f}%")

if __name__ == "__main__":
    run()
