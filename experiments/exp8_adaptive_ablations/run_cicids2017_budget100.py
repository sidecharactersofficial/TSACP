import os, sys, numpy as np, pandas as pd, torch
from math import sqrt
EXP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.tascp_eval.models.nids_classifier import NIDSClassifier
from src.tascp_eval.defenses.tascp import TASCPWrapper
from src.tascp_eval.attacks.pgd import load_config
from src.tascp_eval.attacks.bpda_adaptive import stateful_pgd_attack

def run():
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
    config['iterations'] = 100

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

    print(f"\nRunning Exp 8: CICIDS2017 Ablations, BUDGET=100 ({n_samples} samples)")
    results = {}

    for name, model, is_adaptive in conditions:
        print(f"\n--- Testing {name} ---")
        x_adv, success_flags, queries = stateful_pgd_attack(model, X_sample, y_target, config, adaptive=is_adaptive)
        success_rate = success_flags.float().mean().item()
        if success_flags.any():
            mean_queries = queries[success_flags].mean().item()
        else:
            mean_queries = float('nan')
        mean_l2 = torch.norm(x_adv - X_sample, p=2, dim=1).mean().item()
        print(f"ASR: {success_rate:.4f}  |  Mean Q: {mean_queries:.2f}  |  Mean L2: {mean_l2:.4f}")
        results[name] = {'asr': success_rate, 'queries': mean_queries, 'l2': mean_l2,
                         'n_success': success_flags.sum().item(), 'n_total': n_samples}

    # Undefended at budget=100 from exp5 re-run
    undefended_asr_100 = 0.5620
    undefended_n = 2000

    print("\n" + "="*75)
    print("EXP 8 — CICIDS2017 ABLATIONS — BUDGET=100")
    print("="*75)
    print(f"{'Condition':35s} | {'ASR@100':>8s} {'MeanQ':>6s} {'L2':>8s}")
    print("-"*75)
    spread_asr = []
    for name, _, _ in conditions:
        r = results[name]
        print(f"{name:35s} | {r['asr']*100:6.2f}% {r['queries']:5.1f} {r['l2']:8.4f}")
        spread_asr.append(r['asr'])
    print(f"\nUndefended (budget=100) ASR: {undefended_asr_100*100:.2f}%")
    print(f"Spread (max-min) across 5 conditions: {(max(spread_asr)-min(spread_asr))*100:.2f}pp")

    # Two-proportion z-test: Ablation 2 (Basis) vs Undefended at budget=100
    print("\n--- Two-Proportion Z-Test: Basis-Fixed vs Undefended @ budget=100 ---")
    basis_r = results["TASCP (Ablation 2: Basis)"]
    p1 = basis_r['asr']
    n1 = basis_r['n_total']
    p2 = undefended_asr_100
    n2 = undefended_n
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p1 - p2) / se
    from scipy.stats import norm
    p_value = 2 * norm.sf(abs(z))
    print(f"Basis ASR: {p1*100:.2f}% (n={n1}), Undefended ASR: {p2*100:.2f}% (n={n2})")
    print(f"Z-statistic: {z:.4f}")
    print(f"p-value: {p_value:.4f}")
    print("Significant (p<=0.05)?", "Yes" if p_value <= 0.05 else "No")

    # UNSW-NB15 comparison at budget=20
    print("\n" + "="*75)
    print("CROSS-DATASET COMPARISON (5-condition table)")
    print("="*75)
    print(f"{'Condition':35s} | {'UNSW ASR@20':>11s} | {'CID ASR@100':>11s}")
    print("-"*75)
    unsw_budget20 = {
        "TASCP (Naive)": 0.2295,
        "TASCP (Full Adaptive)": 0.5360,
        "TASCP (Ablation 1: Grad Mag)": 0.6160,
        "TASCP (Ablation 2: Basis)": 0.1030,
        "TASCP (Ablation 3: Proj Only)": 0.9570,
    }
    for name, _, _ in conditions:
        u = unsw_budget20[name]
        c = results[name]['asr']
        print(f"{name:35s} | {u*100:8.2f}%  | {c*100:8.2f}%")

    unsw_spread = max(unsw_budget20.values()) - min(unsw_budget20.values())
    cic_spread = max(spread_asr) - min(spread_asr)
    print(f"\nSpread (max-min): UNSW-NB15 = {unsw_spread*100:.2f}pp, CICIDS2017 = {cic_spread*100:.2f}pp")

if __name__ == "__main__":
    run()
