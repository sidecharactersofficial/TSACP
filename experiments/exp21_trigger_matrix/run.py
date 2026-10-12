"""RQ3: trigger observability x attack-family matrix (generalized B1).

Reviewer concern: "the antithetic-NES blindness is just your momentum
trigger." This experiment builds FIVE trajectory triggers

  momentum            TASCP's windowed ||v||*max(0, cos) statistic
  cosine              mean signed cosine of consecutive displacements
  sign_agreement      fraction of coordinates whose displacement sign repeats
  displacement        directional persistence ||mean v|| / mean ||v||
  antithetic_pair     fraction of steps with cos(v_t, v_{t-1}) < -tau_pair

and runs SEVEN query-efficient attack families against an undefended base
model while recording every query. Each trigger's threshold tau is
calibrated on a benign control session (repeated benign flow re-checks with
tiny jitter) at a ~1% false-positive level, then we report the per-cell
fire rate: fraction of attack sessions in which the trigger ever fires.

Deliverable: the matrix shows trajectory statistics are attack-family
dependent (no trigger fires on all families; the momentum trigger is blind
to antithetic probe structures), establishing that a stateful trigger must
be evaluated against the query-generation statistics of the attacker, not
the designer's intuition of "persistent movement".

CLI: python -m experiments.exp21_trigger_matrix.run [dataset]
Writes results/tables/exp21_trigger_matrix_<ds>.csv,
results/logs/exp21_trigger_matrix/metrics_<ds>.json.
"""

import csv
import json
import os
import sys

import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
for _p in (os.path.join(PROJECT_ROOT, "experiments", "exp9_blackbox_attack"),
           os.path.join(PROJECT_ROOT, "experiments", "exp11_detector_robustness")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.exp9_blackbox_attack.exp9_common import load_data  # noqa: E402
from src.tascp_eval.attacks.trigger_scope import TRIGGER_SCOPE_ATTACKS  # noqa: E402
from experiments.exp11_detector_robustness.run import nes_attack_jitter  # noqa: E402

WINDOW = 5            # trigger window (matches TASCP)
BENIGN_T = 60         # benign session length for calibration
JITTER_SIGMA = 1e-3   # benign re-check jitter
N_SESSIONS = 200      # queries per attack session (per sample)
TAU_FP_LEVEL = 0.99   # benign calibration quantile (raised to per-sample max by calibrate_taus)

SCOPE_CFG = {"epsilon": 0.1, "sigma": 0.05, "alpha": 0.02, "q": 10,
             "max_queries": 150, "spsa_momentum": 0.9, "zoo_fd_h": 1e-2,
             "simba_step": 0.02, "square_p": 0.05, "boundary_step": 0.05}


# ---------------------------------------------------------------------------
# Query recorder: transparent model wrapper logging every query.
# ---------------------------------------------------------------------------
class QueryRecorder(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.seq = []
        self._hist_init = False

    def forward(self, x):
        # stateful passthrough: if the wrapped model maintains query history
        # (TASCP family wrappers), drive it with the deployed-protocol calls
        # so black-box attacks see the real stateful behavior.
        if hasattr(self.base, "reset_history"):
            if not self._hist_init:
                self.base.reset_history(x.size(0), x.device)
                self._hist_init = True
            self.base.add_to_history(x.detach())
        self.seq.append(x.detach().clone())
        return self.base(x)

    def reset(self):
        self.seq = []
        self._hist_init = False

    def session(self):
        """(T, B, d) query tensor for the recorded session."""
        return torch.stack(self.seq, dim=0)


# ---------------------------------------------------------------------------
# Trigger statistics. Each takes seq (T, B, d) and returns per-window stats
# (n_windows, B); fire = any stat > tau. All computed on unit-normalized
# displacement directions where a direction is involved.
# ---------------------------------------------------------------------------
def _displacements(seq):
    return seq[1:] - seq[:-1]  # (T-1, B, d)


def stat_momentum(seq):
    """TASCP's eps_p over a sliding window: mean ||v_i|| * max(0, cos)."""
    vs = _displacements(seq)
    n = vs.norm(p=2, dim=2)  # (T-1, B)
    cos = _cos_seq(vs)  # (T-2, B): cos(v_i, v_{i-1})
    # first displacement has no predecessor -> rectified momentum term 0
    cos = torch.cat([torch.zeros(1, cos.size(1), device=cos.device), cos], dim=0)
    w_terms = (_unfold_guard(n * cos.clamp(min=0.0)))  # (n_w, B, W)
    return w_terms.mean(dim=2)


def stat_cosine(seq):
    vs = _displacements(seq)
    c = _unfold_guard(_cos_seq(vs))
    return c.mean(dim=2)


def stat_sign_agreement(seq):
    vs = _displacements(seq)
    agree = (torch.sign(vs[1:]) == torch.sign(vs[:-1])).float().mean(dim=2)  # (T-2, B)
    return _unfold_guard(agree).mean(dim=2)


def stat_displacement(seq):
    vs = _displacements(seq)
    w = _unfold_guard(vs)  # (n_w, B, W, d)
    mean_vec = w.mean(dim=2)
    mean_norm = w.norm(p=2, dim=3).mean(dim=2)
    return mean_vec.norm(p=2, dim=2) / mean_norm.clamp(min=1e-9)


def stat_antithetic(seq):
    """Antithetic-pair detector: flags consecutive near-opposite,
    equal-magnitude displacements (the probe alternation structure of
    antithetic NES / symmetric finite differences). Per-window score is
    max over pairs of 1 - ||v_t + v_{t-1}|| / (||v_t|| + ||v_{t-1}||):
    near 1 for exact antithetic pairs, near 0 for random/benign motion
    (benign jitter does not pair up, so the benign FP stays low)."""
    vs = _displacements(seq)
    if vs.size(0) < 2:
        return vs.new_zeros(0, vs.size(1))
    n1 = vs[1:].norm(p=2, dim=2)
    n0 = vs[:-1].norm(p=2, dim=2)
    sym = (vs[1:] + vs[:-1]).norm(p=2, dim=2) / (n1 + n0).clamp(min=1e-9)
    anti = 1.0 - sym  # (T-2, B), high = antithetic pair
    return _unfold_guard(anti).max(dim=2).values


TRIGGERS = {
    "momentum": stat_momentum,
    "cosine": stat_cosine,
    "sign_agreement": stat_sign_agreement,
    "displacement": stat_displacement,
    "antithetic_pair": stat_antithetic,
}


def _unfold_guard(t):
    """Sliding-window unfold, robust to sessions shorter than WINDOW.
    2-D input (n, B) -> (n_w, B, W); higher-dim keeps trailing dims."""
    if t.size(0) < WINDOW:
        shape = list(t.shape)
        shape[0] = 0
        shape.insert(2, WINDOW)
        return t.new_zeros(tuple(shape))
    return t.unfold(0, WINDOW, 1)


def _cos_seq(vs):
    """Cosine of consecutive displacements, (T-2, B), 0 where degenerate."""
    n1 = vs[1:].norm(p=2, dim=2)
    n0 = vs[:-1].norm(p=2, dim=2)
    ok = (n1 > 1e-9) & (n0 > 1e-9)
    dot = (vs[1:] * vs[:-1]).sum(dim=2)
    return torch.where(ok, dot / (n1 * n0).clamp(min=1e-9), torch.zeros_like(dot))


# ---------------------------------------------------------------------------
# Attack registry. All signatures (model, x, config, projector=None).
# ---------------------------------------------------------------------------
def random_search_attack(model, x, config, projector=None, seed=42):
    """Query-efficient random search baseline: propose a random direction;
    keep it if the (queried) margin improves. 1 query per candidate."""
    epsilon = config.get("epsilon", 0.1)
    max_q = config.get("max_queries", 150)
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    x0 = x.clone().detach()
    x_adv = (x0 + epsilon).clamp(0.0, 1.0)
    B = x.size(0)
    succ = torch.zeros(B, dtype=torch.bool, device=x.device)
    qts = torch.full((B,), float(max_q), device=x.device)
    out0 = model(x0)
    best = out0.squeeze(-1)  # want margin > 0 (attack class)
    for i in range(max_q // 2):
        u = torch.randn(x0.shape, generator=gen, device=x.device)
        u = u / u.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
        cand = (x_adv + 0.05 * u).clamp(0.0, 1.0)
        out = model(cand)
        m = out.squeeze(-1)
        improve = m > best
        x_adv = torch.where(improve.unsqueeze(1), cand, x_adv)
        best = torch.where(improve, m, best)
        newly = (best > 0) & ~succ
        if newly.any():
            qts[newly] = 2 * (i + 1)
            succ |= newly
    return x_adv, succ, qts, torch.full((B,), float(max_q), device=x.device)


ATTACKS = {name: fn for name, fn in TRIGGER_SCOPE_ATTACKS.items()}
ATTACKS["random_search"] = random_search_attack
ATTACKS["nes_antithetic_jitter"] = nes_attack_jitter


def run_attack(recorder, fn, X, y_model_unused, config):
    recorder.reset()
    out = fn(recorder, X, config)
    if len(out) == 4:
        x_adv, succ, qts, total_q = out
        # total_q may be a scalar (run_condition-style) -> broadcast to a
        # per-sample tensor so downstream .mean() calls stay uniform.
        if not torch.is_tensor(total_q):
            total_q = torch.full((X.size(0),), float(total_q), device=X.device)
    else:  # some legacy signatures return 3
        x_adv, succ, qts = out
        if torch.is_tensor(qts):
            total_q = qts.clone().float()
        else:
            total_q = torch.full((X.size(0),), float(qts), device=X.device)
    seq = recorder.session()
    return seq, succ, qts, total_q


def benign_sessions(base_model, X, n_repeat=BENIGN_T):
    """Benign control: the same benign flow re-checked repeatedly with tiny
    jitter (the realistic NIDS client behavior the defense must not flag)."""
    recorder = QueryRecorder(base_model)
    recorder.reset()
    g = torch.Generator(device=X.device)
    g.manual_seed(123)
    for _ in range(n_repeat):
        recorder(X + JITTER_SIGMA * torch.randn(X.shape, generator=g, device=X.device))
    return recorder.session()


def calibrate_taus(benign_seq, fp_level=TAU_FP_LEVEL):
    """Per-trigger tau: max(benign quantile, benign max) per BROWSER sample.

    The reviewer calibration requirement is ~1% benign FP across full
    sessions; the 99th percentile of single-window statistics overshoots
    that on heavy-tailed benign traffic (NIDS re-checks are bursty). We take
    the harder guarantee: tau >= max over windows for each sample, i.e. an
    attack fires only where its statistic EXCEEDS every benign window
    statistic observed for that flow."""
    taus = {}
    for name, stat_fn in TRIGGERS.items():
        s = stat_fn(benign_seq)  # (n_windows, B)
        per_sample_max = s.max(dim=0).values  # (B,)
        q = torch.quantile(s.flatten(), fp_level).item()
        tau = float(torch.maximum(per_sample_max, torch.full_like(per_sample_max, q)).max().item())
        taus[name] = tau
    return taus


def fire_rate(stat, tau):
    """Fraction of samples whose statistic exceeds tau in ANY window."""
    return float((stat > tau).any(dim=0).float().mean().item())


def main(dataset="unsw_nb15", n_samples=100, seed=42):
    torch.manual_seed(seed)
    base_model, feature_cols, spec, levels, X = load_data(dataset, n_samples, seed)

    recorder = QueryRecorder(base_model)

    # benign calibration
    benign_seq = benign_sessions(base_model, X)
    taus = calibrate_taus(benign_seq)
    benign_fp = {t: fire_rate(fn(benign_seq), taus[t]) for t, fn in TRIGGERS.items()}
    print(f"benign FP at calibrated taus: "
          + ", ".join(f"{t}={v*100:.1f}%" for t, v in benign_fp.items()))

    rows = []
    for atk_name, atk_fn in ATTACKS.items():
        seq, succ, qts, total_q = run_attack(recorder, atk_fn, X, None, SCOPE_CFG)
        row = {
            "attack": atk_name,
            "asr_undefended": float(succ.float().mean().item()),
            "mean_queries": float(total_q.mean().item()),
        }
        for trig_name, stat_fn in TRIGGERS.items():
            s = stat_fn(seq)
            row[f"fire_{trig_name}"] = fire_rate(s, taus[trig_name])
            row[f"maxstat_{trig_name}"] = float(s.max().item())
        rows.append(row)
        cells = " | ".join(f"{t}:{row[f'fire_{t}']*100:5.1f}%" for t in TRIGGERS)
        print(f"{atk_name:22s} ASR {row['asr_undefended']*100:5.1f}% | {cells}")

    out = {
        "dataset": dataset,
        "n_samples": X.size(0),
        "window": WINDOW,
        "benign_session_len": BENIGN_T,
        "taus": taus,
        "benign_fp": benign_fp,
        "rows": rows,
    }
    log_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp21_trigger_matrix")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"metrics_{dataset}.json"), "w") as f:
        json.dump(out, f, indent=2)

    table_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(table_dir, exist_ok=True)
    cols = (["attack", "asr_undefended", "mean_queries"]
            + [f"fire_{t}" for t in TRIGGERS] + [f"maxstat_{t}" for t in TRIGGERS])
    with open(os.path.join(table_dir, f"exp21_trigger_matrix_{dataset}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] if not isinstance(r[c], float) else f"{r[c]:.6g}" for c in cols])

    print("\n=== EXP 21 SUMMARY (RQ3 trigger x attack-family matrix) ===")
    print(f"{'attack':22s} " + " ".join(f"{t[:12]:>13s}" for t in TRIGGERS))
    for r in rows:
        print(f"{r['attack']:22s} " + " ".join(f"{r[f'fire_{t}']*100:12.1f}%" for t in TRIGGERS))
    n_blind_mom = sum(1 for r in rows if r["fire_momentum"] < 0.1)
    print(f"\nattacks firing momentum trigger <10% of sessions: {n_blind_mom}/{len(rows)}")
    all_fire = [t for t in TRIGGERS if all(r[f"fire_{t}"] > 0.5 for r in rows)]
    print(f"triggers firing on >50% of sessions for EVERY attack: {all_fire or 'none'}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "unsw_nb15"
    main(ds)
