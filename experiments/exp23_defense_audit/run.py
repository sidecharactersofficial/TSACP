"""RQ6: do the three audits distinguish different stateful defenses?

Audits THREE stateful/query-history defenses with the same instruments:

  tascp            trajectory-conditioned transformation (this paper)
  blacklight_sim   Blacklight-STYLE near-duplicate density detector
                   (L2 fingerprint test from exp10 — a simplification of
                   Li et al. USENIX Sec'22, as documented there)
  rate_limit       per-session query-budget limiter (the standard
                   operational rate-limit defense)

Audits (identical protocol per defense):
  A1 coupling       can attacker-controlled state change the verdict
                    computation geometrically? For differentiable
                    transformations: C_norm via autograd. For detectors:
                    empirically, can state evolution flip a verdict in the
                    ALLOW direction? (nu_allow below)
  A2 trigger obs.   fraction of query-efficient attack sessions that
                    complete without EVER tripping the defense's trigger
                    (evasion rate), plus undefended ASR context
  A3 stationarity   verdict flips of a frozen probe while only session
                    state evolves: nu_session (any flip), nu_allow (flips
                    toward allow — the exploitable direction)

Outcome table (defense x audit), the reviewer's "killer experiment":
  TASCP          coupling>0, trigger partially blind, NON-stationary both ways
  detectors      no coupling channel, evasion possible (adaptive query
                 shaping), stationary but monotone-toward-reject
Supports the paper's claim: statefulness is safer when used to
detect/reject than to change the geometry of verdict computation — and the
audits are what distinguish the two.

CLI: python -m experiments.exp23_defense_audit.run [dataset]
Writes results/tables/exp23_defense_audit_<ds>.csv and
results/logs/exp23_defense_audit/metrics_<ds>.json.
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
           os.path.join(PROJECT_ROOT, "experiments", "exp10_hybrid_detection_sim"),
           os.path.join(PROJECT_ROOT, "experiments", "exp11_detector_robustness"),
           os.path.join(PROJECT_ROOT, "experiments", "exp19_trajectory_family"),
           os.path.join(PROJECT_ROOT, "experiments", "exp21_trigger_matrix")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.exp9_blackbox_attack.exp9_common import load_data  # noqa: E402
from experiments.exp10_hybrid_detection_sim.run import FingerprintDetector  # noqa: E402
from experiments.exp21_trigger_matrix.run import (ATTACKS, QueryRecorder,  # noqa: E402
                                                  benign_sessions, run_attack)
from experiments.exp22_arch_split.run import measure_coupling  # noqa: E402
from family import FamilyWrapper  # noqa: E402

AUDIT_ATTACKS = ("nes_independent", "nes_antithetic_jitter", "square", "random_search")
AUDIT_CFG = {"epsilon": 0.1, "sigma": 0.05, "alpha": 0.02, "q": 10,
             "max_queries": 100, "square_p": 0.05, "boundary_step": 0.05}


# ---------------------------------------------------------------------------
# Defense wrappers with a uniform stateful protocol
# (reset_history / add_to_history / forward -> alert logit).
# Verdict semantics: logit > 0 = ALERT (flag/reject), <= 0 = serve.
# ---------------------------------------------------------------------------
class TASCPDefense(torch.nn.Module):
    """TASCP trajectory-conditioned transformation (family wrapper)."""

    def __init__(self, base_model):
        super().__init__()
        self.w = FamilyWrapper(base_model, variant="tascp", tau=0.0, gamma=50.0)
        self.w.adaptive = False

    def forward(self, x):
        return self.w(x)

    def reset_history(self, batch, device):
        self.w.reset_history(batch, device)

    def add_to_history(self, x):
        self.w.add_to_history(x)


class BlacklightSimDefense(torch.nn.Module):
    """Near-duplicate density detector: flags a query when its session window
    contains >= k_min queries within tau_match. Monotone: detection state
    only accumulates; there is no operation that un-flags a session."""

    def __init__(self, base_model, tau_match=0.1, k_min=5, window=50):
        super().__init__()
        self.base = base_model
        self.det = FingerprintDetector(tau_match=tau_match, k_min=k_min, window=window)

    def forward(self, x):
        hit = self.det.check_and_update("client0", x)
        self.last_hit = hit
        out = self.base(x)
        # flagged rows: alert logit (reject). Served rows: base verdict.
        return torch.where(hit.unsqueeze(1), torch.full_like(out, 1e3), out)

    def reset_history(self, batch, device):
        self.det.reset()

    def add_to_history(self, x):
        pass  # detector state updates inside forward (check_and_update)


class RateLimitDefense(torch.nn.Module):
    """Per-session query budget: alert once the session exceeds budget."""

    def __init__(self, base_model, budget=50):
        super().__init__()
        self.base = base_model
        self.budget = budget
        self.count = 0

    def forward(self, x):
        self.count += 1
        out = self.base(x)
        if self.count > self.budget:
            return torch.full_like(out, 1e3)
        return out

    def reset_history(self, batch, device):
        self.count = 0

    def add_to_history(self, x):
        pass


DEFENSES = ("tascp", "blacklight_sim", "rate_limit")


# ---------------------------------------------------------------------------
# A1: coupling + A3: stationarity (shared frozen-probe replay)
# ---------------------------------------------------------------------------
def stationarity_audit(defense, probe, n_steps=12, step_scale=0.3, seed=42):
    """Frozen-probe replay with a persistent-direction session stream.
    The probe pool MIXES benign flows (v0=0) and adversarial points
    (v0=1) so nu_allow — flips from ALERT to serve, the direction an
    attacker exploits — is measured on the subset it applies to
    (normalized by the count of v0==1 probes, 0 if none)."""
    device = probe.device
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    defense.reset_history(probe.size(0), device)
    with torch.no_grad():
        v0 = (defense(probe).squeeze(-1) > 0)
    n_adv0 = int(v0.sum().item())

    u = torch.nn.functional.normalize(torch.randn(probe.shape, generator=g, device=device), dim=1)
    cursor = probe.clone()
    flips = flips_allow = 0
    alerts = 0
    for _ in range(n_steps):
        nxt = (cursor + step_scale * u).clamp(0.0, 1.0)
        defense.add_to_history(nxt)
        with torch.no_grad():
            v = (defense(probe).squeeze(-1) > 0)
        flips += int((v != v0).sum().item())
        flips_allow += int((v0 & ~v).sum().item())
        alerts += int(v.sum().item())
        cursor = nxt
    with torch.no_grad():
        v_final = (defense(probe).squeeze(-1) > 0)
    return {
        "nu_session": flips / (n_steps * probe.size(0)),
        "nu_allow": flips_allow / max(n_adv0 * n_steps, 1) if n_adv0 else 0.0,
        "n_adv0": n_adv0,
        "nu_final": float((v_final != v0).float().mean().item()),
        "alert_rate": alerts / (n_steps * probe.size(0)),
    }


def audit_defense(name, defense, base_model, X):
    r = {"defense": name}
    # A1 coupling: for the differentiable transformation, C_norm; for the
    # detectors there is no differentiable transform to audit (the empirical
    # analog is stat_nu_allow below).
    if name == "tascp":
        C, firing = measure_coupling(base_model, X)
        r["coupling_C_norm"] = C
        r["coupling_firing_frac"] = firing
    else:
        r["coupling_C_norm"] = None
        r["coupling_firing_frac"] = None

    # A3 stationarity on a MIXED probe pool: benign flows + adversarial
    # points (v0=1) so the toward-allow flip rate is measurable.
    adv = build_adversarial_probes(base_model, X)
    pool = torch.cat([X, adv], dim=0)
    stat = stationarity_audit(defense, pool)
    r.update({f"stat_{k}": v for k, v in stat.items()})

    # A2 trigger observability. Semantics per defense class:
    #   tascp          attack SUCCESS rate through the stateful defense
    #                  (the defense perturbs; evasion = still succeeding)
    #   detectors      fraction of attack sessions that complete without the
    #                  defense EVER alerting (undetected rate)
    rec = QueryRecorder(defense)
    evasions = {}
    for atk in AUDIT_ATTACKS:
        rec.reset()
        try:
            if name == "tascp":
                _, succ, _, _ = run_attack(rec, ATTACKS[atk], X, None, AUDIT_CFG)
                evasions[atk] = float(succ.float().mean().item())
            else:
                run_attack(rec, ATTACKS[atk], X, None, AUDIT_CFG)
                evasions[atk] = session_evasion(defense, rec.session(), base_model)
        except Exception as e:
            evasions[atk] = None
            print(f"  [warn] {atk} failed on {name}: {type(e).__name__}: {e}")
    r["evasion"] = evasions
    return r


def session_evasion(defense, seq, base_model):
    """Replay a recorded query session through a FRESH defense of the same
    class; returns fraction of samples whose session never alerts. Stateful
    wrappers get their history driven explicitly (the recorded session IS the
    defender-observed query stream)."""
    import copy

    fresh_def = copy.deepcopy(defense)
    fresh_def.reset_history(seq.size(1), seq.device)
    B = seq.size(1)
    ever_alert = torch.zeros(B, dtype=torch.bool)
    with torch.no_grad():
        for t in range(seq.size(0)):
            if hasattr(fresh_def, "add_to_history") and "tascp" in getattr(fresh_def, "__class__").__name__.lower():
                fresh_def.add_to_history(seq[t])
            out = fresh_def(seq[t])
            ever_alert |= (out.squeeze(-1) > 0)
    return float((~ever_alert).float().mean().item())


def build_adversarial_probes(base_model, X, eps=0.3, alpha=0.01, iters=20, seed=42):
    """PGD-adversarial points (undefended model says ATTACK) used as the
    v0=1 probe subset for the toward-allow stationarity audit."""
    from experiments.exp19_trajectory_family.run import PGD_CFG
    from src.tascp_eval.attacks.bpda_adaptive import stateful_pgd_attack

    y = torch.zeros(X.size(0), 1)
    cfg = dict(PGD_CFG)
    torch.manual_seed(seed)
    x_adv, succ, _ = stateful_pgd_attack(base_model, X, y, cfg, adaptive=False)
    keep = succ
    if keep.sum() < max(10, X.size(0) // 10):
        keep = torch.ones_like(succ)  # fall back to all if too few flipped
    return x_adv[keep].detach()


def main(dataset="unsw_nb15", n_samples=100, seed=42):
    torch.manual_seed(seed)
    base_model, feature_cols, spec, levels, X = load_data(dataset, n_samples, seed)

    defenses = {
        "tascp": TASCPDefense(base_model),
        "blacklight_sim": BlacklightSimDefense(base_model),
        "rate_limit": RateLimitDefense(base_model, budget=50),
    }

    results = {}
    for name, defense in defenses.items():
        print(f"\n=== defense: {name} ===")
        r = audit_defense(name, defense, base_model, X)
        results[name] = r
        for k, v in r.items():
            print(f"  {k}: {v if not isinstance(v, float) else round(v, 4)}")

    # benign control: do the detectors alert on benign re-check sessions?
    benign = benign_sessions(base_model, X, n_repeat=40)
    benign_fp = {}
    for name, defense in defenses.items():
        fp = session_evasion(defense, benign, base_model)  # evasion of benign = not flagged
        benign_fp[name] = 1.0 - fp  # FP = flagged benign sessions
    print("\nbenign false-positive rates:", {k: round(v, 4) for k, v in benign_fp.items()})

    out_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp23_defense_audit")
    os.makedirs(out_dir, exist_ok=True)
    payload = {"dataset": dataset, "n_samples": X.size(0), "defenses": results,
               "benign_fp": benign_fp}
    with open(os.path.join(out_dir, f"metrics_{dataset}.json"), "w") as f:
        json.dump(payload, f, indent=2)

    table_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(table_dir, exist_ok=True)
    cols = ["coupling_C_norm", "stat_nu_session", "stat_nu_allow", "stat_nu_final",
            "stat_alert_rate"] + [f"evasion_{a}" for a in AUDIT_ATTACKS] + ["benign_fp"]
    with open(os.path.join(table_dir, f"exp23_defense_audit_{dataset}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["defense"] + cols)
        for name, r in results.items():
            row = [name]
            for c in cols:
                if c == "benign_fp":
                    row.append(f"{benign_fp[name]:.6g}")
                else:
                    v = r.get(c)
                    row.append(f"{v:.6g}" if isinstance(v, float) else (v if v is not None else ""))
            w.writerow(row)

    print("\n=== EXP 23 SUMMARY (RQ6 defense audit) ===")
    print(f"{'defense':15s} {'coupling':>9s} {'nu_any':>7s} {'nu_allow':>8s} "
          + " ".join(f"{a[:14]:>15s}" for a in AUDIT_ATTACKS) + "   benignFP")
    for name, r in results.items():
        c = f"{r['coupling_C_norm']:.3f}" if r.get("coupling_C_norm") is not None else "n/a"
        ev = " ".join(
            (f"{r['evasion'][a]*100:14.1f}%" if r["evasion"].get(a) is not None else f"{'n/a':>15s}")
            for a in AUDIT_ATTACKS)
        print(f"{name:15s} {c:>9s} {r['stat_nu_session']*100:6.2f}% "
              f"{r['stat_nu_allow']*100:7.2f}% {ev}   {benign_fp[name]*100:.1f}%")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "unsw_nb15"
    main(ds)
