"""Offline-attacker analysis (P1-9): the strongest threat model against a
stateful defense is *offline* — optimize against an undefended replica, then
submit a single flow. If the submitted point differs from the session's first
query, the defense's Δθ is 0 on the first evaluation (no history), so the
defended verdict must equal the undefended one.

We verify this exactly:
  1. bit-identical check: TASCP's output logits on the FIRST query of a
     session are bit-identical to DACM's (static clamp) on the same input —
     and to the undefended model's — because delta_theta = 0 when there is
     no rotation plane (len(history) < 2).
  2. Optimize x* against the undefended replica with white-box PGD, submit
     x* as the single flow of a fresh session to TASCP: verdict must equal
     the replica's (bit-identical logits).
  3. Surrogate-mismatch variant: the replica differs from the deployed model
     (transfer attack). TASCP still cannot help — it never sees the
     optimization trajectory, only the final transfer point; the outcome is
     whatever the surrogate's transfer success is, i.e. the same as DACM's.
"""

import math

import torch


def _fresh_session(model, x, batch_size=None, device="cpu"):
    """Reset the wrapper's history and evaluate x as the FIRST query of a session."""
    b = x.size(0) if batch_size is None else batch_size
    if hasattr(model, "reset_history"):
        model.reset_history(b, x.device)
    with torch.no_grad():
        return model(x)


def offline_attack_bit_identical(base_model, x_star, tau=0.0, gamma=50.0, max_theta=None):
    """Submit x* (optimized offline) as a single fresh-session query.

    Returns dict with:
      tascp_logits, dacm_logits, undefended_logits, tascp_bit_identical_dacm,
      tascp_bit_identical_undefended, max_abs_diff
    """
    import sys
    if max_theta is None:
        max_theta = math.pi / 2
    from src.tascp_eval.defenses.tascp import TASCPWrapper
    from src.tascp_eval.baselines.dacm import DACMWrapper

    tascp = TASCPWrapper(base_model, window_size=5, max_theta=max_theta, tau=tau, gamma=gamma)
    dacm = DACMWrapper(base_model)

    with torch.no_grad():
        undef = base_model(x_star)
        t = _fresh_session(tascp, x_star)
        d = dacm(x_star)
        max_diff_td = (t - d).abs().max().item()
        max_diff_tu = (t - undef).abs().max().item()

    return {
        "tascp_logits": t,
        "dacm_logits": d,
        "undefended_logits": undef,
        "tascp_bit_identical_dacm": torch.equal(t, d),
        "tascp_bit_identical_undefended": torch.equal(t, undef),
        "max_abs_diff_tascp_dacm": max_diff_td,
        "max_abs_diff_tascp_undefended": max_diff_tu,
        "verdict_agreement": bool(torch.equal((t > 0), (undef > 0))),
    }


def offline_attack_surrogate_mismatch(surrogate_model, deployed_model, x, config, seed=42):
    """Optimize against a *different* replica (surrogate), submit to the
    deployed model — fresh-session single query. TASCP cannot rotate (Δθ=0)
    on the first query, so the outcome must match the undefended deployed
    model's verdict on the transferred point (transfer attack baseline)."""
    from src.tascp_eval.defenses.tascp import TASCPWrapper
    from src.tascp_eval.baselines.dacm import DACMWrapper
    from .bpda_adaptive import stateful_pgd_attack

    torch.manual_seed(seed)
    y = torch.zeros(x.size(0), 1)
    # 1. optimize against the surrogate replica (it is undefended)
    x_star, _, _ = stateful_pgd_attack(surrogate_model, x, y, config, adaptive=False)

    # 2. single fresh-session submission to the deployed model
    res = offline_attack_bit_identical(deployed_model, x_star)
    dacm = DACMWrapper(deployed_model)
    with torch.no_grad():
        d_logits = dacm(x_star)
        s_logits = surrogate_model(x_star)
    res["surrogate_mismatch"] = {
        "surrogate_logits": s_logits,
        "dacm_logits": d_logits,
        "transfer_success_undefended": float(((deployed_model(x_star) > 0).squeeze(-1)).float().mean().item()),
        "transfer_success_dacm": float(((d_logits > 0).squeeze(-1)).float().mean().item()),
        "transfer_success_tascp_fresh_session": float(((res["tascp_logits"] > 0).squeeze(-1)).float().mean().item()),
    }
    return res, x_star


def full_offline_analysis(dataset_model, X, config, seeds=(42,), surrogate=None):
    """Run the complete offline-attacker experiment block on real data.

    Conditions:
      A. white-box offline: optimize vs deployed replica (undefended), submit 1 flow
      B. surrogate mismatch: optimize vs surrogate != deployed, submit 1 flow
    Reports bit-identity assertions and transfer ASRs.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    results = {}
    for seed in seeds:
        torch.manual_seed(seed)
        # A. white-box offline
        from .bpda_adaptive import stateful_pgd_attack

        y = torch.zeros(X.size(0), 1)
        x_star, _, _ = stateful_pgd_attack(dataset_model, X, y, config, adaptive=False)
        a = offline_attack_bit_identical(dataset_model, x_star)
        results["whitebox_offline"] = {
            "bit_identical_dacm": a["tascp_bit_identical_dacm"],
            "bit_identical_undefended": a["tascp_bit_identical_undefended"],
            "max_abs_diff": a["max_abs_diff_tascp_dacm"],
            "verdict_agreement": a["verdict_agreement"],
            "submitted_asr": float(((a["undefended_logits"] > 0).squeeze(-1)).float().mean().item()),
        }
        if surrogate is not None:
            b, _ = offline_attack_surrogate_mismatch(surrogate, dataset_model, X, config, seed=seed)
            results["surrogate_mismatch"] = b["surrogate_mismatch"]
            results["surrogate_mismatch"]["bit_identical_dacm"] = b["tascp_bit_identical_dacm"]
    return results
