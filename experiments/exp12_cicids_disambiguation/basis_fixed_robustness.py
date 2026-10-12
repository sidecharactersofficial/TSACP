"""Basis-fixed TASCP robustness tests (P1-7):

1. Reference-vector robustness: exp8's basis-coupling ablation overrides the
   rotation plane with the FIXED pair (e_0, e_1). A reviewer will ask whether
   the no-op result depends on that specific choice. We re-run the adaptive
   BPDA attack against basis-fixed TASCP for R different fixed reference
   pairs (random unit vectors, and the canonical axes) and report the spread
   of ASRs.

2. Basis-fixed on XGBoost: exp8's ablation grid was run on the MLP only. Here
   the same basis-fixed condition runs against the XGBoost-surrogate path
   (gradients through the surrogate MLP, verdicts from the real XGBoost), so
   the "basis-fixed collapses TASCP to the undefended baseline" claim is not
   classifier-specific.
"""

import math
import os
import sys

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
for _p in (
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "experiments"),
    os.path.join(PROJECT_ROOT, "experiments", "exp5_adaptive_bpda_attack"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.tascp_eval.defenses.tascp import TASCPWrapper, _apply_rotation
from src.tascp_eval.attacks.bpda_adaptive import stateful_pgd_attack
from src.tascp_eval.attacks.pgd import load_config


def make_basis_fixed_ref(base_model, ref_pair=None):
    """TASCPWrapper whose basis-coupling ablation uses an arbitrary fixed
    reference pair (u_ref, v_ref) instead of (e_0, e_1). Implemented by
    monkey-patching _apply_rotation's fixed vectors via a closure wrapper:
    we temporarily set the model's ablation and patch the module-level
    function through a small subclass."""
    w = TASCPWrapper(base_model, window_size=5, max_theta=math.pi / 2, tau=0.0, gamma=50.0)
    w.adaptive = True
    w.ablation_mode = "basis_coupling"
    if ref_pair is None:
        return w  # canonical (e_0, e_1) behavior from tascp.py

    u_ref, v_ref = ref_pair  # (d,), (d,) fixed unit vectors

    class _RefFixedTASCP(TASCPWrapper):
        def forward(self, x_t, apply_rotation=True):
            # replicate TASCPWrapper.forward but substitute the fixed basis
            # only at the _apply_rotation call (keep trigger math identical).
            if not apply_rotation or len(self.history) < 2:
                return self.base_model(x_t)
            x_prev = self.history[-1]
            v_t = x_t - x_prev
            if len(self.history) < 2:
                return self.base_model(x_t)
            x_prev_prev = self.history[-2]
            v_prev = x_prev - x_prev_prev
            if not getattr(self, "adaptive", False):
                v_t = v_t.detach()
                v_prev = v_prev.detach()

            norm_t = torch.norm(v_t, p=2, dim=1, keepdim=True)
            norm_prev = torch.norm(v_prev, p=2, dim=1, keepdim=True)
            valid = (norm_t > 1e-9) & (norm_prev > 1e-9)
            dot_product = torch.sum(v_t * v_prev, dim=1, keepdim=True)
            denom = torch.clamp(norm_t * norm_prev, min=1e-9)
            m_t = torch.where(valid, dot_product / denom, torch.zeros_like(norm_t))

            window_v, window_m = [], []
            for i in range(1, len(self.history)):
                curr_x = self.history[i]
                prev_x = self.history[i - 1]
                v_i = curr_x - prev_x
                if i > 1:
                    pp_x = self.history[i - 2]
                    v_i_prev = prev_x - pp_x
                    n_i = torch.norm(v_i, p=2, dim=1, keepdim=True)
                    n_ip = torch.norm(v_i_prev, p=2, dim=1, keepdim=True)
                    val = (n_i > 1e-9) & (n_ip > 1e-9)
                    dot_i = torch.sum(v_i * v_i_prev, dim=1, keepdim=True)
                    m_i = torch.where(val, dot_i / torch.clamp(n_i * n_ip, min=1e-9), torch.zeros_like(n_i))
                else:
                    m_i = torch.zeros_like(norm_t)
                window_v.append(v_i)
                window_m.append(m_i)
            window_v.append(v_t)
            window_m.append(m_t)

            eps_p = torch.zeros_like(norm_t)
            for v_i, m_i in zip(window_v, window_m):
                eps_p += torch.norm(v_i, p=2, dim=1, keepdim=True) * torch.clamp(m_i, min=0.0)
            eps_p /= len(window_v)

            delta_theta = self.gamma * torch.clamp(eps_p - self.tau, min=0.0)
            delta_theta = torch.clamp(delta_theta, max=self.max_theta)
            if not getattr(self, "adaptive", False) or self.ablation_mode == "gradient_magnitude":
                delta_theta = delta_theta.detach()

            # basis-fixed with CUSTOM reference pair
            u = u_ref.expand_as(x_t).contiguous()
            v_perp = (v_ref - (v_ref * u_ref).sum() * u_ref).expand_as(x_t).contiguous()
            comp_u = torch.sum(x_t * u, dim=1, keepdim=True)
            comp_perp = torch.sum(x_t * v_perp, dim=1, keepdim=True)
            ortho = x_t - comp_u * u - comp_perp * v_perp
            c, s = torch.cos(delta_theta), torch.sin(delta_theta)
            x_rot = ortho + (comp_u * c - comp_perp * s) * u + (comp_u * s + comp_perp * c) * v_perp
            x_rot = x_rot.clamp(0.0, 1.0) if False else x_rot

            self.last_delta_theta = delta_theta
            self.last_u = u
            self.last_v_perp = v_perp
            self.last_x_rot = x_rot
            return self.base_model(x_rot)

    out = _RefFixedTASCP(base_model, window_size=5, max_theta=math.pi / 2, tau=0.0, gamma=50.0)
    out.adaptive = True
    out.ablation_mode = "basis_coupling"
    return out


def reference_vector_robustness(base_model, X_sample, y_target, config, n_refs=8, seed=42):
    """Run adaptive BPDA against basis-fixed TASCP with R different fixed
    reference pairs. Report per-reference ASR + spread. Expectation from
    Theorem 1: ALL reference choices collapse to ~the undefended ASR (the
    no-op is not an artifact of (e_0, e_1))."""
    d = X_sample.size(1)
    gen = torch.Generator().manual_seed(seed)
    refs = [None]  # None = canonical (e_0, e_1) from tascp.py
    for _ in range(n_refs - 1):
        a = torch.randn(d, generator=gen)
        a = a / a.norm()
        b = torch.randn(d, generator=gen)
        b = b - (b * a).sum() * a
        b = b / b.norm()
        refs.append((a, b))

    results = {}
    for i, ref in enumerate(refs):
        label = "e0,e1 (canonical)" if ref is None else f"random pair {i}"
        model = make_basis_fixed_ref(base_model, ref)
        torch.manual_seed(seed)
        x_adv, success, queries = stateful_pgd_attack(model, X_sample, y_target, config, adaptive=True)
        asr = success.float().mean().item()
        results[label] = asr
        print(f"  basis-fixed ref [{label:20s}] ASR {asr*100:6.2f}%")
    asrs = list(results.values())
    print(
        f"Basis-fixed reference robustness: mean {np.mean(asrs)*100:.2f}% "
        f"std {np.std(asrs)*100:.2f}% range [{min(asrs)*100:.2f}%, {max(asrs)*100:.2f}%]"
    )
    return results
