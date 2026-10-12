"""Trajectory-conditioned transformation family (RQ1/RQ2 infrastructure).

The reviewer's central question: is the exploitable coupling phenomenon
specific to TASCP's spiral/rotation construction, or does ANY
trajectory-conditioned transformation expose it? This module builds
controlled synthetic members of the same design family

    x_t -> h_t -> T(h_t, x_t) -> f(T(h_t, x_t))

that share the state-extraction stage (h_t from the observed query window)
but differ ONLY in the transformation stage:

  tascp    directional momentum    rotation (Givens, the original)
  tscale   directional momentum    feature-wise scaling
  tshift   directional momentum    translation along running mean direction
  tshear   directional momentum    rank-1 shear (linear map)
  tnoise   trajectory persistence  state-scaled randomized perturbation

All members are deterministic given (h_t, x_t) except tnoise, whose noise is
reparameterized (x' = x + sigma(S) * xi, xi fixed per call from a seeded
generator) so gradients and coupling audits remain well-defined.

Every member exposes the same module contract as TASCPWrapper:
  reset_history(batch, device) / add_to_history(x) / forward(x)
  .adaptive flag (False detaches the state path, True keeps the graph)
  .freeze_state (control: S is clamped to this value -> no trajectory coupling)
so the existing attack harness (stateful_pgd_attack, NES, HSJ, ...) drives
every member without modification.

Coupling audit: C = || d T(h,x) / d h_t-1 ||_F per sample (history entries are
the attacker's past queries, so dh/dx = I by construction; C is therefore the
complete coupling Jacobian norm of the defense's state channel).
"""

import math

import torch
import torch.nn as nn

from src.tascp_eval.defenses.tascp import _apply_rotation

FAMILY_VARIANTS = ("tascp", "tscale", "tshift", "tshear", "tnoise")

# Strength constants: chosen so that at max state the transform's input
# displacement is comparable to TASCP's max_theta = pi/2 rotation (which moves
# a unit-norm point by up to sqrt(2)). Same-impact scaling across variants.
SCALE_ALPHA = 0.5   # tscale: x' = x * (1 + alpha * S * |v_hat|), S in [0, S_MAX]
SHIFT_BETA = 0.5    # tshift: x' = x + beta * S * r_hat
SHEAR_KAPPA = 0.5   # tshear: x' = (I + kappa * S * wv^T) x
NOISE_SIGMA = 0.5   # tnoise: x' = x + sigma * S * xi
S_MAX = 2.0         # state saturation (matches TASCP's max_theta scale)


def _window_state(history, x_t):
    """TASCP's exact state extraction: windowed directional momentum.

    Returns (v_t, v_prev, eps_p) where eps_p is the windowed
    sum ||v_i|| * max(0, M_i) averaged over the window (the same eps_p the
    TASCP trigger uses). Everything is differentiable in x_t and (optionally)
    in the history entries.
    """
    x_prev = history[-1]
    v_t = x_t - x_prev
    x_pp = history[-2]
    v_prev = x_pp - history[-3] if len(history) >= 3 else x_prev - x_prev

    norm_t = torch.norm(v_t, p=2, dim=1, keepdim=True)
    norm_prev = torch.norm(v_prev, p=2, dim=1, keepdim=True)
    valid = (norm_t > 1e-9) & (norm_prev > 1e-9)
    dot = torch.sum(v_t * v_prev, dim=1, keepdim=True)
    m_t = torch.where(valid, dot / torch.clamp(norm_t * norm_prev, min=1e-9), torch.zeros_like(norm_t))

    window_terms = []
    for i in range(1, len(history)):
        v_i = history[i] - history[i - 1]
        if i > 1:
            v_im1 = history[i - 1] - history[i - 2]
            n_i = torch.norm(v_i, p=2, dim=1, keepdim=True)
            n_im1 = torch.norm(v_im1, p=2, dim=1, keepdim=True)
            val = (n_i > 1e-9) & (n_im1 > 1e-9)
            m_i = torch.where(val, torch.sum(v_i * v_im1, dim=1, keepdim=True) / torch.clamp(n_i * n_im1, min=1e-9), torch.zeros_like(n_i))
        else:
            m_i = torch.zeros_like(norm_t)
            n_i = torch.norm(v_i, p=2, dim=1, keepdim=True)
        window_terms.append(torch.norm(v_i, p=2, dim=1, keepdim=True) * torch.clamp(m_i, min=0.0))
    window_terms.append(norm_t * torch.clamp(m_t, min=0.0))

    eps_p = sum(window_terms) / max(len(window_terms), 1)
    return v_t, v_prev, eps_p


def _normalize_rows(z, eps=1e-9):
    return z / torch.clamp(torch.norm(z, p=2, dim=1, keepdim=True), min=eps)


# ---------------------------------------------------------------------------
# Pluggable state functions (RQ2): the transformation family keeps the same
# geometry (v_t, v_prev) but the scalar state S can come from any trajectory
# statistic. Each returns S in [0, S_MAX] (batch, 1), differentiable in the
# history entries and x_t.
# ---------------------------------------------------------------------------

def _disp_seqs(history, x_t):
    """All consecutive displacement vectors of the window incl. x_t."""
    seq = list(history) + [x_t]
    return [seq[i] - seq[i - 1] for i in range(1, len(seq))]


def _state_cosine(history, x_t):
    """Mean signed cosine similarity of consecutive displacements, rescaled
    to [0, S_MAX]: fires on any consistent direction (signed, not rectified)."""
    vs = _disp_seqs(history, x_t)
    terms = []
    for i in range(1, len(vs)):
        n_i = torch.norm(vs[i], p=2, dim=1, keepdim=True)
        n_j = torch.norm(vs[i - 1], p=2, dim=1, keepdim=True)
        ok = (n_i > 1e-9) & (n_j > 1e-9)
        c = torch.where(ok, torch.sum(vs[i] * vs[i - 1], dim=1, keepdim=True)
                        / torch.clamp(n_i * n_j, min=1e-9), torch.zeros_like(n_i))
        terms.append(c)
    mean_c = sum(terms) / max(len(terms), 1)
    return S_MAX * torch.clamp(mean_c, min=0.0, max=1.0)


def _state_disp_mag(history, x_t):
    """Mean query displacement magnitude (no direction information)."""
    vs = _disp_seqs(history, x_t)
    mean_n = sum(torch.norm(v, p=2, dim=1, keepdim=True) for v in vs) / len(vs)
    return torch.clamp(mean_n, max=S_MAX)


def _state_running_mean_dir(history, x_t):
    """Directional persistence: ||mean displacement|| / mean ||disp|| x S_MAX
    (translation-invariant consistency of the session's movement)."""
    vs = _disp_seqs(history, x_t)
    stack = torch.stack(vs, dim=0)
    mean_vec = stack.mean(dim=0)
    mean_n = stack.norm(p=2, dim=2).mean(dim=0, keepdim=True).t()
    ratio = torch.norm(mean_vec, p=2, dim=1, keepdim=True) / torch.clamp(mean_n, min=1e-9)
    return S_MAX * torch.clamp(ratio, max=1.0)


def _state_ema(history, x_t):
    """Deviation of x_t from the exponential moving average of the session's
    queries (h_t = 0.5 h_{t-1} + 0.5 x_t): S = ||x_t - h_t|| clipped."""
    lam = 0.5
    h = history[0]
    for i in range(1, len(history)):
        h = lam * h + (1 - lam) * history[i]
    return torch.clamp(torch.norm(x_t - h, p=2, dim=1, keepdim=True), max=S_MAX)


def _state_curvature(history, x_t):
    """Change in direction: S_MAX x (1 - cos(v_t, v_prev))/2 — high when the
    session reverses (antithetic query patterns)."""
    vs = _disp_seqs(history, x_t)
    if len(vs) < 2:
        return torch.zeros(x_t.size(0), 1, device=x_t.device)
    v_t, v_p = vs[-1], vs[-2]
    n_t = torch.norm(v_t, p=2, dim=1, keepdim=True)
    n_p = torch.norm(v_p, p=2, dim=1, keepdim=True)
    ok = (n_t > 1e-9) & (n_p > 1e-9)
    c = torch.where(ok, torch.sum(v_t * v_p, dim=1, keepdim=True)
                    / torch.clamp(n_t * n_p, min=1e-9), torch.zeros_like(n_t))
    return S_MAX * torch.clamp((1.0 - c) / 2.0, min=0.0, max=1.0)


STATE_FNS = {
    "momentum": None,  # TASCP's eps_p (handled specially in _state_scalar)
    "cosine": _state_cosine,
    "disp_mag": _state_disp_mag,
    "running_mean_dir": _state_running_mean_dir,
    "ema": _state_ema,
    "curvature": _state_curvature,
}
STATE_FN_NAMES = tuple(STATE_FNS.keys())


class FamilyWrapper(nn.Module):
    """One member of the trajectory-conditioned transformation family."""

    def __init__(self, base_model, variant="tascp", window_size=5,
                 tau=0.0, gamma=50.0, max_theta=math.pi / 2, seed=42,
                 state_fn="momentum"):
        super().__init__()
        assert variant in FAMILY_VARIANTS, variant
        assert state_fn in STATE_FNS, state_fn
        self.base_model = base_model
        self.variant = variant
        self.state_fn = state_fn
        self.window_size = window_size
        self.tau = tau
        self.gamma = gamma
        self.max_theta = max_theta
        self.history = None
        self.adaptive = False
        # fixed-state control: when not None, S is clamped to this value
        self.freeze_state = None
        self._noise_gen = torch.Generator()
        self._noise_gen.manual_seed(seed)
        self.last_S = None
        self.last_x_transformed = None

    # -- stateful protocol ------------------------------------------------
    def reset_history(self, batch_size, device):
        self.history = []

    def add_to_history(self, x):
        self.history.append(x.detach().clone())
        if len(self.history) > self.window_size + 1:
            self.history.pop(0)

    # -- core -------------------------------------------------------------
    def _state_scalar(self, x_t, detach_state):
        """Shared state extraction: returns S in [0, S_MAX] (batch, 1)."""
        if self.state_fn == "momentum":
            v_t, v_prev, s = _window_state(self.history, x_t)
        else:
            v_t = x_t - self.history[-1]
            v_prev = (self.history[-2] - self.history[-3]
                      if len(self.history) >= 3 else torch.zeros_like(v_t))
            s = STATE_FNS[self.state_fn](self.history, x_t)
        if self.variant == "tnoise":
            # state = persistence of direction, unscaled by displacement
            # magnitude (noise strength should not scale with |v| alone)
            n_t = torch.norm(v_t, p=2, dim=1, keepdim=True)
            S = torch.where(n_t > 1e-9, torch.ones_like(n_t), torch.zeros_like(n_t)) * s
        else:
            S = s
        S = torch.clamp(S, max=S_MAX)
        if self.freeze_state is not None:
            S = torch.full_like(S, float(self.freeze_state))
        if detach_state:
            S = S.detach()
        return S, v_t, v_prev

    def _transform(self, x_t, v_t, v_prev, S):
        """T(h_t, x_t): the transformation stage. Differentiable in x_t, v_t,
        v_prev, S unless the caller detached them."""
        if self.variant == "tascp":
            delta_theta = torch.clamp(self.gamma * torch.clamp(S - self.tau, min=0.0), max=self.max_theta)
            x_out, _, _ = _apply_rotation(x_t, v_t, v_prev, delta_theta)
            return x_out
        if self.variant == "tscale":
            # feature-wise scaling driven by the current displacement direction
            feat_dir = torch.abs(_normalize_rows(v_t))
            gate = torch.clamp(self.gamma * torch.clamp(S - self.tau, min=0.0), max=self.max_theta) / self.max_theta
            return x_t * (1.0 + SCALE_ALPHA * gate * feat_dir)
        if self.variant == "tshift":
            # translation along the running mean displacement direction
            disp = torch.stack([self.history[i] - self.history[i - 1] for i in range(1, len(self.history))], dim=0)
            r_hat = _normalize_rows(disp.mean(dim=0))
            offset = SHIFT_BETA * torch.clamp(self.gamma * torch.clamp(S - self.tau, min=0.0), max=self.max_theta) / self.max_theta
            return x_t + offset * r_hat
        if self.variant == "tshear":
            # rank-1 shear: (I + kappa * gate * w v^T) x with w = v_prev_perp
            gate = torch.clamp(self.gamma * torch.clamp(S - self.tau, min=0.0), max=self.max_theta) / self.max_theta
            v_hat = _normalize_rows(v_t)
            w = v_prev - torch.sum(v_prev * v_hat, dim=1, keepdim=True) * v_hat
            w_hat = _normalize_rows(w)
            # x' = x + kappa*gate * w_hat * <v_hat, x>  (row w_hat, col v_hat)
            return x_t + SHEAR_KAPPA * gate * w_hat * torch.sum(v_hat * x_t, dim=1, keepdim=True)
        if self.variant == "tnoise":
            # state-dependent randomized perturbation (reparameterized)
            gate = torch.clamp(S, max=S_MAX) / S_MAX
            d = x_t.size(1)
            xi = torch.randn(x_t.size(0), d, generator=self._noise_gen)
            return x_t + NOISE_SIGMA * gate * xi
        raise ValueError(self.variant)

    def forward(self, x_t):
        if self.history is None or len(self.history) < 2:
            return self.base_model(x_t)
        detach_state = not getattr(self, "adaptive", False)
        S, v_t, v_prev = self._state_scalar(x_t, detach_state=detach_state)
        if not getattr(self, "adaptive", False):
            v_t = v_t.detach()
            v_prev = v_prev.detach()
        x_out = self._transform(x_t, v_t, v_prev, S)
        self.last_S = S
        self.last_x_transformed = x_out
        return self.base_model(x_out)

    # -- coupling audit -----------------------------------------------------
    @torch.enable_grad()
    def coupling_norm(self, x_t, normalized=False):
        """C = || d T(h,x) / d x_{t-1} ||_F per sample.

        The history entries ARE the attacker's past queries, so dh/dx = I and
        this Jacobian is the complete state->transform channel magnitude.
        Uses the adaptive path (graph through S and v_t/v_prev). Returns
        (batch,) tensor. With normalized=True, C is divided by the median
        per-sample transform displacement median_i ||T(x_i) - x_i||_2
        (same-batch geometry), giving a scale-free relative coupling
        comparable across transformations of very different Jacobian
        magnitude (rotations/shears vs translations/noise).
        """
        was_adaptive = self.adaptive
        self.adaptive = True
        saved = [h.clone() for h in self.history]
        try:
            hist_leaf = self.history[-1].detach().clone().requires_grad_(True)
            self.history[-1] = hist_leaf
            with torch.no_grad():
                pass
            S, v_t, v_prev = self._state_scalar(x_t, detach_state=False)
            x_out = self._transform(x_t.detach().requires_grad_(True), v_t, v_prev, S)
            self.last_S = S
            grads = torch.autograd.grad(x_out.sum(), hist_leaf, retain_graph=False, allow_unused=True)[0]
            if grads is None:
                return torch.zeros(x_t.size(0), device=x_t.device)
            C = torch.norm(grads, p=2, dim=1)
            if normalized:
                # Relative coupling. The naive denominator ||T(x)|| is
                # dominated by ||h|| for large-mean feature spaces (the
                # rotation channel then shows as ~1e-6, indistinguishable
                # from the no-op noise baseline). Subtracting x first
                # isolates the CHANGE the transform applies, and per-sample
                # distances are compared against the batch's natural
                # inter-sample spacing (same-batch geometry, no ||h|| term).
                dx = torch.norm(x_out - x_t.detach(), p=2, dim=1)
                med = torch.median(dx)
                if not torch.isfinite(med) or med <= 1e-9:
                    med = torch.clamp(dx.mean(), min=1e-9)
                C = C / torch.clamp(med, min=1e-9)
            return C
        finally:
            self.history = saved
            self.adaptive = was_adaptive


# ---------------------------------------------------------------------------
# Stationarity replay (RQ4): freeze x, evolve state with an independent
# persistent-direction session, measure verdict flips of the frozen probe.
# ---------------------------------------------------------------------------

@torch.no_grad()
def stationarity_replay(wrapper, probe, n_steps=12, step_scale=0.3, seed=42):
    """Replay the same probe while only the session state evolves.

    The session stream is a persistent-direction walk (the stream that
    reliably activates trajectory state), independent of the probe. Returns
    dict with nu_session (verdict flip rate of the probe across the replay),
    nu_final (probe verdict at final state vs initial state), and the
    per-step state magnitudes.
    """
    device = probe.device
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    batch, d = probe.shape

    if hasattr(wrapper, "reset_history"):
        wrapper.reset_history(batch, device)
        wrapper.add_to_history(probe.clone())
    with torch.no_grad():
        v0 = (wrapper(probe) > 0).squeeze(-1)

    u = torch.randn(batch, d, generator=g, device=device)
    u = _normalize_rows(u)
    cursor = probe.clone()
    flips = 0
    states = []
    for t in range(n_steps):
        nxt = (cursor + step_scale * u).clamp(0.0, 1.0)
        wrapper.add_to_history(nxt)
        with torch.no_grad():
            out = wrapper(probe)
            v = (out > 0).squeeze(-1)
        flips += int((v != v0).sum().item())
        if hasattr(wrapper, "last_S") and wrapper.last_S is not None:
            states.append(float(wrapper.last_S.mean().item()))
        cursor = nxt
    with torch.no_grad():
        v_final = (wrapper(probe) > 0).squeeze(-1)
    nu_final = float((v_final != v0).float().mean().item())
    return {
        "nu_session": flips / (n_steps * batch),
        "nu_final": nu_final,
        "mean_state": sum(states) / max(len(states), 1),
        "n_steps": n_steps,
    }
