"""Tests for the trajectory-conditioned transformation family (exp19-24)."""

import math

import torch

from experiments.exp19_trajectory_family.family import (
    FAMILY_VARIANTS,
    STATE_FN_NAMES,
    FamilyWrapper,
    stationarity_replay,
)
from src.tascp_eval.attacks.stateful import init_stateful_history, observe_query


class LinearMargin(torch.nn.Module):
    def __init__(self, d=6):
        super().__init__()
        self.w = torch.zeros(1, d)
        self.w[0, 0] = 8.0
        self.b = torch.tensor([-4.0])

    def forward(self, x):
        return x @ self.w.T + self.b


def _benign_batch(n=8, d=6):
    x = torch.zeros(n, d)
    x[:, 0] = 0.3 + 0.1 * torch.rand(n)
    return x


def _active_history(wrapper, X, steps=3, step=0.15):
    """Seed a persistent-direction walk so the trajectory trigger can fire."""
    wrapper.reset_history(X.size(0), X.device)
    wrapper.add_to_history(X.clone())
    u = torch.nn.functional.normalize(torch.rand_like(X), dim=1)
    cursor = X.clone()
    for _ in range(steps):
        cursor = (cursor + step * u).clamp(0.0, 1.0)
        wrapper.add_to_history(cursor)
    return cursor


def test_all_variants_forward_and_protocol():
    base = LinearMargin()
    X = _benign_batch()
    for variant in FAMILY_VARIANTS:
        for state_fn in STATE_FN_NAMES:
            w = FamilyWrapper(base, variant=variant, state_fn=state_fn)
            _active_history(w, X)
            out = w(X)
            assert out.shape == (X.size(0), 1)
            assert w.last_S is not None and w.last_S.shape == (X.size(0), 1)
            assert torch.isfinite(out).all()


def test_freeze_state_equals_identity_channel():
    """freeze_state=0 makes every variant act as the identity transform."""
    base = LinearMargin()
    X = _benign_batch()
    for variant in FAMILY_VARIANTS:
        w = FamilyWrapper(base, variant=variant)
        w.freeze_state = 0.0
        _active_history(w, X)
        out = w(X)
        ref = base(X)
        assert torch.allclose(out, ref, atol=1e-5), variant


def test_coupling_norm_zero_when_silent_positive_when_firing():
    base = LinearMargin()
    X = _benign_batch()
    # silent random walk: state silent -> coupling 0 (identity transform)
    w = FamilyWrapper(base, variant="tascp")
    w.reset_history(X.size(0), X.device)
    w.add_to_history(X.clone())
    w.add_to_history(X.clone() + 0.01 * torch.randn_like(X))
    C_silent = w.coupling_norm(X)
    assert float(C_silent.abs().max().item()) < 1e-6
    # active persistent-direction walk: coupling positive
    _active_history(w, X)
    C_active = w.coupling_norm(X)
    assert float(C_active.mean().item()) > 0.0
    # normalized variant is finite and positive where firing
    Cn = w.coupling_norm(X, normalized=True)
    assert torch.isfinite(Cn).all() and float(Cn.mean().item()) > 0.0


def test_stationarity_replay_shapes():
    base = LinearMargin()
    X = _benign_batch()
    w = FamilyWrapper(base, variant="tascp")
    stat = stationarity_replay(w, X, n_steps=6)
    assert set(stat) >= {"nu_session", "nu_final", "mean_state", "n_steps"}
    assert 0.0 <= stat["nu_session"] <= 1.0
    assert 0.0 <= stat["nu_final"] <= 1.0
    assert stat["n_steps"] == 6


def test_adaptive_path_keeps_graph():
    """adaptive=True must keep the state path differentiable (B2 channel)."""
    base = LinearMargin()
    X = _benign_batch().requires_grad_(True)
    w = FamilyWrapper(base, variant="tscale")
    _active_history(w, X)
    w.adaptive = True
    out = w(X)
    g = torch.autograd.grad(out.sum(), X, allow_unused=True)[0]
    assert g is not None and torch.isfinite(g).all()
    w.adaptive = False
    X2 = X.detach().clone().requires_grad_(True)
    out2 = w(X2)
    g2 = torch.autograd.grad(out2.sum(), X2, allow_unused=True)[0]
    # detached state path: gradient only flows through x_t (may be nonzero)
    assert g2 is not None
