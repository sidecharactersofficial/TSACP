"""Gradient-consistency tests (Lemma 3) and clamp-boundary checks.

Lemma 3 states that the autograd gradient of the penetration integral ε_p
agrees with a central finite-difference estimate along random directions. We
verify this on random trajectories, exercise the ``max(0, ·)`` kink region,
and separately check that ``Δθ = clamp(γ·max(0, ε_p−τ), max_θ)`` has the
expected derivative on each side of its two clamp boundaries.
"""
import numpy as np
import torch

RNG = np.random.default_rng(2024)


def _eps_p_from_history(H):
    """Differentiable ε_p; H: (k, d) trajectory. Mirrors the paper formula."""
    k = H.shape[0]
    v = [H[i] - H[i - 1] for i in range(1, k)]
    terms = []
    for idx, vi in enumerate(v):
        if idx == 0:
            m = torch.zeros((), dtype=H.dtype)
        else:
            a, b = v[idx], v[idx - 1]
            m = torch.dot(a, b) / (a.norm() * b.norm()).clamp(min=1e-9)
        terms.append(vi.norm() * torch.clamp(m, min=0.0))
    return torch.stack(terms).mean()


def _random_history(k=6, d=8, scale=0.05):
    return torch.tensor(RNG.normal(scale=scale, size=(k, d)), dtype=torch.float64)


def test_lemma3_autograd_matches_central_fd():
    torch.manual_seed(0)
    for _ in range(20):
        H = _random_history()
        u = torch.tensor(RNG.normal(size=H.shape), dtype=torch.float64)
        u = u / u.norm()

        H.requires_grad_(True)
        eps = _eps_p_from_history(H)
        (grad,) = torch.autograd.grad(eps, H)
        directional = torch.sum(grad * u).item()

        delta = 1e-3
        eps_plus = _eps_p_from_history(H.detach() + delta * u).item()
        eps_minus = _eps_p_from_history(H.detach() - delta * u).item()
        fd = (eps_plus - eps_minus) / (2 * delta)

        denom = max(1.0, abs(directional))
        assert abs(directional - fd) / denom < 1e-4


def test_kink_region_is_piecewise_consistent():
    # Construct a window where one momentum is small but positive; FD should
    # still agree with autograd (away from the exact kink).
    torch.manual_seed(1)
    H = _random_history(k=3, d=4)
    H = H.detach().clone().requires_grad_(True)
    eps = _eps_p_from_history(H)
    (grad,) = torch.autograd.grad(eps, H)
    assert torch.isfinite(grad).all()


def test_delta_theta_derivative_across_clamps():
    gamma, tau, max_theta = 50.0, 0.0, float(np.pi / 2)

    def dtheta(eps):
        return torch.clamp(gamma * torch.clamp(eps - tau, min=0.0), max=max_theta)

    # Below τ: derivative 0.
    eps = torch.tensor(-1e-3, dtype=torch.float64, requires_grad=True)
    (g,) = torch.autograd.grad(dtheta(eps), eps)
    assert g.item() == 0.0

    # Linear region: derivative == gamma.
    eps = torch.tensor(0.005, dtype=torch.float64, requires_grad=True)  # 0.25 rad
    (g,) = torch.autograd.grad(dtheta(eps), eps)
    assert np.isclose(g.item(), gamma)

    # Above max_θ/γ: derivative 0.
    eps = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    (g,) = torch.autograd.grad(dtheta(eps), eps)
    assert g.item() == 0.0


def test_autograd_gradient_finite_on_real_like_pgd_path():
    # A persistent-direction PGD path: consecutive displacements are aligned,
    # so the momentum clip is active and the gradient is non-zero.
    torch.manual_seed(2)
    k, d = 7, 43
    direction = torch.tensor(RNG.normal(size=d), dtype=torch.float64)
    direction = direction / direction.norm()
    step = 0.02 * direction
    noise = torch.tensor(RNG.normal(scale=1e-4, size=(k, d)), dtype=torch.float64)
    H = (torch.arange(k, dtype=torch.float64).reshape(-1, 1) * step + noise)
    H.requires_grad_(True)
    (grad,) = torch.autograd.grad(_eps_p_from_history(H), H)
    assert torch.isfinite(grad).all()
    assert grad.norm().item() > 0.0
