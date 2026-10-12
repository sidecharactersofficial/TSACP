"""Projection equivariance and the Prop. 5 chord identity for the rotation.

The rotation acts only inside the plane span(u, p); therefore it commutes with
the orthogonal projection onto that plane (equivariance), and the displacement
it induces is exactly the chord ``2·r·sin(Δθ/2)`` used in Prop. 5's flip-set
width. The oracle is ``src/tascp_eval/rotation.py``; the SUT is the PyTorch
implementation in ``defenses/tascp.py``.
"""
import numpy as np
import torch

from src.tascp_eval.rotation import (
    apply_rotation,
    orthonormal_basis,
    in_plane_components,
)
from src.tascp_eval.defenses.tascp import _apply_rotation as torch_apply_rotation
from src.tascp_eval.metrics import rotation_displacement_norm

RNG = np.random.default_rng(99)


def _plane(d=8):
    v_t = RNG.normal(size=d)
    v_prev = RNG.normal(size=d)
    return v_t, v_prev


def test_rotation_commutes_with_plane_projection():
    for _ in range(25):
        v_t, v_prev = _plane()
        theta = RNG.uniform(-np.pi, np.pi)
        u, p = orthonormal_basis(v_t, v_prev)
        x = RNG.normal(size=8)
        proj = np.dot(x, u) * u + np.dot(x, p) * p

        rotated_proj = apply_rotation(proj, v_t, v_prev, theta)
        proj_of_rotated = np.dot(apply_rotation(x, v_t, v_prev, theta), u) * u + np.dot(
            apply_rotation(x, v_t, v_prev, theta), p
        ) * p
        assert np.allclose(rotated_proj, proj_of_rotated, atol=1e-9)


def test_rotation_of_orthogonal_residual_is_identity():
    v_t, v_prev = _plane()
    u, p = orthonormal_basis(v_t, v_prev)
    x = RNG.normal(size=8)
    _, _, x_orth = in_plane_components(x, u, p)
    theta = RNG.uniform(-np.pi, np.pi)
    assert np.allclose(apply_rotation(x_orth, v_t, v_prev, theta), x_orth, atol=1e-9)


def test_prop5_chord_identity():
    # ||R(θ)x - x|| = 2 r sin(θ/2), r = in-plane radius.
    for _ in range(25):
        v_t, v_prev = _plane()
        u, p = orthonormal_basis(v_t, v_prev)
        x = RNG.normal(size=8)
        a, b, _ = in_plane_components(x, u, p)
        r = np.hypot(a, b)
        theta = RNG.uniform(-np.pi, np.pi)
        measured = np.linalg.norm(apply_rotation(x, v_t, v_prev, theta) - x)
        predicted = rotation_displacement_norm(r, theta)
        assert np.isclose(measured, predicted, atol=1e-9)


def test_prop5_halfwidth_scales_as_sin_half_angle():
    # Halving Δθ at small angle roughly halves the induced displacement.
    r = 1.0
    small = rotation_displacement_norm(r, 0.02)
    half = rotation_displacement_norm(r, 0.01)
    assert np.isclose(half / small, 0.5, atol=1e-4)


def test_pytorch_matches_oracle_on_orthogonal_residual():
    v_t, v_prev = _plane()
    u, p = orthonormal_basis(v_t, v_prev)
    x = RNG.normal(size=8)
    _, _, x_orth = in_plane_components(x, u, p)
    theta = RNG.uniform(-np.pi, np.pi)
    out, _, _ = torch_apply_rotation(
        torch.tensor(x_orth[None, :], dtype=torch.float32),
        torch.tensor(v_t[None, :], dtype=torch.float32),
        torch.tensor(v_prev[None, :], dtype=torch.float32),
        torch.tensor([[theta]], dtype=torch.float32),
    )
    assert np.allclose(out.detach().numpy().ravel(), x_orth, atol=1e-5)
