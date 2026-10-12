"""Independent tests for the TASCP planar (Givens) rotation.

The oracle lives in ``src/tascp_eval/rotation.py`` (written from the paper's
equations); the system under test is the PyTorch implementation actually used
by the experiments, ``src/tascp_eval/defenses/tascp.py::_apply_rotation``.
"""
import numpy as np
import torch

from src.tascp_eval.rotation import (
    apply_rotation,
    orthonormal_basis,
    rotation_matrix,
)
from src.tascp_eval.defenses.tascp import _apply_rotation as torch_apply_rotation

RNG = np.random.default_rng(1234)


def _random_plane(d=8):
    v_t = RNG.normal(size=d)
    v_prev = RNG.normal(size=d)
    return v_t, v_prev, RNG.uniform(-np.pi, np.pi)


def test_rotation_matrix_is_orthogonal_and_proper():
    for _ in range(25):
        v_t, v_prev, theta = _random_plane()
        u, p = orthonormal_basis(v_t, v_prev)
        R = rotation_matrix(u, p, theta)
        assert np.allclose(R.T @ R, np.eye(len(u)), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_rotation_is_orthonormal_basis():
    for _ in range(25):
        v_t, v_prev, _ = _random_plane()
        u, p = orthonormal_basis(v_t, v_prev)
        assert np.isclose(np.linalg.norm(u), 1.0, atol=1e-12)
        assert np.isclose(np.linalg.norm(p), 1.0, atol=1e-12)
        assert np.isclose(np.dot(u, p), 0.0, atol=1e-12)


def test_rotation_preserves_norm():
    for _ in range(25):
        v_t, v_prev, theta = _random_plane()
        x = RNG.normal(size=8)
        x_rot = apply_rotation(x, v_t, v_prev, theta)
        assert np.isclose(np.linalg.norm(x_rot), np.linalg.norm(x), atol=1e-9)


def test_rotation_matches_pytorch_implementation():
    for _ in range(25):
        v_t, v_prev, theta = _random_plane()
        x = RNG.normal(size=8)
        ref = apply_rotation(x, v_t, v_prev, theta)

        out, _, _ = torch_apply_rotation(
            torch.tensor(x[None, :], dtype=torch.float32),
            torch.tensor(v_t[None, :], dtype=torch.float32),
            torch.tensor(v_prev[None, :], dtype=torch.float32),
            torch.tensor([[theta]], dtype=torch.float32),
        )
        assert np.allclose(out.detach().numpy().ravel(), ref, atol=1e-5)


def test_rotation_is_involution_at_pi():
    # Rotating by 2*pi returns the original point (identity).
    v_t, v_prev, _ = _random_plane()
    x = RNG.normal(size=8)
    assert np.allclose(apply_rotation(x, v_t, v_prev, 2 * np.pi), x, atol=1e-9)


def test_degenerate_plane_returns_input_unchanged():
    x = RNG.normal(size=8)
    # v_t = 0 -> no plane
    assert np.allclose(apply_rotation(x, np.zeros(8), RNG.normal(size=8), 1.0), x)
    # v_prev parallel to v_t -> no orthogonal axis
    v = RNG.normal(size=8)
    assert np.allclose(apply_rotation(x, v, 3.0 * v, 1.0), x)


def test_orthogonal_component_is_invariant():
    v_t, v_prev, theta = _random_plane()
    u, p = orthonormal_basis(v_t, v_prev)
    x = RNG.normal(size=8)
    x_orth = x - np.dot(x, u) * u - np.dot(x, p) * p
    x_rot = apply_rotation(x, v_t, v_prev, theta)
    # The component orthogonal to the rotation plane must be untouched.
    assert np.allclose(x_rot - (np.dot(x_rot, u) * u + np.dot(x_rot, p) * p), x_orth, atol=1e-9)
