"""Independent NumPy reference for the TASCP Givens rotation.

Written from the equations in the paper (rotation in the plane spanned by the
current displacement ``v_t`` and the previous displacement ``v_prev``), *not*
copied from the PyTorch implementation used by the experiments
(``src/tascp_eval/defenses/tascp.py``). It therefore serves as an independent
oracle for ``tests/test_rotation.py`` and ``tests/test_projection.py``.

Convention (matches the paper and ``_apply_rotation``):

    u      = v_t / ||v_t||                      (in-plane axis 1)
    p      = normalize(v_prev - (v_prev·u) u)   (in-plane axis 2, ⊥ u)
    x_rot  = x_orth + (a cosθ - b sinθ) u + (a sinθ + b cosθ) p
    where  a = x·u, b = x·p, x_orth = x - a u - b p

If ``v_t`` or the orthogonalised ``v_prev`` is (near) zero the rotation plane
is undefined and the input is returned unchanged.
"""

import numpy as np


def orthonormal_basis(v_t, v_prev, eps=1e-9):
    """Return the orthonormal in-plane basis ``(u, p)`` or ``(None, None)``."""
    v_t = np.asarray(v_t, dtype=float)
    v_prev = np.asarray(v_prev, dtype=float)
    n_t = np.linalg.norm(v_t)
    if n_t < eps:
        return None, None
    u = v_t / n_t
    w = v_prev - np.dot(v_prev, u) * u
    n_w = np.linalg.norm(w)
    if n_w < eps:
        return None, None
    return u, w / n_w


def rotation_matrix(u, p, theta):
    """Dense ``d x d`` planar rotation matrix ``R(u, p, theta)``.

    ``R = I + (cosθ-1)(u uᵀ + p pᵀ) + sinθ (p uᵀ - u pᵀ)``
    """
    u = np.asarray(u, dtype=float).reshape(-1, 1)
    p = np.asarray(p, dtype=float).reshape(-1, 1)
    d = u.shape[0]
    return (
        np.eye(d)
        + (np.cos(theta) - 1.0) * (u @ u.T + p @ p.T)
        + np.sin(theta) * (p @ u.T - u @ p.T)
    )


def in_plane_components(x, u, p):
    """Return ``(a, b, x_orth)``: in-plane coords and the orthogonal residual."""
    x = np.asarray(x, dtype=float)
    a = np.dot(x, u)
    b = np.dot(x, p)
    x_orth = x - a * u - b * p
    return a, b, x_orth


def apply_rotation(x, v_t, v_prev, theta):
    """Apply the TASCP rotation to ``x``; identity when the plane is undefined."""
    u, p = orthonormal_basis(v_t, v_prev)
    if u is None:
        return np.asarray(x, dtype=float).copy()
    return rotation_matrix(u, p, theta) @ np.asarray(x, dtype=float)
