"""Independent NumPy reference for the penetration integral ε_p.

Definition::

    ε_p(h) = (1/|W|) Σ_{i∈W} ||v_i||_2 · max(0, M_i)

where ``W`` is the trailing window of displacements ``v_i = x_i - x_{i-1}`` and
``M_i = cos∠(v_i, v_{i-1})``. This module is an independent oracle for
``tests/test_penetration.py`` and ``tests/test_gradients.py``; it does not
import the module under test.
"""

import numpy as np


def compute_penetration(window, eps=1e-9):
    """ε_p over ``window`` = list of ``(v_i, M_i)`` tuples.

    Contributions with ``M_i <= 0`` are clipped to zero (the ``max(0, ·)``).
    The empty window has penetration 0.
    """
    if not window:
        return 0.0
    total = 0.0
    for v_i, m_i in window:
        total += np.linalg.norm(np.asarray(v_i, dtype=float)) * max(0.0, float(m_i))
    return total / len(window)


def delta_theta(eps_p, gamma, tau=0.0, max_theta=np.pi / 2):
    """``Δθ = clamp(γ · max(0, ε_p - τ), max=max_theta)``."""
    return float(np.clip(gamma * max(0.0, eps_p - tau), 0.0, max_theta))
