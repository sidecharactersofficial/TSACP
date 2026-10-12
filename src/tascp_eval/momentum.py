"""Independent NumPy reference for trajectory momentum (Eq. 4-5).

The single-step momentum of a query trajectory is the cosine similarity of
consecutive displacements::

    M_t = cos∠(v_t, v_{t-1}) = <v_t, v_{t-1}> / (||v_t|| ||v_{t-1}||)

with ``M_t = 0`` when either displacement is (near) zero. This is an
independent oracle for ``tests/test_momentum.py``; it deliberately does not
import the module under test.
"""

import numpy as np


def compute_momentum(v_t, v_prev, eps=1e-9):
    """Cosine similarity of two displacements; 0.0 for a (near) zero vector."""
    v_t = np.asarray(v_t, dtype=float)
    v_prev = np.asarray(v_prev, dtype=float)
    n_t = np.linalg.norm(v_t)
    n_p = np.linalg.norm(v_prev)
    if n_t < eps or n_p < eps:
        return 0.0
    return float(np.dot(v_t, v_prev) / (n_t * n_p))


def windowed_penetration(window):
    """``ε_p = (1/|W|) Σ_{i∈W} ||v_i|| · max(0, M_i)`` (see ``penetration.py``)."""
    if not window:
        return 0.0
    total = 0.0
    for v_i, m_i in window:
        total += np.linalg.norm(np.asarray(v_i, dtype=float)) * max(0.0, float(m_i))
    return total / len(window)
