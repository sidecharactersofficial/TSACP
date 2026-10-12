"""Trajectory metrics with hand-checkable oracles.

These pure functions encode the quantities used by the stationarity audit
(Prop. 5) and the antithetic-blindness argument (Cor. 3). They are used by
``tests/test_metrics.py`` as an independent oracle.
"""

import numpy as np


def nu_session(n_flips, n_queries):
    """Stationarity measure ν_session = (#verdict flips) / (#trajectory queries).

    An undefended / static-projection control has ν ≡ 0 because Δθ ≡ 0.
    """
    if n_queries <= 0:
        return 0.0
    return n_flips / n_queries


def rotation_displacement_norm(in_plane_radius, delta_theta):
    """‖R(Δθ)x − x‖ = 2·r·|sin(Δθ/2)| for in-plane radius ``r`` (Prop. 5 chord).

    The flip-set width is proportional to this quantity, hence ∝ sin(Δθ/2).
    """
    return 2.0 * in_plane_radius * abs(np.sin(delta_theta / 2.0))


def antithetic_mean_momentum(n_pairs=200000, sigma=1.0, rng=None):
    """Monte-Carlo E[M_t] for the antithetic NES sequence (Cor. 3).

    Expectation is −2/√5 ≈ −0.894; because it is negative, ``max(0, M_t) = 0``
    and the trigger stays silent.
    """
    rng = rng or np.random.default_rng(0)
    d = 8
    u = rng.normal(size=(n_pairs, d))
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    u_prev = rng.normal(size=(n_pairs, d))
    u_prev = u_prev / np.linalg.norm(u_prev, axis=1, keepdims=True)
    v_t = -2.0 * sigma * u
    v_prev = 2.0 * sigma * u + sigma * u_prev
    num = np.sum(v_t * v_prev, axis=1)
    den = np.linalg.norm(v_t, axis=1) * np.linalg.norm(v_prev, axis=1)
    return float(np.mean(num / den))
