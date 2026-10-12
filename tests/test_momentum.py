"""Independent tests for trajectory momentum (Eq. 4-5) and Cor. 3.

Oracle: ``src/tascp_eval/momentum.py`` and ``src/tascp_eval/metrics.py``.
"""
import numpy as np

from src.tascp_eval.momentum import compute_momentum, windowed_penetration
from src.tascp_eval.metrics import antithetic_mean_momentum

RNG = np.random.default_rng(7)


def test_aligned_vectors_momentum_is_one():
    a = RNG.normal(size=10)
    assert np.isclose(compute_momentum(a, 2.0 * a), 1.0, atol=1e-12)


def test_orthogonal_vectors_momentum_is_zero():
    a = np.zeros(10)
    a[0] = 1.0
    b = np.zeros(10)
    b[1] = 1.0
    assert np.isclose(compute_momentum(a, b), 0.0, atol=1e-12)


def test_opposite_vectors_momentum_is_minus_one():
    a = RNG.normal(size=10)
    assert np.isclose(compute_momentum(a, -a), -1.0, atol=1e-12)


def test_zero_displacement_momentum_is_zero():
    a = RNG.normal(size=10)
    assert compute_momentum(np.zeros(10), a) == 0.0
    assert compute_momentum(a, np.zeros(10)) == 0.0


def test_momentum_clip_zeroes_antithetic_direction():
    # A single antithetic pair has negative momentum, so max(0, M) = 0.
    sigma = 0.05
    u = RNG.normal(size=10)
    u = u / np.linalg.norm(u)
    v_t = -2.0 * sigma * u
    v_prev = 2.0 * sigma * u  # pure antithetic step
    assert compute_momentum(v_t, v_prev) < 0.0
    assert np.isclose(max(0.0, compute_momentum(v_t, v_prev)), 0.0)


def test_antithetic_mean_momentum_matches_corollary():
    # E[M_t] = -2/sqrt(5) ~= -0.894 (Cor. 3).
    m = antithetic_mean_momentum(n_pairs=100000, rng=np.random.default_rng(3))
    assert abs(m - (-2.0 / np.sqrt(5.0))) < 0.01
    assert m < 0.0


def test_windowed_penetration_matches_manual_value():
    v = np.zeros(5)
    v[0] = 2.0
    window = [(v, 0.5) for _ in range(4)]
    # (1/4) * 4 * (2.0 * 0.5) = 1.0
    assert np.isclose(windowed_penetration(window), 1.0, atol=1e-12)


def test_windowed_penetration_is_window_size_invariant():
    v = np.zeros(5)
    v[0] = 1.0
    small = windowed_penetration([(v, 1.0)] * 5)
    large = windowed_penetration([(v, 1.0)] * 50)
    assert np.isclose(small, large, atol=1e-12)
    assert np.isclose(small, 1.0, atol=1e-12)
