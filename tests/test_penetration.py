"""Independent tests for the penetration integral ε_p.

Two independent implementations exist and must agree: the new NumPy oracle
(``src/tascp_eval/penetration.py``) and the long-standing prototype-match
module (``src/tascp_eval/reference_math.py``). Hand-computed values pin the
formula and the ``max(0, ·)`` clip.
"""
import numpy as np

from src.tascp_eval.penetration import compute_penetration, delta_theta
from src.tascp_eval.reference_math import compute_penetration as ref_penetration

RNG = np.random.default_rng(11)


def test_empty_window_is_zero():
    assert compute_penetration([]) == 0.0


def test_all_aligned_full_penetration():
    v = np.zeros(6)
    v[0] = 1.0
    window = [(v, 1.0)] * 5
    assert np.isclose(compute_penetration(window), 1.0, atol=1e-12)


def test_benign_zero_momentum_is_zero():
    v = np.zeros(6)
    v[0] = 1.0
    window = [(v, 0.0)] * 5
    assert np.isclose(compute_penetration(window), 0.0, atol=1e-12)


def test_negative_momentum_is_clipped_to_zero():
    v = np.zeros(6)
    v[0] = 5.0
    assert np.isclose(compute_penetration([(v, -0.9)]), 0.0, atol=1e-12)


def test_hand_computed_mixed_window():
    v1 = np.array([3.0, 0.0])
    v2 = np.array([0.0, 4.0])
    window = [(v1, 0.5), (v2, -0.2), (v1, 0.5)]
    # (1/3) * (3*0.5 + 4*0 + 3*0.5) = (1/3)*3 = 1.0
    assert np.isclose(compute_penetration(window), 1.0, atol=1e-12)


def test_two_implementations_agree_randomly():
    for _ in range(50):
        size = int(RNG.integers(1, 10))
        window = [
            (RNG.normal(size=7), float(RNG.uniform(-1.0, 1.0))) for _ in range(size)
        ]
        assert np.isclose(
            compute_penetration(window), ref_penetration(window), atol=1e-12
        )


def test_delta_theta_clips_at_zero_and_max():
    assert delta_theta(eps_p=0.5, gamma=10.0, tau=1.0) == 0.0
    assert np.isclose(delta_theta(eps_p=1.0, gamma=10.0, tau=0.0, max_theta=np.pi / 2),
                      np.pi / 2)
    assert np.isclose(delta_theta(eps_p=0.02, gamma=50.0, tau=0.0), 1.0)
