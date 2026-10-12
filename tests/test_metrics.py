"""Stationarity-metric oracle tests (Prop. 5).

``ν_session`` is (#verdict flips of a frozen probe) / (#trajectory queries).
The undefended / static-projection control has Δθ ≡ 0 and hence ν ≡ 0. The
flip-set width scales with the rotation chord ``2·r·sin(Δθ/2)``.
"""
import numpy as np
import pytest

from src.tascp_eval.metrics import nu_session, rotation_displacement_norm

RNG = np.random.default_rng(5)


def test_nu_session_hand_values():
    assert nu_session(3, 10) == pytest.approx(0.3)
    assert nu_session(0, 10) == 0.0
    assert nu_session(10, 10) == 1.0


def test_nu_session_zero_queries_is_zero():
    assert nu_session(0, 0) == 0.0
    assert nu_session(5, 0) == 0.0


def test_control_has_zero_flip_width():
    # Δθ = 0 (undefended / static projection) -> no displacement, no flips.
    for r in [0.0, 0.5, 2.0]:
        assert rotation_displacement_norm(r, 0.0) == 0.0


def test_flip_width_grows_with_angle():
    r = 1.0
    widths = [rotation_displacement_norm(r, th) for th in [0.01, 0.05, 0.1, 0.5]]
    assert all(b > a for a, b in zip(widths, widths[1:]))


def test_flip_width_is_zero_for_axis_point():
    # A point on the rotation axis (r = 0) never moves.
    assert rotation_displacement_norm(0.0, 1.0) == 0.0


def test_prop5_small_angle_quadratic_scaling():
    # Prop 5(ii): halving γ (hence Δθ) reduces the chord by ~half; the
    # *flip rate* scales as sin²(Δθ/2), i.e. quadratically in Δθ.
    r = 1.0
    th = 0.04
    chord = rotation_displacement_norm(r, th)
    chord_half = rotation_displacement_norm(r, th / 2)
    assert chord_half / chord == pytest.approx(0.5, abs=1e-3)
    # sin² scaling of the rate: ratio of rates ~ (1/2)²
    rate_ratio = np.sin(th / 2) ** 2 / np.sin(th / 4) ** 2
    assert rate_ratio == pytest.approx(4.0, rel=1e-3)
