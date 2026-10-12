import torch

from src.tascp_eval.attacks.domain_constraints import (
    TabularDomainProjector,
    build_projector_from_dataframe,
)


def _make_projector():
    cols = ["proto", "is_sm_ips_ports", "dur", "meta"]
    levels = {"proto": [0.0, 0.25, 0.5, 1.0]}
    return TabularDomainProjector(
        levels=levels,
        feature_cols=cols,
        binary=["is_sm_ips_ports"],
        discrete=["proto"],
        immutable=["meta"],
    )


def test_binary_snaps_to_0_1():
    p = _make_projector()
    x_adv = torch.tensor([[0.1, 0.7, 0.5, 0.3], [0.1, 0.49, 0.5, 0.3]])
    x_orig = torch.zeros_like(x_adv)
    out = p.project(x_adv, x_orig)
    assert out[:, 1].tolist() == [1.0, 0.0]


def test_discrete_snaps_to_nearest_observed_level():
    p = _make_projector()
    x_adv = torch.tensor([[0.3, 0.0, 0.5, 0.3], [0.7, 0.0, 0.5, 0.3], [0.95, 0.0, 0.5, 0.3]])
    x_orig = torch.zeros_like(x_adv)
    out = p.project(x_adv, x_orig)
    assert out[:, 0].tolist() == [0.25, 0.5, 1.0]


def test_immutable_restores_original():
    p = _make_projector()
    x_adv = torch.tensor([[0.1, 0.0, 0.5, 0.9]])
    x_orig = torch.tensor([[0.0, 0.0, 0.0, 0.42]])
    out = p.project(x_adv, x_orig)
    assert torch.isclose(out[0, 3], x_orig[0, 3])


def test_projection_clamps_to_box():
    p = _make_projector()
    x_adv = torch.tensor([[-0.5, 1.5, 0.5, 0.3]])
    x_orig = torch.zeros_like(x_adv)
    out = p.project(x_adv, x_orig)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_unknown_column_raises():
    try:
        TabularDomainProjector(
            levels={}, feature_cols=["a"], binary=["nope"], discrete=[], immutable=[]
        )
        assert False, "expected KeyError"
    except KeyError as e:
        assert "nope" in str(e)


def test_build_projector_from_dataframe():
    import pandas as pd

    df = pd.DataFrame({"proto": [0.0, 0.5, 0.5, 1.0], "flag": [0.0, 1.0, 0.0, 1.0]})
    p = build_projector_from_dataframe(
        df, ["proto", "flag"], {"binary": ["flag"], "discrete": ["proto"], "immutable": []}
    )
    x_adv = torch.tensor([[0.4, 0.8], [0.6, 0.2]])
    out = p.project(x_adv, torch.zeros_like(x_adv))
    assert out[:, 0].tolist() == [0.5, 0.5]
    assert out[:, 1].tolist() == [1.0, 0.0]


def test_projector_callable_interface_matches_attack_usage():
    """Attacks call projector(x_adv, x_orig) — must behave like project()."""
    p = _make_projector()
    x_adv = torch.tensor([[0.3, 0.6, 0.5, 0.1]])
    x_orig = torch.tensor([[0.3, 0.0, 0.5, 0.7]])
    assert torch.equal(p(x_adv, x_orig), p.project(x_adv, x_orig))
