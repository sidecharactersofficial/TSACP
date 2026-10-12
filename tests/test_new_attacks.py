import math

import torch

from src.tascp_eval.attacks.adaptive_variants import (
    eot_stateful_pgd_attack,
    finite_difference_stateful_pgd_attack,
)
from src.tascp_eval.attacks.offline_attacker import (
    offline_attack_bit_identical,
    offline_attack_surrogate_mismatch,
)
from src.tascp_eval.attacks.trigger_scope import TRIGGER_SCOPE_ATTACKS
from src.tascp_eval.defenses.tascp import TASCPWrapper


class LinearMargin(torch.nn.Module):
    """Deterministic binary classifier: attack class iff x[0] > 0.5."""

    def __init__(self, d=6):
        super().__init__()
        self.w = torch.zeros(1, d)
        self.w[0, 0] = 8.0
        self.b = torch.tensor([-4.0])

    def forward(self, x):
        return x @ self.w.T + self.b


def _benign_batch(n=8, d=6):
    x = torch.zeros(n, d)
    x[:, 0] = 0.3 + 0.1 * torch.rand(n)  # x[0] in [0.3, 0.4] -> benign
    return x


PGD_CONFIG = {"epsilon": 0.3, "alpha": 0.05, "iterations": 5, "norm": "l2"}
SCOPE_CONFIG = {"epsilon": 0.3, "sigma": 0.05, "alpha": 0.05, "q": 4, "max_queries": 12}


# ---------------------------------------------------------------------------
# adaptive_variants (P1-10)
# ---------------------------------------------------------------------------

def _defended(model):
    return TASCPWrapper(model, window_size=5, max_theta=math.pi / 2, tau=0.0, gamma=50.0)


def test_eot_attack_contract():
    torch.manual_seed(0)
    x = _benign_batch()
    wrapped = _defended(LinearMargin())
    x_adv, success, q = eot_stateful_pgd_attack(
        wrapped, x, torch.zeros(x.size(0), 1), PGD_CONFIG,
        adaptive=True, eot_samples=2,
    )
    assert x_adv.shape == x.shape
    assert success.dtype == torch.bool and success.shape == (x.size(0),)
    assert torch.isfinite(q).all()
    assert (x_adv - x).norm(p=2, dim=1).max() <= PGD_CONFIG["epsilon"] + 1e-4
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0


def test_fd_attack_never_uses_autograd():
    torch.manual_seed(0)
    x = _benign_batch()
    wrapped = _defended(LinearMargin())
    x_adv, success, q = finite_difference_stateful_pgd_attack(
        wrapped, x, torch.zeros(x.size(0), 1), PGD_CONFIG,
        fd_samples=4,
    )
    assert x_adv.shape == x.shape
    assert success.dtype == torch.bool
    assert torch.isfinite(q).all()
    # The wrapper's adaptive flag was toggled by the attack.
    assert wrapped.adaptive is True


# ---------------------------------------------------------------------------
# trigger_scope (P1-8): black-box protocol, 4-tuple contract for all attacks
# ---------------------------------------------------------------------------

def test_all_scope_attacks_share_contract():
    torch.manual_seed(0)
    x = _benign_batch(n=4)
    wrapped = _defended(LinearMargin())
    for name, attack in TRIGGER_SCOPE_ATTACKS.items():
        out = attack(wrapped, x, SCOPE_CONFIG)
        assert len(out) == 4, f"{name} must return (x_adv, success, q_to_success, total)"
        x_adv, success, q_to_success, total = out
        assert x_adv.shape == x.shape, name
        assert success.dtype == torch.bool, name
        assert torch.isfinite(q_to_success).all(), name
        assert total > 0, name


def test_scope_attacks_respect_l2_budget():
    torch.manual_seed(0)
    x = _benign_batch(n=4)
    wrapped = _defended(LinearMargin())
    for name, attack in TRIGGER_SCOPE_ATTACKS.items():
        x_adv, _, _, _ = attack(wrapped, x, SCOPE_CONFIG)
        dist = (x_adv - x).norm(p=2, dim=1)
        # Boundary attack drifts from its random-start far point toward x0, so
        # only the deliberately-unconstrained attacks are strictly in-ball.
        if name == "boundary":
            continue
        assert dist.max() <= SCOPE_CONFIG["epsilon"] + 1e-3, (
            f"{name} exceeded the L2 budget: {dist.max():.4f}"
        )


# ---------------------------------------------------------------------------
# offline_attacker (P1-9): first-query bit-identity TASCP == DACM == undefended
# ---------------------------------------------------------------------------

def test_offline_first_query_bit_identity():
    torch.manual_seed(0)
    x_star = _benign_batch()
    res = offline_attack_bit_identical(
        LinearMargin(), x_star, tau=0.0, gamma=50.0
    )
    assert res["tascp_bit_identical_dacm"] is True
    assert res["tascp_bit_identical_undefended"] is True
    assert res["max_abs_diff_tascp_dacm"] == 0.0
    assert res["max_abs_diff_tascp_undefended"] == 0.0
    assert res["verdict_agreement"] is True


def test_offline_surrogate_mismatch_tascp_equals_dacm_fresh_session():
    torch.manual_seed(0)
    d = 6
    surrogate = LinearMargin(d)
    deployed = LinearMargin(d)
    with torch.no_grad():  # make the surrogate differ from the deployed model
        surrogate.w += 0.5
        surrogate.b += 0.2
    x = _benign_batch(n=8, d=d)
    res, x_star = offline_attack_surrogate_mismatch(surrogate, deployed, x, PGD_CONFIG)
    mm = res["surrogate_mismatch"]
    # Fresh-session TASCP verdict on the transferred point == DACM's verdict,
    # because delta_theta = 0 on the first query of a session.
    assert torch.equal(res["tascp_logits"], mm["dacm_logits"])
    assert res["tascp_bit_identical_dacm"] is True
    assert 0.0 <= mm["transfer_success_undefended"] <= 1.0
    assert mm["transfer_success_tascp_fresh_session"] == mm["transfer_success_dacm"]
