import torch

from src.tascp_eval.attacks.nes import nes_attack
from src.tascp_eval.attacks.hopskipjump import hopskipjump_attack


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
    x[:, 0] = 0.3 + 0.1 * torch.rand(n)  # x[0] in [0.3, 0.4] -> benign (logit < -0.8)
    return x


def test_nes_success_on_trivial_boundary():
    torch.manual_seed(0)
    model = LinearMargin()
    x = _benign_batch()
    # need dx0 up to ~0.2; l2 budget 0.3 keeps every sample's boundary in reach
    cfg = dict(epsilon=0.3, sigma=0.05, alpha=0.05, q=8, max_queries=200)
    x_adv, success, queries, _ = nes_attack(model, x, cfg)
    assert success.all(), "NES should flip the trivial boundary"
    assert (queries[success] <= cfg["max_queries"]).all()
    # l2 budget respected
    dists = (x_adv - x).norm(p=2, dim=1)
    assert (dists <= cfg["epsilon"] + 1e-5).all()


def test_nes_respects_query_budget():
    torch.manual_seed(0)
    model = LinearMargin()
    x = _benign_batch()
    cfg = dict(epsilon=0.2, sigma=0.05, alpha=0.05, q=4, max_queries=60)
    _, success, _, total_q = nes_attack(model, x, cfg)
    # per-sample worst case: steps = ceil(60 / (2q+1)); each step costs 2q+1
    assert total_q <= x.size(0) * (2 * cfg["q"] + 1) * (60 // (2 * cfg["q"] + 1) + 1)


def test_nes_applies_domain_projector():
    from src.tascp_eval.attacks.domain_constraints import TabularDomainProjector

    torch.manual_seed(0)
    d = 6
    cols = [f"f{i}" for i in range(d)]
    proj = TabularDomainProjector(
        levels={"f1": [0.0, 0.5, 1.0]},
        feature_cols=cols,
        binary=["f2"],
        discrete=["f1"],
        immutable=[],
    )
    model = LinearMargin(d)
    x = _benign_batch()
    cfg = dict(epsilon=0.2, sigma=0.05, alpha=0.05, q=8, max_queries=200)
    x_adv, success, _, _ = nes_attack(model, x, cfg, projector=proj)
    assert set(x_adv[:, 1].unique().tolist()).issubset({0.0, 0.5, 1.0})
    assert set(x_adv[:, 2].unique().tolist()).issubset({0.0, 1.0})


def test_hsj_finds_adversarial_point_and_refines():
    torch.manual_seed(0)
    model = LinearMargin()
    x = _benign_batch()
    cfg = dict(N=8, b=5, delta_0=0.1, delta_min=0.001, max_queries=400,
               init_radii=[0.25, 0.5, 1.0, 2.0], init_draws=16)
    x_adv, success, queries, _ = hopskipjump_attack(model, x, cfg)
    assert success.all(), "HSJ init should find the trivial boundary"
    # final adversarial points must be adversarial and closer than the init radius
    assert (model(x_adv).squeeze(-1) > 0).all()
    assert ((x_adv - x).norm(p=2, dim=1) < 1.0).all()


def test_hsj_queries_accounted_per_block():
    """Every forward pass counts as one query per sample in the batch."""
    torch.manual_seed(0)
    model = LinearMargin()
    x = _benign_batch(n=4)
    cfg = dict(N=4, b=3, delta_0=0.1, delta_min=0.001, max_queries=80,
               init_radii=[0.5], init_draws=1)
    _, _, _, total_q = hopskipjump_attack(model, x, cfg)
    assert total_q % x.size(0) == 0  # whole (batch, d) blocks only


def test_stateful_wrapper_sees_every_query():
    """Black-box attacks must feed each distinct query into the defense history."""
    from src.tascp_eval.defenses.tascp import TASCPWrapper
    from src.tascp_eval.attacks.stateful import init_stateful_history, observe_query

    class Recorder:
        def __init__(self, base):
            self.base = base

        def __call__(self, x):
            return self.base(x)

    torch.manual_seed(0)
    base = Recorder(LinearMargin())
    wrapper = TASCPWrapper(base, window_size=5, max_theta=3.14159 / 2, tau=0.0, gamma=50.0)
    x = _benign_batch(n=4)
    init_stateful_history(wrapper, x)
    n_calls = 0
    for _ in range(6):
        xq = x + 0.01 * (n_calls + 1)
        wrapper(xq)
        observe_query(wrapper, xq)
        n_calls += 1
        assert len(wrapper.history) == min(n_calls + 1, 6)  # window_size + 1 cap
