import torch

from experiments.exp10_hybrid_detection_sim.run import FingerprintDetector


def test_benign_sparse_sessions_not_flagged():
    """Distinct benign queries per session stay below the density threshold."""
    torch.manual_seed(0)
    det = FingerprintDetector(tau_match=0.1, k_min=5, window=50)
    flags = 0
    total = 0
    for _ in range(20):
        x = torch.rand(64, 43)  # fresh distinct points per session row-slot
        flags += int(det.check_and_update("c", x).sum().item())
        total += 64
    assert flags == 0, f"benign FPR was {flags}/{total}"


def test_attack_cluster_is_flagged():
    """Dense NES-like probe clusters (normalized directions, sigma spacing)
    trip the detector, matching the mechanism in exp10's live runs."""
    torch.manual_seed(0)
    det = FingerprintDetector(tau_match=0.1, k_min=5, window=50)
    anchor = torch.rand(32, 43)
    flagged_any = False
    for i in range(10):
        u = torch.randn(32, 43)
        u = u / u.norm(dim=1, keepdim=True).clamp(min=1e-9)
        probes = anchor + 0.05 * u  # NES probe: sigma * unit direction
        hit = det.check_and_update("c", probes)
        flagged_any |= bool(hit.any()) and i >= 2
    assert flagged_any, "dense NES-like cluster was never flagged"


def test_window_is_bounded():
    """Store never grows beyond the window size per session."""
    det = FingerprintDetector(tau_match=0.1, k_min=5, window=50)
    x = torch.rand(8, 43)
    for _ in range(200):
        det.check_and_update("c", x)
    assert det.store.shape == (8, 50, 43)
    assert det.count == 50
