import csv
import json
import math
import os
import sys

import numpy as np
import torch
import yaml

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "experiments", "exp9_blackbox_attack")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp9_common import load_data, make_tascp, print_summary
from src.tascp_eval.attacks.nes import nes_attack
from src.tascp_eval.attacks.hopskipjump import hopskipjump_attack
from src.tascp_eval.attacks.stateful import init_stateful_history, observe_query
from src.tascp_eval.models.nids_classifier import NIDSClassifier

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


# ---------------------------------------------------------------------------
# Blacklight-inspired detector SIMPLIFICATION: probabilistic perceptual
# hashing (exact Blacklight, Li et al. USENIX Sec 2022) is replaced by an
# L2 near-duplicate test — a query is flagged if it lies within tau_match of
# ANY of the same session's stored window queries. Sessions are per-row
# (each sample = one client session, matching per-client deployment).
# This preserves the load-bearing mechanism (iterative attacks emit
# highly similar queries within a session; benign sessions do not) without
# reproducing Blacklight's exact hash tables or detection rates. tau is set
# above the NES probe spacing (~sigma*sqrt(2)) and below benign inter-query
# distances; the benign control below validates the calibration.
# ---------------------------------------------------------------------------
class FingerprintDetector:
    """Density test: flag a query when its session's window contains >= k_min
    queries within tau_match (attacks emit dense probe clusters; benign
    sessions emit sparse, mostly-distinct queries)."""

    def __init__(self, tau_match=0.1, k_min=5, window=50):
        self.tau_match = tau_match
        self.k_min = k_min
        self.window = window  # Blacklight-style fingerprint window W
        self.store = None  # (B, W, d) per-session query ring buffer
        self.ptr = 0
        self.count = 0

    def reset(self):
        self.store = None
        self.ptr = 0
        self.count = 0

    def check_and_update(self, client_id, x):
        if self.store is None or self.store.size(0) != x.size(0) or self.store.size(2) != x.size(1):
            self.store = torch.zeros(x.size(0), self.window, x.size(1), device=x.device)
            self.ptr = 0
            self.count = 0
        hit = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        if self.count > 0:
            # distances from each query to its own session's window
            diff = self.store - x.unsqueeze(1)  # (B, W, d)
            dist = diff.norm(p=2, dim=2)  # (B, W)
            valid = torch.arange(self.window, device=x.device) < self.count
            neighbors = ((dist < self.tau_match) & valid).sum(dim=1)
            hit = neighbors >= self.k_min
        self.store[:, self.ptr, :] = x.detach()
        self.ptr = (self.ptr + 1) % self.window
        self.count = min(self.count + 1, self.window)
        return hit


# ---------------------------------------------------------------------------
# Query-stream recorder: wraps a model so every forward (query) is both
# fingerprint-checked by the detector and logged, without modifying the
# black-box attack loops.
# ---------------------------------------------------------------------------
class RecordingModel(torch.nn.Module):
    def __init__(self, base, detector, defense_active=False):
        super().__init__()
        self.base = base
        self.detector = detector
        self.defense_active = defense_active
        self.flagged_queries = 0
        self.total_queries = 0
        self.detected_clients = set()
        self._blocked_now = False

    def forward(self, x):
        self.total_queries += x.size(0)
        hit = self.detector.check_and_update("client0", x)
        self.flagged_queries += int(hit.sum().item())
        detected = bool(hit.any().item())
        if detected:
            self.detected_clients.add("client0")
        # Blacklight policy: reject detected queries (fail-closed). Serve the
        # query only if not detected; detection semantics are unaffected.
        self._blocked_now = detected and self.defense_active
        self.last_blocked = hit if self.defense_active else torch.zeros_like(hit)
        if self._blocked_now:
            return torch.zeros(x.size(0), 1)
        return self.base(x)

    def reset_state(self):
        self.flagged_queries = 0
        self.total_queries = 0

    # Stateful-defense plumbing passthrough so black-box attacks can drive the
    # wrapper chain exactly as they drive a raw TASCPWrapper.
    def reset_history(self, batch_size, device):
        if hasattr(self.base, "reset_history"):
            self.base.reset_history(batch_size, device)

    def add_to_history(self, x):
        if hasattr(self.base, "add_to_history"):
            self.base.add_to_history(x)


# ---------------------------------------------------------------------------
# Condition runners. Each wraps the model used by the attack in a
# RecordingModel so the detector sees exactly the attacker's query stream.
# ---------------------------------------------------------------------------
def _run_with_detector(wrapped, model_for_attack, X, attack_fn, cfg, label):
    wrapped.detector.reset()
    wrapped.reset_state()
    # TASCP/defense state (if any) handled by the attack's stateful plumbing.
    torch.manual_seed(42)
    x_adv, success, queries, total_q = attack_fn(model_for_attack, X, cfg)
    n = X.size(0)
    # Post-attack verification: an attack only succeeds if the defender
    # *served* an adversarial verdict on the final point. Queries rejected by
    # the detector return a benign default (zeros), so served & logits>0 is
    # the honest success criterion under a reject policy.
    with torch.no_grad():
        final_logits = wrapped(x_adv).squeeze(-1)
        served = ~wrapped.last_blocked
        verified = served & (final_logits > 0)
    result = {
        "asr": verified.float().mean().item(),
        "asr_attack_internal": success.float().mean().item(),
        "mean_queries_to_success": queries[success].mean().item() if success.any() else float("nan"),
        "total_queries": int(total_q),
        "queries_flagged": wrapped.flagged_queries,
        "flag_rate": wrapped.flagged_queries / max(wrapped.total_queries, 1),
        "client_detected": "client0" in wrapped.detected_clients,
    }
    print(
        f"{label:55s} | ASR {result['asr'] * 100:6.2f}% | flagged {result['flag_rate'] * 100:5.1f}% "
        f"| detected client: {result['client_detected']}"
    )
    return result


def load_cfg():
    p = os.path.join(PROJECT_ROOT, "configs", "attack", "blackbox.yaml")
    with open(p) as f:
        return yaml.safe_load(f)


def run(dataset="unsw_nb15", n_samples=2000, seed=42):
    cfg = load_cfg()
    nes_cfg = dict(cfg["nes"])
    hsj_cfg = dict(cfg["hopskipjump"])

    # Held-out shard for the benign detector calibration (attack samples come
    # from the first n samples; calibration uses a DISJOINT sample so benign
    # flags cannot be inflated by repeated queries at the same points).
    base_model, feature_cols, spec, levels, X_all = load_data(dataset, n_samples + 1280, seed)
    X = X_all[:n_samples]
    X_holdout = X_all[n_samples:]
    print(f"\n=== EXP 10: hybrid detection simulation ({dataset}, {len(X)} samples) ===")

    results = {}

    # --- benign traffic detector calibration --------------------------------
    # Benign clients issue sparse, non-repeating queries. A benign stream is
    # approximated by random correctly-classified benign points (no attack
    # iteration => near-duplicates are rare).
    benign_detector = FingerprintDetector()
    benign_model = RecordingModel(base_model, benign_detector, defense_active=True)
    torch.manual_seed(seed)
    with torch.no_grad():
        for i in range(0, len(X_holdout), 64):
            benign_model(X_holdout[i : i + 64])
    results["benign_calibration"] = {
        "queries": benign_model.total_queries,
        "flag_rate": benign_model.flagged_queries / max(benign_model.total_queries, 1),
        "note": "held-out benign sessions; each row = one client with W-window history",
    }
    print(f"Benign control: flag rate {results['benign_calibration']['flag_rate'] * 100:.2f}% over {benign_model.total_queries} benign queries")

    # --- attacker streams under detection-only, defense-only, hybrid --------
    for attack_name, attack_fn, atk_cfg in [
        ("NES", nes_attack, nes_cfg),
        ("HSJ", hopskipjump_attack, hsj_cfg),
    ]:
        # (1) detection-only
        det = FingerprintDetector()
        rec = RecordingModel(base_model, det, defense_active=True)
        results[f"{attack_name} vs Undefended + Detection"] = _run_with_detector(
            rec, rec, X, attack_fn, atk_cfg, f"{attack_name} + Detection (reject policy)"
        )

        # (2) defense-only (TASCP naive, detector passive)
        det_passive = FingerprintDetector()
        tascp = make_tascp(base_model)
        rec_tascp = RecordingModel(tascp, det_passive, defense_active=False)
        results[f"{attack_name} vs TASCP (Naive) + Passive Detection"] = _run_with_detector(
            rec_tascp, rec_tascp, X, attack_fn, atk_cfg, f"{attack_name} + TASCP only (detector passive)"
        )

        # (3) hybrid: TASCP + detection with reject policy
        det_hybrid = FingerprintDetector()
        tascp_h = make_tascp(base_model)
        rec_h = RecordingModel(tascp_h, det_hybrid, defense_active=True)
        results[f"{attack_name} vs Hybrid (TASCP + Detection)"] = _run_with_detector(
            rec_h, rec_h, X, attack_fn, atk_cfg, f"{attack_name} + Hybrid (TASCP + detect & reject)"
        )

    out_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp10_hybrid")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(PROJECT_ROOT, "results", "tables", "exp10_hybrid_tradeoff.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "asr", "flag_rate", "client_detected"])
        for name, r in results.items():
            if name == "benign_calibration":
                w.writerow(["Benign control (flag rate)", "", r["flag_rate"], ""])
            else:
                w.writerow([name, f"{r['asr']:.4f}", f"{r['flag_rate']:.4f}", r["client_detected"]])

    print("\n=== EXP 10 SUMMARY ===")
    for name, r in results.items():
        if name == "benign_calibration":
            continue
        print(f"{name:55s} | ASR {r['asr'] * 100:6.2f}% | flagged {r['flag_rate'] * 100:5.1f}% | detected {r['client_detected']}")


if __name__ == "__main__":
    run()
