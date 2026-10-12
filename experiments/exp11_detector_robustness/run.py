import csv
import json
import os
import sys

import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
for _p in (
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "experiments"),
    os.path.join(PROJECT_ROOT, "experiments", "exp9_blackbox_attack"),
    os.path.join(PROJECT_ROOT, "experiments", "exp10_hybrid_detection_sim"),
    EXP_DIR,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp9_common import load_data
from exp10_hybrid_detection_sim.run import FingerprintDetector, RecordingModel, load_cfg
from src.tascp_eval.attacks.nes import nes_attack
from src.tascp_eval.attacks.hopskipjump import hopskipjump_attack

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


# ---------------------------------------------------------------------------
# P0-1 infrastructure: identity-rotation-aware recording model
# ---------------------------------------------------------------------------
class RecordingModelMultiClient(RecordingModel):
    """Records each row's query under a rotating client identity:
    client_id(row i, step s) = (i + s) mod n_identities.

    An evader that spoofs 2..10 identities splits one attack session across
    n_identities defender-side windows, so no single window accumulates
    k_min near-duplicates. The DEFENDER does the attribution — the attack
    itself is unchanged — which is exactly how identity rotation works
    against a session-keyed detector."""

    def __init__(self, base, detector, defense_active=False, n_identities=1):
        super().__init__(base, detector, defense_active=defense_active)
        self.n_identities = n_identities
        self._step = 0
        self.detected_clients = set()

    def forward(self, x):
        self.total_queries += x.size(0)
        hit = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        for i in range(x.size(0)):
            cid = f"client_{(i + self._step) % self.n_identities}"
            h_i = self.detector.check_and_update(cid, x[i : i + 1])
            hit[i] = h_i[0]
        self._step += 1
        self.flagged_queries += int(hit.sum().item())
        if bool(hit.any().item()):
            self.detected_clients.add("multi")
        self.last_blocked = hit if self.defense_active else torch.zeros_like(hit)
        if self.defense_active and bool(hit.any().item()):
            return torch.zeros(x.size(0), 1)
        return self.base(x)


class MultiFingerprintDetector(FingerprintDetector):
    """Per-client-id stores (dict-backed), so identity rotation is modelled
    honestly. Each identity keeps its own W-window and count."""

    def __init__(self, tau_match=0.1, k_min=5, window=50):
        super().__init__(tau_match=tau_match, k_min=k_min, window=window)
        self.stores = {}

    def reset(self):
        self.stores = {}

    def check_and_update(self, client_id, x):
        # x is a single row (1, d)
        if client_id not in self.stores:
            self.stores[client_id] = {
                "store": torch.zeros(1, self.window, x.size(1), device=x.device),
                "ptr": 0,
                "count": 0,
            }
        st = self.stores[client_id]
        store, ptr, count = st["store"], st["ptr"], st["count"]
        hit = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        if count > 0:
            diff = store[0, :] - x[0]  # (W, d)
            dist = diff.norm(p=2, dim=1)  # (W,)
            valid = torch.arange(self.window, device=x.device) < count
            neighbors = ((dist < self.tau_match) & valid).sum()
            hit[0] = neighbors >= self.k_min
        store[0, ptr, :] = x.detach()[0]
        st["ptr"] = (ptr + 1) % self.window
        st["count"] = min(count + 1, self.window)
        return hit


class CountingDetector(FingerprintDetector):
    """Records, for every query, the number of window neighbors within
    tau_match (the detector's decision variable) instead of thresholding.
    Used for the empirical ROC over the count threshold k."""

    def __init__(self, tau_match=0.1, window=50):
        super().__init__(tau_match=tau_match, k_min=1, window=window)
        self.counts = []

    def reset(self):
        super().reset()
        self.counts = []

    def check_and_update(self, client_id, x):
        hit_counts = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if self.store is None or self.store.size(0) != x.size(0) or self.store.size(2) != x.size(1):
            self.store = torch.zeros(x.size(0), self.window, x.size(1), device=x.device)
            self.ptr = 0
            self.count = 0
        if self.count > 0:
            diff = self.store - x.unsqueeze(1)
            dist = diff.norm(p=2, dim=2)
            valid = torch.arange(self.window, device=x.device) < self.count
            hit_counts = ((dist < self.tau_match) & valid).sum(dim=1).long()
        self.store[:, self.ptr, :] = x.detach()
        self.ptr = (self.ptr + 1) % self.window
        self.count = min(self.count + 1, self.window)
        self.counts.extend(hit_counts.cpu().tolist())
        return hit_counts >= self.k_min


# ---------------------------------------------------------------------------
# P0-1: adaptive evader attack variants
# ---------------------------------------------------------------------------
def nes_attack_jitter(model, x, config, projector=None, seed=42):
    """NES with query jitter: every probe pair is drawn around a freshly
    randomized center (anchor + jitter noise), so the identical probe point is
    never emitted twice and consecutive queries differ by >= ~jitter_scale.
    Directly attacks the near-duplicate-density mechanism. Costs 0 extra
    queries; the estimator stays unbiased for the local gradient because each
    pair still measures f(center + sigma u) - f(center - sigma u)."""
    epsilon = config.get("epsilon", 0.1)
    sigma = config.get("sigma", 0.05)
    alpha = config.get("alpha", 0.02)
    q = config.get("q", 10)
    max_queries = config.get("max_queries", 250)
    jitter_scale = config.get("jitter_scale", 0.12)  # > tau_match = 0.1

    batch_size, d = x.shape
    device = x.device
    x0 = x.clone().detach()
    x_adv = x0.clone()
    success = torch.zeros(batch_size, dtype=torch.bool, device=device)
    q_to_success = torch.full((batch_size,), float(max_queries), device=device)
    q_per_sample = torch.zeros(batch_size, dtype=torch.long, device=device)
    total_queries = 0
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    from src.tascp_eval.attacks.stateful import init_stateful_history, observe_query

    init_stateful_history(model, x_adv)

    def ev(xq):
        nonlocal total_queries, q_per_sample
        with torch.no_grad():
            logits = model(xq).squeeze(-1)
        observe_query(model, xq)
        total_queries += xq.size(0)
        q_per_sample += 1
        return logits

    def project_ball(y):
        eta = y - x0
        en = eta.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
        return (x0 + eta * torch.min(torch.ones_like(en), epsilon / en)).clamp(0.0, 1.0)

    while q_per_sample.min() < max_queries and not success.all():
        s_plus = torch.zeros(batch_size, q, device=device)
        s_minus = torch.zeros(batch_size, q, device=device)
        us = torch.zeros(batch_size, q, d, device=device)
        for j in range(q):
            u = torch.randn(batch_size, d, device=device, generator=rng)
            u = u / u.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
            us[:, j, :] = u
            center = project_ball(x_adv + jitter_scale * torch.randn(batch_size, d, device=device, generator=rng))
            s_plus[:, j] = ev((center + sigma * u).clamp(0, 1))
            s_minus[:, j] = ev((center - sigma * u).clamp(0, 1))
        grad_est = ((s_plus - s_minus).unsqueeze(2) * us).sum(dim=1) / (2.0 * q * sigma)
        with torch.no_grad():
            gn = grad_est.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
            x_adv = project_ball(x_adv + alpha * (grad_est / gn))
            if projector is not None:
                x_adv = projector(x_adv, x0)
        logits = ev(x_adv)
        newly = (logits > 0) & ~success
        if newly.any():
            q_to_success[newly] = q_per_sample[newly].float()
            success |= newly
        if q_per_sample.min() >= max_queries:
            break
    return x_adv.detach(), success, q_to_success, total_queries


def hsj_attack_min_distance(model, x, config, projector=None, min_dist=0.15, seed=42):
    """HSJ variant enforcing a minimum L2 separation between queries from
    *different* phases: probes are taken at an inflated radius
    delta_eff = max(delta, min_dist) and the interior step never shrinks below
    min_dist. Caveat (reported honestly): bisection chains inherently emit
    near-duplicate midpoint queries — a decision-based attack cannot fully
    avoid near-duplicates without abandoning bisection; this variant measures
    how much of the detection signal survives when only cross-step duplication
    is removed."""
    n_dirs = config.get("N", 16)
    n_bisect = config.get("b", 6)
    delta = config.get("delta_0", 0.1)
    delta_min = config.get("delta_min", 0.001)
    max_queries = config.get("max_queries", 1000)

    batch_size, d = x.shape
    device = x.device
    x0 = x.clone().detach()
    success = torch.zeros(batch_size, dtype=torch.bool, device=device)
    q_to_success = torch.full((batch_size,), float(max_queries), device=device)
    q_per_sample = torch.zeros(batch_size, dtype=torch.long, device=device)
    total_queries = 0

    from src.tascp_eval.attacks.stateful import init_stateful_history, observe_query

    init_stateful_history(model, x0)

    def ev(xq):
        nonlocal total_queries, q_per_sample
        with torch.no_grad():
            logits = model(xq).squeeze(-1)
        observe_query(model, xq)
        total_queries += xq.size(0)
        q_per_sample = q_per_sample + 1
        return logits > 0

    def bisect(adv_point, target, n_iters):
        lo = torch.zeros(batch_size, 1, device=device)
        hi = torch.ones(batch_size, 1, device=device)
        for _ in range(n_iters):
            mid = 0.5 * (lo + hi)
            a = ev(adv_point + mid * (target - adv_point)).unsqueeze(1)
            lo = torch.where(a, mid, lo)
            hi = torch.where(a, hi, mid)
        return lo, adv_point + lo * (target - adv_point)

    x_star = x0.clone()
    found = torch.zeros(batch_size, dtype=torch.bool, device=device)
    radii = config.get("init_radii", [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    draws_per_radius = config.get("init_draws", 2)
    for r in radii:
        for k in range(draws_per_radius):
            u = torch.randn(batch_size, d, device=device)
            u = u / u.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
            cand = (x0 + (r + k * min_dist) * u).clamp(0.0, 1.0)
            adv = ev(cand)
            newly = adv & ~found
            if newly.any():
                q_at = q_per_sample.clone()
                _, boundary = bisect(cand, x0, n_bisect)
                x_star = torch.where(newly.unsqueeze(1), boundary, x_star)
                success |= newly
                q_to_success[newly] = q_at[newly].float()
                found |= newly
        if found.all():
            break

    delta_eff = max(delta, min_dist)
    while found.any() and q_per_sample.min() < max_queries:
        grad_sum = torch.zeros(batch_size, d, device=device)
        for _ in range(n_dirs):
            u = torch.randn(batch_size, d, device=device)
            u = u / u.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
            ray = (x_star + delta_eff * u).clamp(0.0, 1.0)
            ray_len = (ray - x_star).norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
            lo = torch.zeros(batch_size, 1, device=device)
            hi = torch.ones(batch_size, 1, device=device)
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                a = ev(x_star + mid * (ray - x_star)).unsqueeze(1)
                lo = torch.where(a, mid, lo)
                hi = torch.where(a, hi, mid)
            theta = (lo * ray_len).squeeze(1)
            grad_sum += theta.unsqueeze(1) * u
        grad = grad_sum / n_dirs
        gn = grad.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
        x_cand = (x_star + delta_eff * (grad / gn)).clamp(0.0, 1.0)
        if projector is not None:
            x_cand = projector(x_cand, x0)
        adv_cand = ev(x_cand)
        _, new_point = bisect(x_cand, x0, n_bisect)
        improved = adv_cand & ((new_point - x0).norm(p=2, dim=1) < (x_star - x0).norm(p=2, dim=1))
        x_star = torch.where(improved.unsqueeze(1), new_point, x_star)
        if improved.any():
            delta_eff = max(delta_min, min_dist)  # never shrink below min_dist
        if q_per_sample.min() >= max_queries:
            break
    x_adv = torch.where(success.unsqueeze(1), x_star, x0)
    return x_adv.detach(), success, q_to_success, total_queries


def run_adaptive_evaders(base_model, X, nes_cfg, hsj_cfg, dataset):
    """P0-1: run jitter / multi-identity / min-distance evaders against the
    density detector under the reject policy; report detection rate + verified ASR."""
    print(f"\n=== EXP 11 part A: adaptive evaders vs density detector ({dataset}) ===")
    results = {}
    conditions = [
        ("NES baseline (no evasion)", nes_attack, nes_cfg, 1),
        ("NES + query jitter (0.12)", nes_attack_jitter, dict(nes_cfg, jitter_scale=0.12), 1),
        ("NES + multi-identity (5 ids)", nes_attack, nes_cfg, 5),
        ("NES + multi-identity (10 ids)", nes_attack, nes_cfg, 10),
        ("HSJ baseline (no evasion)", hopskipjump_attack, hsj_cfg, 1),
        ("HSJ + min-distance (0.15)", hsj_attack_min_distance, hsj_cfg, 1),
    ]
    for name, fn, cfg, n_ids in conditions:
        if n_ids > 1:
            det = MultiFingerprintDetector()
            rec = RecordingModelMultiClient(base_model, det, defense_active=True, n_identities=n_ids)
        else:
            det = FingerprintDetector()
            rec = RecordingModel(base_model, det, defense_active=True)
        rec.detector.reset()
        rec.reset_state()
        torch.manual_seed(42)
        x_adv, success, queries, total_q = fn(rec, X, cfg)
        with torch.no_grad():
            final_logits = rec(x_adv).squeeze(-1)
            served = ~rec.last_blocked
            verified = served & (final_logits > 0)
        results[name] = {
            "verified_asr": verified.float().mean().item(),
            "internal_asr": success.float().mean().item(),
            "queries_flagged": rec.flagged_queries,
            "total_queries": rec.total_queries,
            "flag_rate": rec.flagged_queries / max(rec.total_queries, 1),
            "client_detected": bool(rec.detected_clients),
        }
        r = results[name]
        print(
            f"{name:38s} | verified ASR {r['verified_asr']*100:6.2f}% | flag rate {r['flag_rate']*100:5.1f}% "
            f"| detected {r['client_detected']}"
        )
    return results


# ---------------------------------------------------------------------------
# P0-2: detector sweep + ROC + benign-boundary FPR
# ---------------------------------------------------------------------------
def benign_boundary_shard(base_model, X_all, n_boundary=2000):
    """Benign flows NEAREST the decision boundary: correctly-classified benign
    points ranked by smallest |margin|. FPR measured here, not on random
    benign flows, is the deployment-relevant false-positive rate."""
    margins = []
    with torch.no_grad():
        for i in range(0, len(X_all), 100_000):
            margins.append(base_model(X_all[i : i + 100_000]).squeeze(-1).abs().cpu())
    margins = torch.cat(margins)
    n = min(n_boundary, len(X_all))
    idx = torch.argsort(margins)[:n]
    return X_all[idx], margins[idx]


def detector_sweep(base_model, X_attack, X_boundary, nes_cfg, hsj_cfg):
    """Grid tau x k: flag rate on attack streams vs FPR on boundary-benign."""
    print("\n=== EXP 11 part B: detector sweep (tau x k) + boundary FPR ===")
    taus = [0.05, 0.1, 0.2, 0.3, 0.5]
    ks = [3, 5, 8, 12, 20]
    n_sweep = min(256, len(X_attack))
    Xs = X_attack[:n_sweep]
    print(f"  attack streams subsampled to {n_sweep} for the 25-point grid x 2 attacks")

    # benign-boundary FPR per (tau, k): cheap passes, no attack
    benign_fpr = {}
    for tau in taus:
        for k in ks:
            det = FingerprintDetector(tau_match=tau, k_min=k, window=50)
            flags, total = 0, 0
            with torch.no_grad():
                for i in range(0, len(X_boundary), 64):
                    h = det.check_and_update("benign", X_boundary[i : i + 64])
                    flags += int(h.sum().item())
                    total += h.size(0)
            benign_fpr[(tau, k)] = flags / max(total, 1)

    # attack flag rate per (tau, k)
    attack_flag = {}
    for attack_name, attack_fn, cfg in [("NES", nes_attack, nes_cfg), ("HSJ", hopskipjump_attack, hsj_cfg)]:
        for tau in taus:
            for k in ks:
                det = FingerprintDetector(tau_match=tau, k_min=k, window=50)
                rec = RecordingModel(base_model, det, defense_active=False)
                rec.detector.reset()
                rec.reset_state()
                torch.manual_seed(42)
                attack_fn(rec, Xs, cfg)
                attack_flag[(attack_name, tau, k)] = rec.flagged_queries / max(rec.total_queries, 1)

    grid = {}
    for tau in taus:
        for k in ks:
            grid[f"tau={tau},k={k}"] = {
                "benign_boundary_fpr": benign_fpr[(tau, k)],
                "NES_flag_rate": attack_flag[("NES", tau, k)],
                "HSJ_flag_rate": attack_flag[("HSJ", tau, k)],
            }
            g = grid[f"tau={tau},k={k}"]
            print(
                f"  tau={tau:<4} k={k:<3} | benign-boundary FPR {g['benign_boundary_fpr']*100:6.2f}% "
                f"| NES flag {g['NES_flag_rate']*100:6.2f}% | HSJ flag {g['HSJ_flag_rate']*100:6.2f}%"
            )

    # ROC over the count threshold k at tau=0.1 on full query streams
    roc = {}
    for attack_name, attack_fn, cfg in [("NES", nes_attack, nes_cfg), ("HSJ", hopskipjump_attack, hsj_cfg)]:
        det = CountingDetector(tau_match=0.1, window=50)
        rec = RecordingModel(base_model, det, defense_active=False)
        rec.detector.reset()
        rec.reset_state()
        torch.manual_seed(42)
        attack_fn(rec, X_attack, cfg)
        s_attack = torch.tensor(det.counts, dtype=torch.float32)

        det_b = CountingDetector(tau_match=0.1, window=50)
        with torch.no_grad():
            for i in range(0, len(X_boundary), 64):
                det_b.check_and_update("benign", X_boundary[i : i + 64])
        s_benign = torch.tensor(det_b.counts, dtype=torch.float32)

        points = []
        for t in range(0, 11):
            tpr = (s_attack > t).float().mean().item() if len(s_attack) else 0.0
            fpr = (s_benign > t).float().mean().item() if len(s_benign) else 0.0
            points.append({"k": t, "tpr": tpr, "fpr": fpr})
        roc[attack_name] = points
        auc = _auc(points)
        print(f"  ROC ({attack_name}, tau=0.1): AUC {auc:.4f} | TPR@FPR=1%: {_tpr_at_fpr(points, 0.01)*100:.1f}%")
    return grid, roc


def _auc(points):
    """AUC via the Mann-Whitney U statistic over (fpr, tpr) points; robust to
    degenerate curves (e.g. all FPRs = 0)."""
    # points are (threshold asc). Convert to (fpr, tpr) pairs incl. (0,0),(1,1)
    pts = [(0.0, 0.0)] + [(p["fpr"], p["tpr"]) for p in points] + [(1.0, 1.0)]
    pts = sorted(set(pts))
    auc = 0.0
    for (f0, t0), (f1, t1) in zip(pts, pts[1:]):
        if f1 > f0:
            auc += (f1 - f0) * 0.5 * (t0 + t1)
    return auc


def _tpr_at_fpr(points, fpr_target):
    best = 0.0
    for p in points:
        if p["fpr"] <= fpr_target:
            best = max(best, p["tpr"])
    return best


# ---------------------------------------------------------------------------
# P0-3: real quantize-then-hash Blacklight-style fingerprinting
# ---------------------------------------------------------------------------
class QuantizeHashDetector:
    """Blacklight-style (Li et al., USENIX Sec 2022) fingerprinting adapted to
    flow features: quantize each feature to a lattice of step `quantum` and
    flag a query when its session's window contains >= k_min queries whose
    quantized vector is within Hamming distance r (Blacklight's "robust"
    matching; r=0 is the exact-match variant)."""

    def __init__(self, quantum=0.05, k_min=5, window=50, hamming_r=0):
        self.quantum = quantum
        self.k_min = k_min
        self.window = window
        self.hamming_r = hamming_r
        self.tables = {}

    def reset(self):
        self.tables = {}

    def _quantize(self, x):
        return torch.floor(x / self.quantum).to(torch.int32)

    def check_and_update(self, client_id, x):
        if client_id not in self.tables:
            self.tables[client_id] = {
                "ring": torch.zeros(self.window, x.size(1), dtype=torch.int32, device=x.device),
                "ptr": 0,
                "count": 0,
            }
        t = self.tables[client_id]
        q = self._quantize(x)  # (B, d)
        hit = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        if t["count"] > 0:
            valid = torch.arange(self.window, device=x.device) < t["count"]
            # Hamming distance of each query to every stored window entry
            neq = (t["ring"].unsqueeze(0) != q.unsqueeze(1)).float().sum(dim=2)  # (B, W)
            neighbors = (((neq <= self.hamming_r) & valid).float().sum(dim=1))
            hit = neighbors >= self.k_min
        t["ring"][t["ptr"], :] = q[0]  # session store: one row per check (RecordingModel batches rows together)
        t["ptr"] = (t["ptr"] + 1) % self.window
        t["count"] = min(t["count"] + x.size(0), self.window)
        return hit


def run_hash_vs_l2(base_model, X, nes_cfg, hsj_cfg, X_boundary):
    """P0-3: compare quantize-then-hash (Blacklight-style, with a small grid
    over quantum and Hamming radius r) vs the L2 proxy on the same attack
    streams, plus benign (random + boundary) controls."""
    print("\n=== EXP 11 part C: quantize-then-hash vs L2 proxy ===")
    results = {}

    def flag_rate(det, X_stream):
        flags, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(X_stream), 64):
                h = det.check_and_update("s", X_stream[i : i + 64])
                flags += int(h.sum().item())
                total += h.size(0)
        return flags / max(total, 1)

    # hash-detector grid: report the best attack-detection operating point
    # subject to ~0 benign-boundary FPR (Blacklight's tuning criterion)
    quantum_grid = [0.02, 0.05, 0.1]
    r_grid = [0, 5, 10, 20]
    hash_grid = {}
    for attack_name, attack_queries in _collect_attack_queries(base_model, X, nes_cfg, hsj_cfg).items():
        best = None
        for quantum in quantum_grid:
            for r in r_grid:
                det = QuantizeHashDetector(quantum=quantum, k_min=5, window=50, hamming_r=r)
                fr = flag_rate(det, attack_queries)
                hash_grid[f"{attack_name}/q={quantum},r={r}"] = fr
                if best is None or fr > best[1]:
                    best = (f"q={quantum},r={r}", fr)
        results[f"{attack_name} / quantize-then-hash (best)"] = {
            "config": best[0],
            "flag_rate": best[1],
            "grid": {k: v for k, v in hash_grid.items() if k.startswith(attack_name)},
        }
        print(f"  {attack_name} / hash best [{best[0]}] flag rate {best[1]*100:6.2f}%")

    # L2 proxy on the same streams (queries replayed identically)
    for attack_name, attack_queries in _collect_attack_queries(base_model, X, nes_cfg, hsj_cfg).items():
        det = FingerprintDetector()
        fr = flag_rate(det, attack_queries)
        results[f"{attack_name} / L2 proxy"] = {"flag_rate": fr}
        print(f"  {attack_name} / L2 proxy     flag rate {fr*100:6.2f}%")

    # benign controls for the hash detector at every grid point
    for ctrl_name, X_ctrl in [("random benign", X[:512]), ("boundary benign", X_boundary[:512])]:
        worst = 0.0
        for quantum in quantum_grid:
            for r in r_grid:
                det = QuantizeHashDetector(quantum=quantum, k_min=5, window=50, hamming_r=r)
                fr = flag_rate(det, X_ctrl)
                worst = max(worst, fr)
        results[f"Benign control ({ctrl_name}) / hash worst-case"] = {"flag_rate": worst}
        print(f"  Benign control ({ctrl_name}) / hash worst-case over grid: {worst*100:.2f}%")
    return results


def _collect_attack_queries(base_model, X, nes_cfg, hsj_cfg):
    """Record each attack's raw query stream once, so every detector variant
    is evaluated on the IDENTICAL query sequence (paired comparison)."""
    streams = {}
    for attack_name, attack_fn, cfg in [("NES", nes_attack, nes_cfg), ("HSJ", hopskipjump_attack, hsj_cfg)]:
        queries = []

        class _Recorder(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, x):
                queries.append(x.detach().cpu().clone())
                return self.base(x)

            def reset_history(self, *a, **k):
                pass

            def add_to_history(self, x):
                pass  # observe_query() calls this after forward; already recorded

        rec = _Recorder(base_model)
        torch.manual_seed(42)
        attack_fn(rec, X, cfg)
        streams[attack_name] = torch.cat(queries, dim=0)
        print(f"  recorded {len(streams[attack_name])} {attack_name} queries for replay")
    return streams


def run(dataset="unsw_nb15", n_samples=2000, seed=42):
    cfg = load_cfg()
    nes_cfg = dict(cfg["nes"])
    hsj_cfg = dict(cfg["hopskipjump"])

    base_model, feature_cols, spec, levels, X_all = load_data(dataset, n_samples + 2560, seed)
    X = X_all[:n_samples]
    print(f"\n=== EXP 11: detector robustness ({dataset}, {len(X)} attack samples) ===")

    X_boundary, boundary_margins = benign_boundary_shard(base_model, X_all, n_boundary=2000)
    print(
        f"Benign boundary shard: {len(X_boundary)} flows, |margin| in "
        f"[{boundary_margins.min():.4f}, {boundary_margins.max():.4f}]"
    )

    results = {}
    results["adaptive_evaders"] = run_adaptive_evaders(base_model, X, nes_cfg, hsj_cfg, dataset)
    grid, roc = detector_sweep(base_model, X, X_boundary, nes_cfg, hsj_cfg)
    results["detector_sweep"] = grid
    results["roc"] = roc
    results["hash_vs_l2"] = run_hash_vs_l2(base_model, X, nes_cfg, hsj_cfg, X_boundary)

    out_dir = os.path.join(PROJECT_ROOT, "results", "logs", "exp11_detector_robustness")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(os.path.join(PROJECT_ROOT, "results", "tables", "exp11_detector_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "k", "benign_boundary_fpr", "NES_flag_rate", "HSJ_flag_rate"])
        for key, v in grid.items():
                tau, k = key.split(",")
                w.writerow([float(tau.replace("tau=", "")), int(k.replace("k=", "")), f"{v['benign_boundary_fpr']:.4f}", f"{v['NES_flag_rate']:.4f}", f"{v['HSJ_flag_rate']:.4f}"])

    with open(os.path.join(PROJECT_ROOT, "results", "tables", "exp11_roc.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attack", "k_threshold", "tpr", "fpr"])
        for attack_name, pts in roc.items():
            for p in pts:
                w.writerow([attack_name, p["k"], f"{p['tpr']:.4f}", f"{p['fpr']:.4f}"])

    with open(os.path.join(PROJECT_ROOT, "results", "tables", "exp11_adaptive_evaders.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["evader", "verified_asr", "internal_asr", "flag_rate", "client_detected"])
        for name, r in results["adaptive_evaders"].items():
            w.writerow([name, f"{r['verified_asr']:.4f}", f"{r['internal_asr']:.4f}", f"{r['flag_rate']:.4f}", r["client_detected"]])

    print("\n=== EXP 11 SUMMARY ===")
    print("Adaptive evaders (verified ASR / flag rate / detected):")
    for name, r in results["adaptive_evaders"].items():
        print(f"  {name:38s} | {r['verified_asr']*100:6.2f}% | {r['flag_rate']*100:5.1f}% | {r['client_detected']}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "unsw_nb15"
    run(dataset=ds)
