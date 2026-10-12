import torch
from .stateful import init_stateful_history, observe_query


def hopskipjump_attack(model, x, config, projector=None):
    """
    Decision-based black-box attack (Chen, Jordan, Wainwright,
    "HopSkipJumpAttack: A Query-Efficient Decision-Based Attack", IEEE S&P 2020).

    Threat model: label-only access. The attacker sees only the predicted class
    (attack / benign) of each query — no scores, no gradients, no weights.

    Simplified batched variant for binary tabular classifiers:
      1. Initialization: random-direction search over an expanding radius
         schedule to find any adversarial point, then bisection toward x.
      2. Each step: sample N boundary directions, locate the decision boundary
         along each ray by bisection, estimate the boundary-normal direction,
         take an interior step, and bisect back toward x.

    Every forward pass counts as a query (stateful defenses see all of them).
    Queries are issued one (batch, d) block at a time so the defender observes
    the true per-session query sequence.

    Args:
        model: callable, model(x) -> logits (batch_size, 1).
        x: (batch_size, d) starting points (assumed benign / correctly classified).
        config: dict with keys N, b, delta_0, delta_min, max_queries.
        projector: optional callable (x_adv, x_orig) -> x' enforcing the tabular
            data domain (see domain_constraints).

    Returns:
        x_adv, success_flags (bool), queries_to_success (float), total_queries (int)
    """
    n_dirs = config.get("N", 16)
    n_bisect = config.get("b", 6)
    delta = config.get("delta_0", 0.1)
    delta_min = config.get("delta_min", 0.001)
    max_queries = config.get("max_queries", 1000)

    batch_size, d = x.shape
    device = x.device
    x0 = x.clone().detach()

    success_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
    queries_to_success = torch.full((batch_size,), float(max_queries), device=device)
    queries_per_sample = torch.zeros(batch_size, dtype=torch.long, device=device)
    total_queries = 0

    init_stateful_history(model, x0)

    def evaluate(x_q):
        """Query the model on one (batch, d) block; the defender logs it."""
        nonlocal total_queries, queries_per_sample
        with torch.no_grad():
            logits = model(x_q).squeeze(-1)
        observe_query(model, x_q)
        total_queries += x_q.size(0)
        queries_per_sample += 1
        return logits > 0  # True = attack class

    def unit_dirs(shape):
        u = torch.randn(*shape, device=device)
        return u / u.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)

    def bisect_toward(adv_point, target, n_iters):
        """Largest t in [0, 1] with adv_point + t*(target - adv_point) adversarial.
        Returns (t, point). Both (batch, 1) / (batch, d)."""
        lo = torch.zeros(batch_size, 1, device=device)
        hi = torch.ones(batch_size, 1, device=device)
        for _ in range(n_iters):
            mid = 0.5 * (lo + hi)
            a = evaluate(adv_point + mid * (target - adv_point)).unsqueeze(1)
            lo = torch.where(a, mid, lo)
            hi = torch.where(a, hi, mid)
        return lo, adv_point + lo * (target - adv_point)

    # --- Phase 1: initialization ---------------------------------------------
    x_star = x0.clone()  # current adversarial point (boundary-adjacent)
    found = torch.zeros(batch_size, dtype=torch.bool, device=device)
    radii = config.get("init_radii", [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    draws_per_radius = config.get("init_draws", 2)

    for r in radii:
        for _ in range(draws_per_radius):
            u = unit_dirs((batch_size, d))
            cand = (x0 + r * u).clamp(0.0, 1.0)
            adv = evaluate(cand)
            newly = adv & ~found
            if newly.any():
                q_at_eval = queries_per_sample.clone()
                # pull candidate to the boundary between cand and x0
                _, boundary = bisect_toward(cand, x0, n_bisect)
                x_star = torch.where(newly.unsqueeze(1), boundary, x_star)
                success_flags |= newly
                queries_to_success[newly] = q_at_eval[newly].float()
                found |= newly
        if found.all():
            break

    # --- Phase 2: boundary refinement ----------------------------------------
    while found.any() and queries_per_sample.min() < max_queries:
        # locate the boundary along each direction, one (batch, d) block at a time
        grad_sum = torch.zeros(batch_size, d, device=device)
        for _ in range(n_dirs):
            u = unit_dirs((batch_size, d))
            ray = (x_star + delta * u).clamp(0.0, 1.0)
            ray_len = (ray - x_star).norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)

            lo = torch.zeros(batch_size, 1, device=device)
            hi = torch.ones(batch_size, 1, device=device)
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                a = evaluate(x_star + mid * (ray - x_star)).unsqueeze(1)
                lo = torch.where(a, mid, lo)
                hi = torch.where(a, hi, mid)

            theta = (lo * ray_len).squeeze(1)  # distance-to-boundary estimate
            grad_sum += theta.unsqueeze(1) * u

        grad = grad_sum / n_dirs
        grad_norm = grad.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)

        # interior step, then bisect back toward x0
        x_cand = (x_star + delta * (grad / grad_norm)).clamp(0.0, 1.0)
        if projector is not None:
            x_cand = projector(x_cand, x0)
        adv_cand = evaluate(x_cand)
        _, new_point = bisect_toward(x_cand, x0, n_bisect)

        improved = adv_cand & ((new_point - x0).norm(p=2, dim=1) < (x_star - x0).norm(p=2, dim=1))
        x_star = torch.where(improved.unsqueeze(1), new_point, x_star)

        # shrink the interior step after a successful refinement (HSJ schedule)
        if improved.any():
            delta = max(delta_min, delta * 0.5)

        if queries_per_sample.min() >= max_queries:
            break

    x_adv = torch.where(success_flags.unsqueeze(1), x_star, x0)
    return x_adv.detach(), success_flags, queries_to_success, total_queries
