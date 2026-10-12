import torch
from .stateful import init_stateful_history, observe_query


def nes_attack(model, x, config, projector=None):
    """
    Score-based black-box attack via Natural Evolution Strategies (NES).

    Threat model: the attacker may only *query* the model (forward passes) and
    read the returned score/logit. No gradients, no weights, no internals.
    Gradient estimates are built from antithetic Gaussian perturbations
    (Ilyas et al., "Black-box Adversarial Attacks with Limited Queries and
    Information", ICML 2019).

    Every forward pass counts as a query — including against stateful defenses,
    where each query feeds the defense's trajectory history. Queries are issued
    one (batch, d) block at a time so the defender observes the true per-session
    query sequence.

    Args:
        model: callable, model(x) -> logits (batch_size, 1).
        x: (batch_size, d) starting points (assumed benign / correctly classified).
        config: dict with keys epsilon, sigma, alpha, q, max_queries.
        projector: optional callable (x_adv, x_orig) -> x' projecting perturbations
            back onto the tabular data domain (see domain_constraints).

    Returns:
        x_adv, success_flags (bool), queries_to_success (float), total_queries (int)
    """
    epsilon = config.get("epsilon", 0.1)
    sigma = config.get("sigma", 0.05)
    alpha = config.get("alpha", 0.02)
    q = config.get("q", 10)  # antithetic pairs per step -> 2q queries/step
    max_queries = config.get("max_queries", 250)

    batch_size, d = x.shape
    device = x.device
    x0 = x.clone().detach()

    x_adv = x0.clone()
    success_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
    queries_to_success = torch.full((batch_size,), float(max_queries), device=device)
    queries_per_sample = torch.zeros(batch_size, dtype=torch.long, device=device)
    total_queries = 0

    init_stateful_history(model, x_adv)

    def evaluate(x_q):
        """Query the model on one (batch, d) block; the defender logs it."""
        nonlocal total_queries, queries_per_sample
        with torch.no_grad():
            logits = model(x_q).squeeze(-1)
        observe_query(model, x_q)
        total_queries += x_q.size(0)
        queries_per_sample += 1
        return logits

    while queries_per_sample.min() < max_queries and not success_flags.all():
        # --- gradient estimation: 2q queries per sample, one direction at a time
        s_plus = torch.zeros(batch_size, q, device=device)
        s_minus = torch.zeros(batch_size, q, device=device)
        us = torch.zeros(batch_size, q, d, device=device)
        for j in range(q):
            u = torch.randn(batch_size, d, device=device)
            u = u / u.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
            us[:, j, :] = u
            s_plus[:, j] = evaluate(x_adv + sigma * u)
            s_minus[:, j] = evaluate(x_adv - sigma * u)

        grad_est = ((s_plus - s_minus).unsqueeze(2) * us).sum(dim=1) / (2.0 * q * sigma)

        # --- step + projection ----------------------------------------------
        with torch.no_grad():
            grad_norm = grad_est.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
            x_adv = x_adv + alpha * (grad_est / grad_norm)

            eta = x_adv - x0
            eta_norm = eta.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
            factor = torch.min(torch.ones_like(eta_norm), epsilon / eta_norm)
            x_adv = x0 + eta * factor
            x_adv = x_adv.clamp(0.0, 1.0)
            if projector is not None:
                x_adv = projector(x_adv, x0)

        # --- success check: 1 query per sample -------------------------------
        logits = evaluate(x_adv)
        preds = logits > 0
        newly_successful = preds & ~success_flags
        if newly_successful.any():
            queries_to_success[newly_successful] = queries_per_sample[newly_successful].float()
            success_flags |= newly_successful

    return x_adv.detach(), success_flags, queries_to_success, total_queries
