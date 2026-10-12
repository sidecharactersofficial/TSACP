"""Score/decision-based attack variants for the Corollary 3 scope test.

Corollary 3 (antithetic blindness) covers gradient estimators built from
*symmetric* (antithetic) probe pairs only. To delimit its scope we implement
the standard non-antithetic score-based attackers and log, per attack, the
fraction of queries on which the stateful trigger actually fires
(delta_theta > 1e-5). Any non-antithetic attacker that fires the trigger
limits the corollary's practical reach, and the paper must say so.

All attacks share the black-box query protocol of nes.py / hopskipjump.py:
every forward pass counts as a query and is fed to the stateful wrapper's
history via observe_query, so the defender observes the true per-session
query stream and trigger-activation statistics are measured on it.

Attacks (all batched, binary margin models, L2 ball):
  nes_independent      - NES with i.i.d. Gaussian probes (NO antithetic pairing)
  spsa                 - SPSA (Spall 1992): one symmetric pair per step, but with
                         the classic momentum-averaged update; included because
                         its probe pattern differs from multi-sample NES
  zoo                  - ZOO (Chen et al. 2017): coordinate-wise symmetric FD
                         probes (deterministic antithetic family, per-coordinate)
  simba                - SimBA (Guo et al. 2019): decision-based random
                         directional search with accept/reject steps
  square               - Square Attack (Andriushchenko et al. 2020): random
                         block-masked sign updates, score-free
  boundary             - Boundary Attack (Brendel et al. 2018): random-walk
                         along the decision boundary (label-only)
"""

import math

import torch

from .stateful import init_stateful_history, observe_query


def _new_state(x, max_queries, device):
    b = x.size(0)
    success = torch.zeros(b, dtype=torch.bool, device=device)
    q_to_success = torch.full((b,), float(max_queries), device=device)
    q_per_sample = torch.zeros(b, dtype=torch.long, device=device)
    return success, q_to_success, q_per_sample


def _check(success, q_to_success, q_per_sample, logits, max_queries):
    newly = (logits > 0) & ~success
    if newly.any():
        q_to_success[newly] = q_per_sample[newly].float()
        success |= newly
    return success, q_to_success


def _project(x_adv, x0, epsilon):
    eta = x_adv - x0
    en = eta.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
    return (x0 + eta * torch.min(torch.ones_like(en), epsilon / en)).clamp(0.0, 1.0)


def _finish_eval(model, x, success, q_to_success, q_per_sample, max_queries, trigger_stats):
    with torch.no_grad():
        logits = model(x).squeeze(-1)
    observe_query(model, x)
    q_per_sample += 1
    success, q_to_success = _check(success, q_to_success, q_per_sample, logits, max_queries)
    return success, q_to_success, logits


def _eval(model, x, q_per_sample):
    """Query the model on one (batch, d) block and record it."""
    with torch.no_grad():
        logits = model(x).squeeze(-1)
    observe_query(model, x)
    q_per_sample += 1
    return logits


def nes_independent_attack(model, x, config, projector=None):
    """NES gradient estimate from q i.i.d. probes (NOT antithetic pairs):
    g = (1/q) sum_j f(x + sigma u_j) u_j / sigma. Non-antithetic: consecutive
    probe directions are independent, so consecutive *anchor* displacements
    are uncorrelated rather than sign-flipped."""
    epsilon = config.get("epsilon", 0.1)
    sigma = config.get("sigma", 0.05)
    alpha = config.get("alpha", 0.02)
    q = config.get("q", 10)
    max_queries = config.get("max_queries", 250)

    device = x.device
    batch_size, d = x.shape
    x0 = x.clone().detach()
    x_adv = x0.clone()
    success, q_to_success, q_per_sample = _new_state(x, max_queries, device)
    total_queries = 0

    init_stateful_history(model, x_adv)
    while q_per_sample.min() < max_queries and not success.all():
        s = torch.zeros(batch_size, q, device=device)
        us = torch.zeros(batch_size, q, d, device=device)
        for j in range(q):
            u = torch.randn(batch_size, d, device=device)
            u = u / u.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
            us[:, j, :] = u
            s[:, j] = _eval(model, x_adv + sigma * u, q_per_sample)
            total_queries += batch_size
        grad_est = (s.unsqueeze(2) * us).sum(dim=1) / (q * sigma)
        with torch.no_grad():
            gn = grad_est.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
            x_adv = _project(x_adv + alpha * (grad_est / gn), x0, epsilon)
            if projector is not None:
                x_adv = projector(x_adv, x0)
        logits = _eval(model, x_adv, q_per_sample)
        total_queries += batch_size
        success, q_to_success = _check(success, q_to_success, q_per_sample, logits, max_queries)
        if q_per_sample.min() >= max_queries:
            break
    return x_adv.detach(), success, q_to_success, total_queries


def spsa_attack(model, x, config, projector=None):
    """SPSA: a single symmetric pair per step (2 queries) with a momentum-
    averaged update direction. Included because SPSA's probe cadence (one pair,
    long anchor steps) differs from NES's (q pairs per anchor)."""
    epsilon = config.get("epsilon", 0.1)
    sigma = config.get("sigma", 0.05)
    alpha = config.get("alpha", 0.02)
    max_queries = config.get("max_queries", 250)
    momentum = config.get("spsa_momentum", 0.9)

    device = x.device
    batch_size, d = x.shape
    x0 = x.clone().detach()
    x_adv = x0.clone()
    success, q_to_success, q_per_sample = _new_state(x, max_queries, device)
    vel = torch.zeros(batch_size, d, device=device)
    total_queries = 0

    init_stateful_history(model, x_adv)
    while q_per_sample.min() < max_queries and not success.all():
        u = torch.randn(batch_size, d, device=device)
        u = u / u.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
        f_plus = _eval(model, x_adv + sigma * u, q_per_sample)
        f_minus = _eval(model, x_adv - sigma * u, q_per_sample)
        total_queries += 2 * batch_size
        grad_est = ((f_plus - f_minus) / (2 * sigma)).unsqueeze(1) * u
        with torch.no_grad():
            vel = momentum * vel + grad_est
            gn = vel.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
            x_adv = _project(x_adv + alpha * (vel / gn), x0, epsilon)
            if projector is not None:
                x_adv = projector(x_adv, x0)
        logits = _eval(model, x_adv, q_per_sample)
        total_queries += batch_size
        success, q_to_success = _check(success, q_to_success, q_per_sample, logits, max_queries)
        if q_per_sample.min() >= max_queries:
            break
    return x_adv.detach(), success, q_to_success, total_queries


def zoo_attack(model, x, config, projector=None, coords=64, seed=42):
    """ZOO (Chen et al. 2017): symmetric finite-difference probes on a random
    coordinate subset per step. Deterministic antithetic family along
    coordinates - the strongest non-Gaussian antithetic baseline."""
    epsilon = config.get("epsilon", 0.1)
    alpha = config.get("alpha", 0.02)
    max_queries = config.get("max_queries", 250)
    fd_h = config.get("zoo_fd_h", 1e-2)

    device = x.device
    batch_size, d = x.shape
    x0 = x.clone().detach()
    x_adv = x0.clone()
    success, q_to_success, q_per_sample = _new_state(x, max_queries, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    total_queries = 0

    init_stateful_history(model, x_adv)
    while q_per_sample.min() < max_queries and not success.all():
        grad_est = torch.zeros_like(x_adv)
        idx = torch.randperm(d, generator=gen)[:coords]
        for j in idx.tolist():
            e = torch.zeros_like(x_adv)
            e[:, j] = 1.0
            f_plus = _eval(model, x_adv + fd_h * e, q_per_sample)
            f_minus = _eval(model, x_adv - fd_h * e, q_per_sample)
            total_queries += 2 * batch_size
            grad_est += ((f_plus - f_minus) / (2 * fd_h)).unsqueeze(1) * e
            if q_per_sample.min() >= max_queries:
                break
        with torch.no_grad():
            gn = grad_est.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
            x_adv = _project(x_adv + alpha * (grad_est / gn), x0, epsilon)
            if projector is not None:
                x_adv = projector(x_adv, x0)
        logits = _eval(model, x_adv, q_per_sample)
        total_queries += batch_size
        success, q_to_success = _check(success, q_to_success, q_per_sample, logits, max_queries)
        if q_per_sample.min() >= max_queries:
            break
    return x_adv.detach(), success, q_to_success, total_queries


def simba_attack(model, x, config, projector=None, seed=42):
    """SimBA (Guo et al. 2019), decision-based: probe a random direction; if
    the margin improves, accept the step, else negate it. 1 query per probe,
    no score differencing, so no symmetric-pair structure."""
    epsilon = config.get("epsilon", 0.1)
    sigma_step = config.get("simba_step", 0.02)
    max_queries = config.get("max_queries", 250)

    device = x.device
    batch_size, d = x.shape
    x0 = x.clone().detach()
    x_adv = x0.clone()
    success, q_to_success, q_per_sample = _new_state(x, max_queries, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    total_queries = 0

    init_stateful_history(model, x_adv)
    # current best margin per sample (queried once)
    f_best = _eval(model, x_adv, q_per_sample)
    total_queries += batch_size
    while q_per_sample.min() < max_queries and not success.all():
        u = torch.randn(batch_size, d, device=device)
        u = u / u.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
        cand = _project(x_adv + sigma_step * u, x0, epsilon)
        f_cand = _eval(model, cand, q_per_sample)
        total_queries += batch_size
        accept = f_cand > f_best
        x_adv = torch.where(accept.unsqueeze(1), cand, x_adv)
        f_best = torch.where(accept, f_cand, f_best)
        # also try the negated direction (still 1 query, no score difference)
        cand2 = _project(x_adv - sigma_step * u, x0, epsilon)
        f_cand2 = _eval(model, cand2, q_per_sample)
        total_queries += batch_size
        accept2 = f_cand2 > f_best
        x_adv = torch.where(accept2.unsqueeze(1), cand2, x_adv)
        f_best = torch.where(accept2, f_cand2, f_best)
        if projector is not None:
            x_adv = projector(x_adv, x0)
        success, q_to_success = _check(success, q_to_success, q_per_sample, f_best, max_queries)
        if q_per_sample.min() >= max_queries:
            break
    return x_adv.detach(), success, q_to_success, total_queries


def square_attack(model, x, config, projector=None, seed=42):
    """Square Attack (Andriushchenko et al. 2020), score-free: random
    low-frequency block masks over feature blocks, sign-based updates,
    keep the candidate whenever the (queried) margin improves."""
    epsilon = config.get("epsilon", 0.1)
    max_queries = config.get("max_queries", 250)
    p_init = config.get("square_p", 0.05)

    device = x.device
    batch_size, d = x.shape
    x0 = x.clone().detach()
    x_adv = _project((x0 + epsilon).clamp(0.0, 1.0), x0, epsilon)
    success, q_to_success, q_per_sample = _new_state(x, max_queries, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    total_queries = 0

    init_stateful_history(model, x_adv)
    f_best = _eval(model, x_adv, q_per_sample)
    total_queries += batch_size
    p = p_init
    while q_per_sample.min() < max_queries and not success.all():
        mask = (torch.rand(x.shape, generator=gen, device=device) < p).float()
        # random sign per block-column (low-frequency structure)
        col_sign = torch.sign(torch.randn(d, generator=gen, device=device)).unsqueeze(0)
        delta = mask * col_sign * (2 * epsilon)
        cand = (x_adv + delta).clamp(0.0, 1.0)
        cand = _project(cand, x0, epsilon)
        f_cand = _eval(model, cand, q_per_sample)
        total_queries += batch_size
        accept = f_cand > f_best
        x_adv = torch.where(accept.unsqueeze(1), cand, x_adv)
        f_best = torch.where(accept, f_cand, f_best)
        success, q_to_success = _check(success, q_to_success, q_per_sample, f_best, max_queries)
        # square schedule: shrink p after every 10 evals
        if q_per_sample.max() % 25 == 0:
            p = max(p / 2, 1.0 / d)
        if q_per_sample.min() >= max_queries:
            break
    return x_adv.detach(), success, q_to_success, total_queries


def boundary_attack(model, x, config, projector=None, seed=42):
    """Boundary Attack (Brendel et al. 2018), label-only: start from an
    adversarial point, alternate orthogonal random-walk steps with step-size
    rejection. 1 query per probe, purely decision-based."""
    epsilon = config.get("epsilon", 0.1)
    max_queries = config.get("max_queries", 250)
    step = config.get("boundary_step", 0.05)

    device = x.device
    batch_size, d = x.shape
    x0 = x.clone().detach()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    total_queries = 0

    success, q_to_success, q_per_sample = _new_state(x, max_queries, device)
    init_stateful_history(model, x0)

    # initialization: walk outward along a random ray until adversarial
    x_adv = x0.clone()
    found = torch.zeros(batch_size, dtype=torch.bool, device=device)
    r = 0.1
    while r <= 4.0 and not found.all() and q_per_sample.min() < max_queries:
        u = torch.randn(batch_size, d, device=device)
        u = u / u.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
        cand = (x0 + r * u).clamp(0.0, 1.0)
        adv = _eval(model, cand, q_per_sample) > 0
        total_queries += batch_size
        newly = adv & ~found
        x_adv = torch.where(newly.unsqueeze(1), cand, x_adv)
        found |= newly
        r *= 2.0
    x_adv = torch.where(found.unsqueeze(1), x_adv, x0)
    success |= found

    while q_per_sample.min() < max_queries and not success.all():
        # orthogonal random walk toward x0
        toward = x0 - x_adv
        tnorm = toward.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
        u = torch.randn(batch_size, d, device=device, generator=gen)
        u = u / u.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
        u = u - (torch.sum(u * toward, dim=1, keepdim=True) / tnorm**2) * toward  # orthogonal
        u = u / u.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
        step_vec = step * tnorm
        # unidirectional trend toward the source: blend walk + approach
        cand = (x_adv + step_vec * (u + 0.5 * toward / tnorm)).clamp(0.0, 1.0)
        adv = _eval(model, cand, q_per_sample) > 0
        total_queries += batch_size
        ok = adv & ((cand - x0).norm(p=2, dim=1) < (x_adv - x0).norm(p=2, dim=1))
        x_adv = torch.where(ok.unsqueeze(1), cand, x_adv)
        # stay adversarial-or-flag: a sample counts as success once adversarial
        success = success | adv
        newly = adv & (q_to_success >= max_queries)
        q_to_success[newly] = q_per_sample[newly].float()
        if q_per_sample.min() >= max_queries:
            break
    return x_adv.detach(), success, q_to_success, total_queries


TRIGGER_SCOPE_ATTACKS = {
    "nes_independent": nes_independent_attack,
    "spsa": spsa_attack,
    "zoo": zoo_attack,
    "simba": simba_attack,
    "square": square_attack,
    "boundary": boundary_attack,
}
