"""Adaptive white-box attack variants that do NOT rely on straight-through
autograd through the stateful defense (closes the "straight-through autograd
only" limitation of exp5).

Variants
--------
EOTStatefulPGD  : Expectation-Over-Transformation (Athalye et al., 2018)
    adaptive BPDA. The attacker has white-box access to the deployed model
    f and *replicates* the deterministic defense D(x; h) locally (threat
    model B2), then optimizes E_{T~U}[f(T(D(x; h)))] where T is the
    deployment's input-transform distribution (here: random instance noise
    and feature dropout, the standard EOT robustness set for tabular NIDS).
    Gradients are taken through the *replicated* defense, so no
    straight-through access to the deployed wrapper is required.

FiniteDifferenceStatefulPGD : zeroth-order adaptive BPDA.
    The attacker never backpropagates through the defense at all: the
    gradient of the defended margin g(x) = f(D(x; h(x))) is estimated with
    central finite differences along random directions (Flaxman et al.,
    2005 style), using only model *outputs*. This is the conservative
    lower bound on an adaptive adversary's capability.

Both attackers drive the deployed wrapper's history exactly like
`stateful_pgd_attack` (reset_history / add_to_history), so stateful defenses
observe the true per-session query sequence.
"""

import torch
import torch.nn as nn

from .pgd import load_config


def _replicate_defense(x_adv, x_prev, x_prev_prev, tau, gamma, max_theta):
    """Local replica of TASCPWrapper.forward's state map (threat model B2:
    the defense is deterministic and its form is known). Returns the
    differentiably rotated input the deployed classifier would see."""
    v_t = x_adv - x_prev
    v_prev = x_prev - x_prev_prev

    norm_t = torch.norm(v_t, p=2, dim=1, keepdim=True)
    norm_prev = torch.norm(v_prev, p=2, dim=1, keepdim=True)
    valid = (norm_t > 1e-9) & (norm_prev > 1e-9)
    dot = torch.sum(v_t * v_prev, dim=1, keepdim=True)
    denom = torch.clamp(norm_t * norm_prev, min=1e-9)
    m_t = torch.where(valid, dot / denom, torch.zeros_like(norm_t))

    # windowed epsilon_p over the replicated window {v_prev_prev..v_t}
    eps_p = torch.zeros_like(norm_t)
    n_terms = 0.0
    vv_prev = v_prev
    vv_prev_prev = x_prev - x_prev_prev
    n_p = torch.norm(vv_prev, p=2, dim=1, keepdim=True)
    n_pp = torch.norm(vv_prev_prev, p=2, dim=1, keepdim=True)
    dot_p = torch.sum(vv_prev * vv_prev_prev, dim=1, keepdim=True)
    m_p = torch.where(
        (n_p > 1e-9) & (n_pp > 1e-9), dot_p / torch.clamp(n_p * n_pp, min=1e-9), torch.zeros_like(n_p)
    )
    eps_p = eps_p + n_p * torch.clamp(m_p, min=0.0)
    n_terms += 1.0
    eps_p = eps_p + norm_t * torch.clamp(m_t, min=0.0)
    n_terms += 1.0
    eps_p = eps_p / n_terms

    delta_theta = gamma * torch.clamp(eps_p - tau, min=0.0)
    delta_theta = torch.clamp(delta_theta, max=max_theta)

    # rotation plane: u = v_t/||v_t||, v_perp = orthogonalized v_prev
    u = v_t / torch.clamp(norm_t, min=1e-9)
    comp = torch.sum(v_prev * u, dim=1, keepdim=True)
    w = v_prev - comp * u
    norm_w = torch.norm(w, p=2, dim=1, keepdim=True)
    v_perp = w / torch.clamp(norm_w, min=1e-9)
    valid_mask = (norm_w > 1e-9) & (norm_t > 1e-9)

    comp_u = torch.sum(x_adv * u, dim=1, keepdim=True)
    comp_perp = torch.sum(x_adv * v_perp, dim=1, keepdim=True)
    ortho = x_adv - comp_u * u - comp_perp * v_perp
    c, s = torch.cos(delta_theta), torch.sin(delta_theta)
    rot_u = comp_u * c - comp_perp * s
    rot_perp = comp_u * s + comp_perp * c
    x_rot = ortho + rot_u * u + rot_perp * v_perp
    return torch.where(valid_mask, x_rot, x_adv), delta_theta


def _eot_transform(x, noise_sigma, dropout_p, generator):
    """Random tabular EOT transformation: instance noise + feature dropout."""
    b, d = x.shape
    noise = torch.randn(x.shape, generator=generator, device=x.device) * noise_sigma
    mask = (torch.rand(x.shape, generator=generator, device=x.device) > dropout_p).float()
    return (x + noise) * mask


def eot_stateful_pgd_attack(model, x, y, config, adaptive=True,
                            eot_samples=8, noise_sigma=0.01, dropout_p=0.05,
                            replicated_defense=True, tau=0.0, gamma=50.0,
                            max_theta=None):
    """EOT adaptive BPDA against a stateful defense.

    adaptive=True  -> gradient through the *local replica* of the defense
                      (no straight-through autograd on the deployed wrapper).
    replicated_defense=False -> plain EOT on the deployed wrapper directly
                      (straight-through fallback, for comparison).
    """
    import math

    epsilon = config.get("epsilon", 0.1)
    alpha_0 = config.get("alpha", 0.01)
    iterations = config.get("iterations", 20)
    norm = config.get("norm", "l2")
    if max_theta is None:
        max_theta = math.pi / 2

    if hasattr(model, "adaptive"):
        model.adaptive = adaptive

    criterion = nn.BCEWithLogitsLoss()
    device = x.device
    batch_size = x.size(0)
    x_adv = x.clone().detach()
    success_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
    queries_to_success = torch.full((batch_size,), float(iterations), device=device)

    if hasattr(model, "reset_history"):
        model.reset_history(batch_size=batch_size, device=device)
        model.add_to_history(x.clone().detach())

    # keep the last two *distinct* queries to replicate the state locally
    hist = [x.clone().detach(), x.clone().detach()]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(torch.initial_seed()) % (2**31))

    for i in range(iterations):
        x_adv.requires_grad_(True)

        if replicated_defense:
            # local replica forward + EOT over the deployed classifier
            x_rot, _ = _replicate_defense(x_adv, hist[-1], hist[-2], tau, gamma, max_theta)
            loss = torch.zeros(1, device=device)
            for _ in range(eot_samples):
                xt = _eot_transform(x_rot, noise_sigma, dropout_p, generator)
                xt = xt.clamp(0.0, 1.0)
                out = model.base_model(xt) if hasattr(model, "base_model") else model(xt)
                loss = loss + criterion(out, y)
            loss = loss / eot_samples
            if x_rot.requires_grad:
                grad_rot = torch.autograd.grad(loss, x_rot, retain_graph=False, allow_unused=True)[0]
            else:
                grad_rot = None
            grad = grad_rot if grad_rot is not None else torch.zeros_like(x_adv)
        else:
            out = model(x_adv)
            loss = criterion(out, y)
            model.zero_grad()
            loss.backward()
            grad = x_adv.grad

        with torch.no_grad():
            # query the deployed defense (this is the defender-observed query)
            with torch.no_grad():
                outputs = model(x_adv)
            preds = (outputs > 0).squeeze()
            newly = preds & ~success_flags
            if newly.any():
                queries_to_success[newly] = i + 1
                success_flags |= newly

            if hasattr(model, "add_to_history"):
                model.add_to_history(x_adv.clone().detach())
            hist.append(x_adv.clone().detach())
            hist = hist[-3:]

            if norm == "l2":
                g = grad
                gnorm = torch.clamp(torch.norm(g, p=2, dim=1, keepdim=True), min=1e-8)
                x_adv = x_adv + alpha_0 * (g / gnorm)
                eta = x_adv - x
                en = torch.clamp(torch.norm(eta, p=2, dim=1, keepdim=True), min=1e-8)
                x_adv = x + eta * torch.min(torch.ones_like(en), epsilon / en)
            else:
                x_adv = x_adv + alpha_0 * grad.sign()
                eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
                x_adv = x + eta
            x_adv = x_adv.clamp(0.0, 1.0)

    # final check
    with torch.no_grad():
        outputs = model(x_adv)
        preds = (outputs > 0).squeeze()
        newly = preds & ~success_flags
        if newly.any():
            queries_to_success[newly] = iterations
            success_flags |= newly

    return x_adv.detach(), success_flags, queries_to_success


def finite_difference_stateful_pgd_attack(model, x, y, config, adaptive=True,
                                          fd_samples=16, fd_sigma=1e-3, seed=42):
    """Zeroth-order adaptive BPDA: estimate the gradient of the defended margin
    with central finite differences along random unit directions.

    Uses only model outputs (no autograd at all), so it works for ANY
    deterministic stateful defense, differentiable or not.
    """
    epsilon = config.get("epsilon", 0.1)
    alpha_0 = config.get("alpha", 0.01)
    iterations = config.get("iterations", 20)
    norm = config.get("norm", "l2")

    if hasattr(model, "adaptive"):
        model.adaptive = adaptive

    device = x.device
    batch_size = x.size(0)
    x_adv = x.clone().detach()
    success_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
    queries_to_success = torch.full((batch_size,), float(iterations), device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    if hasattr(model, "reset_history"):
        model.reset_history(batch_size=batch_size, device=device)
        model.add_to_history(x.clone().detach())

    def margin(xq):
        with torch.no_grad():
            return model(xq).squeeze(-1)

    for i in range(iterations):
        # --- query the current point (defender sees it)
        out = margin(x_adv)
        preds = out > 0
        newly = preds & ~success_flags
        if newly.any():
            queries_to_success[newly] = i + 1
            success_flags |= newly
        if hasattr(model, "add_to_history"):
            model.add_to_history(x_adv.clone().detach())

        # --- central FD gradient estimate: 2 * fd_samples queries per step
        grad_est = torch.zeros_like(x_adv)
        for _ in range(fd_samples):
            u = torch.randn(x.shape, generator=gen, device=device)
            u = u / torch.clamp(u.norm(p=2, dim=1, keepdim=True), min=1e-9)
            f_plus = margin(x_adv + fd_sigma * u)
            f_minus = margin(x_adv - fd_sigma * u)
            if hasattr(model, "add_to_history"):
                model.add_to_history((x_adv + fd_sigma * u).detach())
                model.add_to_history((x_adv - fd_sigma * u).detach())
            grad_est += ((f_plus - f_minus) / (2 * fd_sigma)).unsqueeze(1) * u
        grad_est = grad_est / fd_samples

        with torch.no_grad():
            if norm == "l2":
                gnorm = torch.clamp(grad_est.norm(p=2, dim=1, keepdim=True), min=1e-9)
                x_adv = x_adv + alpha_0 * (grad_est / gnorm)
                eta = x_adv - x
                en = torch.clamp(eta.norm(p=2, dim=1, keepdim=True), min=1e-8)
                x_adv = x + eta * torch.min(torch.ones_like(en), epsilon / en)
            else:
                x_adv = x_adv + alpha_0 * grad_est.sign()
                eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
                x_adv = x + eta
            x_adv = x_adv.clamp(0.0, 1.0)

    with torch.no_grad():
        preds = margin(x_adv) > 0
        newly = preds & ~success_flags
        if newly.any():
            queries_to_success[newly] = iterations
            success_flags |= newly

    return x_adv.detach(), success_flags, queries_to_success
