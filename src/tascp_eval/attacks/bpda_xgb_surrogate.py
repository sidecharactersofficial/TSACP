import torch, torch.nn as nn
import numpy as np

def xgb_stateful_pgd_attack(tascp_model, xgb_model, surrogate_model, x, y, config, adaptive=False):
    """
    BPDA-style attack: gradients flow through `surrogate_model` (wrapped in
    `tascp_model` for rotation), but success is evaluated against the real
    `xgb_model`.  Query counting uses the XGBoost decision.
    """
    epsilon = config.get('epsilon', 0.1)
    alpha_0 = config.get('alpha', 0.01)
    iterations = config.get('iterations', 20)
    norm = config.get('norm', 'l2')

    if hasattr(tascp_model, 'adaptive'):
        tascp_model.adaptive = adaptive

    criterion = nn.BCEWithLogitsLoss()
    x_adv = x.clone().detach().requires_grad_(True)
    batch_size = x.size(0)
    success_flags = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
    queries_to_success = torch.full((batch_size,), iterations, dtype=torch.float32, device=x.device)

    if hasattr(tascp_model, 'reset_history'):
        tascp_model.reset_history(batch_size=batch_size, device=x.device)
        tascp_model.add_to_history(x.clone().detach())

    for i in range(iterations):
        x_adv.requires_grad_(True)
        outputs = tascp_model(x_adv)

        # Check success against REAL XGBoost on the rotated feature
        if hasattr(tascp_model, 'last_x_rot') and tascp_model.last_x_rot is not None:
            x_rot = tascp_model.last_x_rot.detach()
        else:
            x_rot = x_adv.detach()

        xgb_np = x_rot.cpu().numpy()
        xgb_proba = xgb_model.predict_proba(xgb_np)[:, 1]
        xgb_success = torch.from_numpy((xgb_proba > 0.5).astype(np.float32)).to(x.device)

        newly_successful = (xgb_success > 0.5).squeeze() & ~success_flags
        if newly_successful.any():
            queries_to_success[newly_successful] = i + 1
            success_flags = success_flags | newly_successful

        loss = criterion(outputs, y)
        tascp_model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = x_adv.grad
            if hasattr(tascp_model, 'add_to_history'):
                tascp_model.add_to_history(x_adv.clone().detach())

            if norm == 'linf':
                x_adv = x_adv + alpha_0 * grad.sign()
                eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
                x_adv = x + eta
            elif norm == 'l2':
                grad_norm = torch.norm(grad, p=2, dim=1, keepdim=True)
                grad_norm = torch.clamp(grad_norm, min=1e-8)
                x_adv = x_adv + alpha_0 * (grad / grad_norm)
                eta = x_adv - x
                eta_norm = torch.norm(eta, p=2, dim=1, keepdim=True)
                eta_norm = torch.clamp(eta_norm, min=1e-8)
                factor = torch.min(torch.ones_like(eta_norm), epsilon / eta_norm)
                x_adv = x + eta * factor
            else:
                raise ValueError(f"Unknown norm: {norm}")
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # Final check against XGBoost
    if hasattr(tascp_model, 'last_x_rot') and tascp_model.last_x_rot is not None:
        x_rot_final = tascp_model.last_x_rot.detach()
    else:
        x_rot_final = x_adv.detach()
    xgb_np = x_rot_final.cpu().numpy()
    xgb_proba = xgb_model.predict_proba(xgb_np)[:, 1]
    xgb_success = torch.from_numpy((xgb_proba > 0.5).astype(np.float32)).to(x.device)
    newly = (xgb_success > 0.5).squeeze() & ~success_flags
    if newly.any():
        queries_to_success[newly] = iterations
        success_flags = success_flags | newly

    return x_adv, success_flags, queries_to_success
