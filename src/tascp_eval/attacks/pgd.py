import torch
import torch.nn as nn
import numpy as np
import yaml
import os

_DEFAULTS = {"epsilon": 0.1, "alpha": 0.01, "iterations": 20, "norm": "l2"}


def load_config(name="pgd"):
    """Load an attack config from ``configs/attack/<name>.yaml``.

    Missing keys fall back to the canonical white-box defaults above, so the
    recorded protocol always matches what the attacks actually use.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    config_path = os.path.join(root, "configs", "attack", f"{name}.yaml")
    config = dict(_DEFAULTS)
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            loaded = yaml.safe_load(f) or {}
        if "num_steps" in loaded and "iterations" not in loaded:
            loaded["iterations"] = loaded.pop("num_steps")
        config.update({k: v for k, v in loaded.items() if v is not None})
    return config

def pgd_attack(model, x, y, config):
    """
    White-box PGD attack against a binary classifier.
    Returns the full query trajectory of x_t.
    """
    epsilon = config.get('epsilon', 0.1)
    alpha_0 = config.get('alpha', 0.01)
    iterations = config.get('iterations', 20)
    norm = config.get('norm', 'l2')
    mu = config.get('mu', 0.0) # Momentum term for MIM
    step_decay = config.get('gamma', 1.0) # Step size decay factor
    
    x_adv = x.clone().detach().requires_grad_(True)
    trajectory = [x_adv.clone().detach()]
    
    criterion = nn.BCEWithLogitsLoss()
    momentum_grad = torch.zeros_like(x_adv)
    
    for i in range(iterations):
        alpha = alpha_0 * (step_decay ** i)
        
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        
        # We want to maximize the loss of the true label
        loss = criterion(outputs, y)
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad = x_adv.grad
            
            # MIM accumulation
            if mu > 0.0:
                # Standard MIM normalizes the gradient by L1 norm before accumulation
                grad_l1 = torch.norm(grad, p=1, dim=1, keepdim=True)
                grad_l1 = torch.clamp(grad_l1, min=1e-8)
                momentum_grad = mu * momentum_grad + grad / grad_l1
                update_grad = momentum_grad
            else:
                update_grad = grad
                
            if norm == 'linf':
                x_adv = x_adv + alpha * update_grad.sign()
                eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
                x_adv = x + eta
            elif norm == 'l2':
                grad_norm = torch.norm(update_grad, p=2, dim=1, keepdim=True)
                grad_norm = torch.clamp(grad_norm, min=1e-8)
                x_adv = x_adv + alpha * (update_grad / grad_norm)
                
                eta = x_adv - x
                eta_norm = torch.norm(eta, p=2, dim=1, keepdim=True)
                eta_norm = torch.clamp(eta_norm, min=1e-8)
                factor = torch.min(torch.ones_like(eta_norm), epsilon / eta_norm)
                x_adv = x + eta * factor
            else:
                raise ValueError(f"Unknown norm: {norm}")
                
            x_adv = torch.clamp(x_adv, 0.0, 1.0) # Feature vectors were MinMax scaled
            trajectory.append(x_adv.clone().detach())
            
    return trajectory
