import torch
import torch.nn as nn
from .pgd import load_config

def stateful_pgd_attack(model, x, y, config, adaptive=False):
    """
    An iterative PGD attack designed to interface with stateful defenses.
    If adaptive=True and the model supports it, the attack computes the
    UNDETACHED gradient through the defense's dynamic generation logic.
    """
    epsilon = config.get('epsilon', 0.1)
    alpha_0 = config.get('alpha', 0.01)
    iterations = config.get('iterations', 20)
    norm = config.get('norm', 'l2')
    
    # We want to measure the queries-to-success
    # Attack success is when the output flips from benign (0) to attack (1)
    
    if hasattr(model, 'adaptive'):
        model.adaptive = adaptive
        
    criterion = nn.BCEWithLogitsLoss()
    
    x_adv = x.clone().detach().requires_grad_(True)
    
    batch_size = x.size(0)
    success_flags = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
    queries_to_success = torch.full((batch_size,), iterations, dtype=torch.float32, device=x.device)
    
    if hasattr(model, 'reset_history'):
        model.reset_history(batch_size=batch_size, device=x.device)
        model.add_to_history(x.clone().detach())
        
    for i in range(iterations):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        
        # Check success: outputs > 0 means predicted as 1 (attack). y_target is 1.
        preds = (outputs > 0).squeeze()
        newly_successful = preds & ~success_flags
        
        # We assign the query count to newly successful ones. Query count is i+1.
        if newly_successful.any():
            queries_to_success[newly_successful] = i + 1
            success_flags = success_flags | newly_successful
            
        loss = criterion(outputs, y)
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad = x_adv.grad
            
            # Record the current query to the model's history (the defender observed it)
            if hasattr(model, 'add_to_history'):
                model.add_to_history(x_adv.clone().detach())
                
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
                
    # Final check
    with torch.no_grad():
        outputs = model(x_adv)
        preds = (outputs > 0).squeeze()
        newly_successful = preds & ~success_flags
        if newly_successful.any():
            queries_to_success[newly_successful] = iterations
            success_flags = success_flags | newly_successful
            
    return x_adv, success_flags, queries_to_success
