import torch
import torch.nn as nn
import numpy as np
import math

def _apply_rotation(x_t, v_t, v_prev, delta_theta, ablation_mode=None):
    """
    Batched implementation of TASCP Givens rotation in PyTorch.
    x_t, v_t, v_prev: (batch_size, num_features)
    delta_theta: (batch_size, 1)
    """
    if ablation_mode == 'basis_coupling':
        # Override with fixed, data-independent basis vectors
        v_t = torch.zeros_like(x_t)
        v_t[:, 0] = 1.0
        v_prev = torch.zeros_like(x_t)
        v_prev[:, 1] = 1.0

    norm_t = torch.norm(v_t, p=2, dim=1, keepdim=True)
    norm_t_clamped = torch.clamp(norm_t, min=1e-9)
    u = v_t / norm_t_clamped
    
    comp_prev_u = torch.sum(v_prev * u, dim=1, keepdim=True)
    proj = comp_prev_u * u
    
    w = v_prev - proj
    norm_w = torch.norm(w, p=2, dim=1, keepdim=True)
    norm_w_clamped = torch.clamp(norm_w, min=1e-9)
    
    v_t_perp = w / norm_w_clamped
    
    valid_mask = (norm_w > 1e-9) & (norm_t > 1e-9)
    
    comp_x_u = torch.sum(x_t * u, dim=1, keepdim=True)
    comp_x_perp = torch.sum(x_t * v_t_perp, dim=1, keepdim=True)
    
    orthogonal = x_t - (comp_x_u * u) - (comp_x_perp * v_t_perp)
    
    cos_theta = torch.cos(delta_theta)
    sin_theta = torch.sin(delta_theta)
    
    rotated_u = comp_x_u * cos_theta - comp_x_perp * sin_theta
    rotated_perp = comp_x_u * sin_theta + comp_x_perp * cos_theta
    
    rotated_candidate = orthogonal + rotated_u * u + rotated_perp * v_t_perp
    
    rotated = torch.where(valid_mask, rotated_candidate, x_t)
    return rotated, u, v_t_perp

class TASCPWrapper(nn.Module):
    def __init__(self, base_model, window_size=5, max_theta=math.pi/2, tau=0.8, gamma=1.0):
        super().__init__()
        self.base_model = base_model
        self.window_size = window_size
        self.max_theta = max_theta
        self.tau = tau
        self.gamma = gamma
        self.history = None
        self.adaptive = False
        self.ablation_mode = None
        
    def reset_history(self, batch_size, device):
        # We'll store history as a list of tensors of shape (batch_size, num_features)
        self.history = []
        
    def add_to_history(self, x):
        """Add the executed step to history. Called manually during PGD."""
        self.history.append(x.detach().clone())
        if len(self.history) > self.window_size + 1:
            self.history.pop(0)

    def forward(self, x_t, apply_rotation=True):
        """
        Forward pass for a batched sequence.
        Since we need the active history, we rely on the external loop to call add_to_history.
        """
        if not apply_rotation or len(self.history) < 2:
            # If no history, cannot compute momentum or apply rotation
            return self.base_model(x_t)
            
        # compute v_i and M_i from history
        # history has [x_{t-k}, ..., x_{t-1}]
        # Actually x_t is the CURRENT step being evaluated.
        # We need to compute v_t = x_t - x_{t-1}
        # But wait, in TASCP, the rotation is parameterized by the history BEFORE x_t?
        # The paper says Delta_theta_t = gamma * max(0, eps_p - tau).
        # eps_p is computed over the window W.
        # If x_t is the query, the defender observes x_t.
        # The defender computes v_t = x_t - x_{t-1}.
        # Then M_t = cos(v_t, v_{t-1}).
        # Then it adds (v_t, M_t) to the window W to compute eps_p.
        # Then it computes Delta_theta_t.
        # And finally applies the rotation R(Delta_theta_t) to x_t itself!
        
        x_prev = self.history[-1]
        v_t = x_t - x_prev
        
        if len(self.history) < 2:
            return self.base_model(x_t)
            
        x_prev_prev = self.history[-2]
        v_prev = x_prev - x_prev_prev
        
        if not getattr(self, 'adaptive', False):
            v_t = v_t.detach()
            v_prev = v_prev.detach()
        
        # Compute M_t
        norm_t = torch.norm(v_t, p=2, dim=1, keepdim=True)
        norm_prev = torch.norm(v_prev, p=2, dim=1, keepdim=True)
        
        valid = (norm_t > 1e-9) & (norm_prev > 1e-9)
        dot_product = torch.sum(v_t * v_prev, dim=1, keepdim=True)
        denom = torch.clamp(norm_t * norm_prev, min=1e-9)
        m_t = torch.where(valid, dot_product / denom, torch.zeros_like(norm_t))
        
        # Build window to compute eps_p
        window_v = []
        window_m = []
        
        for i in range(1, len(self.history)):
            curr_x = self.history[i]
            prev_x = self.history[i-1]
            v_i = curr_x - prev_x
            if i > 1:
                prev_prev_x = self.history[i-2]
                v_i_prev = prev_x - prev_prev_x
                
                n_i = torch.norm(v_i, p=2, dim=1, keepdim=True)
                n_i_prev = torch.norm(v_i_prev, p=2, dim=1, keepdim=True)
                val = (n_i > 1e-9) & (n_i_prev > 1e-9)
                
                dot_i = torch.sum(v_i * v_i_prev, dim=1, keepdim=True)
                denom_i = torch.clamp(n_i * n_i_prev, min=1e-9)
                m_i = torch.where(val, dot_i / denom_i, torch.zeros_like(n_i))
            else:
                m_i = torch.zeros_like(norm_t)
                
            window_v.append(v_i)
            window_m.append(m_i)
            
        # Add current step
        window_v.append(v_t)
        window_m.append(m_t)
        
        eps_p = torch.zeros_like(norm_t)
        for v_i, m_i in zip(window_v, window_m):
            eps_p += torch.norm(v_i, p=2, dim=1, keepdim=True) * torch.clamp(m_i, min=0.0)
        eps_p /= len(window_v)
        
        delta_theta = self.gamma * torch.clamp(eps_p - self.tau, min=0.0)
        delta_theta = torch.clamp(delta_theta, max=self.max_theta)
        
        if not getattr(self, 'adaptive', False) or self.ablation_mode == 'gradient_magnitude':
            delta_theta = delta_theta.detach()
            
        if getattr(self, 'adaptive', False) and self.ablation_mode == 'projection_only':
            delta_theta = torch.round(delta_theta / 0.2) * 0.2
        
        x_rot, u, v_perp = _apply_rotation(x_t, v_t, v_prev, delta_theta, ablation_mode=self.ablation_mode)
        
        self.last_delta_theta = delta_theta
        self.last_u = u
        self.last_v_perp = v_perp
        self.last_x_rot = x_rot
        
        return self.base_model(x_rot)
