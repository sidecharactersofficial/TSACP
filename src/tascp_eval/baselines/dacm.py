import torch
import torch.nn as nn

class DACMWrapper(nn.Module):
    """
    Static baseline: DACM (Dynamic Adversarial Constraint Manifold).
    Since the dataset is MinMax scaled, the static manifold is essentially
    the hypercube [0,1]^D. We represent the static projection as a clamp.
    Because it is static and differentiable almost everywhere, standard PGD
    can backpropagate straight through it to compute the exact projection bypass.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        x_proj = torch.clamp(x, 0.0, 1.0)
        return self.base_model(x_proj)
