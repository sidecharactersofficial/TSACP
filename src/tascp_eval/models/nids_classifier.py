import torch
import torch.nn as nn

class NIDSClassifier(nn.Module):
    def __init__(self, input_dim=43, hidden_dim=64):
        super(NIDSClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Binary classification (logits)
        )

    def forward(self, x):
        return self.net(x)
