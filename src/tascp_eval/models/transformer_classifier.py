import torch
import torch.nn as nn
import math

class ContinuousEmbedding(nn.Module):
    def __init__(self, num_features, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, d_model) / math.sqrt(d_model))
        self.bias = nn.Parameter(torch.zeros(num_features, d_model))

    def forward(self, x):
        # x is (B, num_features)
        # weight is (num_features, d_model)
        # output is (B, num_features, d_model)
        # x.unsqueeze(-1) is (B, num_features, 1)
        return x.unsqueeze(-1) * self.weight + self.bias

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=43, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embedding = ContinuousEmbedding(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x: (B, input_dim)
        B = x.size(0)
        x_emb = self.embedding(x) # (B, input_dim, d_model)
        
        # add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, d_model)
        x_emb = torch.cat((cls_tokens, x_emb), dim=1) # (B, input_dim + 1, d_model)
        
        # transformer
        out = self.transformer(x_emb) # (B, input_dim + 1, d_model)
        
        # take cls token out
        cls_out = out[:, 0, :] # (B, d_model)
        
        return self.mlp_head(cls_out)

