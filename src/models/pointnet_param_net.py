import torch.nn as nn
import torch


class PointNetParamNet(nn.Module):
    """PointNet encoder + shallow MLP ⇒ 3‑D action in [-1,1]."""
    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, feat_dim), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 3), nn.Tanh()
        )

    def forward(self, pts: torch.Tensor):
        x = self.encoder(pts)         # (B,N,F)
        x = x.max(dim=1).values       # (B,F) global feature
        return self.head(x)           # (B,3)
