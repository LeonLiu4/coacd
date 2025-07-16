from typing import Dict, Any
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PointNetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,          # size of the output vector handed to SB3
        **kwargs: Dict[str, Any],         # keep kwargs so YAML/Hydra configs don’t break
    ):
        super().__init__(observation_space, features_dim)

        # ---- backbone ----
        self.backbone = nn.Sequential(
            nn.Linear(3, 64),  nn.ReLU(),
            nn.Linear(64,128), nn.ReLU(),
            nn.Linear(128,256), nn.ReLU(),
            nn.Linear(256,features_dim), nn.ReLU(),  # (B,N,F)
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # obs comes in as (B, N, 3) if you followed Gymnasium’s convention
        feat_per_point = self.backbone(obs)     # (B,N,F)
        global_feat    = feat_per_point.max(dim=1).values   # (B,F)
        return global_feat                       # SB3 routes this to the actor & critic