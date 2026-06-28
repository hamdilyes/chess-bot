"""The policy network: a small AlphaZero-style residual CNN.

Input  : (B, 17, 8, 8) board planes.
Output : (B, NUM_MOVES) logits over the fixed move vocabulary.

Policy-only by design. The blunder check uses a material search, not a learned value head,
so there is no value head here and training stays a clean single-objective classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .encoding import NUM_MOVES, NUM_PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


class PolicyNet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_planes, cfg.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(cfg.channels) for _ in range(cfg.blocks)])
        # Policy head: 1x1 conv down to a few planes, then a linear map to the vocabulary.
        self.policy_conv = nn.Sequential(
            nn.Conv2d(cfg.channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(32 * 8 * 8, NUM_MOVES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.policy_conv(x)
        x = x.flatten(1)
        return self.policy_fc(x)


def build_model(cfg: ModelConfig | None = None) -> PolicyNet:
    assert NUM_PLANES == (cfg or ModelConfig()).in_planes, "plane count mismatch"
    return PolicyNet(cfg)
