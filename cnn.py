from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class ChessCNN(nn.Module):
    def __init__(self, channels: int, action_space: int):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=128)

        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=128)

        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)

        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=128)

        self.p_conv = nn.Conv2d(128, 32, 1)
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc = nn.Linear(32 * 8 * 8, action_space)

        self.v_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.v_bn = nn.BatchNorm2d(32)
        self.v_fc1 = nn.Linear(32 * 8 * 8, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.view(B, -1)
        policy_logits = self.p_fc(p)

        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(B, -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))

        return policy_logits, v
