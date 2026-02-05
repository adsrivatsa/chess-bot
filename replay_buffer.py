import random
from collections import deque
from typing import List, Tuple

import torch

from device import device


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def save_game(self, game_history: List[Tuple[torch.Tensor, torch.Tensor, float]]):
        """
        game_history: List of (state, pi, z) tuples from ONE game.
        """
        for sample in game_history:
            self.buffer.append(sample)

    def sample_batch(
        self, batch_size
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)

        states, policies, values = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(policies, dtype=torch.float32, device=device),
            torch.tensor(values, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)
