from typing import Tuple, Any

import torch

from rl_arc_3.base.clone import Checkpointable

class BaseMemory(Checkpointable):
    def push(self, transition: Tuple[torch.Tensor]):
        raise ValueError("Abstract method")

    def sample(self, batch_size) -> Tuple[torch.Tensor]:
        raise ValueError("Abstract method")

    def __len__(self) -> int:
        raise ValueError("Abstract method")

