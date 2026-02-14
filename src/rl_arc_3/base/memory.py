from typing import Tuple, Any

import torch

class BaseMemory:
    def push(self, transition: Tuple[torch.Tensor]):
        raise ValueError("Abstract method")

    def sample(self, batch_size) -> Tuple[torch.Tensor]:
        raise ValueError("Abstract method")

    def __len__(self) -> int:
        raise ValueError("Abstract method")

