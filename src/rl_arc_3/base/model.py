from dataclasses import dataclass

import torch.nn as nn
from gymnasium.spaces import Space

from rl_arc_3.base.clone import Checkpointable

class ModelSignature:
    input_shape: list[int]
    output_shape: list[int]

class BaseModel(nn.Module, Checkpointable):
    @property
    def signature(self) -> ModelSignature:
        raise NotImplementedError
