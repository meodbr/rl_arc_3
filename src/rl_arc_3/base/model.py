from dataclasses import dataclass

import torch.nn as nn

from rl_arc_3.base.clone import Checkpointable

@dataclass
class ModelSignature:
    input_shape: list[int]
    output_shape: list[int]

class BaseModel(nn.Module, Checkpointable):
    @property
    def signature(self) -> ModelSignature:
        raise NotImplementedError
    
    def state_dict(self) -> dict:
        return Checkpointable.state_dict(self)
    
    def load_state_dict(self, state: dict):
        return Checkpointable.load_state_dict(self, state)
        