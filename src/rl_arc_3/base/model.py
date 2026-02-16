from dataclasses import dataclass

import torch
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
        state = super().state_dict()
        for k, v in state.items():
            if torch.is_tensor(v):
                v = v.clone().detach()
        for k in ["_init_args", "_init_kwargs"]:
            state[k] = getattr(self, k, "bad")
        return state
    
    def load_state_dict(self, state_dict, strict = True, assign = False):
        state = state_dict.copy()
        del state["_init_args"]
        del state["_init_kwargs"]
        return super().load_state_dict(state, strict, assign)