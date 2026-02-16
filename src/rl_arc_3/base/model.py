from dataclasses import dataclass

import torch
import torch.nn as nn

from rl_arc_3.base.clone import Checkpointable

@dataclass
class ModelSignature:
    input_shape: list[int]
    output_shape: list[int]

class BaseModel(nn.Module, Checkpointable):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        Checkpointable.__init__(self, *args, **kwargs)

    @property
    def signature(self) -> ModelSignature:
        raise NotImplementedError
    
    def state_dict(self, *args, **kwargs) -> dict:
        self.ensure_checkpointable()
        state = super().state_dict(*args, **kwargs)
        for k, v in state.items():
            if torch.is_tensor(v):
                v = v.clone().detach()
        for k in ["_init_args", "_init_kwargs"]:
            state[k] = getattr(self, k, None)
        state["class"] = self.__class__
        return state
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        keys = ["_init_args", "_init_kwargs", "class"]
        state = {k:v for k, v in state_dict.items() if k not in keys}
        return super().load_state_dict(state, *args, **kwargs)