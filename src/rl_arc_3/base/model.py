from dataclasses import dataclass

import torch.nn as nn
from gymnasium.spaces import Space

class ModelSignature:
    input_shape: list[int]
    output_shape: list[int]

class BaseModel(nn.Module):
    @property
    def signature(self) -> ModelSignature:
        raise NotImplementedError

    def clone(self, device=None):
        """
        Return a full clone of this module with fresh memory for all parameters and buffers.
        Optional: move to `device`.
        """
        # Ensure sentinel attr is present
        if not hasattr(self, "is_clonable") or not self.is_clonable:
            raise NotImplementedError(f"Model not clonable ({type(self)}) : define the _is_clonable attr or override clone method.")

        # Recreate module via constructor args if they exist
        if hasattr(self, "_init_args") and hasattr(self, "_init_kwargs"):
            new_model = type(self)(*self._init_args, **self._init_kwargs)
        else:
            # Fallback: create empty instance and copy state_dict
            new_model = type(self)()  

        # Clone parameters and buffers
        new_state = {}
        for k, v in self.state_dict().items():
            cloned = v.clone().detach()
            if device is not None:
                cloned = cloned.to(device)
            new_state[k] = cloned
        new_model.load_state_dict(new_state)

        return new_model