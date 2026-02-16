import copy
import torch

class Checkpointable:
    def __init__(self, *args, **kwargs):
        self._checkpointable = True
        self._init_args = args
        self._init_kwargs = kwargs

    def state_dict(self):
        if not hasattr(self, "_checkpointable") or not self._checkpointable:
            raise RuntimeError("Object is not checkpointable, missing _checkpointable attribute or set to False")
        state = {
            "class": self.__class__, 
            "_init_args": self._init_args,
            "_init_kwargs": self._init_kwargs
        }
        for k, v in self.__dict__.items():
            if k.startswith("_init_") or k == "_checkpointable":
                continue
            if hasattr(v, "state_dict"):
                state[k] = v.state_dict()
            elif torch.is_tensor(v):
                state[k] = v.detach().clone().to("cpu")
            else:
                state[k] = copy.deepcopy(v)
        return state

    def load_state_dict(self, state):
        if not hasattr(self, "_checkpointable") or not self._checkpointable:
            raise RuntimeError("Object is not checkpointable, missing _checkpointable attribute or set to False")

        for k, v in state.items():
            # Ensure class and init args/kwargs match before loading state
            if k == "class" and v != self.__class__:
                raise RuntimeError(f"Cannot load state dict, class mismatch, current: {self.__class__}, incoming: {v}")
            if k == "_init_args" and v != self._init_args:
                raise RuntimeError(f"Cannot load state dict, init args mismatch, current: {self._init_args}, incoming: {v}")
            if k == "_init_kwargs" and v != self._init_kwargs:
                raise RuntimeError(f"Cannot load state dict, init kwargs mismatch, current: {self._init_kwargs}, incoming: {v}")

            # Load state for other attributes
            current = getattr(self, k, None)
            if hasattr(current, "load_state_dict"):
                current.load_state_dict(v)
            else:
                setattr(self, k, copy.deepcopy(v))
    
    def clone(self):
        return self.__class__.from_state_dict(self.state_dict())

    @classmethod
    def from_state_dict(cls, state):
        obj_cls = state.get("class", cls)
        print(obj_cls.__mro__)
        obj = obj_cls(*state["_init_args"], **state["_init_kwargs"])
        obj.load_state_dict(state)
        return obj


# Legacy clonable mixin, to be removed in favor of Checkpointable
class ClonableMixin:
    def clone(self):
        return copy.copy(self).load_state_dict(self.state_dict())
    
    def state_dict(
        self,
    ) -> dict:
        return self.__dict__

    def load_state_dict(
        self,
        state: dict,
    ) -> None:
        if not state.key() == self.__dict__.keys():
            raise RuntimeError(f"Cannot load state dict, keys don't match, current: {self.__dict__.keys()}, incoming: {state.keys()}")
        self.__dict__ = copy.deepcopy(state)
