import os
import copy
import torch

class Checkpointable:
    _checkpointable_attrs = ["_checkpointable", "_is_initialized", "_init_args", "_init_kwargs"]

    def __init__(self, *args, **kwargs):
        self._checkpointable = True
        self._is_initialized = True
        self._init_args = args
        self._init_kwargs = kwargs
    
    def state_dict(self):
        self.ensure_checkpointable()
        self.ensure_initialized()
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
        self.ensure_checkpointable()
        self.ensure_initialized()
        for k, v in state.items():
            # Ensure class and init args/kwargs match before loading state
            if k == "class":
                if v != self.__class__:
                    raise RuntimeError(f"Cannot load state dict, class mismatch, current: {self.__class__}, incoming: {v}")
                continue

            if k in self._checkpointable_attrs:
                if v != getattr(self, k):
                    raise RuntimeError(f"Cannot load state dict, {k} mismatch, current: {getattr(self, k)}, incoming: {v}")
                continue

            # Load state for other attributes
            current = getattr(self, k, None)
            if isinstance(current, Checkpointable):
                if not current.is_initialized():
                    current.__class__.from_state_dict(v)
                else:
                    current.load_state_dict(v)
            elif hasattr(current, "load_state_dict"):
                print(f"proc {os.getpid()}, {self.__class__.__name__}: loading state dict for attribute {k} of type {type(current)}, is_none: {current is None}")
                current.load_state_dict(v)
            else:
                print(f"proc {os.getpid()}, {self.__class__.__name__}: fall back to deepcopy for attribute {k} of type {type(v)}, is_none: {current is None}")
                setattr(self, k, copy.deepcopy(v))
    
    def clone(self):
        return self.__class__.from_state_dict(self.state_dict())

    @classmethod
    def from_state_dict(cls, state):
        obj_cls = state.get("class", cls)
        obj = obj_cls(*state["_init_args"], **state["_init_kwargs"])
        print(f"process {os.getpid()} cloning object of class {obj_cls} with init args {state['_init_args']} and init kwargs {state['_init_kwargs']}")
        obj.load_state_dict(state)
        return obj
    
    def is_checkpointable(self):
        return hasattr(self, "_checkpointable") and self._checkpointable
    
    def ensure_checkpointable(self):
        if not self.is_checkpointable():
            raise RuntimeError("Object is not checkpointable, most likely Checkpointable.__init__ was not called in the constructor")
    
    @classmethod
    def uninitialized(cls):
        obj = cls.__new__(cls)
        obj._checkpointable = False
        obj._is_initialized = False
        return obj
    
    def is_initialized(self):
        return self._is_initialized
    
    def ensure_initialized(self):
        if not self.is_initialized():
            raise RuntimeError("This object is not initialized, cannot use it until it is initialized by calling from_state_dict or by passing init args to the constructor")


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
