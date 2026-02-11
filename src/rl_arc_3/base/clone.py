from copy import deepcopy, copy

class ClonableMixin:
    def clone(self):
        return copy(self).load_state_dict(self.state_dict())
    
    def state_dict(
        self,
    ) -> dict:
        return self.__dict__

    def load_state_dict(
        self,
        state: dict,
    ) -> None:
        if not state.keys() == self.__dict__.keys():
            raise RuntimeError(f"Cannot load state dict, keys don't match, current: {self.__dict__.keys()}, incoming: {state.keys()}")
        self.__dict__ = deepcopy(state)