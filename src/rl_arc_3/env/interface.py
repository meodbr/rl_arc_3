from typing import Tuple, Any
from dataclasses import dataclass

from gymnasium.spaces import Space
import numpy as np

@dataclass(frozen=True)
class Observation:
    state: np.ndarray
    reward: float
    terminated: bool
    info: dict

@dataclass(frozen=True)
class Action:
    id_: int
    coords: Tuple[int, int] | None = None

    def is_complex(self):
        return self.coords is not None

@dataclass(frozen=True)
class Transitions:
    states: Any
    actions: Any
    next_state: Any
    rewards: Any
    terminated: Any

class EnvInterface:
    def reset(self) -> Observation:
        raise NotImplementedError

    def step(self, action: Action) -> Observation:
        raise NotImplementedError
    
    @property
    def observation_space(self) -> Space:
        raise NotImplementedError

    @property
    def action_space(self) -> Space:
        raise NotImplementedError