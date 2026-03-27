from typing import Tuple, Any
from dataclasses import dataclass

from gymnasium.spaces import Space
import numpy as np

Envinfo = Tuple[np.ndarray, float, bool, dict] # state, reward, done, info

@dataclass(frozen=True)
class EnvSignature:
    observation_space: Space
    action_space: Space

class BaseEnv:
    def reset(self) -> Envinfo:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> Envinfo:
        raise NotImplementedError

    def signature(self) -> EnvSignature:
        raise NotImplementedError
