from typing import Protocol

from gymnasium.spaces import Space


class ModelFactory(Protocol):
    def __call__(
        self,
        observation_space: Space,
        action_space: Space,
    ):
        raise NotImplementedError
