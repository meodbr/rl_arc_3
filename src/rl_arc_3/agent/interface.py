from typing import Any
from dataclasses import dataclass, field

import torch.nn as nn

from rl_arc_3.env.interface import Observation, Action, Transitions

@dataclass
class PolicyOutput:
    selected_action: Action
    logits: Any = None
    info: dict = field(default_factory=dict)

@dataclass
class InferenceConfig:
    pass

class ActorInterface:
    def __call__(
        self,
        model: nn.module,
        observation: Observation,
    ) -> Action:
        """Default inference: returns deterministic action"""
        raise NotImplementedError

    def policy(
        self,
        model: nn.Module,
        observation: Observation,
        config: InferenceConfig | None = None,
    ) -> PolicyOutput:
        raise NotImplementedError
    
    def process_transition(
        self,
        observation: Observation,
        action: Action,
        next_observation: Observation,
    ):
        raise NotImplementedError

    def state_dict(
        self,
    ) -> dict:
        raise NotImplementedError
    
    def load_state_dict(
        self,
        state: dict,
    ) -> None:
        raise NotImplementedError

class LearnerInterface:
    def learn(
        self,
        model: nn.Module,
        batch: Any,
    ) -> dict:
        raise NotImplementedError

    def state_dict(
        self,
    ) -> dict:
        raise NotImplementedError
    
    def load_state_dict(
        self,
        state: dict,
    ) -> None:
        raise NotImplementedError