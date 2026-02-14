from typing import Any
from dataclasses import dataclass, field

import torch.nn as nn

from rl_arc_3.base.env import Observation, Action, Transitions
from rl_arc_3.base.clone import ClonableMixin
from rl_arc_3.base.model import BaseModel

@dataclass
class PolicyOutput:
    selected_action: Action
    logits: Any = None
    info: dict = field(default_factory=dict)

@dataclass
class InferenceConfig:
    deterministic: bool = True

class BaseActor(ClonableMixin):
    def __call__(
        self,
        model: BaseModel,
        observation: Observation,
    ) -> Action:
        """Default inference: returns deterministic action"""
        raise NotImplementedError

    def policy(
        self,
        model: BaseModel,
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

class BaseLearner(ClonableMixin):
    def learn(
        self,
        model: nn.Module,
        batch: Any,
    ) -> dict:
        raise NotImplementedError