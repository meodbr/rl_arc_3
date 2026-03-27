from typing import Any
from dataclasses import dataclass, field

import torch.nn as nn

from rl_arc_3.base.env import Envinfo
from rl_arc_3.base.checkpointable import Checkpointable
from rl_arc_3.base.model import BaseModel

@dataclass
class PolicyOutput:
    selected_action: Any
    logits: Any = None
    action_tensor: Any = None
    info: dict = field(default_factory=dict)

@dataclass
class InferenceConfig:
    deterministic: bool = True

class BaseActor(Checkpointable):
    def __call__(
        self,
        model: BaseModel,
        observation: Envinfo,
    ) -> Any:
        """Default inference: returns deterministic action"""
        raise NotImplementedError

    def policy(
        self,
        model: BaseModel,
        observation: Envinfo,
        config: InferenceConfig | None = None,
    ) -> PolicyOutput:
        raise NotImplementedError
    
    def process_transition(
        self,
        observation: Envinfo,
        action: PolicyOutput,
        next_observation: Envinfo,
    ):
        raise NotImplementedError

class BaseLearner(Checkpointable):
    def learn(
        self,
        batch: Any,
        global_step: int,
        return_metrics: bool = False,
    ) -> dict | None:
        raise NotImplementedError
    
    def get_target_model(self) -> BaseModel:
        raise NotImplementedError