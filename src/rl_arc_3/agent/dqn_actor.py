import logging
import random
import math
from typing import Any
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Tuple, Discrete, Box

from rl_arc_3.model.conv_basic import ConvBasicModule
from rl_arc_3.model.memory import TensorMemory, DequeMemory

from rl_arc_3.base.env import EnvSignature, BaseEnv, Envinfo
from rl_arc_3.base.model import BaseModel, ModelSignature
from rl_arc_3.base.agent import (
    InferenceConfig,
    PolicyOutput,
    BaseActor,
)
from rl_arc_3.agent.adapters import ModelAdapter
from rl_arc_3.base.trainer import DQNTrainingArgs
from rl_arc_3.utils.utils import get_model_device

logger = logging.getLogger(__name__)

class DQNActor(BaseActor):
    def __init__(
        self,
        config: DQNTrainingArgs,
        model_adapter: ModelAdapter,
    ):
        self.config = config
        self.model_adapter = model_adapter
        self.action_count = 0

    def __call__(
        self,
        model: BaseModel,
        observation: Envinfo,
    ) -> Any:
        """Default inference: returns deterministic action"""
        return self.policy(
            model=model,
            observation=observation,
        ).selected_action

    def policy(
        self,
        model: BaseModel,
        observation: Envinfo,
        config: InferenceConfig | None = None,
    ) -> PolicyOutput:
        if config is None:
            config = InferenceConfig(deterministic=True)

        p = random.random()
        epsilon = self.get_epsilon()
        state, _, _, _ = observation

        if p < epsilon and not config.deterministic:
            type = "random_action"
            action = self.model_adapter.env_signature.action_space.sample()
        else:
            type = "model_action"
            inputs = self.model_adapter.observation_to_tensor(
                state, device=get_model_device(model)
            )
            with torch.no_grad():
                logits = model.forward(inputs)
            
            action = self.model_adapter.tensor_to_action(logits)
        return PolicyOutput(
            selected_action=action,
            logits=logits,
            info={"type": type},
        )

    def process_transition(
        self,
        observation: Envinfo,
        policy_output: PolicyOutput,
        next_observation: Envinfo,
    ) -> Any:
        state, _, _, _ = observation
        next_state, reward, done, info = next_observation
        return (
            self.model_adapter.observation_to_tensor(state),
            policy_output.selected_action,
            self.model_adapter.observation_to_tensor(next_state),
            reward,
            done,
        )

    def get_epsilon(self):
        return self.config.eps_min + (
            self.config.eps_max - self.config.eps_min
        ) * math.exp(-1 * (self.action_count / self.config.eps_decay))
