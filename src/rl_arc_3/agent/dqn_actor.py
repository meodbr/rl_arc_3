import logging
import random
import math
from typing import Any
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium.spaces import Tuple, Discrete, Box

from rl_arc_3.base.env import EnvSignature, BaseEnv, Envinfo
from rl_arc_3.base.model import BaseModel, ModelSignature
from rl_arc_3.base.model_adapter import ModelAdapter
from rl_arc_3.base.trainer import DQNTrainingArgs
from rl_arc_3.base.agent import (
    InferenceConfig,
    PolicyOutput,
    BaseActor,
)

from rl_arc_3.utils.utils import get_model_device

logger = logging.getLogger(__name__)


class DQNActor(BaseActor):
    def __init__(
        self,
        config: DQNTrainingArgs,
        model_adapter: ModelAdapter,
        **kwargs,
    ):
        super().__init__(config=config, model_adapter=model_adapter, **kwargs)
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
                action_tensor = torch.argmax(logits, dim=1)
                action = self.model_adapter.tensor_to_action(action_tensor)

        return PolicyOutput(
            selected_action=action,
            logits=logits,
            action_tensor=action_tensor,
            info={"type": type},
        )

    def process_transition(
        self,
        observation: Envinfo,
        policy_output: PolicyOutput,
        next_observation: Envinfo,
    ) -> Any:
        state, _, _, _ = observation
        next_state, reward, done, _ = next_observation

        next_state = (
            next_state if not done else np.zeros(shape=state.shape, dtype=state.dtype)
        )

        logger.debug(
            "Processing transition with reward: %s, done: %s, action: %s",
            reward,
            done,
            policy_output.action_tensor,
        )

        state = self.model_adapter.observation_to_tensor(state).unsqueeze(0)
        state = self.model_adapter.compress_obs(state, batched=True)
        next_state = self.model_adapter.observation_to_tensor(next_state).unsqueeze(0)
        next_state = self.model_adapter.compress_obs(next_state, batched=True)

        action = policy_output.action_tensor.view(1, -1).numpy()
        reward = torch.tensor(reward, dtype=torch.float32).view(1, -1).numpy()
        done = torch.tensor(done, dtype=torch.bool).view(1).numpy()

        res = (
            state,
            action,
            reward,
            next_state,
            done,
        )
        logger.debug("Transition shapes: %s", [t.shape for t in res])
        return res

    def get_epsilon(self):
        return self.config.eps_min + (
            self.config.eps_max - self.config.eps_min
        ) * math.exp(-1 * (self.action_count / self.config.eps_decay))
