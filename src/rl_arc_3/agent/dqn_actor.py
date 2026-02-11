import logging
import random
import math
from typing import Any
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_arc_3.model.conv_basic import ConvBasicModule
from rl_arc_3.model.memory import TensorMemory, DequeMemory

from rl_arc_3.base.env import Observation, Action, EnvSignature, EnvInterface
from rl_arc_3.base.agent import (
    InferenceConfig,
    PolicyOutput,
    BaseActor,
)
from rl_arc_3.base.model import BaseModel, ModelSignature
from rl_arc_3.trainer.dqn import DQNTrainingArgs
from rl_arc_3.utils.utils import get_model_device

logger = logging.getLogger(__name__)

class DQNModelAdapter(BaseModel):
    def __init__(
        self,
        env_signature: EnvSignature,
        model_signature: ModelSignature | None = None
    ):
        self.env_signature = env_signature
        self.env_act = env_signature.action_space
        self.env_obs = env_signature.observation_space

        computed_m_sig = self.compute_model_signature(env_signature)

        if model_signature is not None and computed_m_sig != model_signature:
            raise ValueError(f"Wrong Model signature: {model_signature} != {computed_m_sig}")

        self.model_signature = computed_m_sig
        self.m_input = self.model_signature.input_shape
        self.m_output = self.model_signature.output_shape
    
    @staticmethod
    def compute_model_signature(env_signature: EnvSignature) -> ModelSignature:
        raise NotImplementedError

    def observation_to_tensor(obs: Observation) -> torch.Tensor:
        raise NotImplementedError

    def tensor_to_action(obs: Observation) -> torch.Tensor:
        raise NotImplementedError

class DQNActor(BaseActor):
    def __init__(
        self,
        config: DQNTrainingArgs,
    ):
        self.config = config
        self.action_count = 0

    def __call__(
        self,
        model: nn.Module,
        observation: Observation,
    ) -> Action:
        """Default inference: returns deterministic action"""
        return self.policy(
            observation=observation,
        ).selected_action

    def policy(
        self,
        model: nn.Module,
        observation: Observation,
        config: InferenceConfig | None = None,
    ) -> PolicyOutput:
        p = random.random()
        epsilon = self.get_epsilon()

        if p < epsilon:
            action =  Action(torch.randint(0, self.model.input_size, (1,)).item())
        else:
            with torch.no_grad():
                inputs = torch.tensor(observation.state, device=get_model_device(model))
                logits = model.forward(inputs)
                # print(f"logits: {logits}")
                action = Action(logits.argmax().item())
        return PolicyOutput(
            selected_action=self.select_action(logits),
            logits=logits,
            info={},
        )
    
    def process_transition(
        self,
        observation: Observation,
        policy_output: PolicyOutput,
        next_observation: Observation,
    ) -> Any:
        return (
            observation.state,
            policy_output.selected_action,
            next_observation.state,
            next_observation.reward,
            next_observation.terminated,
        )

    def get_epsilon(self):
        return self.config.eps_min + (
            self.config.eps_max - self.config.eps_min
        ) * math.exp(-1 * (self.action_count / self.config.eps_decay))

    def _old_select_action(self, observations: torch.Tensor, action_space_size: int) -> int:
        p = random.random()
        epsilon = self.get_epsilon()

        if p < epsilon:
            return torch.randint(0, action_space_size, (1,)).item()
        else:
            print(f"observations shape: {observations.shape}")
            print(f"observations dtype: {observations.dtype}")
            with torch.no_grad():
                logits = self.model(observations)
                print(f"logits: {logits}")
                return logits.argmax().item()

    def _old_store_transition(self, transition: Tuple[torch.Tensor]):
        for elem in transition:
            if not isinstance(elem, torch.Tensor):
                ValueError("All elements of transition tuple must be tensors")

        ### Auto convertion code
        # transition = tuple(
        #     torch.as_tensor(elem, device=self.device) if not isinstance(elem, torch.Tensor)
        #     else elem.to(self.device)
        #     for elem in transition
        # )

        self.memory.push(transition)
        self.action_count += 1


def preprocess_frame(frame, device="cpu"):
    # Convert to tensor and add channel dimension (C, H, W)
    # 1 channel per color
    frame = torch.tensor(frame, dtype=torch.long, device=device)
    frame = F.one_hot(frame, num_classes=16).permute(2, 0, 1).float()  # (C, H, W)
    return frame