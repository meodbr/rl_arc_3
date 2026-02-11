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

from rl_arc_3.env.interface import Observation, Action, Transitions
from rl_arc_3.agent.interface import (
    InferenceConfig,
    PolicyOutput,
    ActorInterface,
)
from rl_arc_3.model.interface import ModelInterface
from rl_arc_3.trainer.dqn import DQNTrainingArgs
from rl_arc_3.utils.utils import get_model_device

logger = logging.getLogger(__name__)


class DQNInferenceConfig(InferenceConfig):
    weights: str = "target"

class DQNActor(ActorInterface):
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