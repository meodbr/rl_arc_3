from typing import Tuple, Any
import logging
import random
import copy
import math

import torch

from rl_arc_3.base.clone import Checkpointable
from rl_arc_3.base.model import BaseModel
from rl_arc_3.base.agent import BaseLearner
from rl_arc_3.base.trainer import DQNTrainingArgs

from rl_arc_3.agent.adapters import ModelAdapter

logger = logging.getLogger(__name__)


class DQNLearner(BaseLearner):
    def __init__(
        self,
        config: DQNTrainingArgs,
        model: BaseModel | None = None,
        model_adapter: ModelAdapter | None = None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)

        self.config = config

        if model is None:
            model = Checkpointable.uninitialized()

        self.device = self.config.device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Learner using device : {self.device}")

        self.model = model.to(self.device) if model.is_initialized() else model

        self.model_adapter = model_adapter

        self.target_model = Checkpointable.uninitialized()
        self.optimizer = Checkpointable.uninitialized()

        if self.model.is_initialized():
            self.target_model = self.model.clone()
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.lr
            )
        

        self.current_step = 0
    
    def load_state_dict(self, state):
        self.model = BaseModel.from_state_dict(state["model"]).to(self.device)
        self.config = copy.deepcopy(state["config"])
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr
        )
        self.optimizer.load_state_dict(state["optimizer"])
        self.target_model = BaseModel.from_state_dict(state["target_model"]).to(self.device)
        filtered_state = {k:v for k, v in state.items() if k not in ["model", "config", "optimizer", "target_model"]}
        return super().load_state_dict(filtered_state)
    
    def state_dict(self):
        state = super().state_dict()
        del state["device"]
        return state

    def learn(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, Any],
    ):
        self.model.train()
        self.optimizer.zero_grad()

        x, x_hat = self.compute_sample_batch(batch)

        loss = self.model.loss(x, x_hat)

        loss.backward()
        self.optimizer.step()

        self.current_step += 1
        if self.current_step % self.config.target_update_steps == 0:
            self.update_target_model()
    
    def get_target_model(self):
        return self.target_model

    def compute_sample_batch(self, batch):
        assert self.config.batch_size == batch[0].shape[0], "Batch size mismatch: expected {}, got {}".format(self.config.batch_size, batch[0].shape[0])
        batch_size = self.config.batch_size

        batch = tuple(torch.from_numpy(tensor) if not torch.is_tensor(tensor) else tensor for tensor in batch)
        batch = tuple(tensor.to(self.device) for tensor in batch)

        logger.debug("Batch tensors classes %s", [type(tensor) for tensor in batch])

        states, actions, rewards, next_states, dones = batch

        # Compute: predicted = Q(s, a)
        predicted = self.model(states).gather(1, actions)

        # Compute: expected = r + gamma * max_a(Q'(s',a))
        with torch.no_grad():
            next_state_rewards = torch.zeros((batch_size, 1), device=self.device)
            next_state_rewards[~dones] = (
                self.target_model(next_states[~dones]).max(1).values.unsqueeze(1)
            )
            logger.debug("Next state rewards shape: %s, rewards shape: %s", next_state_rewards.shape, rewards.shape)
            expected = rewards + self.config.gamma * next_state_rewards
        
        logger.debug("Sizes: expected %s, predicted %s", expected.shape, predicted.shape)

        return (expected, predicted)

    def get_epsilon(self):
        return self.config.eps_min + (
            self.config.eps_max - self.config.eps_min
        ) * math.exp(-1 * (self.current_step / self.config.eps_decay))

    @torch.no_grad()
    def update_target_model(self):
        for target_param, policy_param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * policy_param.data
                + (1.0 - self.config.tau) * target_param.data
            )
