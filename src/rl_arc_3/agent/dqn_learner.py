from typing import Tuple, Any
import logging
import random
import math

import torch

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
        self.model = model
        self.model_adapter = model_adapter

        self.target_model = None
        self.optimizer = None

        if self.model is not None:
            self.target_model = self.model.clone()
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.lr
            )

        self.current_step = 0

    def learn(
        self,
        batch: Tuple[torch.Tensor, Any, torch.Tensor, Any, Any],
    ):
        self.model.train()
        self.optimizer.zero_grad()

        x_hat, x = self.compute_sample_batch(batch)

        loss = self.model.loss(x, x_hat)

        loss.backward()
        self.optimizer.step()

        self.current_step += 1
        if self.current_step % self.config.target_update_steps == 0:
            self.update_target_model()
    
    def get_target_model(self):
        return self.target_model

    def compute_sample_batch(self, batch):
        batch_size = self.config.batch_size
        for tensor in batch:
            if not tensor.device == self.device:
                tensor = tensor.to(self.device) # TODO: this is a bit hacky

        states, actions, rewards, next_states, dones = batch

        # Compute: predicted = Q(s, a)
        predicted = self.model(states).gather(1, actions)

        # Compute: expected = r + gamma * max_a(Q'(s',a))
        with torch.no_grad():
            next_state_rewards = torch.zeros((batch_size, 1), device=self.device)
            next_state_rewards[~dones] = (
                self.target_model(next_states[~dones]).max(1).values.unsqueeze(1)
            )
            expected = rewards + self.config.gamma * next_state_rewards

        return (predicted, expected)

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
