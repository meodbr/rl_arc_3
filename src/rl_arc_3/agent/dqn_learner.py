from typing import Tuple, Any
import logging
import random
import copy
import math

import torch

from rl_arc_3.base.checkpointable import Checkpointable
from rl_arc_3.base.model import BaseModel
from rl_arc_3.base.agent import BaseLearner
from rl_arc_3.base.trainer import DQNTrainingArgs
from rl_arc_3.base.model_adapter import ModelAdapter

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
        self.acc_count = torch.tensor(0, dtype=torch.float, device=self.device)
        self.acc_loss = torch.tensor(0, dtype=torch.float, device=self.device)
        self.acc_q_value = torch.tensor(0, dtype=torch.float, device=self.device)
        self.acc_q_value_target = torch.tensor(0, dtype=torch.float, device=self.device)
        self.acc_td_error = torch.tensor(0, dtype=torch.float, device=self.device)

    def load_state_dict(self, state):
        self.model = BaseModel.from_state_dict(state["model"]).to(self.device)
        self.config = copy.deepcopy(state["config"])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self.optimizer.load_state_dict(state["optimizer"])
        self.target_model = BaseModel.from_state_dict(state["target_model"]).to(
            self.device
        )
        filtered_state = {
            k: v
            for k, v in state.items()
            if k not in ["model", "config", "optimizer", "target_model"]
        }
        return super().load_state_dict(filtered_state)

    def state_dict(self):
        state = super().state_dict()
        del state["device"]
        return state

    def learn(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, Any],
        global_step: int,
        return_metrics: bool = False,
    ):
        self.current_step = global_step
        self.model.train()
        self.optimizer.zero_grad()

        batch = self.convert_transition(batch)
        x, x_hat = self.compute_sample_batch(batch)

        loss = self.model.loss(x, x_hat)

        loss.backward()
        self.optimizer.step()

        batch_size = batch[0].shape[0]
        self.update_metrics_tensors(loss, x, x_hat, batch_size)

        if global_step % self.config.target_update_steps == 0:
            logger.debug(self.state_dict())
            self.update_target_model()

        if return_metrics:
            return self.compute_metrics(global_step)
    
    def update_metrics_tensors(self, loss, x, x_hat, batch_size):
        td_error = (x - x_hat).sum()
        self.acc_count += batch_size
        self.acc_loss += loss.detach() * batch_size
        self.acc_q_value += x_hat.detach().sum()
        self.acc_q_value_target += x.detach().sum()
        self.acc_td_error += td_error.detach().sum()

    def compute_metrics(self, global_step):
        count = self.acc_count

        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item()

        metrics = {
            "global_step": global_step,
            "train/loss": (self.acc_loss / count).item(),
            "train/td_error": (self.acc_td_error / count).item(),
            "train/mean_q_value": (self.acc_q_value / count).item(),
            "train/mean_q_value_target": (self.acc_q_value_target / count).item(),
            "train/grad_norm": grad_norm,
        }

        self.acc_count = self.acc_count.zero_()
        self.acc_loss = self.acc_loss.zero_()
        self.acc_q_value = self.acc_q_value.zero_()
        self.acc_q_value_target = self.acc_q_value_target.zero_()
        self.acc_td_error = self.acc_td_error.zero_()

        return metrics


    def get_target_model(self):
        return self.target_model

    def convert_transition(self, trs: Tuple) -> Tuple:
        logger.debug("Batch tensors classes %s", [type(tensor) for tensor in trs])
        obs_index = [0, 3]

        res = tuple(
            (
                self.model_adapter.uncompress_obs(t, batched=True, device=self.device)
                if i in obs_index
                else t
            )
            for i, t in enumerate(trs)
        )
        res = tuple(
            torch.tensor(t, device=self.device) if not torch.is_tensor(t) else t
            for t in res
        )
        res = tuple(t.to(self.device) for t in res)
        return res

    def compute_sample_batch(self, batch):
        assert (
            self.config.batch_size == batch[0].shape[0]
        ), "Batch size mismatch: expected {}, got {}".format(
            self.config.batch_size, batch[0].shape[0]
        )
        batch_size = self.config.batch_size

        states, actions, rewards, next_states, dones = batch

        # Compute: predicted = Q(s, a)
        predicted = self.model(states).gather(1, actions)

        # Compute: expected = r + gamma * max_a(Q'(s',a))
        with torch.no_grad():
            next_state_rewards = torch.zeros((batch_size, 1), device=self.device)
            next_state_rewards[~dones] = (
                self.target_model(next_states[~dones]).max(1).values.unsqueeze(1)
            )
            logger.debug(
                "Next state rewards shape: %s, rewards shape: %s",
                next_state_rewards.shape,
                rewards.shape,
            )
            expected = rewards + self.config.gamma * next_state_rewards

        logger.debug(
            "Sizes: expected %s, predicted %s", expected.shape, predicted.shape
        )

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
