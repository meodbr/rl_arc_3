from collections import namedtuple, deque
from typing import Type
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pydantic import BaseModel
import matplotlib.pyplot as plt
import time
from typing import Tuple

from rl_arc_3.utils import utils
from rl_arc_3.models.memory import Memory

class EpisodeStatistics(BaseModel):
    score: list[int]
    duration: list[int]


class DQNModel:
    """
    Class to wrap DQN training process
    """
    GAMMA: float     = 0.99
    LR: float        = 1e-3
    EPS_MAX: float   = 0.9
    EPS_MIN: float   = 0.02
    EPS_DECAY: float = 25000
    TAU: float       = 0.005
    BATCH_SIZE: int  = 128

    def __init__(self, model_class: Type[nn.Module], memory: Memory, model_instantation_args={}, device=None):
        if device is None:
            device = self.get_available_device()
        
        print(f"Using device: {device}")
        self.device = device

        self.model = model_class(**model_instantation_args)
        self.target_model = model_class(**model_instantation_args)
        self.model.to(device)
        self.target_model.to(device)
        self.target_model.eval()
        self.memory = memory
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.LR)

        self.action_count = 0
        self.fig = None
        self.statistics = {}
        self.tprof = {
            "tensor_conversion_time": [],
            "transition_zip_time": [],
            "prediction_time": [],
            "expected_computation_time": [],
            "statistics_computation_time": []
        }

    @staticmethod
    def get_available_device():
        return torch.device(
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )
    
    def compute_sample_batch(self, batch_size):
        # Sample memory
        transitions = self.memory.sample(batch_size)
        for tensor in transitions:
            if not tensor.device == self.device:
                tensor.to(self.device)
        state, action, next_state, reward, is_final = transitions
        # print(f"state shape: {state.shape}, dtype: {state.dtype}, device: {state.device}")

        # Compute: predicted = Q(s, a)
        predicted = self.model(state).gather(1, action)

        # Compute: expected = r + gamma * max_a(Q'(s',a))
        with torch.no_grad():
            next_state_reward = torch.zeros((batch_size, 1), device=self.device)
            next_state_reward[~is_final] = self.target_model(next_state[~is_final]).max(1).values.unsqueeze(1)
            expected = reward + self.GAMMA * next_state_reward

        return (predicted, expected)
    
    def train_iterations(self, n_iterations, batch_size=None) -> None:
        if not batch_size:
            batch_size = self.BATCH_SIZE

        if len(self.memory) < batch_size*4:
            return

        self.model.train()
        for _ in range(n_iterations):
            self.train_step(batch_size)
    
    def train_step(self, batch_size=None):
        if not batch_size:
            batch_size = self.BATCH_SIZE

        if len(self.memory) < batch_size*4:
            return

        self.model.train()
        self.optimizer.zero_grad()

        x_hat, x = self.compute_sample_batch(batch_size)

        loss = self.model.loss(x, x_hat)

        loss.backward()
        self.optimizer.step()

        self.update_target_model() 

    
    def get_epsilon(self):
        return self.EPS_MIN + (self.EPS_MAX - self.EPS_MIN) * math.exp(-1 * (self.action_count/self.EPS_DECAY))

    
    def select_action(self, observations: torch.Tensor, action_space_size: int) -> int:
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


    def store_transition(self, transition: Tuple[torch.Tensor]):
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
    
    # Generic version of store_episode_statistics
    def store_episode_statistics(self, statistics: dict):
        """
        Store episode statistics in the statistics dictionary.
        If the key does not exist, it will be created.
        """
        for key, value in statistics.items():
            if key not in self.statistics:
                self.statistics[key] = []
            self.statistics[key].append(value)

    
    @torch.no_grad()
    def update_target_model(self):
        for target_param, policy_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                self.TAU * policy_param.data + (1.0 - self.TAU) * target_param.data
            )
    

    def plot_statistics(self):
        """
        Plot statistics collected during training
        """
        if self.fig is None:
            self.fig = plt.figure(1, figsize=(14, 9))
        self.fig.clf()
        ax = self.fig.subplots(len(self.statistics)//2 + 1, 2, sharex=True)

        x_axis = []
        sum = 0
        for x in self.statistics['duration']:
            sum += x
            x_axis.append(sum)

        for i, (key, values) in enumerate(self.statistics.items()):
            ax[i//2, i%2].set_title(key)
            ax[i//2, i%2].plot(x_axis, np.array(values), label=key)

        self.fig.tight_layout()
        plt.pause(0.001)

        print("")
        print("Tprofiler statistics:")
        for key, times in self.tprof.items():
            values = np.array(times)
            print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f} seconds")


class ConvBasicModule(nn.Module):
    """
    Basic Conv2D module
    """
    def __init__(self, size=32, channels=3):
        super().__init__()
        self.input_size = size*size
        self.layer1 = nn.Conv2d(channels, 16, kernel_size=5, stride=1, padding=2)
        self.layer2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.layer3 = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.flattened_size = (size // 8) * (size // 8) * 32  # After 3 poolings

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.layer1(x)))
        x = self.pool(F.relu(self.layer2(x)))
        x = self.pool(F.relu(self.layer3(x)))

        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(x, x_hat)
    

class ConvBasic(DQNModel):
    pass