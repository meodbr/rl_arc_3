import torch
from collections import deque, namedtuple
import random
from typing import Tuple

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class Memory:
    def push(self, transition: Tuple[torch.Tensor]):
        raise ValueError("Abstract method")

    def sample(self, batch_size) -> Tuple[torch.Tensor]:
        raise ValueError("Abstract method")

    def __len__(self) -> int:
        raise ValueError("Abstract method")


class DequeMemory(Memory):
    def __init__(self, size: int):
        self.transitions = deque([], maxlen=size)

    def push(self, transition):
        self.transitions.append(transition)

    def sample(self, n: int) -> list[Transition]:
        return random.sample(self.transitions, k=n)

    def __len__(self):
        return len(self.transitions)


class TensorMemory(Memory):
    def __init__(self, capacity, state_shape, device="cpu"):
        self.capacity = capacity
        self.device = device

        self.states = torch.zeros(
            (capacity, *state_shape), dtype=torch.float32, device=device
        )
        self.next_states = torch.zeros(
            (capacity, *state_shape), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros((capacity, 1), dtype=torch.long, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity,), dtype=torch.bool, device=device)

        self.index = 0
        self.size = 0

    def push(self, transition):
        state, action, next_state, reward, done = transition
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size) -> Tuple[torch.Tensor]:
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.states[idxs],
            self.actions[idxs],
            self.next_states[idxs],
            self.rewards[idxs],
            self.dones[idxs],
        )

    def __len__(self):
        return self.size
