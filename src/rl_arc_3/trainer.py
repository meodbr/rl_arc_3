from typing import Any

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized

from rl_arc_3.env.interface import EnvInterface, Observation, Action
from rl_arc_3.model import ConvBasicModule
from rl_arc_3.agent.interface import (
    Transitions,
)


class TrainingArgs:
    num_episodes: int
    num_workers: int
    max_steps_per_episode: int
    memory_capacity: int
    target_update_frequency: int
    log_interval: int
    save_interval: int
    plot_interval: int
    device: str | None = None


class DQNTrainingArgs(TrainingArgs):
    gamma: float = 0.99
    lr: float = 1e-3
    eps_max: float = 0.9
    eps_min: float = 0.02
    eps_decay: int = 25000
    tau: float = 0.005
    batch_size: int = 128


class DQNTrainer:
    def __init__(self, env_factory, training_args: TrainingArgs):
        self.env_factory = env_factory
        self.training_args = training_args

    @staticmethod
    def worker(
        env_factory,
        shared_model: nn.Module,
        shared_model_version: Synchronized[Any],
        replay_queue: mp.Queue,
        num_episodes: int = 10,
        max_steps_per_episode: int = 1000,
    ):
        env = env_factory()
        local_model = shared_model.__class__()
        local_model.load_state_dict(shared_model.state_dict())
        local_model_version = shared_model_version.value

        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            for step in range(max_steps_per_episode):
                action = Action(0)  # Placeholder for action selection logic
                next_obs = env.step(action)

                transition = (
                    obs,
                    action,
                    next_obs,
                )  # Placeholder for transition processing logic
                replay_queue.put(transition)

                obs = next_obs

            # Check for model updates
            with shared_model_version.get_lock():
                if shared_model_version.value > local_model_version:
                    local_model.load_state_dict(shared_model.state_dict())
                    local_model_version = shared_model_version.value

    @staticmethod
    def learner(
        shared_model: nn.Module,
        shared_model_version: Synchronized[Any],
        learner_queue: mp.Queue,
        max_steps: int = 10000,
        update_frequency: int = 100,
    ):
        local_model = shared_model.__class__()
        local_model.load_state_dict(shared_model.state_dict())

        for i in range(max_steps):
            batch = learner_queue.get()  # Placeholder for batch retrieval logic
            # Placeholder for learning logic using the batch

            if i % update_frequency == 0:
                with shared_model_version.get_lock():
                    shared_model.load_state_dict(local_model.state_dict())
                    shared_model_version.value += 1

    @staticmethod
    def memory(replay_queue: mp.Queue, learner_queue: mp.Queue):
        pass

    def train(self):
        model = ConvBasicModule()
        shared_model = model.clone().share_memory_()
        shared_model_version = mp.Value("i", 0)
        replay_queue = mp.Queue(maxsize=100)
        learner_queue = mp.Queue(maxsize=100)

        workers = [
            mp.Process(
                target=self.__class__.worker,
                args=(
                    self.env_factory,
                    shared_model,
                    shared_model_version,
                    replay_queue,
                    self.training_args.num_episodes,
                    self.training_args.max_steps_per_episode,
                ),
            )
            for _ in range(self.training_args.num_workers)
        ]
        learner = mp.Process(
            target=self.__class__.learner,
            args=(shared_model, shared_model_version, learner_queue),
        )
        memory = mp.Process(
            target=self.__class__.memory, args=(replay_queue, learner_queue)
        )

        for p in workers + [learner, memory]:
            p.start()

        for p in workers + [learner, memory]:
            p.join()
