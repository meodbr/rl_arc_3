import time
from typing import Any, Callable
from itertools import count
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized

from rl_arc_3.base.env import EnvInterface
from rl_arc_3.base.agent import BaseActor, BaseLearner
from rl_arc_3.base.model import BaseModel
from rl_arc_3.base.trainer import BaseTrainer, OffPolicyTrainingArgs
from rl_arc_3.model.memory import BaseMemory

from rl_arc_3.agent.adapters import FullModelAdapter, KeyboardOnlyModelAdapter
from rl_arc_3.utils.utils import push_with_stop, get_with_stop


class OffPolicyTrainer(BaseTrainer):
    def __init__(
        self,
        training_args: OffPolicyTrainingArgs,
        env_factory: Callable[[], EnvInterface],
        model: BaseModel,
        actor: BaseActor,
        learner: BaseLearner,
        memory_factory: Callable[[int], BaseMemory],
    ):
        self.env_factory = env_factory
        self.actor = actor
        self.learner = learner
        self.memory_factory = memory_factory
        self.training_args = training_args
        self.model = model

    @staticmethod
    def worker_process(
        shared_model: BaseModel,
        shared_model_version: Synchronized[Any],
        stop_event: Synchronized[Any],
        env_factory: Callable[[], EnvInterface],
        actor: BaseActor,
        replay_queue: mp.Queue,
        config: OffPolicyTrainingArgs,
    ):
        env = env_factory()
        local_model = shared_model.clone()
        local_model_version = shared_model_version.value

        # while not stop_event.is_set():
        for episode in count():
            obs = env.reset()
            done = False

            # Check for model updates
            with shared_model_version.get_lock():
                if shared_model_version.value > local_model_version:
                    local_model = shared_model.clone()
                    local_model_version = shared_model_version.value

            for step in range(config.max_steps_per_episode):
                policy_output = actor.policy(local_model, obs)
                action = policy_output.selected_action
                next_obs = env.step(action)

                transition = actor.process_transition(
                    obs,
                    action,
                    next_obs,
                )

                obs = next_obs

                pushed = push_with_stop(replay_queue, transition, stop_event)

                if done or not pushed:
                    break

            if stop_event.is_set():
                return

    @staticmethod
    def learner_process(
        shared_model: BaseModel,
        shared_model_version: Synchronized[Any],
        stop_event: Synchronized[Any],
        learner_queue: mp.Queue,
        learner: BaseLearner,
        config: OffPolicyTrainingArgs,
    ):
        local_model = shared_model.clone()

        for i in range(config.max_steps):
            batch = get_with_stop(
                learner_queue, stop_event
            )  # Placeholder for batch retrieval logic
            if batch is None:
                return

            metrics = learner.learn(local_model, batch)

            if i % config.target_update_steps == 0:
                with shared_model_version.get_lock():
                    shared_model.load_state_dict(local_model.state_dict())
                    shared_model_version.value += 1

            if stop_event.is_set():
                return

        stop_event.set()  # Signal workers to stop after learning is done

    @staticmethod
    def memory_process(
        stop_event: Synchronized[Any],
        replay_queue: mp.Queue,
        learner_queue: mp.Queue,
        memory_factory: Callable[[int], BaseMemory],
        config: OffPolicyTrainingArgs,
    ):
        train_step = 0
        explore_steps = 0
        memory = memory_factory()
        while not stop_event.is_set():
            if (
                train_step > explore_steps * config.train_explore_ratio
                or len(memory) < config.batch_size
            ):
                transition = get_with_stop(replay_queue, stop_event)
                memory.push(*transition)
                explore_steps += 1
            else:
                batch = memory.sample(config.batch_size)
                pushed = push_with_stop(learner_queue, batch, stop_event)
                if pushed:
                    train_step += 1

            if train_step % config.log_steps == 0:
                replay_qsize = replay_queue.qsize()
                learner_qsize = learner_queue.qsize()
                print(
                    f"Memory: train_step={train_step}\treplay_qsize={replay_qsize}\tlearner_qsize={learner_qsize}"
                )

    def train(self, resume_from_checkpoint: str | None = None):
        shared_model = self.model.clone().share_memory_()
        replay_queue = mp.Queue(maxsize=100)
        learner_queue = mp.Queue(maxsize=100)

        shared_model_version = mp.Value("i", 0)
        stop_event = mp.Event()

        workers = [
            mp.Process(
                target=self.__class__.worker,
                kwargs={
                    "shared_model": shared_model,
                    "shared_model_version": shared_model_version,
                    "stop_event": stop_event,
                    "env_factory": self.env_factory,
                    "actor": self.actor.clone(),
                    "replay_queue": replay_queue,
                    "config": self.training_args,
                },
            )
            for _ in range(self.training_args.num_workers)
        ]
        learner = mp.Process(
            target=self.__class__.learner_process,
            kwargs={
                "shared_model": shared_model,
                "shared_model_version": shared_model_version,
                "stop_event": stop_event,
                "learner_queue": learner_queue,
                "learner": self.learner.clone(),
                "config": self.training_args,
            },
        )
        memory = mp.Process(
            target=self.__class__.memory_process,
            kwargs={
                "stop_event": stop_event,
                "replay_queue": replay_queue,
                "learner_queue": learner_queue,
                "memory_factory": self.memory_factory,
                "config": self.training_args,
            },
        )

        processes = workers + [learner, memory]

        for p in processes:
            p.start()

        try:
            while any(p.is_alive() for p in processes):
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Training interrupted. Stopping processes...")
            stop_event.set()

        for p in processes:
            p.join()

        replay_queue.close()
        learner_queue.close()
