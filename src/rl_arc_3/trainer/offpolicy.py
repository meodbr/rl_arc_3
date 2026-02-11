import time
from typing import Any, Callable
from itertools import count
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized

from rl_arc_3.base.env import EnvInterface, Observation, Action
from rl_arc_3.base.agent import ActorInterface, LearnerInterface
from rl_arc_3.base.model import ModelInterface
from rl_arc_3.base.trainer import TrainerInterface, TrainingArgs

from rl_arc_3.utils.utils import push_with_stop, get_with_stop


class OffPolicyTrainingArgs(TrainingArgs):
    train_explore_ratio: int
    target_update_steps: int
    memory_capacity: int


class OffPolicyTrainer(TrainerInterface):
    def __init__(
        self,
        model: ModelInterface,
        env_factory: Callable[[], EnvInterface],
        training_args: OffPolicyTrainingArgs,
    ):
        self.model = model
        self.env_factory = env_factory
        self.training_args = training_args

    @staticmethod
    def worker_process(
        shared_model: ModelInterface,
        shared_model_version: Synchronized[Any],
        stop_event: Synchronized[Any],
        env_factory: Callable[[], EnvInterface],
        actor: ActorInterface,
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
        shared_model: ModelInterface,
        shared_model_version: Synchronized[Any],
        stop_event: Synchronized[Any],
        learner_queue: mp.Queue,
        learner: LearnerInterface,
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
        memory_class,
        config: OffPolicyTrainingArgs,
    ):
        train_step = 0
        explore_steps = 0
        memory = memory_class(config.memory_capacity)
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
                args=(
                    self.env_factory,
                    shared_model,
                    shared_model_version,
                    replay_queue,
                    stop_event,
                    self.training_args.num_episodes,
                    self.training_args.max_steps_per_episode,
                ),
            )
            for _ in range(self.training_args.num_workers)
        ]
        learner = mp.Process(
            target=self.__class__.learner,
            args=(shared_model, shared_model_version, stop_event, learner_queue),
        )
        memory = mp.Process(
            target=self.__class__.memory, args=(stop_event, replay_queue, learner_queue)
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
