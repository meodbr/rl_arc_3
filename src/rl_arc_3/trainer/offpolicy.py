import os
import time
import logging
from typing import Any, Callable
from itertools import count
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event

from rl_arc_3.base.env import BaseEnv
from rl_arc_3.base.agent import BaseActor, BaseLearner
from rl_arc_3.base.model import BaseModel
from rl_arc_3.base.trainer import BaseTrainer, OffPolicyTrainingArgs
from rl_arc_3.model.memory import BaseMemory

from rl_arc_3.utils.utils import push_with_stop, get_with_stop, setup_logging
from rl_arc_3.settings import settings


class OffPolicyTrainer(BaseTrainer):
    """
    This class is not meant to be used directly, but rather serve as a base for specific off-policy algorithms like DQNTrainer.
    """

    def __init__(
        self,
        training_args: OffPolicyTrainingArgs,
        env_factory: Callable[[], BaseEnv],
        **kwargs,
    ):
        super().__init__(training_args, env_factory, **kwargs)

        self.training_args = training_args
        self.env_factory = env_factory

        self.actors_states: list[dict] = None
        self.learner_state: dict = None
        self.memory_state: dict = None

    def validate_states_integrity(self):
        if self.learner_state is None:
            raise RuntimeError("Learner state is not set, cannot checkpoint")
        if (
            not self.actors_states
            or len(self.actors_states) != self.training_args.num_workers
        ):
            raise RuntimeError(
                "Actors states is not properly initialized, cannot checkpoint"
            )
        if self.memory_state is None:
            raise RuntimeError("Memory state is not set, cannot checkpoint")

    @staticmethod
    def worker_process(
        process_id: int,
        shared_model: BaseModel,
        shared_model_version: "Synchronized[int]",
        stop_event: Event,
        env_factory: Callable[[], BaseEnv],
        actor_state: dict,
        replay_queue: mp.Queue,
        config: OffPolicyTrainingArgs,
    ):
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting worker process n°%d at pid %d", process_id, os.getpid())

        env = env_factory()
        local_model = shared_model.clone()
        local_model_version = shared_model_version.value
        actor = BaseActor.from_state_dict(actor_state)

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
                    policy_output,
                    next_obs,
                )

                obs = next_obs

                logger.debug("Pushing transition to replay queue")
                pushed = push_with_stop(replay_queue, transition, stop_event)
                logger.debug("Push result: %s", pushed)

                if done or not pushed:
                    break

            logger.info("Episode %d finished after %d steps.", episode, step + 1)

            if stop_event.is_set():
                return

    @staticmethod
    def learner_process(
        shared_model: BaseModel,
        shared_model_version: "Synchronized[int]",
        stop_event: Event,
        learner_queue: mp.Queue,
        learner_state: dict,
        config: OffPolicyTrainingArgs,
    ):
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting learner process at pid %d", os.getpid())

        logger.debug("Loading learner state from dict: %s", learner_state.keys())
        learner = BaseLearner.from_state_dict(learner_state)

        for i in range(config.max_steps):
            logger.debug("Learner step %d waiting for batch...", i)
            batch = get_with_stop(
                learner_queue, stop_event
            )  # Placeholder for batch retrieval logic
            logger.debug("Learner step %d received batch: %s", i, batch is not None)
            if batch is None:
                return

            metrics = learner.learn(batch)

            if i % config.target_update_steps == 0:
                logger.info("Updating shared model at step %d, metrics: %s", i, metrics)
                with shared_model_version.get_lock():
                    shared_model.load_state_dict(learner.target_model.state_dict())
                    shared_model_version.value += 1

            if stop_event.is_set():
                return

        stop_event.set()  # Signal workers to stop after learning is done

    @staticmethod
    def memory_process(
        stop_event: Event,
        replay_queue: mp.Queue,
        learner_queue: mp.Queue,
        memory_state: dict,
        config: OffPolicyTrainingArgs,
    ):
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting memory process at pid %d", os.getpid())

        train_step = 0
        explore_steps = 0
        log_step = -1
        memory = BaseMemory.from_state_dict(memory_state)
        while not stop_event.is_set():
            if (
                train_step > explore_steps * config.train_explore_ratio
                or len(memory) < config.batch_size
            ):
                transition = get_with_stop(replay_queue, stop_event)
                logger.debug("Ingesting transition %s", transition)
                memory.push(transition)
                explore_steps += 1
            else:
                batch = memory.sample(config.batch_size)
                pushed = push_with_stop(learner_queue, batch, stop_event)
                if pushed:
                    train_step += 1

            if train_step % config.log_steps == 0 and train_step // config.log_steps > log_step:
                log_step = train_step // config.log_steps
                replay_qsize = replay_queue.qsize()
                learner_qsize = learner_queue.qsize()
                logger.info(
                    "Memory: train_step=%d\treplay_qsize=%d\tlearner_qsize=%d",
                    train_step,
                    replay_qsize,
                    learner_qsize,
                )

    def train(self, resume_from_checkpoint: str | None = None):
        self.validate_states_integrity()
        setup_logging()

        mp.set_start_method("spawn")

        shared_model = BaseModel.from_state_dict(self.learner_state["target_model"])
        replay_queue = mp.Queue(maxsize=100)
        learner_queue = mp.Queue(maxsize=100)

        shared_model_version = mp.Value("i", 0)
        stop_event = mp.Event()

        workers = [
            mp.Process(
                name=f"Worker{i:02d}",
                target=self.__class__.worker_process,
                kwargs={
                    "process_id": i,
                    "shared_model": shared_model,
                    "shared_model_version": shared_model_version,
                    "stop_event": stop_event,
                    "env_factory": self.env_factory,
                    "actor_state": self.actors_states[i],
                    "replay_queue": replay_queue,
                    "config": self.training_args,
                },
            )
            for i in range(self.training_args.num_workers)
        ]
        learner = mp.Process(
            target=self.__class__.learner_process,
            name="Learner_",
            kwargs={
                "shared_model": shared_model,
                "shared_model_version": shared_model_version,
                "stop_event": stop_event,
                "learner_queue": learner_queue,
                "learner_state": self.learner_state,
                "config": self.training_args,
            },
        )
        memory = mp.Process(
            target=self.__class__.memory_process,
            name="Memory__",
            kwargs={
                "stop_event": stop_event,
                "replay_queue": replay_queue,
                "learner_queue": learner_queue,
                "memory_state": self.memory_state,
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
            print("Training interrupted. Sending stop event to processes...")
            stop_event.set()

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("Forcefully terminating processes...")
            print("Processes that were still alive:", [p.pid for p in processes if p.is_alive()])
            for p in processes:
                p.terminate()

        replay_queue.close()
        learner_queue.close()

    def state_dict(self) -> dict:
        return {
            "config": self.training_args,
            "learner_state": self.learner_state,
            "actors_states": self.actors_states,
            "memory_state": self.memory_state,
        }

    def load_state_dict(self, state: dict):
        self.training_args = state["config"]
        self.learner_state = state["learner_state"]
        self.actors_states = state["actors_states"]
        self.memory_state = state["memory_state"]
