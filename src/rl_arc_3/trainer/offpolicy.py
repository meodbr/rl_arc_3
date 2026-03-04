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

logger = logging.getLogger(__name__)

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
        self.is_running = False

        self.checkpoint_version = mp.Value("i", 0)

        self.last_learner_ref = None
        self.last_memory_ref = None

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
        checkpoint_version: "Synchronized[int]",
        stop_event: Event,
        learner_queue: mp.Queue,
        learner_state: dict,
        config: OffPolicyTrainingArgs,
    ):
        setup_logging()
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
            
            if i % config.save_steps == 0:
                with checkpoint_version.get_lock():
                    checkpoint_version.value += 1
                learner_ref = OffPolicyTrainer.learner_ref(checkpoint_version.value)
                learner.save_checkpoint(learner_ref)
                logger.info("Saved learner checkpoint at step %d, v%d : %s", i, checkpoint_version.value, learner_ref)

            if stop_event.is_set():
                return

        stop_event.set()  # Signal workers to stop after learning is done

    @staticmethod
    def memory_process(
        checkpoint_version: "Synchronized[int]",
        stop_event: Event,
        replay_queue: mp.Queue,
        learner_queue: mp.Queue,
        memory_state: dict,
        config: OffPolicyTrainingArgs,
    ):
        setup_logging()
        logger.info("Starting memory process at pid %d", os.getpid())

        local_checkpoint_version = checkpoint_version.value
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
            
            if checkpoint_version.value > local_checkpoint_version:
                local_checkpoint_version = checkpoint_version.value
                memory_ref = OffPolicyTrainer.memory_ref(local_checkpoint_version)
                memory.save_checkpoint(memory_ref)
                logger.info("Saved memory checkpoint at train_step %d, v%d : %s", train_step, local_checkpoint_version, memory_ref)


    def train(self, resume_from_checkpoint: str | None = None):
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)

        self.validate_states_integrity()

        # print("Starting training with spawn multiprocessing method...")
        # mp.set_start_method("spawn")

        shared_model = BaseModel.from_state_dict(self.learner_state["target_model"])
        replay_queue = mp.Queue(maxsize=10)
        learner_queue = mp.Queue(maxsize=10)

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
                "checkpoint_version": self.checkpoint_version,
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
                "checkpoint_version": self.checkpoint_version,
                "replay_queue": replay_queue,
                "learner_queue": learner_queue,
                "memory_state": self.memory_state,
                "config": self.training_args,
            },
        )

        processes = workers + [learner, memory]

        self._pre_run()
        for p in processes:
            p.start()

        checkpoint = self.checkpoint_version.value
        try:
            while any(p.is_alive() for p in processes):
                time.sleep(0.5)
                if self.checkpoint_version.value > checkpoint:
                    checkpoint = self.checkpoint_version.value
                    self.on_checkpoint(checkpoint)
        except KeyboardInterrupt:
            print("Training interrupted. Sending stop event to processes...")
            stop_event.set()

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("Forcefully terminating processes...")
            print("Processes that were still alive:", [p.name for p in processes if p.is_alive()])
            for p in processes:
                p.terminate()

        replay_queue.close()
        learner_queue.close()
        self._post_run()
    
    def state_dict(self) -> dict:
        return {
            "config": self.training_args,
            "learner_ref": self.last_learner_ref,
            "memory_ref": self.last_memory_ref,
        }

    def load_state_dict(self, state: dict):
        if self.is_running:
            raise RuntimeError("Cannot load state while training is running")
        self.training_args = state["config"]
        self.learner_state = BaseLearner.read_checkpoint(state["learner_ref"])
        self.memory_state = BaseMemory.read_checkpoint(state["memory_ref"])
    
    def on_checkpoint(self, checkpoint_version: int):
        logger.info(f"Checkpoint version updated to {checkpoint_version}")
        self.last_learner_ref = self.learner_ref(checkpoint_version)
        self.last_memory_ref = self.memory_ref(checkpoint_version)
        self.save_checkpoint(self.trainer_ref())
    
    @staticmethod
    def learner_ref(checkpoint_version: int):
        return os.path.join(settings.CHECKPOINT_DIR, "learner", f"learner_checkpoint_{checkpoint_version}.pth")

    @staticmethod
    def memory_ref(checkpoint_version: int):
        return os.path.join(settings.CHECKPOINT_DIR, "memory", f"memory_snapshot{checkpoint_version}.pth")
    
    def trainer_ref(self):
        return os.path.join(settings.CHECKPOINT_DIR, f"checkpoint_{self.checkpoint_version.value}.pth")

    def _pre_run(self):
        self.is_running = True
        os.makedirs(settings.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(os.path.join(settings.CHECKPOINT_DIR, "learner"), exist_ok=True)
        os.makedirs(os.path.join(settings.CHECKPOINT_DIR, "memory"), exist_ok=True)
    
    def _post_run(self):
        self.is_running = False