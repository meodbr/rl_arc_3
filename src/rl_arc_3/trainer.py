import time
from typing import Any
from itertools import count

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized

from rl_arc_3.env.interface import EnvInterface, Observation, Action
from rl_arc_3.model import ConvBasicModule
from rl_arc_3.utils.utils import push_with_stop, get_with_stop
from rl_arc_3.agent.interface import (
    Transitions,
)


class TrainingArgs:
    num_episodes: int
    num_workers: int
    max_steps_per_episode: int
    memory_capacity: int
    target_update_steps: int
    log_steps: int
    save_steps: int
    max_steps: int
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
        stop_event: Synchronized[Any],
        replay_queue: mp.Queue,
        max_steps_per_episode: int = 1000,
    ):
        env = env_factory()
        local_model = shared_model.__class__()
        local_model.load_state_dict(shared_model.state_dict())
        local_model_version = shared_model_version.value

        # while not stop_event.is_set():
        for episode in count():
            obs = env.reset()
            done = False

            # Check for model updates
            with shared_model_version.get_lock():
                if shared_model_version.value > local_model_version:
                    local_model.load_state_dict(shared_model.state_dict())
                    local_model_version = shared_model_version.value


            for step in range(max_steps_per_episode):
                action = Action(0)  # Placeholder for action selection logic
                next_obs = env.step(action)

                transition = (
                    obs,
                    action,
                    next_obs,
                )  # Placeholder for transition processing logic


                obs = next_obs

                pushed = push_with_stop(replay_queue, transition, stop_event)

                if done or not pushed:
                    break
            
            if stop_event.is_set():
                return

    @staticmethod
    def learner(
        shared_model: nn.Module,
        shared_model_version: Synchronized[Any],
        stop_event: Synchronized[Any],
        learner_queue: mp.Queue,
        max_steps: int = 10000,
        update_frequency: int = 100,
    ):
        local_model = shared_model.__class__()
        local_model.load_state_dict(shared_model.state_dict())

        for i in range(max_steps):
            batch = get_with_stop(learner_queue, stop_event)  # Placeholder for batch retrieval logic
            # Placeholder for learning logic using the batch

            if i % update_frequency == 0:
                with shared_model_version.get_lock():
                    shared_model.load_state_dict(local_model.state_dict())
                    shared_model_version.value += 1
            
            if stop_event.is_set():
                return
        
        stop_event.set()  # Signal workers to stop after learning is done

    @staticmethod
    def memory(
        stop_event: Synchronized[Any],
        replay_queue: mp.Queue,
        learner_queue: mp.Queue,
        memory_class,
        memory_capacity: int = 10000,
        batch_size: int = 128,
        train_explore_ratio: float = 0.5,
        logging_steps: int = 1000,
    ):
        train_step = 0
        explore_steps = 0
        memory = memory_class(memory_capacity)
        while not stop_event.is_set():
            if train_step > explore_steps * train_explore_ratio or len(memory) < batch_size:
                transition = get_with_stop(replay_queue, stop_event)
                memory.push(*transition)
                explore_steps += 1
            else:
                batch = memory.sample(batch_size)
                pushed = push_with_stop(learner_queue, batch, stop_event)
                if pushed:
                    train_step += 1
            
            if train_step % logging_steps == 0:
                replay_qsize = replay_queue.qsize()
                learner_qsize = learner_queue.qsize()
                print(f"Memory: train_step={train_step}\treplay_qsize={replay_qsize}\tlearner_qsize={learner_qsize}")


    def train(self):
        model = ConvBasicModule()
        shared_model = model.clone().share_memory_()
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
        