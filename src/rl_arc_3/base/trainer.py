from typing import Any
from dataclasses import dataclass

import pandas as pd

from rl_arc_3.base.checkpointable import Checkpointable
from rl_arc_3.base.utils import compute_run_name

class BaseTrainer(Checkpointable):
    def train(
        self,
        resume_from_checkpoint: dict | None = None,
    ):
        raise NotImplementedError

    def eval(
        self,
    ):
        raise NotImplementedError


@dataclass(kw_only=True)
class TrainingArgs:
    output_dir: str
    num_episodes: int = 10
    num_workers: int = 1
    max_steps_per_episode: int = 1000
    log_steps: int = 100
    save_steps: int = 5000
    max_steps: int = 10
    lr: float = 1e-3
    batch_size: int = 64
    device: str | None = None
    run: str | None = None
    model_adapter: str = "full"
    metric_hub: str = "csv"

    def __post_init__(self):
        self.run = compute_run_name(self.output_dir) if self.run is None else self.run

@dataclass(kw_only=True)
class OffPolicyTrainingArgs(TrainingArgs):
    train_explore_ratio: int = 1
    target_update_steps: int = 1000
    memory_capacity: int = 1000


@dataclass(kw_only=True)
class DQNTrainingArgs(OffPolicyTrainingArgs):
    gamma: float = 0.99
    eps_max: float = 0.9
    eps_min: float = 0.02
    eps_decay: int = 25000
    tau: float = 0.005


class BaseMetricHub:
    """
    Saves and reads metrics, stateless
    """
    def save(self, data: dict, run: str, emitter: Any) -> None:
        raise NotImplementedError
    
    def get(self, run: str) -> pd.DataFrame:
        raise NotImplementedError
    
    def plot(self, run: str, metric: str) -> None:
        raise NotImplementedError