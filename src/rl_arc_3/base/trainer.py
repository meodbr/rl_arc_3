from dataclasses import dataclass

from rl_arc_3.base.clone import Checkpointable

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
    num_episodes: int = 10
    num_workers: int = 1
    max_steps_per_episode: int = 1000
    log_steps: int = 100
    save_steps: int = 5000
    max_steps: int = 10
    max_epochs: float = 1.0
    lr: float = 1e-3
    batch_size: int = 64
    device: str | None = None
    model_adapter: str = "full"



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
