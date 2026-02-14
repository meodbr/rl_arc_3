from dataclasses import dataclass


class BaseTrainer:
    def train(
        self,
        resume_from_checkpoint: dict | None = None,
    ):
        raise NotImplementedError

    def eval(
        self,
    ):
        raise NotImplementedError


@dataclass
class TrainingArgs:
    num_episodes: int
    num_workers: int
    max_steps_per_episode: int
    log_steps: int
    save_steps: int
    max_steps: int
    max_epochs: float
    lr: float
    batch_size: int
    device: str | None = None
    model_adapter: str = "full"


class OffPolicyTrainingArgs(TrainingArgs):
    train_explore_ratio: int
    target_update_steps: int
    memory_capacity: int


class DQNTrainingArgs(OffPolicyTrainingArgs):
    gamma: float = 0.99
    eps_max: float = 0.9
    eps_min: float = 0.02
    eps_decay: int = 25000
    tau: float = 0.005
