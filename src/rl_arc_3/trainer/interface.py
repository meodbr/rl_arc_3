from dataclasses import dataclass


@dataclass
class TrainingArgs:
    num_episodes: int
    num_workers: int
    max_steps_per_episode: int
    log_steps: int
    save_steps: int
    max_steps: int
    lr: float = 1e-3
    batch_size: int = 128
    device: str | None = None


class TrainerInterface:
    def train(
        self,
        resume_from_checkpoint: dict | None = None,
    ):
        raise NotImplementedError

    def eval(
        self,
    ):
        raise NotImplementedError
