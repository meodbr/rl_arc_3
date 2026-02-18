import logging
from functools import partial

from rl_arc_3.base.trainer import DQNTrainingArgs

from rl_arc_3.trainer.dqn_trainer import DQNTrainer
from rl_arc_3.env.fake import FakeEnv
from rl_arc_3.utils.utils import setup_logging


def main():
    training_args = DQNTrainingArgs(
        num_episodes=10,
        max_steps_per_episode=1000,
        memory_capacity=1000,
        target_update_steps=1000,
        log_steps=5,
        save_steps=5000,
        device="cpu",
    )

    trainer = DQNTrainer(
        training_args=training_args,
        env_factory=partial(FakeEnv, game="fake_game"),
    )
    trainer.train()


if __name__ == "__main__":
    setup_logging()
    main()
