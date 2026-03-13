import logging
import sys
from functools import partial

import torch.multiprocessing as mp

from rl_arc_3.base.trainer import DQNTrainingArgs

from rl_arc_3.trainer.dqn_trainer import DQNTrainer
from rl_arc_3.env.arc import ArcEnv
from rl_arc_3.env.gym import AtariEnv
from rl_arc_3.utils.utils import setup_logging


def main():
    training_args = DQNTrainingArgs(
        output_dir="data/dqn_draft",
        num_episodes=10,
        num_workers=5,
        max_steps=80,
        max_steps_per_episode=1000,
        memory_capacity=1000,
        target_update_steps=200,
        log_steps=5,
        save_steps=100,
        # device="cpu",
    )

    trainer = DQNTrainer(
        training_args=training_args,
        env_factory=partial(AtariEnv, game="Pong"),
        # env_factory=partial(ArcEnv, game="ls20", render_mode=None),
        # env_factory=partial(ArcEnv, game="ls20", render_mode="terminal-fast"),
        # env_factory=partial(ArcEnv, game="ft09", render_mode="terminal-fast"),
    )
    if len(sys.argv) > 1:
        trainer.train(resume_from_checkpoint=sys.argv[1])
    else:
        trainer.train()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    setup_logging()
    main()
