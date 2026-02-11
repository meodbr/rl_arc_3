from rl_arc_3.base.trainer import Trainer, TrainingArgs


def main():
    training_args = TrainingArgs(
        num_episodes=100000,
        max_steps_per_episode=1000,
        memory_capacity=100000,
        target_update_frequency=1000,
        log_interval=100,
        save_interval=5000,
        plot_interval=500,
        device=None,
    )

    trainer = Trainer(training_args)
    trainer.train()
    # Optionnaly
    # trainer.train(resume_from_checkpoint="path/to/checkpoint.pth")

    trainer.eval()


if __name__ == "__main__":
    main()
