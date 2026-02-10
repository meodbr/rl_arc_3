from rl_arc_3.trainer.offpolicy import OffPolicyTrainer, OffPolicyTrainingArgs


class DQNTrainingArgs(OffPolicyTrainingArgs):
    gamma: float = 0.99
    eps_max: float = 0.9
    eps_min: float = 0.02
    eps_decay: int = 25000
    tau: float = 0.005


class DQNTrainer(OffPolicyTrainer):
    pass
