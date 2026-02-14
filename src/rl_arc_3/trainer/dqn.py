from pyexpat import model
from typing import Callable, Tuple

from rl_arc_3.trainer.offpolicy import OffPolicyTrainer, OffPolicyTrainingArgs
from rl_arc_3.base.model import ModelSignature
from rl_arc_3.base.env import BaseEnv, EnvSignature

from rl_arc_3.agent.dqn_actor import DQNActor
from rl_arc_3.model.conv_basic import ConvBasicModel
from rl_arc_3.model.memory import TensorMemory


class DQNTrainingArgs(OffPolicyTrainingArgs):
    gamma: float = 0.99
    eps_max: float = 0.9
    eps_min: float = 0.02
    eps_decay: int = 25000
    tau: float = 0.005

def get_model_sig_for_env(env_signature: EnvSignature) -> ModelSignature:
    return ModelSignature(
        state_shape=env_signature.observation_space.shape,
        action_shape=env_signature.action_space.n,
    )

class DQNTrainer(OffPolicyTrainer):
    def __init__(
        self,
        env_factory: Callable[[], BaseEnv],
        training_args: DQNTrainingArgs,
    ):
        model = ConvBasicModel(get_model_sig_for_env(env_factory().signature))
        actor = DQNActor(model)
        learner = DQNLearner(model, training_args)
        super().__init__(model, env_factory, actor, learner, training_args)
