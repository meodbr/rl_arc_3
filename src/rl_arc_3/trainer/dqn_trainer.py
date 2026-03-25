from typing import Callable, Tuple

from rl_arc_3.trainer.offpolicy import OffPolicyTrainer
from rl_arc_3.base.trainer import DQNTrainingArgs
from rl_arc_3.base.model import ModelSignature, BaseModel
from rl_arc_3.base.env import BaseEnv, EnvSignature
from rl_arc_3.base.model_adapter import ModelAdapter

from rl_arc_3.adapters.utils import get_model_adapter
from rl_arc_3.agent.dqn_actor import DQNActor
from rl_arc_3.agent.dqn_learner import DQNLearner
from rl_arc_3.model.conv_basic import ConvBasicModule
from rl_arc_3.model.memory import TensorMemory, DequeMemory, DequeNumpyMemory


class DQNTrainer(OffPolicyTrainer):
    def __init__(
        self,
        training_args: DQNTrainingArgs,
        env_factory: Callable[[], BaseEnv],
        model: BaseModel | None = None,
        **kwargs,
    ):
        super().__init__(
            training_args=training_args,
            env_factory=env_factory,
            **kwargs,
        )

        model_sig: ModelSignature = model.signature if model is not None else None
        model_adapter: ModelAdapter = get_model_adapter(training_args.model_adapter, env_factory().signature(), model_sig)

        if model is None:
            model = ConvBasicModule(model_adapter.model_signature)


        actor = DQNActor(training_args, model_adapter)
        learner = DQNLearner(training_args, model, model_adapter)
        memory = DequeNumpyMemory(training_args.memory_capacity)

        self.actors_states = [actor.state_dict() for _ in range(training_args.num_workers)]
        self.learner_state = learner.state_dict()
        self.memory_state = memory.state_dict()
