from typing import Callable, Tuple

from rl_arc_3.trainer.offpolicy import OffPolicyTrainer
from rl_arc_3.base.trainer import DQNTrainingArgs
from rl_arc_3.base.model import ModelSignature, BaseModel
from rl_arc_3.base.env import BaseEnv, EnvSignature

from rl_arc_3.agent.adapters import ModelAdapter, FullModelAdapter, KeyboardOnlyModelAdapter
from rl_arc_3.agent.dqn_actor import DQNActor
from rl_arc_3.agent.dqn_learner import DQNLearner
from rl_arc_3.model.conv_basic import ConvBasicModule
from rl_arc_3.model.memory import TensorMemory


class DQNTrainer(OffPolicyTrainer):
    def __init__(
        self,
        training_args: DQNTrainingArgs,
        env_factory: Callable[[], BaseEnv],
        model: BaseModel | None = None,
    ):
        model_sig = model.signature if model is not None else None
        model_adapter = self._get_model_adapter(training_args, model_sig)

        if model is None:
            model = ConvBasicModule(model_adapter.model_signature)

        actor = DQNActor(training_args, model_adapter)
        learner = DQNLearner(training_args, model, model_adapter) # TODO: pass dict

        super().__init__(
            training_args=training_args,
            env_factory=env_factory,
            model=model,
            actor=actor,
            learner=learner,
            memory_factory=None, # TODO
        )

    @staticmethod
    def _get_model_adapter(training_args: DQNTrainingArgs, model_sig: ModelSignature | None = None) -> ModelAdapter:
        if training_args.env_adapter == "full":
            return FullModelAdapter(model_sig)
        elif training_args.env_adapter == "keyboard_only":
            return KeyboardOnlyModelAdapter(model_sig)
        else:
            raise ValueError(f"Unknown env_adapter: {training_args.env_adapter}")