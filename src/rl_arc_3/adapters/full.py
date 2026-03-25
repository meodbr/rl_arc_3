from typing import Any, Tuple, Iterable
import logging

import torch
import numpy as np
from gymnasium.spaces import Dict, Discrete

from rl_arc_3.base.env import EnvSignature
from rl_arc_3.base.model import ModelSignature

from rl_arc_3.adapters.arc_style import ArcStyleModelAdapter
from rl_arc_3.utils.utils import unwrap_if_single

logger = logging.getLogger(__name__)


class FullModelAdapter(ArcStyleModelAdapter):
    """
    This adapter uses the mouse action by splitting the mouse action into 1 action for each pixel
    """

    def __init__(
        self, env_signature: EnvSignature, model_signature: ModelSignature | None = None
    ):
        super().__init__(env_signature, model_signature)

        if not self._is_action_env_discrete:
            self.key_n = self.env_act.spaces["key"].n
            self.mouse_n = self.env_act.spaces["mouse"].n
            self.mouse_action_id = self.env_act.spaces["key"].n - 1

    @staticmethod
    def compute_model_signature(env_signature: EnvSignature) -> ModelSignature:
        env_obs_info = ArcStyleModelAdapter._get_env_obs_info(env_signature)
        input_shape = env_obs_info["model_input_shape"]
        env_act = env_signature.action_space
        if isinstance(env_act, Discrete):
            return ModelSignature(
                input_shape=input_shape,
                output_shape=[env_act.n],
            )

        lenghts = {}
        for name, subspace in env_act.spaces.items():
            lenghts[name] = subspace.n

        return ModelSignature(
            input_shape=input_shape,
            output_shape=[
                lenghts["key"] - 1 + lenghts["mouse"]
            ],  # -1 because mouse action is represented as 1 action for each pixel
        )

    def observation_to_tensor(
        self, obs: np.ndarray, device: str = None
    ) -> torch.Tensor:
        return self.read_obs(obs, device=device)

    def tensor_to_action(self, x: torch.Tensor) -> Any:
        action_list = x.tolist()

        if self._is_action_env_discrete:
            return unwrap_if_single(action_list)

        num_key_actions = self.key_n - 1  # Exclude the "mouse" action
        key_action = [
            a if a < num_key_actions else self.mouse_action_id for a in action_list
        ]
        mouse_action = [
            a - num_key_actions if a >= self.key_n else 0 for a in action_list
        ]

        actions = [
            {
                "key": k_a,
                "mouse": m_a,
            }
            for k_a, m_a in zip(key_action, mouse_action)
        ]

        return unwrap_if_single(actions)
