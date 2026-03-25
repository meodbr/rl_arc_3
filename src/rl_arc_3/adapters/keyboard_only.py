from typing import Any, Tuple, Iterable
import logging

import torch
import numpy as np
from gymnasium.spaces import Dict, Discrete, Box, Space

from rl_arc_3.base.env import EnvSignature
from rl_arc_3.base.model import ModelSignature
from rl_arc_3.base.model_adapter import ModelAdapter

from rl_arc_3.adapters.arc_style import ArcStyleModelAdapter
from rl_arc_3.utils.utils import unwrap_if_single

logger = logging.getLogger(__name__)


class KeyboardOnlyModelAdapter(ArcStyleModelAdapter):
    """
    This adapter does not use the "mouse" subspace (always sets it to 0) and only outputs actions for the "key" subspace.
    """

    @staticmethod
    def compute_model_signature(env_signature: EnvSignature) -> ModelSignature:
        env_obs_info = ArcStyleModelAdapter._get_env_obs_info(env_signature)
        input_shape = env_obs_info["model_input_shape"]
        if isinstance(env_signature.action_space, Discrete):
            return ModelSignature(
                input_shape=input_shape,
                output_shape=[env_signature.action_space.n],
            )
        else:
            return ModelSignature(
                input_shape=input_shape,
                output_shape=[env_signature.action_space.spaces["key"].n],
            )

    def observation_to_tensor(
        self, obs: np.ndarray, device: str = None
    ) -> torch.Tensor:
        return self.read_obs(obs, device=device)

    def tensor_to_action(self, x: torch.Tensor) -> Any:
        key_action = x.tolist()

        if self._is_action_env_discrete:
            return unwrap_if_single(key_action)

        actions = [
            {
                "key": k_a,
                "mouse": 0,  # Mouse action not used in this adapter
            }
            for k_a in key_action
        ]

        return unwrap_if_single(actions)
