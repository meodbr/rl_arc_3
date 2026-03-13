from typing import Any, Tuple
import logging

import torch
import numpy as np

from rl_arc_3.base.env import EnvSignature
from rl_arc_3.base.model import ModelSignature

logger = logging.getLogger(__name__)


class ModelAdapter:
    def __init__(
        self, env_signature: EnvSignature, model_signature: ModelSignature | None = None
    ):
        self.env_signature = env_signature
        self.env_act = env_signature.action_space
        self.env_obs = env_signature.observation_space

        computed_m_sig = self.compute_model_signature(env_signature)

        if model_signature is not None and computed_m_sig != model_signature:
            raise ValueError(
                f"Wrong Model signature: {model_signature} != {computed_m_sig}"
            )

        logger.debug(
            "Associating env signature: %s, with model signature: %s",
            env_signature,
            computed_m_sig,
        )

        self.model_signature = computed_m_sig
        self.m_input = self.model_signature.input_shape
        self.m_output = self.model_signature.output_shape

    @staticmethod
    def compute_model_signature(env_signature: EnvSignature) -> ModelSignature:
        raise NotImplementedError

    def observation_to_tensor(self, obs: Any, compressed: bool = False, device: str = None) -> torch.Tensor:
        raise NotImplementedError

    def tensor_to_action(self, array: torch.Tensor) -> Any:
        raise NotImplementedError

    def compress(self, t: torch.Tensor) -> np.ndarray:
        raise NotImplementedError

    def uncompress(self, t: np.ndarray) -> torch.Tensor:
        raise NotImplementedError
    