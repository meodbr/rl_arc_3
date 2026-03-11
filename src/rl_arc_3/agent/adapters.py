from typing import Any, Tuple
import logging

import torch
import numpy as np
from gymnasium.spaces import Dict, Discrete, Box

from rl_arc_3.base.env import EnvSignature
from rl_arc_3.base.model import ModelSignature
from rl_arc_3.utils.constants import MOUSE_ACTION_ID

logger = logging.getLogger(__name__)


class ModelAdapter:
    def __init__(
        self, env_signature: EnvSignature, model_signature: ModelSignature | None = None
    ):
        self.env_signature = env_signature
        self.env_act = env_signature.action_space
        self.env_obs = env_signature.observation_space
        self._is_action_env_discrete = isinstance(env_signature.action_space, Discrete)

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

    def observation_to_tensor(self, obs: Any, device=None) -> torch.Tensor:
        raise NotImplementedError

    def tensor_to_action(self, array: torch.Tensor) -> Any:
        raise NotImplementedError


def get_model_adapter(
    name: str,
    env_signature: EnvSignature,
    model_signature: ModelSignature | None = None,
) -> ModelAdapter:
    if name == "full":
        return FullModelAdapter(env_signature, model_signature)
    elif name == "keyboard_only":
        return KeyboardOnlyModelAdapter(env_signature, model_signature)
    else:
        raise ValueError(f"Unknown model adapter name: {name}")


class KeyboardOnlyModelAdapter(ModelAdapter):
    """
    Adapter for environments with Dict action spaces containing "key" and "mouse" Discrete subspaces.
    This adapter does not use the "mouse" subspace (always sets it to 0) and only outputs actions for the "key" subspace.
    """

    @staticmethod
    def compute_model_signature(env_signature: EnvSignature) -> ModelSignature:
        if not isinstance(env_signature.observation_space, Box):
            raise NotImplementedError("Only Box observation spaces are supported")

        if not isinstance(env_signature.action_space, Dict) and not isinstance(
            env_signature.action_space, Discrete
        ):
            raise NotImplementedError(
                "Only Dict or Discrete action spaces are supported"
            )
        
        if isinstance(env_signature.action_space, Discrete):
            return ModelSignature(
                input_shape=env_signature.observation_space.shape,
                output_shape=[env_signature.action_space.n]
            )

        for name, subspace in env_signature.action_space.spaces.items():
            if not isinstance(subspace, Discrete):
                raise NotImplementedError(
                    "Only Discrete action subspaces are supported"
                )
            if name not in ["key", "mouse"]:
                raise NotImplementedError(f"Unsupported action subspace name: {name}")

        return ModelSignature(
            input_shape=env_signature.observation_space.shape,
            output_shape=[env_signature.action_space.spaces["key"].n],
        )

    def observation_to_tensor(self, obs: Any, device=None) -> torch.Tensor:
        if device is None:
            device = torch.device("cpu")

        tensor = torch.tensor(obs, device=device, dtype=torch.long)

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    def tensor_to_action(self, x: torch.Tensor) -> Any:
        key_action = x.tolist()

        if self._is_action_env_discrete:
            return key_action

        actions = [
            {
                "key": k_a,
                "mouse": 0,  # Mouse action not used in this adapter
            }
            for k_a in key_action
        ]

        if len(actions) == 1:
            return actions[0]
        else:
            return actions


class FullModelAdapter(ModelAdapter):
    def __init__(
        self, env_signature: EnvSignature, model_signature: ModelSignature | None = None
    ):
        super().__init__(env_signature, model_signature)

        if self._is_action_env_discrete:
            self.key_n = self.env_act.spaces["key"].n
            self.mouse_n = self.env_act.spaces["mouse"].n
            self.mouse_action_id = self.env_act.spaces["key"].n - 1

    @staticmethod
    def compute_model_signature(env_signature: EnvSignature) -> ModelSignature:
        if not isinstance(env_signature.observation_space, Box):
            raise NotImplementedError("Only Box observation spaces are supported")

        if not isinstance(env_signature.action_space, Dict) or isinstance(env_signature.action_space, Discrete):
            raise NotImplementedError("Only Dict and Discrete action spaces are supported")
        
        if isinstance(env_signature.action_space, Discrete):
            return ModelSignature(
                input_shape=env_signature.observation_space.shape,
                output_shape=env_signature.action_space.n
            )

        lenghts = {}
        for name, subspace in env_signature.action_space.spaces.items():
            if not isinstance(subspace, Discrete):
                raise NotImplementedError(
                    "Only Discrete action subspaces are supported"
                )
            if name not in ["key", "mouse"]:
                raise NotImplementedError(f"Unsupported action subspace name: {name}")
            lenghts[name] = subspace.n

        return ModelSignature(
            input_shape=env_signature.observation_space.shape,
            output_shape=[lenghts["key"] - 1 + lenghts["mouse"]],
        )

    def observation_to_tensor(self, obs: Any, device=None) -> torch.Tensor:
        if device is None:
            device = torch.device("cpu")

        tensor = torch.tensor(obs, device=device, dtype=torch.long)

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    def tensor_to_action(self, x: torch.Tensor) -> Any:
        action = x.tolist()
        num_key_actions = self.key_n - 1  # Exclude the "mouse" action
        key_action = [
            a if a < num_key_actions else self.mouse_action_id
            for a in action
        ]
        mouse_action = [
            a - num_key_actions if a >= self.key_n else 0
            for a in action
        ]

        actions = [
            {
                "key": k_a,
                "mouse": m_a,
            }
            for k_a, m_a in zip(key_action, mouse_action)
        ]

        if len(actions) == 1:
            return actions[0]
        else:
            return actions
