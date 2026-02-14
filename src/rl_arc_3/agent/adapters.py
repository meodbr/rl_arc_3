from typing import Any, Tuple

import torch
import numpy as np
from gymnasium.spaces import Dict, Discrete, Box

from rl_arc_3.base.env import EnvSignature
from rl_arc_3.base.model import ModelSignature

class ModelAdapter:
    def __init__(
        self,
        env_signature: EnvSignature,
        model_signature: ModelSignature | None = None
    ):
        self.env_signature = env_signature
        self.env_act = env_signature.action_space
        self.env_obs = env_signature.observation_space

        computed_m_sig = self.compute_model_signature(env_signature)

        if model_signature is not None and computed_m_sig != model_signature:
            raise ValueError(f"Wrong Model signature: {model_signature} != {computed_m_sig}")

        self.model_signature = computed_m_sig
        self.m_input = self.model_signature.input_shape
        self.m_output = self.model_signature.output_shape
    
    @staticmethod
    def compute_model_signature(env_signature: EnvSignature) -> ModelSignature:
        raise NotImplementedError

    def observation_to_tensor(self, obs: Any, device = None) -> torch.Tensor:
        raise NotImplementedError
    
    def tensor_to_action(self, array: torch.Tensor) -> Any:
        raise NotImplementedError

        
class DiscreteModelAdapter(ModelAdapter):
    @staticmethod
    def compute_model_signature(env_signature: EnvSignature) -> ModelSignature:
        if not isinstance(env_signature.observation_space, Box):
            raise NotImplementedError("Only Box observation spaces are supported")
        if not isinstance(env_signature.action_space, Discrete):
            raise NotImplementedError("Only Discrete action spaces are supported")
        
        return ModelSignature(
            input_shape=env_signature.observation_space.shape,
            output_shape=[env_signature.action_space.n],
        )

    def observation_to_tensor(self, obs: Any, device = None) -> torch.Tensor:
        if device is None:
            device = torch.device("cpu")
        tensor = torch.tensor(obs, device=device, dtype=torch.float32)
        return tensor

    def tensor_to_action(self, array: torch.Tensor) -> Any:
        action_index = torch.argmax(array).item()
        return action_index


class TupleModelAdapter(ModelAdapter):
    def __init__(
        self,
        env_signature: EnvSignature,
        model_signature: ModelSignature | None = None
    ):
        super().__init__(env_signature, model_signature)

        self.key_n = self.env_act.spaces["key"].n
        self.mouse_n = self.env_act.spaces["mouse"].n   

    @staticmethod
    def compute_model_signature(env_signature: EnvSignature) -> ModelSignature:
        if not isinstance(env_signature.observation_space, Box):
            raise NotImplementedError("Only Dict action spaces are supported")
        if not isinstance(env_signature.action_space, Dict):
            raise NotImplementedError("Only Dict action spaces are supported")

        lenghts = {}
        for name, subspace in env_signature.action_space.spaces.items():
            if not isinstance(subspace, Discrete):
                raise NotImplementedError("Only Discrete action subspaces are supported")
            if name not in ["key", "mouse"]:
                raise NotImplementedError(f"Unsupported action subspace name: {name}")
            lenghts[name] = subspace.n
        
        return ModelSignature(
            input_shape=env_signature.observation_space.shape,
            output_shape=[lenghts["key"] + lenghts["mouse"]],
        )


    def observation_to_tensor(self, obs: Any, device = None) -> torch.Tensor:
        if device is None:
            device = torch.device("cpu")
        tensor = torch.tensor(obs, device=device, dtype=torch.float32)
        return tensor

    def tensor_to_action(self, array: torch.Tensor) -> Any:
        key = array[:self.key_n]
        mouse = array[self.key_n:self.key_n + self.mouse_n]
        return {
            "key": torch.argmax(key).item(),
            "mouse": torch.argmax(mouse).item(),
        }
