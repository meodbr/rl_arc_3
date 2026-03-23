from typing import Any, Tuple
import logging

import torch
import numpy as np
from gymnasium.spaces import Dict, Discrete, Box, Space

from rl_arc_3.base.env import EnvSignature
from rl_arc_3.base.model import ModelSignature
from rl_arc_3.base.model_adapter import ModelAdapter

from rl_arc_3.utils.utils import unwrap_if_single

logger = logging.getLogger(__name__)


class ArcStyleModelAdapter(ModelAdapter):
    """
    Adapter for "arc style" environments:
        - With Dict action spaces containing "key" and "mouse" Discrete subspaces.
        - Or with Discrete action spaces, i.e. no mouse (ex: Atari envs)
    """

    def __init__(
        self, env_signature: EnvSignature, model_signature: ModelSignature | None = None
    ):
        env_act, env_obs = self.validate_env_spaces(env_signature)

        self._is_action_env_discrete = isinstance(env_act, Discrete)

        self._is_env_channel_dim_compressed = len(env_obs.shape) == 2
        if self._is_env_channel_dim_compressed:
            low = env_obs.low[0]
            high = env_obs.high[0]
            assert low == 0, f"Env obs space should have 0 as lowest pixel value, got {low}"
            self._env_num_channels = high - low
        else:
            self._env_num_channels = env_obs.shape[2]
        
        if self._env_num_channels > 32:
            logger.warning("Detected number of input channels: %d, seems too large", self._env_num_channels)

        super().__init__(env_signature, model_signature)
    
    def validate_env_spaces(self, env_signature: EnvSignature) -> Tuple[Space, Space]:
        env_act = env_signature.action_space
        env_obs = env_signature.observation_space

        # Observation space
        if not isinstance(env_obs, Box):
            raise RuntimeError(f"Only Box observation spaces are supported, not {type(env_obs)}")

        if len(env_obs.shape) not in [2, 3]:
            raise RuntimeError(f"{env_obs.shape} not in supported obs shapes : (H,W) or (H,W,C)")

        # Action space
        if not isinstance(env_act, Dict) and not isinstance(env_act, Discrete):
            raise RuntimeError(f"Only Dict and Discrete action spaces are supported, not {type(env_act)}")
        
        if isinstance(env_act, Dict):
            for name, subspace in env_act.spaces.items():
                if not isinstance(subspace, Discrete):
                    raise RuntimeError(
                        f"Only Discrete action subspaces are supported, not {type(subspace)}"
                    )
                if name not in ["key", "mouse"]:
                    raise RuntimeError(f"Unsupported action subspace name: {name}")

        return env_act, env_obs

    def compress(self, t: torch.Tensor) -> np.ndarray:
        raise NotImplementedError #TODO: Implement compression
        # assert t.dim() == 3
        # return torch.argmax(t, dim=1)

    def uncompress(self, t: np.ndarray) -> torch.Tensor:
        raise NotImplementedError #TODO: Implement compression
    
    @staticmethod
    def compute_model_signature(env_signature: EnvSignature) -> ModelSignature:
        raise NotImplementedError

    def observation_to_tensor(self, obs: Any, compressed: bool = False, device: str = None) -> torch.Tensor:
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


class KeyboardOnlyModelAdapter(ArcStyleModelAdapter):
    """
    This adapter does not use the "mouse" subspace (always sets it to 0) and only outputs actions for the "key" subspace.
    """

    @staticmethod
    def compute_model_signature(env_signature: EnvSignature) -> ModelSignature:
        if isinstance(env_signature.action_space, Discrete):
            return ModelSignature(
                input_shape=env_signature.observation_space.shape,
                output_shape=[env_signature.action_space.n]
            )
        else:
            return ModelSignature(
                input_shape=env_signature.observation_space.shape,
                output_shape=[env_signature.action_space.spaces["key"].n],
            )

    def observation_to_tensor(self, obs: Any, compressed: bool = False, device: str = None) -> torch.Tensor:
        # TODO: Implement compression
        if device is None:
            device = torch.device("cpu")

        tensor = torch.tensor(obs, device=device, dtype=torch.long)

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

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
        if isinstance(env_signature.action_space, Discrete):
            return ModelSignature(
                input_shape=env_signature.observation_space.shape,
                output_shape=[env_signature.action_space.n]
            )

        lenghts = {}
        for name, subspace in env_signature.action_space.spaces.items():
            lenghts[name] = subspace.n

        return ModelSignature(
            input_shape=env_signature.observation_space.shape,
            output_shape=[lenghts["key"] - 1 + lenghts["mouse"]], # -1 because mouse action is represented as 1 action for each pixel
        )

    def observation_to_tensor(self, obs: Any, compressed: bool = False, device: str = None) -> torch.Tensor:
        # TODO: Implement compression
        if device is None:
            device = torch.device("cpu")

        tensor = torch.tensor(obs, device=device, dtype=torch.long)

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    def tensor_to_action(self, x: torch.Tensor) -> Any:
        action_list = x.tolist()

        if self._is_action_env_discrete:
            return unwrap_if_single(action_list)

        num_key_actions = self.key_n - 1  # Exclude the "mouse" action
        key_action = [
            a if a < num_key_actions else self.mouse_action_id
            for a in action_list
        ]
        mouse_action = [
            a - num_key_actions if a >= self.key_n else 0
            for a in action_list
        ]

        actions = [
            {
                "key": k_a,
                "mouse": m_a,
            }
            for k_a, m_a in zip(key_action, mouse_action)
        ]

        return unwrap_if_single(actions)
