from typing import Any, Tuple, Iterable
import logging

import torch
import torch.nn.functional as F
import numpy as np
from gymnasium.spaces import Dict, Discrete, Box, Space

from rl_arc_3.base.env import EnvSignature
from rl_arc_3.base.model import ModelSignature
from rl_arc_3.base.model_adapter import ModelAdapter

from rl_arc_3.utils.utils import unwrap_if_single
from rl_arc_3.utils.constants import ENV_OBS_CHANNEL_DIM

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
        env_act, _ = self.validate_env_spaces(env_signature)

        self._is_action_env_discrete = isinstance(env_act, Discrete)

        env_obs_info = self._get_env_obs_info(env_signature)
        for key, val in env_obs_info.items():
            setattr(self, f"_env_obs_{key}", val)

        super().__init__(env_signature, model_signature)

    def validate_env_spaces(self, env_signature: EnvSignature) -> Tuple[Space, Space]:
        env_act = env_signature.action_space
        env_obs = env_signature.observation_space

        # Observation space
        if not isinstance(env_obs, Box):
            raise RuntimeError(
                f"Only Box observation spaces are supported, not {type(env_obs)}"
            )

        if len(env_obs.shape) not in [2, 3]:
            raise RuntimeError(
                f"{env_obs.shape} not in supported obs shapes : (H,W) or (H,W,C)"
            )

        # Action space
        if not isinstance(env_act, Dict) and not isinstance(env_act, Discrete):
            raise RuntimeError(
                f"Only Dict and Discrete action spaces are supported, not {type(env_act)}"
            )

        if isinstance(env_act, Dict):
            for name, subspace in env_act.spaces.items():
                if not isinstance(subspace, Discrete):
                    raise RuntimeError(
                        f"Only Discrete action subspaces are supported, not {type(subspace)}"
                    )
                if name not in ["key", "mouse"]:
                    raise RuntimeError(f"Unsupported action subspace name: {name}")

        return env_act, env_obs

    @staticmethod
    def _get_env_obs_info(env_signature: EnvSignature):
        env_obs = env_signature.observation_space

        is_channel_dim_compressed = len(env_obs.shape) == 2
        dtype = env_obs.dtype

        high = env_obs.high.max()
        low = env_obs.low.min()

        is_integer = np.issubdtype(env_obs.dtype, np.integer)
        is_binary = low == 0 and high == 1 if is_integer else False
        should_rescale = is_integer and not is_channel_dim_compressed
        rescale_factor = 1.0 / (high - low) if should_rescale else 1.0

        is_compression_supported = False

        logger.info(
            "Env obs shape=%s dtype=%s, range=(%s, %s)",
            env_obs.shape,
            env_obs.dtype,
            low,
            high,
        )

        if is_channel_dim_compressed:
            assert (
                low == 0
            ), f"Env obs space should have 0 as lowest pixel value, got {low}"
            num_channels = high - low + 1

            uncompressed_shape = (*env_obs.shape, num_channels)
            is_compression_supported = True
        else:
            num_channels = env_obs.shape[ENV_OBS_CHANNEL_DIM]

            uncompressed_shape = env_obs.shape
            is_compression_supported = is_binary

        if num_channels > 32:
            logger.warning(
                "Detected number of input channels: %d, seems too large",
                num_channels,
            )

        model_input_shape = ArcStyleModelAdapter.reorder_channel_dim(uncompressed_shape)

        return {
            "is_channel_dim_compressed": is_channel_dim_compressed,
            "dtype": dtype,
            "high": high,
            "low": low,
            "is_integer": is_integer,
            "is_binary": is_binary,
            "should_rescale": should_rescale,
            "rescale_factor": rescale_factor,
            "is_compression_supported": is_compression_supported,
            "num_channels": num_channels,
            "uncompressed_shape": uncompressed_shape,
            "model_input_shape": model_input_shape,
        }

    @staticmethod
    def reorder_channel_dim(shape: Iterable[int]) -> Tuple:
        chan_dim = ENV_OBS_CHANNEL_DIM
        s = list(shape)
        return tuple([s[chan_dim]] + s[:chan_dim] + s[chan_dim + 1 :])

    def compress_obs(self, t: torch.Tensor, batched: bool = False) -> np.ndarray:
        if not self._env_obs_is_compression_supported:
            return t.numpy()
        assert t.dim() == (
            4 if batched else 3
        ), "B={}, shape={}, shape should be (B, C, H, W) if batched else (C, H, W)".format(
            batched, t.shape()
        )
        return torch.argmax(t, dim=1 if batched else 0).numpy().astype(np.uint8)

    def uncompress_obs(
        self, array: np.ndarray, batched: bool = False, device: str = None
    ) -> torch.Tensor:
        if not self._env_obs_is_compression_supported:
            return torch.tensor(array, device=device).float()

        assert array.ndim == (
            3 if batched else 2
        ), "B={}, shape={}, shape should be (B, H, W) if batched else (H, W)".format(
            batched, array.shape
        )
        t = torch.tensor(array, dtype=torch.long, device=device)
        if batched:
            return (
                F.one_hot(t, num_classes=self._env_obs_num_channels)
                .permute(0, 3, 1, 2)
                .float()
            )  # (B, C, H, W)
        else:
            return (
                F.one_hot(t, num_classes=self._env_obs_num_channels)
                .permute(2, 0, 1)
                .float()
            )  # (C, H, W)

    def read_obs(self, obs: np.ndarray, device: str = None):
        assert ENV_OBS_CHANNEL_DIM == 2, "Channel dim not supported: {}".format(
            ENV_OBS_CHANNEL_DIM
        )

        if not self._env_obs_is_channel_dim_compressed:
            res = torch.tensor(obs.transpose(2, 0, 1), device=device, dtype=torch.float)
            if self._env_obs_should_rescale:
                res = res * self._env_obs_rescale_factor
            return res

        res = torch.tensor(obs, device=device, dtype=torch.long)
        logger.debug(res)
        logger.debug("read_obs res shape: %s", res.shape)
        logger.debug("env_obs_num_channels: %s", self._env_obs_num_channels)
        res = (
            F.one_hot(res, num_classes=self._env_obs_num_channels)
            .permute(2, 0, 1)
            .float()
        )  # (C, H, W)
        return res

    @staticmethod
    def compute_model_signature(env_signature: EnvSignature) -> ModelSignature:
        raise NotImplementedError

    def observation_to_tensor(
        self, obs: Any, compressed: bool = False, device: str = None
    ) -> torch.Tensor | np.ndarray:
        raise NotImplementedError

    def tensor_to_action(self, array: torch.Tensor) -> Any:
        raise NotImplementedError
