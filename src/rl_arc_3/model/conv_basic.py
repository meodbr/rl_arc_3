import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_arc_3.base.model import BaseModel, ModelSignature
from rl_arc_3.base.checkpointable import Checkpointable

logger = logging.getLogger(__name__)


class ConvBasicModule(BaseModel):
    """
    Basic Conv2D module
    """

    def __init__(self, signature: ModelSignature, **kwargs):
        super().__init__(signature=signature, **kwargs)

        assert (
            len(signature.input_shape) == 3
        ), "Input shape must be (num_channels, height, width), got : {}".format(signature.input_shape)
        assert (
            len(signature.output_shape) == 1
        ), "Only 1D output is supported, got : {}".format(signature.output_shape)

        input_shape = signature.input_shape
        output_size = signature.output_shape[0]

        self._signature = signature
        self.flattened_size = (
            (input_shape[1] // 8) * (input_shape[2] // 8) * 32
        )  # After 3 poolings

        # Layers
        self.layer1 = nn.Conv2d(input_shape[0], 16, kernel_size=5, stride=1, padding=2)
        self.layer2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.layer3 = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug(f"Input to model: {x.shape}") # Should be (B, C, H, W)
        # x = (
        #     F.one_hot(x, num_classes=self.NUM_COLORS).permute(0, 3, 1, 2).float()
        # )  # (B, C, H, W)
        x = self.pool(F.relu(self.layer1(x)))
        x = self.pool(F.relu(self.layer2(x)))
        x = self.pool(F.relu(self.layer3(x)))

        x = x.reshape(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    @property
    def signature(self) -> ModelSignature:
        return self._signature

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(x, x_hat)
