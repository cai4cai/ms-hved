"""
Shared building blocks for MS-HVED encoder and decoder.

RegressionResBlock: Residual block optimized for regression/SR tasks.
Following EDSR/SRResNet/ESRGAN design principles:
- LeakyReLU(0.2) instead of ReLU (avoids dead neurons)
- Residual scaling (stabilizes deep networks)
And NVAE:
- Spectral regularization for unbounded KL divergences
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

class RegressionResBlock(nn.Module):
    """
    Residual block for regression/super-resolution.

    Structure: Conv3d -> LeakyReLU -> Conv3d -> * residual_scale -> + skip

    Key design choices:
    - LeakyReLU(0.2, inplace=False) avoids dead neurons and gradient corruption
    - Residual scaling (default 0.2) prevents instability in deep networks (EDSR)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        residual_scale: float = 0.2,
    ):
        super().__init__()

        padding = kernel_size // 2
        self.residual_scale = residual_scale

        # First conv path
        layers1 = []
        layers1.append(spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)))
        layers1.append(nn.LeakyReLU(0.2, inplace=False))
        self.path1 = nn.Sequential(*layers1)

        # Second conv path
        layers2 = []
        layers2.append(spectral_norm(nn.Conv3d(out_channels, out_channels, kernel_size, 1, padding)))
        self.path2 = nn.Sequential(*layers2)

        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = spectral_norm(nn.Conv3d(in_channels, out_channels, 1, stride))
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.path1(x)
        out = self.path2(out)
        return identity + out * self.residual_scale


class UpsampleBlock(nn.Module):
    """
    Upsampling block with trilinear interpolation or transposed convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        mode: str = 'trilinear',
    ):
        super().__init__()

        self.mode = mode
        self.scale_factor = scale_factor

        if mode == 'trilinear':
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=False)
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        elif mode == 'transpose':
            self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = nn.Identity()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        return x
