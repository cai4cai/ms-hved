"""
Encoder for MS-HVED

Uses RegressionResBlock for regression/SR tasks.
- LeakyReLU(0.2) instead of ReLU (avoids dead neurons)
- Residual scaling for training stability
- Spectral regularization for kl wieght norm and kl stability
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from torch.nn.utils.parametrizations import spectral_norm

from .blocks import RegressionResBlock


class EncoderBlock(nn.Module):
    """
    Encoder block with RegressionResBlock residual blocks.
    Outputs variational parameters (mu, logvar).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        downsample: bool = True,
    ):
        super().__init__()

        # First block handles channel change and optional downsampling
        stride = 2 if downsample else 1
        blocks = [RegressionResBlock(in_channels, out_channels, stride=stride)]

        # Additional blocks at same resolution
        for _ in range(num_blocks - 1):
            blocks.append(RegressionResBlock(out_channels, out_channels))

        self.blocks = nn.Sequential(*blocks)

        # Variational projection: features -> (mu, logvar)
        self.variational_proj = spectral_norm(nn.Conv3d(out_channels, out_channels * 2, kernel_size=1))
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            features: For skip connections
            mu: Variational mean
            logvar: Variational log-variance (clamped to [-5, 5])
        """
        features = self.blocks(x)
        params = self.variational_proj(features)

        mu = params[:, :self.out_channels]
        logvar = torch.clamp(params[:, self.out_channels:], -5.0, 5.0)

        return features, mu, logvar


class Encoder(nn.Module):
    """
    Multi-scale encoder for MS-HVED.

    Architecture:
        Input -> InitConv -> [RegressionResBlock x n] per scale with downsampling
        Each scale outputs (features, mu, logvar)
    """

    def __init__(
        self,
        in_channels: int = 1,
        init_filters: int = 32,
        num_scales: int = 4,
        blocks_per_scale: Tuple[int, ...] = (1, 2, 2, 4),
    ):
        super().__init__()

        self.num_scales = num_scales

        # Pad blocks_per_scale if needed
        if len(blocks_per_scale) < num_scales:
            blocks_per_scale = blocks_per_scale + (blocks_per_scale[-1],) * (num_scales - len(blocks_per_scale))

        # Initial convolution (no downsampling)
        self.init_conv = spectral_norm(nn.Conv3d(in_channels, init_filters, kernel_size=3, padding=1))

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        in_ch = init_filters

        for i in range(num_scales):
            out_ch = init_filters * (2 ** i)
            downsample = (i > 0)  # First scale stays at input resolution

            self.encoder_blocks.append(
                EncoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_blocks=blocks_per_scale[i],
                    downsample=downsample,
                )
            )
            in_ch = out_ch

        self.hidden_dims = [init_filters * (2 ** i) for i in range(num_scales)]

    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Returns:
            List of dicts per scale with 'features', 'mu', 'logvar'
        """
        h = self.init_conv(x)

        outputs = []
        for block in self.encoder_blocks:
            features, mu, logvar = block(h)
            outputs.append({'features': features, 'mu': mu, 'logvar': logvar})
            h = features

        return outputs


class MultiModalEncoder(nn.Module):
    """
    Multi-orientation encoder using RegressionResBlock blocks.
    Each orientation encoded independently, then fused via Product of Gaussians.
    """

    def __init__(
        self,
        num_orientations: int = 4,
        in_channels: int = 1,
        init_filters: int = 32,
        num_scales: int = 4,
        blocks_per_scale: Tuple[int, ...] = (1, 2, 2, 4),
        share_weights: bool = False,
    ):
        super().__init__()

        self.num_orientations = num_orientations
        self.share_weights = share_weights
        self.num_scales = num_scales

        if share_weights:
            self.encoder = Encoder(
                in_channels, init_filters, num_scales, blocks_per_scale
            )
            self.hidden_dims = self.encoder.hidden_dims
        else:
            self.encoders = nn.ModuleList([
                Encoder(in_channels, init_filters, num_scales, blocks_per_scale)
                for _ in range(num_orientations)
            ])
            self.hidden_dims = self.encoders[0].hidden_dims

    def forward(
        self,
        orientations: List[torch.Tensor],
        orientation_mask: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Dict[int, torch.Tensor]]]:
        """
        Encode all orientations.

        Returns:
            List (per scale) of dicts with 'mu', 'logvar', 'features'
            each mapping orientation_idx -> tensor
        """
        scale_outputs = [
            {'mu': {}, 'logvar': {}, 'features': {}}
            for _ in range(self.num_scales)
        ]

        mask_is_batched = orientation_mask is not None and orientation_mask.dim() == 2

        for idx, tensor in enumerate(orientations):
            # Skip if globally masked out
            if orientation_mask is not None and not mask_is_batched:
                if not orientation_mask[idx]:
                    continue

            encoder = self.encoder if self.share_weights else self.encoders[idx]
            enc_out = encoder(tensor)

            for scale_idx, scale_data in enumerate(enc_out):
                scale_outputs[scale_idx]['mu'][idx] = scale_data['mu']
                scale_outputs[scale_idx]['logvar'][idx] = scale_data['logvar']
                scale_outputs[scale_idx]['features'][idx] = scale_data['features']

        return scale_outputs
