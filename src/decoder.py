"""
SegResNet-style Decoder for MS-HVED

Uses RegressionResBlock (normalization-free by default) for regression/SR tasks.
- No GroupNorm on final output head (prevents contrast shifts)
- Hardtanh 'clamp' activation option (gradient=1 in [0,1], unlike sigmoid max 0.25)
- LeakyReLU(0.2) instead of ReLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .blocks import RegressionResBlock, UpsampleBlock


class SegResDecoderBlock(nn.Module):
    """
    Decoder block with upsampling, skip fusion, and RegressionResBlock blocks.

    Uses additive skip connections (after channel alignment) rather than concat.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        upsample_mode: str = 'trilinear',
        num_groups: int = 8,
        use_norm: bool = False,
    ):
        super().__init__()

        # Upsample
        self.upsample = UpsampleBlock(in_channels, out_channels, mode=upsample_mode)

        # Skip projection (align channels for additive fusion)
        self.skip_proj = nn.Identity() if skip_channels == out_channels else nn.Conv3d(skip_channels, out_channels, 1)

        # Residual blocks
        blocks = [RegressionResBlock(out_channels, out_channels, num_groups=num_groups, use_norm=use_norm) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)

        if skip is not None:
            # Match spatial dims if needed
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=False)
            # Additive fusion (SegResNet style)
            x = x + self.skip_proj(skip)

        x = self.blocks(x)
        return x


class SegResDecoder(nn.Module):
    """
    Multi-scale SegResNet-style decoder for MS-HVED.

    Architecture:
        Deepest latent -> [Upsample + Skip + RegressionResBlocks] x (num_scales-1) -> Output

    Final output head: Conv1x1 only (no GroupNorm/ReLU before output).
    """

    def __init__(
        self,
        out_channels: int = 1,
        init_filters: int = 32,
        num_scales: int = 4,
        blocks_per_scale: Tuple[int, ...] = (1, 1, 1),
        upsample_mode: str = 'trilinear',
        num_groups: int = 8,
        final_activation: str = 'clamp',
        use_norm: bool = False,
    ):
        """
        Args:
            out_channels: Output channels
            init_filters: Base filter count (matches encoder)
            num_scales: Number of scales (matches encoder)
            blocks_per_scale: Residual blocks per decoder level
            upsample_mode: 'trilinear' or 'transpose'
            num_groups: Groups for GroupNorm (only used when use_norm=True)
            final_activation: 'clamp' (recommended), 'sigmoid', 'tanh', or 'none'
            use_norm: Whether to use GroupNorm in residual blocks (default False)
        """
        super().__init__()

        self.num_scales = num_scales

        # Pad blocks_per_scale
        if len(blocks_per_scale) < num_scales - 1:
            blocks_per_scale = blocks_per_scale + (blocks_per_scale[-1],) * (num_scales - 1 - len(blocks_per_scale))

        # Channel dims at each scale (matching encoder)
        self.scale_channels = [init_filters * (2 ** i) for i in range(num_scales)]

        # Decoder blocks (coarse -> fine)
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_scales - 1, 0, -1):
            in_ch = self.scale_channels[i]
            skip_ch = self.scale_channels[i - 1]
            out_ch = self.scale_channels[i - 1]
            block_idx = num_scales - 1 - i

            self.decoder_blocks.append(
                SegResDecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    num_blocks=blocks_per_scale[block_idx],
                    upsample_mode=upsample_mode,
                    num_groups=num_groups,
                    use_norm=use_norm,
                )
            )

        # Final output: Conv1x1 only (no GroupNorm/ReLU — they shift intensities)
        self.final_conv = nn.Conv3d(init_filters, out_channels, kernel_size=1)

        if final_activation == 'clamp':
            self.final_activation = nn.Hardtanh(min_val=0.0, max_val=1.0)
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_activation == 'tanh':
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Identity()

    def forward(
        self,
        latent_samples: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            latent_samples: Multi-scale latents [scale_0, scale_1, ..., scale_N] (fine to coarse)

        Returns:
            Decoded output
        """
        # Start from deepest (coarsest) latent
        x = latent_samples[-1]

        for i, block in enumerate(self.decoder_blocks):
            scale_idx = self.num_scales - 2 - i

            # Get skip from latent samples
            skip = latent_samples[scale_idx] if scale_idx < len(latent_samples) else None

            x = block(x, skip)

        # Final output (no norm/relu before conv)
        x = self.final_conv(x)
        x = self.final_activation(x)

        return x


class MultiOutputSegResDecoder(nn.Module):
    """
    Decoder producing multiple outputs:
    - Main output (e.g., super-resolved image)
    - Orientation reconstructions (for ELBO loss)
    """

    def __init__(
        self,
        num_orientations: int = 4,
        out_channels: int = 1,
        init_filters: int = 32,
        num_scales: int = 4,
        blocks_per_scale: Tuple[int, ...] = (1, 1, 1),
        upsample_mode: str = 'trilinear',
        num_groups: int = 8,
        share_decoder: bool = False,
        final_activation: str = 'clamp',
        use_norm: bool = False,
    ):
        super().__init__()

        self.num_orientations = num_orientations
        self.share_decoder = share_decoder

        decoder_kwargs = dict(
            out_channels=out_channels,
            init_filters=init_filters,
            num_scales=num_scales,
            blocks_per_scale=blocks_per_scale,
            upsample_mode=upsample_mode,
            num_groups=num_groups,
            final_activation=final_activation,
            use_norm=use_norm,
        )

        # Main decoder
        self.main_decoder = SegResDecoder(**decoder_kwargs)

        # Orientation decoders
        if share_decoder:
            self.orientation_decoder = SegResDecoder(**decoder_kwargs)
        else:
            self.orientation_decoders = nn.ModuleList([
                SegResDecoder(**decoder_kwargs)
                for _ in range(num_orientations)
            ])

    def forward(
        self,
        latent_samples: List[torch.Tensor],
        reconstruct_orientations: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            main_output: Main decoded output
            orientation_outputs: List of reconstructed orientations
        """
        main_output = self.main_decoder(latent_samples)

        orientation_outputs = []
        if reconstruct_orientations:
            for i in range(self.num_orientations):
                decoder = self.orientation_decoder if self.share_decoder else self.orientation_decoders[i]
                orientation_outputs.append(decoder(latent_samples))

        return main_output, orientation_outputs
