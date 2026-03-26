"""
MS-HVED with Regression-Optimized Architecture

Changes from original SegResNet-style:
- RegressionResBlock: no GroupNorm, LeakyReLU(0.2), residual scaling
- Kaiming weight initialization with small variational projection init
- Global residual learning (network predicts residual, not full output)
- Hardtanh 'clamp' activation (gradient=1 in [0,1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union

from .encoder import MultiModalSegResEncoder
from .decoder import SegResDecoder, MultiOutputSegResDecoder
from .fusion import MultiScaleFusion


class MSHVED(nn.Module):
    """
    MS-HVED with regression-optimized architecture.

    Key features:
    - Normalization-free residual blocks (optional GroupNorm via use_norm)
    - Global residual learning: output = mean(inputs) + network_prediction
    - Kaiming weight initialization
    - Small variational projection init (prevents KL explosion)
    """

    def __init__(
        self,
        num_orientations: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        init_filters: int = 32,
        num_scales: int = 4,
        blocks_down: Tuple[int, ...] = (1, 2, 2, 4),
        blocks_up: Tuple[int, ...] = (1, 1, 1),
        num_groups: int = 8,
        share_encoder: bool = False,
        share_decoder: bool = False,
        use_prior: bool = True,
        upsample_mode: str = 'trilinear',
        reconstruct_orientations: bool = True,
        final_activation: str = 'clamp',
        use_norm: bool = False,
        global_residual: bool = True,
    ):
        """
        Args:
            num_orientations: Number of input orientations
            in_channels: Channels per orientation
            out_channels: Output channels
            init_filters: Initial filter count (doubles each scale)
            num_scales: Number of hierarchical scales
            blocks_down: Residual blocks per encoder scale
            blocks_up: Residual blocks per decoder scale
            num_groups: Groups for GroupNorm (only used when use_norm=True)
            share_encoder: Share encoder across orientations
            share_decoder: Share decoder for orientation reconstructions
            use_prior: Include prior in Product of Gaussians fusion
            upsample_mode: 'trilinear' (default) or 'transpose'
            reconstruct_orientations: Decode orientation reconstructions
            final_activation: 'clamp' (recommended), 'sigmoid', 'tanh', or 'none'
            use_norm: Enable GroupNorm in residual blocks (default False)
            global_residual: Enable global residual learning (default True)
        """
        super().__init__()

        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.reconstruct_orientations = reconstruct_orientations
        self.global_residual = global_residual

        # When using global residual, the network should output raw residuals
        # (positive and negative), so final activation should be 'none'.
        # The clamp is applied after adding the reference.
        decoder_activation = 'none' if global_residual else final_activation

        # Encoder
        self.encoder = MultiModalSegResEncoder(
            num_orientations=num_orientations,
            in_channels=in_channels,
            init_filters=init_filters,
            num_scales=num_scales,
            blocks_per_scale=blocks_down,
            num_groups=num_groups,
            share_weights=share_encoder,
            use_norm=use_norm,
        )

        # Fusion (unchanged from original MS-HVED)
        self.fusion = MultiScaleFusion(num_scales=num_scales, use_prior=use_prior)

        # Decoder
        if reconstruct_orientations:
            self.decoder = MultiOutputSegResDecoder(
                num_orientations=num_orientations,
                out_channels=out_channels,
                init_filters=init_filters,
                num_scales=num_scales,
                blocks_per_scale=blocks_up,
                upsample_mode=upsample_mode,
                num_groups=num_groups,
                share_decoder=share_decoder,
                final_activation=decoder_activation,
                use_norm=use_norm,
            )
        else:
            self.decoder = SegResDecoder(
                out_channels=out_channels,
                init_filters=init_filters,
                num_scales=num_scales,
                blocks_per_scale=blocks_up,
                upsample_mode=upsample_mode,
                num_groups=num_groups,
                final_activation=decoder_activation,
                use_norm=use_norm,
            )

        self.hidden_dims = self.encoder.hidden_dims

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for conv layers, small init for variational projections."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Small initialization for variational projections to prevent KL explosion
        for module in self.modules():
            if hasattr(module, 'variational_proj'):
                nn.init.normal_(module.variational_proj.weight, 0, 0.02)
                nn.init.zeros_(module.variational_proj.bias)
                # Negative logvar bias: initial std ≈ 0.37 (near-deterministic start)
                out_ch = module.variational_proj.bias.shape[0] // 2
                module.variational_proj.bias.data[out_ch:] = -2.0

        # Zero-init final conv: output starts as pure reference (global residual)
        if self.global_residual:
            if isinstance(self.decoder, MultiOutputSegResDecoder):
                nn.init.zeros_(self.decoder.main_decoder.final_conv.weight)
                nn.init.zeros_(self.decoder.main_decoder.final_conv.bias)
                if hasattr(self.decoder, 'orientation_decoders'):
                    for od in self.decoder.orientation_decoders:
                        nn.init.zeros_(od.final_conv.weight)
                        nn.init.zeros_(od.final_conv.bias)
                elif hasattr(self.decoder, 'orientation_decoder'):
                    nn.init.zeros_(self.decoder.orientation_decoder.final_conv.weight)
                    nn.init.zeros_(self.decoder.orientation_decoder.final_conv.bias)
            else:
                nn.init.zeros_(self.decoder.final_conv.weight)
                nn.init.zeros_(self.decoder.final_conv.bias)

    def _compute_reference(
        self,
        orientations: List[torch.Tensor],
        orientation_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute reference image as mean of available orientation inputs."""
        mask_is_batched = orientation_mask is not None and orientation_mask.dim() == 2

        if mask_is_batched:
            # Per-sample masking: weighted mean
            batch_size = orientations[0].shape[0]
            ref = torch.zeros_like(orientations[0])
            count = torch.zeros(batch_size, 1, 1, 1, 1, device=ref.device, dtype=ref.dtype)

            for idx, tensor in enumerate(orientations):
                mask_val = orientation_mask[:, idx].view(batch_size, 1, 1, 1, 1).float()
                ref = ref + tensor * mask_val
                count = count + mask_val

            ref = ref / count.clamp(min=1.0)
        else:
            # Simple mean over available orientations
            available = []
            for idx, tensor in enumerate(orientations):
                if orientation_mask is not None:
                    if not orientation_mask[idx]:
                        continue
                available.append(tensor)

            if len(available) == 0:
                return torch.zeros_like(orientations[0])
            ref = torch.stack(available, dim=0).mean(dim=0)

        return ref

    def encode(
        self,
        orientations: List[torch.Tensor],
        orientation_mask: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Dict[int, torch.Tensor]]]:
        """Encode all orientations."""
        return self.encoder(orientations, orientation_mask)

    def fuse(
        self,
        encoder_outputs: List[Dict[str, Dict[int, torch.Tensor]]],
        orientation_mask: Optional[torch.Tensor] = None,
        interp_masks: Optional[List[torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Fuse via Product of Gaussians and sample."""
        return self.fusion(encoder_outputs, orientation_mask, interp_masks, deterministic)

    def decode(
        self,
        latent_samples: List[torch.Tensor],
        encoder_outputs: Optional[List[Dict[str, Dict[int, torch.Tensor]]]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Decode latent samples."""

        if self.reconstruct_orientations:
            return self.decoder(latent_samples, reconstruct_orientations=True)
        else:
            return self.decoder(latent_samples)

    def forward(
        self,
        orientations: List[torch.Tensor],
        orientation_mask: Optional[torch.Tensor] = None,
        interp_masks: Optional[List[torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Full forward pass.

        Args:
            orientations: List of 3 orientation volumes, each (B, C, D, H, W)
            orientation_mask: Binary mask for which orientations are present
            interp_masks: List of 3 interpolation masks, each (B, 1, D, H, W).
                         1 = interpolated slice, 0 = acquired slice.
            deterministic: If True, return means without sampling

        Returns:
            Dict with 'sr_output', 'orientation_outputs', 'posteriors', 'latent_samples'
        """
        # Encode
        encoder_outputs = self.encode(orientations, orientation_mask)

        # Fuse
        latent_samples, posteriors = self.fuse(
            encoder_outputs, orientation_mask, interp_masks, deterministic
        )

        # Decode
        if self.reconstruct_orientations:
            sr_output, orientation_outputs = self.decode(latent_samples, encoder_outputs)
        else:
            sr_output = self.decode(latent_samples, encoder_outputs)
            orientation_outputs = []

        # Global residual: output = reference + network_prediction, then clamp
        if self.global_residual:
            reference = self._compute_reference(orientations, orientation_mask)
            # Match spatial dims if needed
            if reference.shape[2:] != sr_output.shape[2:]:
                reference = F.interpolate(reference, size=sr_output.shape[2:], mode='trilinear', align_corners=False)
            sr_output = torch.clamp(sr_output + reference, 0.0, 1.0)

            # Also apply to orientation outputs
            for i in range(len(orientation_outputs)):
                if orientation_outputs[i].shape[2:] != reference.shape[2:]:
                    ref_resized = F.interpolate(reference, size=orientation_outputs[i].shape[2:], mode='trilinear', align_corners=False)
                else:
                    ref_resized = reference
                orientation_outputs[i] = torch.clamp(orientation_outputs[i] + ref_resized, 0.0, 1.0)

        return {
            'sr_output': sr_output,
            'orientation_outputs': orientation_outputs,
            'posteriors': posteriors,
            'latent_samples': latent_samples
        }


def create_mshved(config: str = 'default', **kwargs) -> nn.Module:
    """
    Factory function for MS-HVED models.

    Configs:
        'default': Standard 4-scale model
        'small': Lighter model for smaller GPUs
        'deep': More blocks per scale
    """
    configs = {
        'default': {
            'init_filters': 32,
            'num_scales': 4,
            'blocks_down': (1, 2, 2, 4),
            'blocks_up': (1, 1, 1),
        },
        'small': {
            'init_filters': 16,
            'num_scales': 3,
            'blocks_down': (1, 1, 2),
            'blocks_up': (1, 1),
        },
        'deep': {
            'init_filters': 32,
            'num_scales': 4,
            'blocks_down': (2, 2, 4, 4),
            'blocks_up': (2, 2, 2),
        }
    }

    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Available: {list(configs.keys())}")

    cfg = configs[config].copy()
    cfg.update(kwargs)

    return MSHVED(**cfg)


if __name__ == "__main__":
    print("=" * 60)
    print("MS-HVED Regression-Optimized Shape Analysis")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_mshved('default', num_orientations=3).to(device)
    model.eval()
    print(model)

    # Test input
    batch_size = 2
    spatial_size = (64, 64, 32)
    orientations = [torch.randn(batch_size, 1, *spatial_size).to(device) for _ in range(3)]

    print(f"\nInput: {len(orientations)} orientations, shape {orientations[0].shape}")

    with torch.no_grad():
        outputs = model(orientations)
        print(f"\nOutput shapes:")
        print(f"  sr_output: {outputs['sr_output'].shape}")
        print(f"  orientation_outputs: {len(outputs['orientation_outputs'])} x {outputs['orientation_outputs'][0].shape if outputs['orientation_outputs'] else 'N/A'}")
        print(f"  posteriors: {len(outputs['posteriors'])} scales")
        for i, (mu, lv) in enumerate(outputs['posteriors']):
            print(f"    Scale {i}: mu={mu.shape}, logvar={lv.shape}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Test global residual
    print(f"\nGlobal residual: {model.global_residual}")
    sr_out = outputs['sr_output']
    print(f"  SR output range: [{sr_out.min():.4f}, {sr_out.max():.4f}]")
    print("=" * 60)
