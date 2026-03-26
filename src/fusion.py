"""
Product of Gaussians Fusion for MS-HVED

This module implements the key innovation of U-HVED: fusing variational
distributions from multiple orientations via the product of Gaussians.

The idea is that each orientation provides its own estimate of the latent
representation as a Gaussian distribution (mu, sigma). By taking the
product of these distributions, we get a fused distribution that combines
information from all available orientations.

Key property: The product of Gaussians is also a Gaussian, and its
parameters can be computed in closed form:
    T_fused = sum(T_i)  where T_i = 1/var_i (precision)
    mu_fused = sum(mu_i * T_i) / T_fused

This allows the network to handle missing orientations gracefully - we simply
exclude missing orientations from the sum.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ProductOfGaussians(nn.Module):
    """
    Fuses variational parameters from multiple orientations via Product of Gaussians.

    Given mu and logvar from each orientation, computes the fused posterior
    using precision-weighted combination with an optional prior.
    """

    def __init__(
        self,
        use_prior: bool = True,
        prior_mu: float = 0.0,
        prior_logvar: float = 0.0,
        eps: float = 1e-7,
        interp_attenuation: float = 0.5,
    ):
        """
        Args:
            use_prior: Whether to include a weak prior in the product
            prior_mu: Prior mean (default 0)
            prior_logvar: Prior log-variance (default 0, i.e., var=1)
            eps: Small constant for numerical stability
            interp_attenuation: How much to reduce precision for interpolated
                voxels (0.0 = no reduction, 1.0 = zero out completely).
                Default 0.5 means interpolated voxels contribute half precision.
        """
        super().__init__()

        self.use_prior = use_prior
        self.prior_mu = prior_mu
        self.prior_logvar = prior_logvar
        self.eps = eps
        self.interp_attenuation = interp_attenuation

    def forward(
        self,
        mus: Dict[int, torch.Tensor],
        logvars: Dict[int, torch.Tensor],
        orientation_mask: Optional[torch.Tensor] = None,
        interp_masks: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute fused posterior via product of Gaussians.

        Args:
            mus: Dict mapping orientation index to mu tensor (B, C, D, H, W)
            logvars: Dict mapping orientation index to logvar tensor (B, C, D, H, W)
            orientation_mask: Optional boolean tensor indicating which orientations to include
                          - Shape (num_orientations,): same mask for all batch elements
                          - Shape (B, num_orientations): different mask per batch element
            interp_masks: Optional dict mapping orientation index to interpolation mask
                          (B, 1, D, H, W). 1 = interpolated slice, 0 = acquired slice.
                          Interpolated voxels have their precision attenuated.

        Returns:
            posterior_mu: Fused mean (B, C, D, H, W)
            posterior_logvar: Fused log-variance (B, C, D, H, W)
        """
        if len(mus) == 0:
            raise ValueError("At least one orientation must be present")

        # Force fp32 — precision-weighted fusion overflows fp16 (max 65504)
        input_dtype = next(iter(mus.values())).dtype
        mus = {k: v.float() for k, v in mus.items()}
        logvars = {k: v.float() for k, v in logvars.items()}

        # Get reference tensor for shape
        ref_tensor = next(iter(mus.values()))
        device = ref_tensor.device
        dtype = ref_tensor.dtype
        batch_size = ref_tensor.shape[0]

        # Determine mask type
        mask_is_batched = orientation_mask is not None and orientation_mask.dim() == 2

        # Initialize accumulators for precision-weighted sum
        precision_sum = torch.zeros_like(ref_tensor)
        weighted_mu_sum = torch.zeros_like(ref_tensor)

        # Accumulate contributions from each orientation
        for mod_idx, mu in mus.items():
            logvar = logvars[mod_idx]

            # Handle masking
            if orientation_mask is not None:
                if mask_is_batched:
                    # Batched mask: (B, num_orientations)
                    # Create mask with shape (B, 1, 1, 1, 1) for broadcasting
                    batch_mask = orientation_mask[:, mod_idx].view(batch_size, 1, 1, 1, 1).float()
                else:
                    # Global mask: (num_orientations,)
                    # Skip this orientation entirely if masked out
                    if not orientation_mask[mod_idx]:
                        continue
                    batch_mask = 1.0
            else:
                batch_mask = 1.0

            # Compute precision (inverse variance)
            # Clamp logvar to prevent overflow/underflow: exp(-20) to exp(20)
            logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
            precision = 1.0 / (torch.exp(logvar_clamped) + self.eps)

            # Attenuate precision for interpolated voxels
            if interp_masks is not None and mod_idx in interp_masks:
                # interp_mask: (B, 1, D, H, W), 1=interpolated, 0=acquired
                # confidence: 1.0 for acquired, (1 - attenuation) for interpolated
                confidence = 1.0 - interp_masks[mod_idx].float() * self.interp_attenuation
                precision = precision * confidence

            # Apply mask to precision (zeros out contribution from masked batch elements)
            if isinstance(batch_mask, torch.Tensor):
                precision = precision * batch_mask
                weighted_mu = mu * precision
            else:
                weighted_mu = mu * precision

            # Accumulate
            precision_sum = precision_sum + precision
            weighted_mu_sum = weighted_mu_sum + weighted_mu

        # Add prior contribution if enabled
        if self.use_prior:
            prior_precision = 1.0 / (torch.exp(torch.tensor(self.prior_logvar, device=device, dtype=dtype)) + self.eps)
            precision_sum = precision_sum + prior_precision
            weighted_mu_sum = weighted_mu_sum + self.prior_mu * prior_precision

        # Compute fused posterior
        posterior_var = 1.0 / (precision_sum + self.eps)
        posterior_mu = weighted_mu_sum * posterior_var
        posterior_mu = torch.clamp(posterior_mu, min=-10.0, max=10.0)
        posterior_logvar = torch.log(posterior_var + self.eps)

        # Safety: Clamp output to prevent NaN propagation
        posterior_logvar = torch.clamp(posterior_logvar, min=-10.0, max=10.0)

        return posterior_mu.to(input_dtype), posterior_logvar.to(input_dtype)


class GaussianSampler(nn.Module):
    """
    Samples from Gaussian distribution using the reparameterization trick.

    During training: z = mu + sigma * epsilon, where epsilon ~ N(0, 1)
    During inference: z = mu (deterministic)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        deterministic: bool = False,
        eps: float = 1e-7
    ) -> torch.Tensor:
        """
        Sample from N(mu, exp(logvar)).

        Args:
            mu: Mean tensor
            logvar: Log-variance tensor
            deterministic: If True, return mean without sampling
            eps: Small constant for numerical stability

        Returns:
            Sampled latent tensor
        """
        if deterministic or not self.training:
            return mu

        # Force fp32 — exp/sqrt operations overflow fp16
        input_dtype = mu.dtype
        mu = mu.float()
        logvar = logvar.float()

        # Reparameterization trick
        # Clamp logvar to prevent overflow: exp(0.5 * 20) is still manageable
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar_clamped) + eps
        noise = torch.randn_like(mu)
        return (mu + std * noise).to(input_dtype)


class MultiScaleFusion(nn.Module):
    """
    Applies Product of Gaussians fusion at multiple scales.

    This is the main fusion module used in U-HVED, combining variational
    parameters from all orientations at each spatial scale.
    """

    def __init__(
        self,
        num_scales: int = 4,
        use_prior: bool = True
    ):
        """
        Args:
            num_scales: Number of spatial scales
            use_prior: Whether to use prior in PoG fusion
        """
        super().__init__()

        self.num_scales = num_scales
        self.fusion = ProductOfGaussians(use_prior=use_prior)
        self.sampler = GaussianSampler()

    def forward(
        self,
        encoder_outputs: List[Dict[str, Dict[int, torch.Tensor]]],
        orientation_mask: Optional[torch.Tensor] = None,
        interp_masks: Optional[List[torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Fuse and sample at all scales.

        Args:
            encoder_outputs: List (per scale) of dicts with 'mu' and 'logvar'
                            dicts mapping orientation indices to tensors
            orientation_mask: Boolean tensor indicating present orientations
            interp_masks: Optional list of 3 interpolation masks at full resolution,
                         each (B, 1, D, H, W). Downsampled to match each scale.
            deterministic: If True, return means without sampling

        Returns:
            samples: List of sampled latent tensors at each scale
            posteriors: List of (mu, logvar) tuples for KL computation
        """
        samples = []
        posteriors = []

        for scale_idx, scale_data in enumerate(encoder_outputs):
            mus = scale_data['mu']
            logvars = scale_data['logvar']

            # Skip if no orientations present at this scale
            if len(mus) == 0:
                continue

            # Downsample interpolation masks to match this scale's spatial dims
            scale_interp_masks = None
            if interp_masks is not None:
                ref_spatial = next(iter(mus.values())).shape[2:]  # (D, H, W) at this scale
                scale_interp_masks = {}
                for ori_idx, mask in enumerate(interp_masks):
                    if ori_idx not in mus:
                        continue
                    if list(mask.shape[2:]) != list(ref_spatial):
                        # max_pool preserves interpolated markers (1s) through downsampling
                        mask = F.adaptive_max_pool3d(mask.float(), ref_spatial)
                    scale_interp_masks[ori_idx] = mask

            # Fuse via Product of Gaussians
            fused_mu, fused_logvar = self.fusion(
                mus, logvars, orientation_mask, interp_masks=scale_interp_masks
            )

            # Sample
            z = self.sampler(fused_mu, fused_logvar, deterministic)

            samples.append(z)
            posteriors.append((fused_mu, fused_logvar))

        return samples, posteriors
