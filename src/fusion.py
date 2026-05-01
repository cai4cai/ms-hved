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
        fov_attenuation: float = 1.0,
        smooth_fov: bool = True,
        fov_transition_width: float = 3.0,
    ):
        """
        Args:
            use_prior: Whether to include a weak prior in the product
            prior_mu: Prior mean (default 0)
            prior_logvar: Prior log-variance (default 0, i.e., var=1)
            eps: Small constant for numerical stability
            fov_attenuation: Controls the *depth* of precision suppression for
                missing (out-of-FOV) voxels, i.e. how much precision is removed
                at the worst point. Applied as: confidence = 1 - mask * attenuation.
                1.0 = fully suppress missing voxels (confidence → 0).
                0.5 = retain half precision for missing voxels.
                0.0 = no suppression (FOV masking effectively disabled).
                Independent of smooth_fov, which controls the *shape* (hard vs
                gradual) of the transition at FOV boundaries.
            smooth_fov: Whether to apply Gaussian smoothing to FOV masks.
                False = original hard binary boundary. Default True.
            fov_transition_width: Sigma (in voxels) of the Gaussian kernel used
                to smooth FOV boundaries. Only used when smooth_fov=True.
                Default 3.0.
        """
        super().__init__()

        self.use_prior = use_prior
        self.prior_mu = prior_mu
        self.prior_logvar = prior_logvar
        self.eps = eps
        self.fov_attenuation = fov_attenuation
        self.smooth_fov = smooth_fov
        self.fov_transition_width = fov_transition_width
        self._kernel_cache: Dict[Tuple[float, str], torch.Tensor] = {}

    @staticmethod
    def _make_gaussian_kernel_3d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create a normalised 3D Gaussian kernel (1,1,K,K,K) for conv3d."""
        radius = max(int(3 * sigma + 0.5), 1)  # 3-sigma rule
        size = 2 * radius + 1
        coords = torch.arange(size, device=device, dtype=dtype) - radius
        g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        g3d = g1d[:, None, None] * g1d[None, :, None] * g1d[None, None, :]
        g3d = g3d / g3d.sum()
        return g3d.reshape(1, 1, size, size, size)

    def smooth_fov_mask(self, fov_mask: torch.Tensor) -> torch.Tensor:
        """
        Smooth a binary FOV mask into a continuous confidence map.

        The idea:
        - Convolve the binary *missing* mask (1=missing) with a Gaussian kernel.
        - The result tells each voxel "how much missingness is nearby".
        - confidence = 1 - blurred_missing * attenuation

        Properties:
        - Voxels deep inside the valid FOV → confidence ≈ 1 (unaffected)
        - Voxels near the FOV boundary    → confidence smoothly decreases
        - Voxels fully outside FOV         → confidence ≈ 0 (when attenuation=1)
        - Background boundaries are NOT affected because we only blur the
          geometric FOV mask, not any content. The falloff lives strictly
          within the FOV transition zone.

        Args:
            fov_mask: (B, 1, D, H, W), 1=missing/out-of-FOV, 0=valid

        Returns:
            confidence: (B, 1, D, H, W) in [0, 1]
        """
        sigma = self.fov_transition_width
        if not self.smooth_fov or sigma <= 0:
            # No smoothing — original hard boundary
            return 1.0 - fov_mask.float() * self.fov_attenuation

        # Cache kernel per (sigma, device) to avoid re-creation every call
        cache_key = (sigma, str(fov_mask.device))
        if cache_key not in self._kernel_cache:
            self._kernel_cache[cache_key] = self._make_gaussian_kernel_3d(
                sigma, fov_mask.device, torch.float32
            )
        kernel = self._kernel_cache[cache_key]

        mask_f = fov_mask.float()
        pad = kernel.shape[-1] // 2
        # Replicate-pad so FOV edges beyond the volume don't introduce zeros
        blurred = F.conv3d(
            F.pad(mask_f, [pad] * 6, mode='replicate'),
            kernel
        )
        # blurred is in [0, 1]: 0 = fully valid neighbourhood, 1 = fully missing
        blurred = blurred.clamp(0.0, 1.0)

        confidence = 1.0 - blurred * self.fov_attenuation
        return confidence

    def forward(
        self,
        mus: Dict[int, torch.Tensor],
        logvars: Dict[int, torch.Tensor],
        orientation_mask: Optional[torch.Tensor] = None,
        fov_masks: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute fused posterior via product of Gaussians.

        Args:
            mus: Dict mapping orientation index to mu tensor (B, C, D, H, W)
            logvars: Dict mapping orientation index to logvar tensor (B, C, D, H, W)
            orientation_mask: Optional boolean tensor indicating which orientations to include
                          - Shape (num_orientations,): same mask for all batch elements
                          - Shape (B, num_orientations): different mask per batch element
            fov_masks: Optional dict mapping orientation index to FOV mask
                          (B, 1, D, H, W). 1 = missing (out-of-FOV), 0 = valid.
                          Missing voxels have their precision attenuated.

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
            # Clamp logvar to prevent overflow/underflow: exp(-10) to exp(10)
            logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
            precision = 1.0 / (torch.exp(logvar_clamped) + self.eps)

            # Attenuate precision for missing (out-of-FOV) voxels with smooth falloff
            if fov_masks is not None and mod_idx in fov_masks:
                confidence = self.smooth_fov_mask(fov_masks[mod_idx])
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
        # Clamp logvar to prevent overflow: exp(0.5 * 10) is still manageable
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
        use_prior: bool = True,
        smooth_fov: bool = True,
        fov_attenuation: float = 1.0,
        fov_transition_width: float = 3.0,
    ):
        """
        Args:
            num_scales: Number of spatial scales
            use_prior: Whether to use prior in PoG fusion
            smooth_fov: Whether to smooth FOV mask boundaries
            fov_attenuation: Attenuation factor for FOV masks (0=no attenuation, 1=full attenuation)
            fov_transition_width: Gaussian sigma for FOV smoothing (voxels)
        """
        super().__init__()

        self.num_scales = num_scales
        self.fusion = ProductOfGaussians(
            use_prior=use_prior,
            smooth_fov=smooth_fov,
            fov_attenuation=fov_attenuation,
            fov_transition_width=fov_transition_width,
        )
        self.sampler = GaussianSampler()

    def forward(
        self,
        encoder_outputs: List[Dict[str, Dict[int, torch.Tensor]]],
        orientation_mask: Optional[torch.Tensor] = None,
        fov_masks: Optional[List[torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Fuse and sample at all scales.

        Args:
            encoder_outputs: List (per scale) of dicts with 'mu' and 'logvar'
                            dicts mapping orientation indices to tensors
            orientation_mask: Boolean tensor indicating present orientations
            fov_masks: Optional list of 3 FOV masks at full resolution,
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

            # Downsample FOV masks to match this scale's spatial dims
            scale_fov_masks = None
            if fov_masks is not None:
                ref_spatial = next(iter(mus.values())).shape[2:]  # (D, H, W) at this scale
                scale_fov_masks = {}
                for ori_idx, mask in enumerate(fov_masks):
                    if ori_idx not in mus:
                        continue
                    if list(mask.shape[2:]) != list(ref_spatial):
                        # max_pool preserves missing markers (1s) through downsampling
                        mask = F.adaptive_max_pool3d(mask.float(), ref_spatial)
                    scale_fov_masks[ori_idx] = mask

            # Fuse via Product of Gaussians
            fused_mu, fused_logvar = self.fusion(
                mus, logvars, orientation_mask, fov_masks=scale_fov_masks
            )

            # Sample
            z = self.sampler(fused_mu, fused_logvar, deterministic)

            samples.append(z)
            posteriors.append((fused_mu, fused_logvar))

        return samples, posteriors