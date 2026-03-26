"""
Loss Functions for MS-HVED Super-Resolution

The training objective combines:
1. Reconstruction loss: How well the network reconstructs the target
2. KL divergence: Regularization for the variational latent space
3. Perceptual loss: High-level feature matching for better visual quality
4. Adversarial loss (optional): For GAN-based training

For super-resolution, we adapt the original U-HVED losses:
- Instead of segmentation loss, we use image reconstruction loss
- KL divergence remains the same (regularizes the shared latent space)
- Added perceptual and adversarial losses for better SR quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Literal
import warnings
from monai.losses import PerceptualLoss, SSIMLoss

class KLDivergence(nn.Module):
    """
    KL divergence between posterior q(z|x) and prior p(z).

    For standard VAE with N(0,1) prior:
    KL(q||p) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Supports multi-scale KL computation for hierarchical VAE.
    """

    def __init__(self, reduction: str = 'mean', free_bits: float = 0.0):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
            free_bits: Minimum free nats per latent dimension before contributing
                       to loss. Prevents both KL collapse and KL explosion.
                       Recommended: 1.0
        """
        super().__init__()
        self.reduction = reduction
        self.free_bits = free_bits

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence against N(0,1) prior.

        Args:
            mu: Posterior mean
            logvar: Posterior log-variance

        Returns:
            KL divergence loss
        """
        # Force fp32 — reduction over large 5D tensors accumulates fp16 rounding errors
        mu = mu.float()
        logvar = logvar.float()

        # Clamp logvar before exp to prevent overflow
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
        mu_clamped = torch.clamp(mu, min=-10.0, max=10.0)
        kl = -0.5 * (1 + logvar_clamped - mu_clamped.pow(2) - logvar_clamped.exp())

        # Free-bits: allow each latent dimension a minimum free amount of KL
        if self.free_bits > 0:
            kl = torch.clamp(kl - self.free_bits, min=0.0)

        if self.reduction == 'mean':
            return kl.mean()
        elif self.reduction == 'sum':
            return kl.sum()
        else:
            return kl

    def multi_scale(
        self,
        posteriors: List[Tuple[torch.Tensor, torch.Tensor]],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Compute weighted sum of KL divergence across scales.

        Args:
            posteriors: List of (mu, logvar) tuples at each scale
            weights: Optional weights for each scale (default: equal weighting)

        Returns:
            Total KL loss
        """
        if weights is None:
            weights = [1.0] * len(posteriors)

        total_kl = 0.0
        for (mu, logvar), w in zip(posteriors, weights):
            total_kl = total_kl + w * self.forward(mu, logvar)

        return total_kl


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for super-resolution.

    Supports multiple loss types:
    - L1 (MAE): Good for preserving edges
    - L2 (MSE): Standard choice
    - Charbonnier: Differentiable approximation of L1
    """

    def __init__(
        self,
        loss_type: str = 'l1',
        reduction: str = 'mean',
        eps: float = 1e-6
    ):
        """
        Args:
            loss_type: 'l1', 'l2', or 'charbonnier'
            reduction: 'mean' or 'sum'
            eps: Epsilon for Charbonnier loss
        """
        super().__init__()

        self.loss_type = loss_type
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.

        Args:
            pred: Predicted image
            target: Ground truth image

        Returns:
            Reconstruction loss
        """
        if self.loss_type == 'l1':
            loss = F.l1_loss(pred, target, reduction=self.reduction)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(pred, target, reduction=self.reduction)
        elif self.loss_type == 'charbonnier':
            diff = pred - target
            loss = torch.sqrt(diff.pow(2) + self.eps ** 2)
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class SSIM3DLoss(nn.Module):
    """
    3D Structural Similarity Index (SSIM) loss using MONAI.

    MONAI provides optimized 3D SSIM implementation that properly
    handles volumetric medical imaging data.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
    ):
        """
        Args:
            spatial_dims: Number of spatial dimensions (3 for 3D volumes)
        """
        super().__init__()
        self.ssim_loss = SSIMLoss(
            spatial_dims=spatial_dims,

        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute 3D SSIM loss.

        Args:
            pred: Predicted volume (B, C, D, H, W)
            target: Ground truth volume (B, C, D, H, W)

        Returns:
            SSIM loss (scalar)
        """
        return self.ssim_loss(pred, target)


class MSHVEDLoss(nn.Module):
    """
    Combined 3D loss function for MS-HVED Super-Resolution on volumetric data.

    Total loss = recon_weight * L_recon
               + kl_weight * L_kl
               + perceptual_weight * L_perceptual_3d
               + ssim_weight * L_ssim_3d
               + orientation_weight * L_orientation_recon

    Features:
    - Supports multiple 3D perceptual loss backends
    - Uses MONAI's SSIM3D
    - Maintains all original loss components
    - KL annealing support
    """

    def __init__(
        self,
        recon_loss_type: str = 'l1',
        recon_weight: float = 1.0,
        kl_weight: float = 0.001,
        perceptual_weight: float = 0.1,
        ssim_weight: float = 0.0,
        orientation_weight: float = 0.5,
        use_perceptual: bool = True,
        use_ssim: bool = False,
        perceptual_network: str = 'alex',
        is_fake_3d: bool = False,
        kl_annealing: bool = True,
        kl_anneal_steps: int = 10000,
        free_bits: float = 1.0,
    ):
        """
        Args:
            recon_loss_type: Type of reconstruction loss ('l1', 'l2', 'charbonnier')
            recon_weight: Weight for main reconstruction loss
            kl_weight: Weight for KL divergence
            perceptual_weight: Weight for 3D perceptual loss
            ssim_weight: Weight for 3D SSIM loss
            orientation_weight: Weight for orientation reconstruction loss
            use_perceptual: Whether to use 3D perceptual loss
            use_ssim: Whether to use 3D SSIM loss
            perceptual_network: MONAI network for perceptual loss
                - 'alex': AlexNet (default, lightweight LPIPS standard)
                - 'vgg': VGG network
                - 'squeeze': SqueezeNet
                - 'radimagenet': RadImageNet ResNet-50 pretrained (maps to radimagenet_resnet50)
                - 'medicalnet': MedicalNet ResNet-10 pretrained (maps to medicalnet_resnet10_23datasets)
                - 'resnet50': ResNet-50
            is_fake_3d: Use 2.5D (fake 3D) mode for perceptual loss
                - False: Full 3D processing (default, more accurate)
                - True: 2.5D slice-based processing (faster, lower memory)
            kl_annealing: Whether to anneal KL weight
            kl_anneal_steps: Number of steps for KL annealing
        """
        super().__init__()

        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.orientation_weight = orientation_weight
        self.kl_annealing = kl_annealing
        self.kl_anneal_steps = kl_anneal_steps

        # Reconstruction loss (works with any dimension)
        self.recon_loss = ReconstructionLoss(loss_type=recon_loss_type)

        # KL divergence (works with any dimension)
        self.kl_loss = KLDivergence(free_bits=free_bits)

        # 3D Perceptual loss
        if use_perceptual:
            try:
                # Map user-friendly names to full MONAI network names
                network_mapping = {
                    'radimagenet': 'radimagenet_resnet50',
                    'medicalnet': 'medicalnet_resnet10_23datasets',  # Default to ResNet10
                }
                monai_network_name = network_mapping.get(perceptual_network, perceptual_network)

                self.perceptual_loss = PerceptualLoss(
                    spatial_dims=3,
                    network_type=monai_network_name,
                    is_fake_3d=is_fake_3d,
                    pretrained=True
                )
                mode_str = "2.5D (fake 3D)" if is_fake_3d else "full 3D"
                actual_name = f"{perceptual_network} ({monai_network_name})" if perceptual_network in network_mapping else perceptual_network
                print(f"✓ MONAI Perceptual Loss initialized with network: {actual_name} ({mode_str})")
            except ImportError:
                raise ImportError(
                    "MONAI PerceptualLoss not found. Please install/update MONAI:\n"
                    "  pip install --upgrade monai\n"
                    "MONAI >= 1.3.0 required"
                )
            except Exception as e:
                warnings.warn(f"Failed to initialize 3D perceptual loss: {e}")
                self.perceptual_loss = None
        else:
            self.perceptual_loss = None

        # 3D SSIM loss
        if use_ssim:
            try:
                self.ssim_loss = SSIM3DLoss()
                print("✓ 3D SSIM Loss initialized using MONAI")
            except Exception as e:
                warnings.warn(f"Failed to initialize 3D SSIM loss: {e}")
                self.ssim_loss = None
        else:
            self.ssim_loss = None

        # Current step for KL annealing
        self.register_buffer('current_step', torch.tensor(0))

    def get_kl_weight(self) -> float:
        """Get current KL weight with optional annealing."""
        if not self.kl_annealing:
            return self.kl_weight

        # Linear annealing from 0 to kl_weight
        progress = min(self.current_step.item() / self.kl_anneal_steps, 1.0)
        return self.kl_weight * progress

    def _maybe_move_perceptual(self, device: torch.device) -> None:
        if not isinstance(self.perceptual_loss, nn.Module):
            return
        ref_tensor = next(self.perceptual_loss.parameters(), None)
        if ref_tensor is None:
            ref_tensor = next(self.perceptual_loss.buffers(), None)
        if ref_tensor is None or ref_tensor.device == device:
            return
        self.perceptual_loss.to(device)

    def forward(
        self,
        sr_output: torch.Tensor,
        sr_target: torch.Tensor,
        posteriors: List[Tuple[torch.Tensor, torch.Tensor]],
        orientation_outputs: Optional[List[torch.Tensor]] = None,
        orientation_targets: Optional[List[torch.Tensor]] = None,
        return_components: bool = False
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Compute total 3D loss.

        Args:
            sr_output: Super-resolved output (B, C, D, H, W)
            sr_target: Ground truth high-resolution volume (B, C, D, H, W)
            posteriors: List of (mu, logvar) from encoder
            orientation_outputs: Reconstructed orientations (optional)
            orientation_targets: Target orientations (optional)
            return_components: If True, return dict of individual losses

        Returns:
            Total loss (or dict of losses if return_components=True)
        """
        losses = {}

        # Main reconstruction loss (L1/L2/Charbonnier)
        losses['reconstruction'] = self.recon_loss(sr_output, sr_target) * self.recon_weight

        # KL divergence (multi-scale)
        kl_weight = self.get_kl_weight()
        losses['kl'] = self.kl_loss.multi_scale(posteriors) * kl_weight

        # 3D Perceptual loss
        if self.perceptual_loss is not None:
            try:
                self.perceptual_loss.to(sr_output.device)
                losses['perceptual'] = self.perceptual_loss(sr_output, sr_target) * self.perceptual_weight
            except Exception as e:
                warnings.warn(f"Perceptual loss computation failed: {e}")
                losses['perceptual'] = torch.tensor(0.0, device=sr_output.device)
        else:
            losses['perceptual'] = torch.tensor(0.0, device=sr_output.device)

        # 3D SSIM loss
        if self.ssim_loss is not None:
            try:
                losses['ssim'] = self.ssim_loss(sr_output, sr_target) * self.ssim_weight
            except Exception as e:
                warnings.warn(f"SSIM loss computation failed: {e}")
                losses['ssim'] = torch.tensor(0.0, device=sr_output.device)
        else:
            losses['ssim'] = torch.tensor(0.0, device=sr_output.device)

        # orientation reconstruction loss
        if orientation_outputs is not None and orientation_targets is not None:
            mod_loss = torch.tensor(0.0, device=sr_output.device)
            for idx, (mod_out, mod_target) in enumerate(zip(orientation_outputs, orientation_targets)):
                if mod_target is not None:
                    mod_loss = mod_loss + self.recon_loss(mod_out, mod_target)
            losses['orientation'] = mod_loss * self.orientation_weight
        else:
            losses['orientation'] = torch.tensor(0.0, device=sr_output.device)

        # Total loss
        total_loss = sum(losses.values())

        # Update step counter
        if self.training:
            self.current_step += 1

        if return_components:
            losses['total'] = total_loss
            return losses

        return total_loss


def spectral_regularization(model: nn.Module) -> torch.Tensor:
    """
    Compute spectral regularization loss (NVAE, Vahdat & Kautz 2020).

    Sums the largest singular value of each Conv3d/ConvTranspose3d weight matrix
    in the model. Constrains the Lipschitz constant of the network, keeping
    encoder outputs (mu, logvar) bounded — critical for stable KL in
    normalization-free architectures.

    L_SR = Σᵢ s⁽ⁱ⁾

    where s⁽ⁱ⁾ is the largest singular value of the i-th conv layer,
    estimated via a single power iteration step.

    Returns:
        Scalar spectral regularization loss (unweighted — caller applies λ).
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            w = module.weight
            # Reshape to 2D: (out_features, in_features * kernel)
            w_mat = w.reshape(w.shape[0], -1)
            # Single power iteration to estimate largest singular value
            # Use detached random vector for stability (no grad through SVD estimate)
            if not hasattr(module, '_sr_u'):
                module._sr_u = F.normalize(torch.randn(w_mat.shape[0], device=w.device), dim=0)
            u = module._sr_u
            with torch.no_grad():
                v = F.normalize(w_mat.t() @ u, dim=0)
                u = F.normalize(w_mat @ v, dim=0)
                module._sr_u = u
            # Compute sigma with grad (through w_mat, not u/v)
            sigma = u @ w_mat @ v
            total = total + sigma
    return total


# Convenience function to create 3D loss
def create_loss(
    network_type: str = 'alex',
    use_perceptual: bool = True,
    use_ssim: bool = False,
    **kwargs
) -> MSHVEDLoss:
    """
    Factory function to create MSHVEDLoss with sensible defaults.

    Args:
        network_type: MONAI network type ('alex', 'vgg', 'squeeze',
                     'radimagenet', 'medicalnet', 'resnet50')
        use_perceptual: Whether to use 3D perceptual loss
        use_ssim: Whether to use 3D SSIM loss
        **kwargs: Additional arguments for MSHVEDLoss

    Returns:
        MSHVEDLoss instance

    Example:
        >>> # Basic usage with L1 + KL only
        >>> loss_fn = create_loss(use_perceptual=False, use_ssim=False)

        >>> # With AlexNet perceptual loss (recommended)
        >>> loss_fn = create_loss(network_type='alex', use_perceptual=True)

        >>> # With VGG perceptual + SSIM
        >>> loss_fn = create_loss(
        ...     network_type='vgg',
        ...     use_perceptual=True,
        ...     use_ssim=True,
        ...     ssim_weight=0.1
        ... )
    """
    return MSHVEDLoss(
        perceptual_network=network_type,
        use_perceptual=use_perceptual,
        use_ssim=use_ssim,
        **kwargs
    )
