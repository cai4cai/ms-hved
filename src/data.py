"""
Data loading and generation for ms-hved training.
Includes comprehensive k-space artifact simulation and Physics-based PSF blurring.

This module simulates the MRI acquisition pipeline to generate synthetic Low-Resolution (LR) images
from High-Resolution (HR) ground truth. It models the physical degradation process including:
1.  **Slice Profile & Point Spread Function (PSF):** The physical blurring caused by RF excitation profiles and signal decay.
2.  **B1 Field Inhomogeneity:** Bias fields that cause smooth intensity variations.
3.  **Contrast Variation:** Gamma correction to simulate different T1/T2 weightings.
4.  **K-Space Artifacts:** Motion ghosting and spikes in the frequency domain.
5.  **Sampling Limits:** Resolution loss via FFT cropping (simulating limited k-space acquisition).
"""
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict

from monai.data import Dataset, CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, CropForegroundd, RandSpatialCropd, SpatialPadd,
    ToTensord, ScaleIntensityRangePercentiles,
    CenterSpatialCropd, RandAffined,
)



class SliceProfilePhysics(nn.Module):
    """
    Simulates the physical blurring caused by the MRI scanner's slice selection profile and in-plane sampling.

    **MRI Physical Representation:**
    In a real MRI scanner, a 2D slice is not a perfect geometric plane. It has a thickness determined
    by the Radio Frequency (RF) excitation pulse.
    - **Through-Plane (Slice Direction):** The RF pulse excites spins within a "slab." The sensitivity profile
      across this slab is rarely a perfect rectangle (Boxcar). It is often trapezoidal or Gaussian due to
      hardware limits on the RF pulse duration (truncated Sinc pulses). This causes "Partial Volume Effects"
      where signal from adjacent tissues bleeds into the slice.
    - **In-Plane (Phase/Frequency Directions):** Blurring occurs due to T2* relaxation during readout
      and finite sampling windows, typically modeled as a Point Spread Function (PSF).

    This class replaces generic Gaussian blurring with physically programmable kernels to accurately
    model these distinct behaviors.

    Args:
        profile_type (str): The shape of the slice sensitivity profile.
            - 'boxcar': Ideal rectangular profile (perfect slice selection).
            - 'gaussian': Standard approximation (often used in simple simulations).
            - 'trapezoid': Realistic profile for most clinical scanners (flat top with fading edges).
        edge_width (float): For 'trapezoid', the fraction of the slice thickness that is the "slope"
            (transition region). Represents the imperfect sharp cutoff of the RF pulse.
    """
    
    def __init__(self, profile_type='trapezoid', edge_width=0.1):
        """
        Args:
            profile_type: 'gaussian', 'boxcar' (ideal), or 'trapezoid' (realistic).
            edge_width: For trapezoid, how much of the slice is the "slope" (0.0-0.5).
                        0.1 means 10% on left and 10% on right are fading out.
        """
        super().__init__()
        self.profile_type = profile_type
        self.edge_width = edge_width

    def get_slice_kernel(self, thickness_mm, current_res_mm, device):
        """
        Generates the 1D convolution kernel representing the slice sensitivity profile.
        
        Args:
            thickness_mm (float): The target slice thickness to simulate.
            current_res_mm (float): The current resolution of the input image.
            device (torch.device): Device to create tensors on.
            
        Returns:
            torch.Tensor: Normalized 1D kernel.
        """
        
        # Calculate kernel size in voxels
        # We need enough support to capture the profile
        scale = thickness_mm / current_res_mm
        kernel_size = int(math.ceil(scale * 3)) 
        if kernel_size % 2 == 0: kernel_size += 1
        
        grid = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size, device=device)
        
        # Normalize grid relative to slice thickness (0.5 = half thickness)
        x = grid / scale 

        if self.profile_type == 'boxcar':
            # Ideal rectangular profile: 1 inside [-0.5, 0.5], 0 outside
            kernel = (x.abs() <= 0.5).float()
            
        elif self.profile_type == 'gaussian':
            # Standard approximation (FWHM = thickness)
            # sigma corresponding to FWHM=1 is 1 / 2.355 = 0.4246
            sigma = 0.4246
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            
        elif self.profile_type == 'trapezoid':
            # Realistic profile: Flat top with sloping edges
            # Width of the flat top
            flat_width = 0.5 - self.edge_width
            
            # Mask for flat region
            flat_mask = (x.abs() <= flat_width).float()
            
            # Mask for slopes
            slope_mask = ((x.abs() > flat_width) & (x.abs() <= 0.5)).float()
            
            # Linear decay on slopes
            # Dist from edge start / edge width
            slope_val = 1.0 - (x.abs() - flat_width) / self.edge_width
            
            kernel = flat_mask + slope_mask * slope_val
            
        else:
            raise ValueError(f"Unknown profile: {self.profile_type}")

        # Energy conservation (area under curve must be 1)
        return kernel / kernel.sum()

    def forward(self, img, resolution, thickness):
        """
        Applies the physics-based blurring to the input volume.

        This method identifies the "Slice Select" direction (the one with the lowest resolution/highest thickness)
        and applies the specific Slice Profile kernel. For the other two directions (In-Plane), it applies
        a standard PSF blur.

        Args:
            img (torch.Tensor): Input image tensor (C, D, H, W).
            resolution (torch.Tensor): Current voxel size of the input [res_D, res_H, res_W].
            thickness (torch.Tensor): Target slice thickness to simulate [thick_D, thick_H, thick_W].
        """
        device = img.device

        # Identify the slice dimension (the one with largest thickness/resolution ratio)
        # Usually MRI stacks are anisotropic, so the "thick" axis is the slice axis.
        factors = thickness / resolution
        slice_dim_idx = torch.argmax(factors).item()
        
        # We process dimensions 1, 2, 3 (D, H, W)
        for i in range(3):
            
            # CASE A: Through-Plane (Slice Selection Axis)
            # Apply the specific slice profile (Boxcar/Trapezoid/Gaussian)
            if i == slice_dim_idx and factors[i] > 1.1: # Only if actually thick
                kernel = self.get_slice_kernel(thickness[i], resolution[i], device)
                
            # CASE B: In-Plane (Frequency/Phase Encoding Axes)
            # Apply standard PSF (Gaussian) due to T2* decay and sampling
            else:
                # Use standard Gaussian approximation for in-plane PSF
                # Sigma is small (approx 0.5-0.8 pixels) for in-plane
                sigma = 0.42  # ~0.42 pixels (T2* decay PSF)
                
                k_size = 5
                k_grid = torch.arange(k_size, device=device) - k_size//2
                kernel = torch.exp(-0.5 * (k_grid / sigma) ** 2)
                kernel = kernel / kernel.sum()

            # --- Convolve ---
            # Reshape kernel for conv1d: (1, 1, K)
            kernel = kernel.view(1, 1, -1)
            padding = kernel.shape[-1] // 2
            
            # Permute dimensions to apply 1D conv on the current axis
            if i == 0:   # D
                img_in = img.permute(0, 2, 3, 1) # (C, H, W, D)
            elif i == 1: # H
                img_in = img.permute(0, 1, 3, 2) # (C, D, W, H)
            else:        # W
                img_in = img  # W is already the last dim
            
            # Flatten non-active dims into batch
            shape_before = img_in.shape
            # Combine all dims except the last one (the active one)
            img_flat = img_in.reshape(-1, 1, shape_before[-1]) 
            
            # Apply Convolution
            img_filtered = F.conv1d(img_flat, kernel, padding=padding)
            
            # Un-flatten and un-permute
            img_out = img_filtered.view(shape_before)
            
            if i == 0:
                img = img_out.permute(0, 3, 1, 2)
            elif i == 1:
                img = img_out.permute(0, 1, 3, 2)
            else:
                img = img_out 
                
        return img


class BiasFieldCorruption(nn.Module):
    """
    Simulates MRI bias field (B1 inhomogeneity) artifacts.

    **MRI Physical Representation:**
    In MRI, the transmit/receive coils are not perfectly uniform. The sensitivity of the
    Radio Frequency (RF) coils varies across space, especially in older scanners or with
    surface coils. This causes low-frequency intensity variations where some parts of the
    brain appear brighter or darker than others, despite having the same tissue type.
    This is often called "intensity non-uniformity" (INU) or "shading."

    The bias field is modeled as a multiplicative field that varies smoothly over the image volume.

    Args:
        bias_field_std (float): Standard deviation of the bias field coefficients.
            Higher values create stronger shading effects (simulating poorer coil homogeneity).
        bias_scale (float): Scale factor for the bias field resolution.
            Controls the "frequency" of the shading. Smaller values mean very smooth,
            gradual shading (typical of body coils); larger values allow for more localized
            variations (typical of multi-channel surface coils).
        prob (float): Probability of applying this corruption.

    Input Shape:
        (B, C, D, H, W) - Batch of 3D volumes

    Output Shape:
        (B, C, D, H, W) - Corrupted volumes with bias field applied

    Example:
        >>> bias_corruptor = BiasFieldCorruption(
        ...     bias_field_std=0.3,
        ...     bias_scale=0.025,
        ...     prob=0.98
        ... )
        >>> input_volume = torch.randn(2, 1, 128, 128, 128)
        >>> corrupted = bias_corruptor(input_volume)
        >>> # Corrupted volume has smooth intensity variations
    """

    def __init__(
        self, bias_field_std: float = 0.3, bias_scale: float = 0.025, prob: float = 0.98
    ):
        super().__init__()
        self.bias_field_std = bias_field_std
        self.bias_scale = bias_scale
        self.prob = prob

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies a multiplicative bias field to the input volume.

        Args:
            image: Input tensor of shape (B, C, D, H, W)

        Returns:
            Corrupted tensor of shape (B, C, D, H, W) with multiplicative bias field
        """
        if torch.rand(1).item() > self.prob:
            return image

        batch_size = image.shape[0]
        spatial_shape = image.shape[2:]
        device = image.device
        outputs = []

        for b in range(batch_size):
            img = image[b : b + 1]

            # Generate low-resolution bias field coefficients
            bias_shape = [max(1, int(s * self.bias_scale)) for s in spatial_shape]
            bias_coeffs = (
                torch.randn(1, 1, *bias_shape, device=device) * self.bias_field_std
            )

            # Upsample to image size (creates smooth bias field)
            bias_field = F.interpolate(
                bias_coeffs, size=list(spatial_shape), mode="trilinear", align_corners=False
            )

            # Convert to multiplicative field and apply
            bias_field = torch.exp(bias_field)
            img = img * bias_field
            outputs.append(img)

        return torch.cat(outputs, dim=0)


class IntensityAugmentation(nn.Module):
    """
    Simulates variations in MRI contrast mechanisms and sensor dynamics.

    **MRI Physical Representation:**
    1. **Clipping:** Simulates the dynamic range limits of the MRI receiver/ADC (Analog-to-Digital Converter).
       Extremely high signal intensities (e.g., from fat or flow artifacts) can saturate the sensor.
    2. **Gamma Correction:** Simulates variations in tissue contrast (T1/T2 weighting).
       Different pulse sequences (TE, TR settings) produce different contrast curves.
       A power-law transform approximates these non-linear relationships between proton density
       and final pixel intensity.
       - Gamma < 1: Simulates images with brighter mid-tones (e.g., PD-weighted).
       - Gamma > 1: Simulates images with darker mid-tones (higher contrast).

    Gamma correction: I_out = I_in^(exp(γ)) where γ ~ N(0, gamma_std)

    Args:
        clip: Clipping bounds. Options:
             - float: Clip to [0, clip]
             - tuple: Clip to [clip[0], clip[1]]
             - False: No clipping
             Default: 300
        gamma_std: Standard deviation of gamma parameter for gamma correction.
                  Gamma is sampled from N(0, gamma_std). Higher = stronger variation.
                  Default: 0.5
        channel_wise: If True, apply different gamma per channel. If False, same
                     gamma for all channels. Default: False
        prob_gamma: Probability of applying gamma correction. Default: 0.95

    Input Shape:
        (B, C, D, H, W) - Batch of 3D volumes

    Output Shape:
        (B, C, D, H, W) - Augmented volumes

    Example:
        >>> intensity_aug = IntensityAugmentation(
        ...     clip=300,
        ...     gamma_std=0.5,
        ...     prob_gamma=0.95
        ... )
        >>> input_volume = torch.randn(2, 1, 128, 128, 128).abs() * 100
        >>> augmented = intensity_aug(input_volume)
        >>> # Augmented volume has clipped values and modified contrast
    """

    def __init__(
        self,
        clip: Union[float, Tuple[float, float], bool] = 300,
        gamma_std: float = 0.5,
        channel_wise: bool = False,
        prob_gamma: float = 0.95,
    ):
        super().__init__()
        self.clip = clip
        self.gamma_std = gamma_std
        self.channel_wise = channel_wise
        self.prob_gamma = prob_gamma

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply intensity augmentations to input volumes.

        Args:
            image: Input tensor of shape (B, C, D, H, W)

        Returns:
            Augmented tensor of shape (B, C, D, H, W)
        """
        batch_size, n_channels = image.shape[:2]
        ndims = len(image.shape) - 2
        device = image.device

        # 1. Clip outliers
        if self.clip:
            if isinstance(self.clip, (int, float)):
                image = torch.clamp(image, 0, self.clip)
            else:
                image = torch.clamp(image, self.clip[0], self.clip[1])

        # 2. Gamma augmentation (power-law transform)
        if self.gamma_std > 0 and torch.rand(1).item() < self.prob_gamma:
            if self.channel_wise:
                # Different gamma per channel
                gamma = (
                    torch.randn(batch_size, n_channels, *([1] * ndims), device=device)
                    * self.gamma_std
                )
            else:
                # Same gamma for all channels
                gamma = (
                    torch.randn(batch_size, 1, *([1] * ndims), device=device)
                    * self.gamma_std
                )

            # Apply power transform: I^(exp(gamma))
            image = torch.pow(image.clamp(min=1e-7), torch.exp(gamma))

        return image


class SampleResolution(nn.Module):
    """
    Simulates the selection of MRI acquisition protocols (Field of View and Matrix Size).

    Note: This class is a standalone utility and is NOT used by the main training pipeline.
    Resolution sampling in HRLRDataGenerator is handled by _create_orthogonal_resolutions().

    **MRI Physical Representation:**
    MRI scanners are configured by technicians to acquire data at specific resolutions.
    - **Isotropic:** High-resolution 3D scans (e.g., MP-RAGE) often have 1x1x1 mm voxels.
    - **Anisotropic:** Fast clinical 2D scans (e.g., T2-weighted turbo spin echo) typically
      have high in-plane resolution (e.g., 0.5 mm) but thick slices (e.g., 5.0 mm) to save time.

    This class randomizes these parameters to train the model to handle diverse
    clinical scenarios, from high-quality research scans to rapid emergency protocols.

    This is crucial for training models that must handle diverse acquisition protocols,
    such as:
    - T1 scans: typically 1×1×1mm isotropic
    - T2 scans: often 0.5×0.5×3mm anisotropic
    - Clinical scans: highly variable (1×1×5mm common)

    Args:
        min_resolution: Minimum resolution (highest quality) in mm for each axis.
                       Example: [1.0, 1.0, 1.0]
        max_res_iso: Maximum isotropic resolution (lowest quality) in mm.
                    Example: [1.0, 1.0, 1.0] for 1mm isotropic
                    Can be None if only anisotropic is used.
        max_res_aniso: Maximum anisotropic resolution in mm for each axis.
                      Example: [9.0, 9.0, 9.0]
                      Can be None if only isotropic is used.
        prob_iso: Probability of sampling isotropic resolution (vs anisotropic).
                 Default: 0.05 (95% anisotropic, 5% isotropic)
        prob_min: Probability of using minimum (highest quality) resolution.
                 Default: 0.05
        return_thickness: If True, also returns slice thickness (can differ from resolution).
                         Default: True

    Returns:
        If return_thickness=False:
            resolution: Tensor of shape (batch_size, 3) with sampled resolutions
        If return_thickness=True:
            (resolution, thickness): Tuple of tensors, both shape (batch_size, 3)

    Example:
        >>> res_sampler = SampleResolution(
        ...     min_resolution=[1.0, 1.0, 1.0],
        ...     max_res_iso=[1.0, 1.0, 1.0],
        ...     max_res_aniso=[9.0, 9.0, 9.0],
        ...     prob_iso=0.02,
        ...     prob_min=0.1
        ... )
        >>> resolution, thickness = res_sampler(batch_size=4)
        >>> print(resolution.shape)  # (4, 3)
        >>> # Example: [[1.0, 1.0, 5.2], [1.0, 1.0, 7.8], ...]
    """

    def __init__(
        self,
        min_resolution: List[float],
        max_res_iso: Optional[List[float]] = None,
        max_res_aniso: Optional[List[float]] = None,
        prob_iso: float = 0.05,
        prob_min: float = 0.05,
        return_thickness: bool = True,
    ):
        super().__init__()
        self.min_res = torch.tensor(min_resolution, dtype=torch.float32)
        self.max_res_iso = (
            torch.tensor(max_res_iso, dtype=torch.float32) if max_res_iso else None
        )
        self.max_res_aniso = (
            torch.tensor(max_res_aniso, dtype=torch.float32) if max_res_aniso else None
        )
        self.prob_iso = prob_iso
        self.prob_min = prob_min
        self.return_thickness = return_thickness
        self.n_dims = len(min_resolution)

        assert (max_res_iso is not None) or (max_res_aniso is not None), (
            "At least one of max_res_iso or max_res_aniso must be provided"
        )

    def forward(
        self, batch_size: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample random resolutions for a batch.

        Args:
            batch_size: Number of resolution samples to generate

        Returns:
            If return_thickness=False: resolution tensor of shape (batch_size, 3)
            If return_thickness=True: tuple of (resolution, thickness), both (batch_size, 3)
        """
        device = self.min_res.device

        # Determine which samples are isotropic vs anisotropic
        if (self.max_res_iso is not None) and (self.max_res_aniso is not None):
            use_iso = torch.rand(batch_size, device=device) < self.prob_iso
        elif self.max_res_iso is not None:
            use_iso = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            use_iso = torch.zeros(batch_size, dtype=torch.bool, device=device)

        resolution = torch.zeros(batch_size, self.n_dims, device=device)

        for b in range(batch_size):
            if use_iso[b]:
                # Isotropic: same resolution for all dimensions
                res_val = (
                    torch.rand(1, device=device) * (self.max_res_iso - self.min_res)
                    + self.min_res
                )
                resolution[b] = res_val[0]
            else:
                # Anisotropic: one dimension has high res, others have low res
                high_res_dim = torch.randint(0, self.n_dims, (1,), device=device).item()
                for d in range(self.n_dims):
                    if d == high_res_dim:
                        resolution[b, d] = (
                            torch.rand(1, device=device)
                            * (self.max_res_aniso[d] - self.min_res[d])
                            + self.min_res[d]
                        )
                    else:
                        resolution[b, d] = self.min_res[d]

        # Apply minimum resolution override
        use_min = torch.rand(batch_size, device=device) < self.prob_min
        resolution[use_min] = self.min_res.unsqueeze(0).expand(use_min.sum(), -1)

        if self.return_thickness:
            # Slice thickness >= voxel resolution (physically correct)
            thickness = resolution.clone()
            return resolution, thickness
        else:
            return resolution


# ---  Artifact Helpers (Motion, Spikes, Aliasing) ---

def apply_kspace_motion_ghosting(volume: torch.Tensor, axis: int, intensity: float = 0.5, num_ghosts: int = 2) -> torch.Tensor:
    """
    Simulates motion artifacts (ghosting) in K-space.

    **MRI Physical Representation:**
    Patient movement during the acquisition (especially during the Phase Encoding step) causes
    positional inconsistencies in the frequency data. This manifests as "ghosts" or faint copies
    of the anatomy propagated along the Phase Encoding direction.
    
    This function applies a phase error modulation in K-space to mathematically reproduce this effect.
    """
    k_space = torch.fft.fftn(volume, dim=(1, 2, 3))
    k_space = torch.fft.fftshift(k_space, dim=(1, 2, 3))
    dims = volume.shape[1:]
    phase_axis_len = dims[axis]
    indices = torch.arange(phase_axis_len, device=volume.device)
    phase_error = torch.exp(1j * intensity * torch.sin(2 * np.pi * num_ghosts * indices / phase_axis_len))
    view_shape = [1, 1, 1, 1]
    view_shape[axis + 1] = phase_axis_len
    phase_error = phase_error.view(*view_shape)
    k_space_corrupted = k_space * phase_error
    k_space_corrupted = torch.fft.ifftshift(k_space_corrupted, dim=(1, 2, 3))
    return torch.abs(torch.fft.ifftn(k_space_corrupted, dim=(1, 2, 3)))

def apply_kspace_spike(volume: torch.Tensor, intensity: float = 5.0) -> torch.Tensor:
    """
    Simulates RF spikes (zipper artifacts).

    **MRI Physical Representation:**
    Stray radio frequency (RF) interference (e.g., from a light bulb or unshielded equipment)
    can appear as a high-intensity "spike" at a specific point in K-space.
    When reconstructed via Inverse FFT, a single point in K-space transforms into a 
    periodic stripe or "herringbone" pattern across the entire image.
    """
    k_space = torch.fft.fftn(volume, dim=(1, 2, 3))
    C, D, H, W = volume.shape
    rd, rh, rw = torch.randint(0, D, (1,)), torch.randint(0, H, (1,)), torch.randint(0, W, (1,))
    spike_val = torch.max(torch.abs(k_space)) * intensity
    k_space[:, rd, rh, rw] += spike_val
    return torch.abs(torch.fft.ifftn(k_space, dim=(1, 2, 3)))

def apply_aliasing(volume: torch.Tensor, axis: int, fold_pct: float = 0.2) -> torch.Tensor:
    """
    Simulates wrap-around aliasing (fold-over artifacts).

    **MRI Physical Representation:**
    If the Field of View (FOV) is smaller than the anatomy in the Phase Encoding direction,
    signal from outside the FOV "wraps around" to the opposite side of the image.
    This is common in abdominal or shoulder MRI where the body extends beyond the selected box.
    """
    dims = list(volume.shape)
    spatial_axis = axis + 1
    original_size = dims[spatial_axis]
    shift = int(original_size * fold_pct / 2)
    wrapped = torch.roll(volume, shifts=shift, dims=spatial_axis) * 0.5 + \
              torch.roll(volume, shifts=-shift, dims=spatial_axis) * 0.5
    return (volume + wrapped) / 1.5


# ---  Main Simulator Class ---

class MRIArtifactSimulator(torch.nn.Module):
    """
    The physics engine that orchestrates the degradation pipeline.

    **Simulation Pipeline:**
    1.  **Slice Profile (Physics-Based):** Applies realistic slice blurring (Trapezoidal/Boxcar) before downsampling.
    2.  **K-Space Corruptions:** Transforms data to frequency domain to add Motion Ghosts and RF Spikes.
    3.  **Aliasing:** Simulates FOV wrap-around in spatial domain.
    4.  **Sampling (Resolution Loss):** Performs FFT cropping. This is the physically correct way to
        simulate "Low Resolution." MRI resolution is defined by how far out in K-space we sample (k-max).
        Cropping the high frequencies in K-space is exactly what happens when a scanner acquires a lower matrix size.
    5.  **Thermal Noise:** Adds Rician/Gaussian noise to simulate electronic noise in the receive coils.
    """

    def __init__(
        self,
        volume_res: List[float],
        target_res: List[float],
        output_shape: List[int] = None,  # Make optional
        prob_motion: float = 0.2,
        prob_spike: float = 0.1,
        prob_aliasing: float = 0.1,
        prob_noise: float = 0.95,
        noise_std: float = 0.05,
        motion_intensity: float = 1.5,
        spike_intensity: float = 0.04,
        upsample_mode: str = "nearest",
        preserve_input_shape: bool = False,
    ):
        super().__init__()
        self.volume_res = torch.tensor(volume_res, dtype=torch.float32)
        self.target_res = torch.tensor(target_res, dtype=torch.float32)
        self.output_shape = output_shape
        self.prob_motion = prob_motion
        self.prob_spike = prob_spike
        self.prob_aliasing = prob_aliasing
        self.prob_noise = prob_noise
        self.noise_std = noise_std
        self.motion_intensity = motion_intensity
        self.spike_intensity = spike_intensity
        self.upsample_mode = upsample_mode
        self.preserve_input_shape = preserve_input_shape
        self.physics_engine = SliceProfilePhysics(profile_type='trapezoid', edge_width=0.1)

    def forward(
        self,
        image: torch.Tensor,
        acquisition_res: torch.Tensor,
        thickness: Optional[torch.Tensor] = None,
        enable_motion: Optional[torch.Tensor] = None,
        enable_spike: Optional[torch.Tensor] = None,
        enable_aliasing: Optional[torch.Tensor] = None,
        enable_noise: Optional[torch.Tensor] = None,
        motion_axis: Optional[torch.Tensor] = None,
        aliasing_axis: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        """
        Apply MRI artifact simulation.

        Args:
            image: Input volume (B, C, D, H, W)
            acquisition_res: Resolution per batch (B, 3) or (3,)
            thickness: Slice thickness per batch (B, 3) or (3,)
            enable_motion: Pre-sampled bool mask (B,) for motion artifacts
            enable_spike: Pre-sampled bool mask (B,) for spike artifacts
            enable_aliasing: Pre-sampled bool mask (B,) for aliasing
            enable_noise: Pre-sampled bool mask (B,) for noise
            motion_axis: Pre-sampled axis (B,) for motion direction
            aliasing_axis: Pre-sampled axis (B,) for aliasing direction

        Returns:
            Tuple of (simulated_lr, interp_masks) where interp_masks (B, 1, D, H, W) marks
            interpolated slices (1) vs acquired slices (0). If return_intermediate=True,
            returns (simulated_lr, true_lr, interp_masks).
        """
        batch_size = image.shape[0]
        device = image.device
        outputs = []
        interp_mask_outputs = []
        true_lr_outputs = [] if return_intermediate else None

        for b in range(batch_size):
            img = image[b]  # (C, D, H, W)
            original_input_shape = img.shape[1:]  # Store original spatial shape (D, H, W)
            acq_res = acquisition_res[b] if acquisition_res.ndim > 1 else acquisition_res
            acq_res = acq_res.to(device)

            if thickness is not None:
                thk = thickness[b] if thickness.ndim > 1 else thickness
                thk = thk.to(device)
            else:
                thk = acq_res

            # --- STEP 1: PSF Blurring (Slice Profile / Partial Volume) ---
            img = self.physics_engine(
                img,
                resolution=self.volume_res.to(device), # Current HR resolution
                thickness=thk # Target slice thickness
            )

            # --- STEP 2: K-Space Artifacts (Motion / Spike) ---
            # Use pre-sampled decisions if provided, otherwise sample randomly
            if enable_motion is not None:
                should_apply_motion = enable_motion[b].item()
            else:
                should_apply_motion = torch.rand(1).item() < self.prob_motion

            if should_apply_motion:
                if motion_axis is not None:
                    axis = motion_axis[b].item()
                else:
                    axis = torch.randint(0, 3, (1,)).item()
                img = apply_kspace_motion_ghosting(img, axis=axis, intensity=self.motion_intensity)

            if enable_spike is not None:
                should_apply_spike = enable_spike[b].item()
            else:
                should_apply_spike = torch.rand(1).item() < self.prob_spike

            if should_apply_spike:
                img = apply_kspace_spike(img, intensity=self.spike_intensity)

            # --- STEP 3: Aliasing ---
            if enable_aliasing is not None:
                should_apply_aliasing = enable_aliasing[b].item()
            else:
                should_apply_aliasing = torch.rand(1).item() < self.prob_aliasing

            if should_apply_aliasing:
                if aliasing_axis is not None:
                    axis = aliasing_axis[b].item()
                else:
                    axis = torch.randint(0, 3, (1,)).item()
                img = apply_aliasing(img, axis=axis, fold_pct=0.15)

            # --- STEP 4: Resolution Reduction (FFT Downsample) ---
            factors = acq_res / self.volume_res.to(device)
            downsample_axis = torch.argmax(factors).item()
            factor = factors[downsample_axis].item()

            true_lr_img = None
            if factor > 1.1:
                # Downsample via FFT cropping (simulates acquisition matrix limit)
                original_shape = img.shape
                spatial_axis = downsample_axis + 1
                new_size = int(round(original_shape[spatial_axis] / factor))

                fft_volume = torch.fft.fftn(img, dim=(1, 2, 3))
                fft_volume = torch.fft.fftshift(fft_volume, dim=(1, 2, 3))

                center_idx = original_shape[spatial_axis] // 2
                crop_start = center_idx - new_size // 2
                crop_end = crop_start + new_size

                if downsample_axis == 0:
                    cropped_fft = fft_volume[:, crop_start:crop_end, :, :]
                elif downsample_axis == 1:
                    cropped_fft = fft_volume[:, :, crop_start:crop_end, :]
                else:
                    cropped_fft = fft_volume[:, :, :, crop_start:crop_end]

                cropped_fft = torch.fft.ifftshift(cropped_fft, dim=(1, 2, 3))
                scale_factor = new_size / original_shape[spatial_axis]
                img = torch.real(torch.fft.ifftn(cropped_fft, dim=(1, 2, 3))) * scale_factor

                # Capture true LR if requested (before upsample)
                if return_intermediate:
                    true_lr_img = img.clone()

                # Determine target shape based on configuration
                if self.preserve_input_shape:
                    # Upsample back to original input shape
                    target_shape = original_input_shape
                elif self.output_shape is not None:
                    # Use fixed output shape (original behavior)
                    target_shape = self.output_shape
                else:
                    # No target specified and no flag set - keep current size
                    target_shape = None

                # Upsample if target_shape is specified and differs from current
                if target_shape is not None and list(img.shape[1:]) != list(target_shape):
                    img = torch.nn.functional.interpolate(
                        img.unsqueeze(0),
                        size=target_shape,
                        mode=self.upsample_mode,
                    ).squeeze(0)

            # Build interpolation mask: 1 = interpolated slice, 0 = acquired slice
            if factor > 1.1:
                interp_mask = torch.ones(1, *img.shape[1:], device=device)
                upsampled_size = img.shape[spatial_axis]
                acquired_indices = sorted(set(
                    min(round(i * upsampled_size / new_size), upsampled_size - 1)
                    for i in range(new_size)
                ))
                acquired_indices = torch.tensor(acquired_indices, device=device)
                if downsample_axis == 0:
                    interp_mask[:, acquired_indices, :, :] = 0
                elif downsample_axis == 1:
                    interp_mask[:, :, acquired_indices, :] = 0
                else:
                    interp_mask[:, :, :, acquired_indices] = 0
            else:
                interp_mask = torch.zeros(1, *img.shape[1:], device=device)
            interp_mask_outputs.append(interp_mask.unsqueeze(0))

            # Store true LR output if capturing intermediates
            if return_intermediate:
                if true_lr_img is not None:
                    true_lr_outputs.append(true_lr_img.unsqueeze(0))
                else:
                    # No downsampling occurred, use current img
                    true_lr_outputs.append(img.clone().unsqueeze(0))

            # --- STEP 5: Noise ---
            if enable_noise is not None:
                should_apply_noise = enable_noise[b].item()
            else:
                should_apply_noise = self.prob_noise > 0 and torch.rand(1).item() < self.prob_noise

            if should_apply_noise:
                n1 = torch.randn_like(img) * self.noise_std
                n2 = torch.randn_like(img) * self.noise_std
                img = torch.sqrt((img + n1)**2 + n2**2)

            outputs.append(img.unsqueeze(0))

        final_output = torch.cat(outputs, dim=0)
        interp_masks = torch.cat(interp_mask_outputs, dim=0)

        if return_intermediate:
            true_lr_output = torch.cat(true_lr_outputs, dim=0)
            return final_output, true_lr_output, interp_masks
        else:
            return final_output, interp_masks


class FOVSliceDrop(nn.Module):
    """
    Simulates incomplete FOV by dropping slices from the through-plane axis.
    
    This correctly mimics real clinical scenarios where:
    - Axial stacks may miss top (vertex) or bottom (skull base) slices
    - Coronal stacks may miss anterior or posterior slices
    - Sagittal stacks may miss lateral slices
    
    IMPORTANT: This should be applied to the UPSAMPLED LR stacks.
    The operation simply zeros out slices - no resizing, no interpolation,
    no distortion to the remaining anatomy.
    
    Args:
        prob: Probability of applying FOV drop to each orientation
        min_keep_fraction: Minimum fraction of slices to keep (e.g., 0.5 = keep at least 50%)
        max_keep_fraction: Maximum fraction to keep (e.g., 0.95 = keep at most 95%)
    """
    
    def __init__(
        self,
        prob: float = 0.4,
        min_keep_fraction: float = 0.5,
        max_keep_fraction: float = 0.95,
        force_both_sides: bool = False,
    ):
        super().__init__()
        self.prob = prob
        self.min_keep_fraction = min_keep_fraction
        self.max_keep_fraction = max_keep_fraction
        self.force_both_sides = force_both_sides
        
        # Map orientation index to through-plane axis
        # Axial (0) → W axis (S direction)
        # Coronal (1) → H axis (A direction)
        # Sagittal (2) → D axis (R direction)
        self.orientation_to_axis = {0: 4, 1: 3, 2: 2}  # In (B, C, D, H, W) format
    
    def forward(
        self,
        image: torch.Tensor,
        orientation_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply FOV slice dropping.
        
        Args:
            image: Input volume (B, C, D, H, W) - the upsampled LR stack
            orientation_idx: 0=Axial, 1=Coronal, 2=Sagittal
        
        Returns:
            cropped_image: Image with some slices zeroed out (same shape as input)
            spatial_mask: Binary mask (B, 1, D, H, W) indicating valid slices (1=valid, 0=dropped)
        """
        B, C, D, H, W = image.shape
        device = image.device
        
        # Initialize output and mask
        output = image.clone()
        spatial_mask = torch.ones(B, 1, D, H, W, device=device)
        
        # Get the through-plane axis for this orientation
        axis = self.orientation_to_axis[orientation_idx]
        axis_size = image.shape[axis]
        
        for b in range(B):
            # Decide whether to apply cropping to this sample
            if torch.rand(1).item() > self.prob:
                continue
            
            # Sample how many slices to keep
            keep_fraction = (
                torch.rand(1).item() * (self.max_keep_fraction - self.min_keep_fraction) 
                + self.min_keep_fraction
            )
            n_keep = max(1, int(axis_size * keep_fraction))
            n_drop = axis_size - n_keep
            
            if n_drop == 0:
                continue
            
            # Decide which end to drop from (or both)
            if self.force_both_sides:
                drop_mode = 2  # Always both sides
            else:
                drop_mode = torch.randint(0, 3, (1,)).item()  # 0=start, 1=end, 2=both
            
            if drop_mode == 0:
                # Drop from start (e.g., bottom of head for axial)
                drop_start, drop_end = 0, n_drop
            elif drop_mode == 1:
                # Drop from end (e.g., top of head for axial)
                drop_start, drop_end = axis_size - n_drop, axis_size
            else:
                # Drop from both ends
                drop_each = n_drop // 2
                drop_start_left, drop_end_left = 0, drop_each
                drop_start_right = axis_size - (n_drop - drop_each)
                drop_end_right = axis_size
            
            # Create slice objects for zeroing
            # This is cleaner than creating full masks
            if axis == 2:  # D axis
                if drop_mode == 2:
                    output[b, :, :drop_each, :, :] = 0
                    output[b, :, drop_start_right:, :, :] = 0
                    spatial_mask[b, :, :drop_each, :, :] = 0
                    spatial_mask[b, :, drop_start_right:, :, :] = 0
                else:
                    output[b, :, drop_start:drop_end, :, :] = 0
                    spatial_mask[b, :, drop_start:drop_end, :, :] = 0
                    
            elif axis == 3:  # H axis
                if drop_mode == 2:
                    output[b, :, :, :drop_each, :] = 0
                    output[b, :, :, drop_start_right:, :] = 0
                    spatial_mask[b, :, :, :drop_each, :] = 0
                    spatial_mask[b, :, :, drop_start_right:, :] = 0
                else:
                    output[b, :, :, drop_start:drop_end, :] = 0
                    spatial_mask[b, :, :, drop_start:drop_end, :] = 0
                    
            elif axis == 4:  # W axis
                if drop_mode == 2:
                    output[b, :, :, :, :drop_each] = 0
                    output[b, :, :, :, drop_start_right:] = 0
                    spatial_mask[b, :, :, :, :drop_each] = 0
                    spatial_mask[b, :, :, :, drop_start_right:] = 0
                else:
                    output[b, :, :, :, drop_start:drop_end] = 0
                    spatial_mask[b, :, :, :, drop_start:drop_end] = 0
        
        return output, spatial_mask


class MultiOrientationFOVDrop(nn.Module):
    """
    Applies FOV slice dropping to multiple orientation stacks.
    
    Ensures that not ALL orientations lose the same region - this would
    make reconstruction impossible. At least one orientation should
    cover each anatomical region.
    
    Args:
        prob_per_orientation: Probability of dropping slices for each orientation
        min_keep_fraction: Minimum fraction of slices to keep
        max_keep_fraction: Maximum fraction of slices to keep
        ensure_coverage: If True, ensures complementary coverage (at least one
                        orientation covers each region)
    """
    
    def __init__(
        self,
        prob_per_orientation: float = 0.4,
        min_keep_fraction: float = 0.5,
        max_keep_fraction: float = 0.95,
        ensure_coverage: bool = True,
        force_both_sides: bool = False,
    ):
        super().__init__()
        self.prob = prob_per_orientation
        self.min_keep = min_keep_fraction
        self.max_keep = max_keep_fraction
        self.ensure_coverage = ensure_coverage
        self.force_both_sides = force_both_sides
        
        self.dropper = FOVSliceDrop(
            prob=1.0,  # We control probability at this level
            min_keep_fraction=min_keep_fraction,
            max_keep_fraction=max_keep_fraction,
            force_both_sides=force_both_sides
        )
    
    def forward(
        self,
        lr_stacks: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Apply FOV dropping to a list of orientation stacks.
        
        Args:
            lr_stacks: List of 3 tensors [axial, coronal, sagittal], each (B, C, D, H, W)
        
        Returns:
            dropped_stacks: List of 3 tensors with FOV dropping applied
            spatial_masks: List of 3 mask tensors (B, 1, D, H, W)
        """
        B = lr_stacks[0].shape[0]
        device = lr_stacks[0].device
        
        dropped_stacks = []
        spatial_masks = []
        
        # Decide which orientations to drop FOR EACH BATCH ELEMENT
        for b in range(B):
            # Sample which orientations get dropped for this sample
            drop_decisions = [torch.rand(1).item() < self.prob for _ in range(3)]
            
            # If ensure_coverage and all would be dropped, keep one
            if self.ensure_coverage and all(drop_decisions):
                # Keep one random orientation fully intact
                keep_idx = torch.randint(0, 3, (1,)).item()
                drop_decisions[keep_idx] = False
        
            # Store decisions for this batch element
            if b == 0:
                batch_drop_decisions = [drop_decisions]
            else:
                batch_drop_decisions.append(drop_decisions)
        
        # Apply dropping per orientation
        for orient_idx in range(3):
            stack = lr_stacks[orient_idx].clone()
            B, C, D, H, W = stack.shape
            mask = torch.ones(B, 1, D, H, W, device=device)
            
            for b in range(B):
                if batch_drop_decisions[b][orient_idx]:
                    # Apply dropping to this sample
                    single_stack = stack[b:b+1]
                    dropped, single_mask = self.dropper(single_stack, orient_idx)
                    stack[b:b+1] = dropped
                    mask[b:b+1] = single_mask
            
            dropped_stacks.append(stack)
            spatial_masks.append(mask)
        
        return dropped_stacks, spatial_masks


class HRLRDataGenerator:
    """
    Domain randomization pipeline using frequency-domain downsampling.

    This is an alternative to the spatial-domain approach in data.py.
    Uses FFT-based k-space cropping for more realistic MRI simulation.

    Args:
        atlas_res: Resolution of input HR images in mm [x, y, z]
        target_res: Target output resolution in mm [x, y, z]
        output_shape: Output spatial shape [D, H, W]
        min_resolution: Minimum (highest quality) resolution [x, y, z]
        max_res_aniso: Maximum anisotropic resolution [x, y, z]
        randomise_res: If True, randomize acquisition resolution
        apply_intensity_aug: If True, apply intensity augmentation to LR
        clip_to_unit_range: If True, clip outputs to [0, 1] range
        orientation_dropout_prob: Probability of applying orientation dropout (0.0-1.0).
                                 When applied, randomly drops 1-2 orientations to simulate
                                 missing views during inference. Default: 0.0 (no dropout)
        min_orientations: Minimum number of orientations to keep after dropout (1-3).
                         Default: 1 (allows training with single views)
        drop_orientations: Specific orientations to drop (0=Axial, 1=Coronal, 2=Sagittal).
                          If specified, these orientations will ALWAYS be dropped (deterministic).
                          Mutually exclusive with random orientation_dropout_prob. Default: None
    """

    def __init__(
        self,
        atlas_res: list = [1.0, 1.0, 1.0],
        target_res: list = [1.0, 1.0, 1.0],
        output_shape: list = None,  
        # Probabilities
        prob_motion: float = 0.2,
        prob_spike: float = 0.05,
        prob_aliasing: float = 0.1,
        prob_bias_field: float = 0.5,
        prob_noise: float = 0.8,
        fov_augmentation_prob: float = 0.7,
        # Resolution simulation
        min_resolution: list = [1.0, 1.0, 1.0],
        max_res_aniso: list = [9.0, 9.0, 9.0],
        randomise_res: bool = True,
        # Toggles
        apply_intensity_aug: bool = True,
        clip_to_unit_range: bool = True,
        # Orientation dropout
        orientation_dropout_prob: float = 0.0,
        min_orientations: int = 1,
        drop_orientations: list = None,
        # Interpolation mode
        upsample_mode: str = "nearest",
        preserve_input_shape: bool = False,
        # LR stack saving
        return_intermediate: bool = True,
        # Motion
        motion_intensity: float = 0.5,
    ):
        self.atlas_res = atlas_res
        self.target_res = target_res
        self.output_shape = output_shape
        self.randomise_res = randomise_res
        self.apply_intensity_aug = apply_intensity_aug
        self.clip_to_unit_range = clip_to_unit_range

        self.prob_bias_field = prob_bias_field
        self.fov_augmentation_prob = fov_augmentation_prob
        self.upsample_mode = upsample_mode
        self.preserve_input_shape = preserve_input_shape  
        self.return_intermediate = return_intermediate

        # Orientation dropout parameters
        self.orientation_dropout_prob = orientation_dropout_prob
        self.min_orientations = max(1, min(min_orientations, 3))  # Clamp to [1, 3]
        self.drop_orientations = drop_orientations

        # Validate drop_orientations if specified
        if self.drop_orientations is not None:
            if len(self.drop_orientations) >= 3:
                raise ValueError("Cannot drop all orientations. Must keep at least one.")
            if any(idx not in [0, 1, 2] for idx in self.drop_orientations):
                raise ValueError("drop_orientations must contain indices 0, 1, or 2")

        # 1. Resolution Sampler
        if randomise_res:
            self.res_sampler = SampleResolution(
                min_resolution=min_resolution,
                max_res_iso=None,
                max_res_aniso=max_res_aniso,
                prob_iso=0.0, 
                prob_min=0.05,
                return_thickness=True,
            )

        # 2. Bias Field
        self.bias = BiasFieldCorruption(
            bias_field_std=0.3, bias_scale=0.025, prob=1.0 
        )

        # 3. Intensity Augmentation
        if apply_intensity_aug:
            # Note: Gamma is applied to normalized data, which works fine for [0,1]
            self.intensity_aug = IntensityAugmentation(
                clip=False, # Don't clip here, we handle it globally
                gamma_std=0.5, 
                channel_wise=False, 
                prob_gamma=0.5
            )

        # 4. MRI Physics Simulator
        self.artifact_simulator = MRIArtifactSimulator(
            volume_res=atlas_res,
            target_res=target_res,
            output_shape=output_shape,
            prob_motion=prob_motion,
            prob_spike=prob_spike,
            prob_aliasing=prob_aliasing,
            prob_noise=prob_noise,
            noise_std=0.02,
            motion_intensity=motion_intensity,
            upsample_mode=upsample_mode,
            preserve_input_shape=preserve_input_shape,
        )

        # Normalization helper
        # We only use this ONCE on the HR image
        self.normalizer = ScaleIntensityRangePercentiles(
            lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        )

        # FOV augmentation (cached to avoid per-call object creation)
        if fov_augmentation_prob > 0:
            self.fov_augmenter = MultiOrientationFOVDrop(
                prob_per_orientation=fov_augmentation_prob,
                min_keep_fraction=0.40,
                max_keep_fraction=0.70,
                ensure_coverage=True,
                force_both_sides=True,
            )

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image to [0, 1] range using percentile scaling."""
        return self.normalizer(image)

    def _create_orientation_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Create orientation mask for dropout (simulating missing views).

        Two modes:
        1. Deterministic: If drop_orientations is specified, always drop those indices
        2. Random: If orientation_dropout_prob > 0, randomly drop orientations

        Args:
            batch_size: Number of volumes in batch
            device: Device to create tensors on

        Returns:
            Boolean mask of shape (batch_size, 3) where True indicates orientation is present.
            Always ensures at least one orientation is present per sample.
        """
        # Start with all orientations present
        mask = torch.ones(batch_size, 3, dtype=torch.bool, device=device)

        # Mode 1: Deterministic dropout (takes precedence)
        if self.drop_orientations is not None and len(self.drop_orientations) > 0:
            for idx in self.drop_orientations:
                mask[:, idx] = False
            return mask

        # Mode 2: Random dropout (existing behavior)
        if self.orientation_dropout_prob > 0.0:
            for b in range(batch_size):
                # Decide whether to apply dropout for this sample
                if torch.rand(1).item() < self.orientation_dropout_prob:
                    # Randomly select how many orientations to keep
                    # Keep between min_orientations and 3
                    num_keep = torch.randint(
                        self.min_orientations,
                        4,  # Upper bound is exclusive, so this gives [min_orientations, 3]
                        (1,)
                    ).item()

                    if num_keep < 3:
                        # Randomly select which orientations to keep
                        indices = torch.randperm(3)[:num_keep]
                        mask[b, :] = False
                        mask[b, indices] = True

        return mask

    def _create_orthogonal_resolutions(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Create three orthogonal anisotropic resolution configurations.

        For MRI thick-slice acquisition, we sample:
        - high_res (in-plane, isotropic across 2 axes)
        - low_res (through-plane, anisotropic along 1 axis)

        Each stack has 2 high-res axes (in-plane) and 1 low-res axis (through-plane):
        - Stack 0 (Axial): [high, high, low] - through-plane is S (axis 2)
        - Stack 1 (Coronal): [high, low, high] - through-plane is A (axis 1)
        - Stack 2 (Sagittal): [low, high, high] - through-plane is R (axis 0)

        After RAS orientation:
          Axis 0 = R (Right-Left)      → Sagittal slices stack here
          Axis 1 = A (Anterior-Post.)  → Coronal slices stack here
          Axis 2 = S (Superior-Inf.)   → Axial slices stack here

        Args:
            batch_size: Number of volumes in batch
            device: Device to create tensors on

        Returns:
            Tuple of (resolutions_list, thickness_list), each containing 3 tensors
            of shape (batch_size, 3) for the three orthogonal stacks
        """
        # Handle min_res - could be tensor or list
        if hasattr(self, 'res_sampler'):
            min_res_value = self.res_sampler.min_res
            if isinstance(min_res_value, torch.Tensor):
                min_res = min_res_value.detach().clone().to(device)
            else:
                min_res = torch.tensor(min_res_value, device=device)
        else:
            min_res = torch.tensor([1.0, 1.0, 1.0], device=device)

        # Handle max_res - could be tensor or list
        if hasattr(self, 'res_sampler'):
            max_res_value = self.res_sampler.max_res_aniso
            if isinstance(max_res_value, torch.Tensor):
                max_res = max_res_value.detach().clone().to(device)
            else:
                max_res = torch.tensor(max_res_value, device=device)
        else:
            max_res = torch.tensor([9.0, 9.0, 9.0], device=device)

        # Use min of min_res as the high resolution (isotropic in-plane)
        high_res_value = min_res.min().item()

        # Sample low resolution (through-plane) ONCE per patient
        low_res_samples = []
        for b in range(batch_size):
            if self.randomise_res:
                # Sample from range [high_res_value, max(max_res)]
                low_res = torch.rand(1, device=device).item() * \
                         (max_res.max() - high_res_value) + high_res_value
            else:
                low_res = max_res.max().item()
            low_res_samples.append(low_res)

        # Mapping stack index to through-plane axis (after RAS orientation)
        # Stack 0 (Axial) → through-plane is axis 2 (S)
        # Stack 1 (Coronal) → through-plane is axis 1 (A)
        # Stack 2 (Sagittal) → through-plane is axis 0 (R)
        stack_to_through_plane = [2, 1, 0]

        resolutions = []
        thicknesses = []

        for stack_idx in range(3):
            through_plane_axis = stack_to_through_plane[stack_idx]

            res_batch = []
            thick_batch = []

            for b in range(batch_size):
                low_res = low_res_samples[b]  # Use same low_res for this patient across all stacks

                res = torch.zeros(3, device=device)
                thick = torch.zeros(3, device=device)

                for axis in range(3):
                    if axis == through_plane_axis:
                        # Through-plane: LOW resolution (thick slices)
                        res[axis] = low_res
                        thick[axis] = low_res
                    else:
                        # In-plane: HIGH resolution (sharp images)
                        res[axis] = high_res_value
                        thick[axis] = high_res_value

                res_batch.append(res)
                thick_batch.append(thick)

            resolutions.append(torch.stack(res_batch, dim=0))  # (batch_size, 3)
            thicknesses.append(torch.stack(thick_batch, dim=0))  # (batch_size, 3)

        return resolutions, thicknesses

    @staticmethod
    def _normalize_lr_volume(lr_b: torch.Tensor) -> torch.Tensor:
        """Soft-clip and min-max normalize a single LR volume to [0, 1]."""
        low = torch.quantile(lr_b, 0.005)
        high = torch.quantile(lr_b, 0.995)
        lr_b = torch.clamp(lr_b, low, high)
        min_val = lr_b.min()
        max_val = lr_b.max()
        return (lr_b - min_val) / (max_val - min_val + 1e-8)

    @torch.no_grad()
    def generate_paired_data(
        self,
        hr_images: torch.Tensor,
        return_resolution: bool = False,
        return_intermediate: bool = False,
        sample_info: Optional[Dict] = None,
    ):
        """
        Generate paired LR-HR training data with three orthogonal LR stacks.

        Creates three low-resolution stacks from each high-resolution volume.
        Input is assumed to be RAS-oriented (from Orientationd transform).

        After RAS orientation:
          Axis 0 = R (Right-Left)      → Sagittal slices stack here
          Axis 1 = A (Anterior-Post.)  → Coronal slices stack here
          Axis 2 = S (Superior-Inf.)   → Axial slices stack here

        Each stack simulates thick-slice MRI with 2 in-plane high-res axes and 1 through-plane low-res axis:
        - Stack 0 (Axial): High-res in R,A (axes 0,1), Low-res in S (axis 2)
        - Stack 1 (Coronal): High-res in R,S (axes 0,2), Low-res in A (axis 1)
        - Stack 2 (Sagittal): High-res in A,S (axes 1,2), Low-res in R (axis 0)

        Args:
            hr_images: High-resolution input images (B, C, D, H, W) in RAS orientation
            return_resolution: If True, also return resolution and thickness info

        Returns:
            If return_resolution=False:
                (lr_stack_list, hr_augmented, orientation_mask)
                where lr_stack_list contains 3 LR volumes and orientation_mask is (B, 3)
            If return_resolution=True:
                (lr_stack_list, hr_augmented, resolutions, thicknesses, orientation_mask)
        """
        batch_size = hr_images.shape[0]
        device = hr_images.device

        # === STEP 1: NORMALIZE HR TO FIXED TARGET DOMAIN ===
        hr_augmented = self._normalize_image(hr_images)

        # Compute HR normalization stats for consistent LR normalization
        hr_norm_stats = []
        for b in range(batch_size):
            hr_b = hr_augmented[b]
            low = torch.quantile(hr_b, 0.005)
            high = torch.quantile(hr_b, 0.995)
            hr_norm_stats.append((low, high))

        # === STEP 2: CREATE THREE ORTHOGONAL LR STACKS ===
        # Generate resolution configurations for three orthogonal orientations
        resolutions, thicknesses = self._create_orthogonal_resolutions(batch_size, device)

        # === STEP 3: PRE-SAMPLE ARTIFACT DECISIONS (ONCE PER PATIENT) ===
        # These will be applied consistently across all 3 orthogonal stacks

        # A. Bias field corruption - sample per patient
        apply_bias_field = torch.rand(batch_size, device=device) < self.prob_bias_field

        # B. Intensity augmentation - apply to all if enabled
        apply_intensity_aug = self.apply_intensity_aug

        # C. Physics simulation artifacts - sample per patient
        apply_motion = torch.rand(batch_size, device=device) < self.artifact_simulator.prob_motion
        apply_spike = torch.rand(batch_size, device=device) < self.artifact_simulator.prob_spike
        apply_aliasing = torch.rand(batch_size, device=device) < self.artifact_simulator.prob_aliasing
        apply_noise = torch.rand(batch_size, device=device) < self.artifact_simulator.prob_noise

        # Sample random axes for motion and aliasing (same for all stacks per patient)
        motion_axis = torch.randint(0, 3, (batch_size,), device=device)
        aliasing_axis = torch.randint(0, 3, (batch_size,), device=device)

        # === STEP 4: APPLY DEGRADATIONS ONCE (SHARED ACROSS ALL STACKS) ===

        # A. Bias Field (Multiplicative shading) - apply per-volume with pre-sampled decisions
        hr_degraded = hr_augmented.clone()
        for b in range(batch_size):
            if apply_bias_field[b]:
                hr_degraded[b:b+1] = self.bias(hr_degraded[b:b+1])

        # B. Intensity Augmentation (Gamma) - apply per-volume if enabled
        if apply_intensity_aug:
            for b in range(batch_size):
                hr_degraded[b:b+1] = self.intensity_aug(hr_degraded[b:b+1])

        lr_stacks = []
        true_lr_stacks = []
        interp_masks = []

        for stack_idx in range(3):
            lr_images = hr_degraded.clone()

            # C. Physics Simulation (PSF, downsampling, noise, motion, aliasing)
            # Pass pre-sampled decisions to ensure consistency
            resolution = resolutions[stack_idx]
            thickness = thicknesses[stack_idx]

            result = self.artifact_simulator(
                lr_images,
                resolution,
                thickness,
                enable_motion=apply_motion,
                enable_spike=apply_spike,
                enable_aliasing=apply_aliasing,
                enable_noise=apply_noise,
                motion_axis=motion_axis,
                aliasing_axis=aliasing_axis,
                return_intermediate=return_intermediate,
            )
            if return_intermediate:
                lr_images, true_lr_images, stack_interp_masks = result
            else:
                lr_images, stack_interp_masks = result
            interp_masks.append(stack_interp_masks)

            # === STEP 5: REALISTIC LR INTENSITY NORMALIZATION ===
            # Use HR-derived normalization stats for contrast consistency
            if self.clip_to_unit_range:
                lr_norm = []
                for b in range(batch_size):
                    lr_b = lr_images[b:b+1]
                    low, high = hr_norm_stats[b]
                    lr_b = torch.clamp(lr_b, low, high)
                    lr_b = (lr_b - low) / (high - low + 1e-8)
                    lr_norm.append(lr_b)
                lr_images = torch.cat(lr_norm, dim=0)

                if return_intermediate:
                    true_lr_norm = []
                    for b in range(batch_size):
                        true_lr_b = true_lr_images[b:b+1]
                        low, high = hr_norm_stats[b]
                        true_lr_b = torch.clamp(true_lr_b, low, high)
                        true_lr_b = (true_lr_b - low) / (high - low + 1e-8)
                        true_lr_norm.append(true_lr_b)
                    true_lr_images = torch.cat(true_lr_norm, dim=0)

            lr_stacks.append(lr_images)
            if return_intermediate:
                true_lr_stacks.append(true_lr_images)

        # HR is already normalized by percentiles; clip tiny float drift
        hr_augmented = torch.clamp(hr_augmented, 0.0, 1.0)

        # === STEP 6: CREATE ORIENTATION DROPOUT MASK (if enabled) ===
        # Create orientation mask for simulating missing views
        # The mask indicates which orientations are "present" for each sample
        # This mask is passed to the fusion mechanism, which handles the dropout
        # by excluding masked-out orientations from the Product of Gaussians fusion
        orientation_mask = self._create_orientation_mask(batch_size, device)
        
        if self.fov_augmentation_prob > 0:
            lr_stacks, spatial_masks = self.fov_augmenter(lr_stacks)
        else:
            # Full coverage masks
            spatial_masks = [
                torch.ones(batch_size, 1, *lr_stacks[0].shape[-3:], device=device)
                for _ in range(3)
        ]

        if return_resolution and return_intermediate:
            return lr_stacks, true_lr_stacks, hr_augmented, resolutions, thicknesses, orientation_mask, spatial_masks, interp_masks
        elif return_resolution:
            return lr_stacks, hr_augmented, resolutions, thicknesses, orientation_mask, spatial_masks, interp_masks
        else:
            return lr_stacks, hr_augmented, orientation_mask, spatial_masks, interp_masks



# --- Dataset Wrappers ---

class GeneratorDataset(torch.utils.data.Dataset):
    """
    Wrapper that applies the HRLRDataGenerator to a base dataset.

    Returns three orthogonal low-resolution stacks for each high-resolution volume:
    - lr_axial: High resolution in depth (D) axis
    - lr_coronal: High resolution in height (H) axis
    - lr_sagittal: High resolution in width (W) axis
    """
    def __init__(
        self,
        base_dataset,
        generator,
        return_resolution,
        balanced_orientation_combos: bool = False,
    ):
        self.base_dataset = base_dataset
        self.generator = generator
        self.return_resolution = return_resolution
        self.balanced_orientation_combos = balanced_orientation_combos
        self._orientation_combo_schedule = None
        self._epoch = 0
        self._last_built_epoch = None
        self._shared_epoch = None

        if self.balanced_orientation_combos:
            import multiprocessing as mp
            self._shared_epoch = mp.Value("i", 0)
            self._build_orientation_combo_schedule()

    def _build_orientation_combo_schedule(self) -> None:
        import random

        combos = [
            torch.tensor([1, 0, 1], dtype=torch.bool),  # ax+sag
            torch.tensor([1, 1, 0], dtype=torch.bool),  # ax+cor
            torch.tensor([0, 1, 1], dtype=torch.bool),  # cor+sag
            torch.tensor([1, 1, 1], dtype=torch.bool),  # all three
        ]
        total = len(self.base_dataset)
        base_count = total // len(combos)
        remainder = total % len(combos)

        schedule = []
        for combo in combos:
            schedule.extend([combo.clone() for _ in range(base_count)])
        for i in range(remainder):
            schedule.append(combos[i].clone())

        rng = random.Random(self._epoch)
        rng.shuffle(schedule)
        self._orientation_combo_schedule = schedule
        self._last_built_epoch = self._epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        if self._shared_epoch is not None:
            self._shared_epoch.value = epoch
        if self.balanced_orientation_combos:
            self._build_orientation_combo_schedule()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        if self.balanced_orientation_combos and self._shared_epoch is not None:
            shared_epoch = self._shared_epoch.value
            if self._last_built_epoch != shared_epoch:
                self._epoch = shared_epoch
                self._build_orientation_combo_schedule()
        data = self.base_dataset[idx]
        hr_image = data["image"]

        # Extract volume name from metadata for saving
        sample_info = None
        if hasattr(self.base_dataset, 'data') and idx < len(self.base_dataset.data):
            volume_path = self.base_dataset.data[idx].get("image", "")
            volume_name = Path(volume_path).stem if volume_path else f"sample_{idx}"
            sample_info = {'volume_name': [volume_name]}  # List for batch consistency

        if hr_image.ndim == 3:
            hr_image = hr_image.unsqueeze(0)
        hr_image = hr_image.unsqueeze(0)  # (1, C, D, H, W)

        result = self.generator.generate_paired_data(
            hr_image,
            return_resolution=self.return_resolution,
            sample_info=sample_info,
        )

        if self.return_resolution:
            lr_stacks, hr_augmented, resolutions, thicknesses, orientation_mask, spatial_masks, interp_masks = result
            if self.balanced_orientation_combos:
                orientation_mask = self._orientation_combo_schedule[idx]
            # lr_stacks is a list of 3 tensors, each (1, C, D, H, W)
            # Return them as separate orientations
            return (
                [stack.squeeze(0) for stack in lr_stacks],  # List of 3 orientations
                hr_augmented.squeeze(0),  # HR ground truth
                [res.squeeze(0) for res in resolutions],  # List of 3 resolution configs
                [thick.squeeze(0) for thick in thicknesses],  # List of 3 thickness configs
                orientation_mask.squeeze(0),  # Orientation mask (3,)
                [mask.squeeze(0) for mask in spatial_masks],  # List of 3 spatial masks
                [mask.squeeze(0) for mask in interp_masks]  # List of 3 interpolation masks
            )
        else:
            lr_stacks, hr_augmented, orientation_mask, spatial_masks, interp_masks = result
            if self.balanced_orientation_combos:
                orientation_mask = self._orientation_combo_schedule[idx]
            # Return the three LR stacks as a list
            return (
                [stack.squeeze(0) for stack in lr_stacks],
                hr_augmented.squeeze(0),
                orientation_mask.squeeze(0),
                [mask.squeeze(0) for mask in spatial_masks],  # List of 3 spatial masks
                [mask.squeeze(0) for mask in interp_masks]  # List of 3 interpolation masks
            )


def create_dataset(
    image_paths: List[str],
    generator: HRLRDataGenerator,
    target_shape: Optional[List[int]] = None,
    target_spacing: Optional[List[float]] = None,
    use_cache: bool = False,
    return_resolution: bool = False,
    is_training: bool = True,
    balanced_orientation_combos: bool = False,
):
    """
    Creates the training dataset using HRLRDataGenerator.
    """
    data_dicts = [{"image": img_path} for img_path in image_paths]

    transforms = [
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
    ]

    if target_spacing is not None:
        transforms.append(Spacingd(keys=["image"], pixdim=target_spacing, mode="bilinear"))

    if target_shape is not None:
        transforms.append(CropForegroundd(keys=["image"], source_key="image"))
        if is_training:
            transforms.append(RandSpatialCropd(keys=["image"], roi_size=target_shape, random_size=False))
            transforms.append(SpatialPadd(keys=["image"], spatial_size=target_shape))
        else:
            transforms.append(CenterSpatialCropd(keys=["image"], roi_size=target_shape))
            transforms.append(SpatialPadd(keys=["image"], spatial_size=target_shape))
    
    if is_training:
        transforms.append(
            RandAffined(
                keys=["image"],
                prob=0.3,                       
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                shear_range=None,
                translate_range=(5, 5, 5),     # optional
                mode="bilinear",
                padding_mode="border",
            )
        )
    transforms.append(ToTensord(keys=["image"]))
    transform = Compose(transforms)

    if use_cache:
        dataset = CacheDataset(data=data_dicts, transform=transform, cache_rate=1.0, num_workers=4)
    else:
        dataset = Dataset(data=data_dicts, transform=transform)

    return GeneratorDataset(
        dataset,
        generator,
        return_resolution,
        balanced_orientation_combos=balanced_orientation_combos,
    )
