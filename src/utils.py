"""
Utility functions for the MS-HVED architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import pandas as pd
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union
from monai.losses import SSIMLoss as MonaiSSIM, PerceptualLoss

# =============================================================================
# Padding Utilities for Inference
# =============================================================================

def pad_to_multiple_of_32(
    volume: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pad volume to make all dimensions multiples of 32 (centered padding).

    This is used for inference with UNet models that require input dimensions
    to be divisible by 2^n_levels (e.g., 32 for 5 levels).

    Args:
        volume: Input volume array (D, H, W)

    Returns:
        Tuple of (padded volume, padding before, original shape)

    Example:
        >>> volume = np.random.rand(100, 120, 90)
        >>> padded, pad_before, orig_shape = pad_to_multiple_of_32(volume)
        >>> print(padded.shape)  # (128, 128, 96) - next multiples of 32
        >>> # Later unpad: volume = padded[pad_before[0]:pad_before[0]+orig_shape[0], ...]
    """
    shape = np.array(volume.shape)
    # Calculate target shape (next multiple of 32)
    target_shape = (np.ceil(shape / 32.0) * 32).astype("int")

    # Calculate padding (centered)
    padding = target_shape - shape
    pad_before = np.floor(padding / 2).astype("int")
    pad_after = padding - pad_before

    # Pad the volume
    padded = np.pad(
        volume,
        [(pad_before[i], pad_after[i]) for i in range(3)],
        mode="constant",
        constant_values=0,
    )

    return padded, pad_before, shape


def unpad_volume(
    volume: np.ndarray, pad_before: np.ndarray, original_shape: np.ndarray
) -> np.ndarray:
    """
    Remove padding from volume that was added by pad_to_multiple_of_32.

    Args:
        volume: Padded volume array
        pad_before: Padding amounts before (from pad_to_multiple_of_32)
        original_shape: Original shape before padding (from pad_to_multiple_of_32)

    Returns:
        Unpadded volume with original shape

    Example:
        >>> padded, pad_before, orig_shape = pad_to_multiple_of_32(volume)
        >>> # ... process padded volume ...
        >>> result = unpad_volume(processed, pad_before, orig_shape)
    """
    return volume[
        pad_before[0] : pad_before[0] + original_shape[0],
        pad_before[1] : pad_before[1] + original_shape[1],
        pad_before[2] : pad_before[2] + original_shape[2],
    ]


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_image_paths_from_csv(
    csv_path: Union[str, Path],
    base_dir: Union[str, Path],
    split: str = "train",
    acquisition_types: Optional[List[str]] = ["3D"],
    mri_classifications: Optional[List[str]] = None,
    filter_4d: bool = True,
    log_filtered_path: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """
    Load image paths from a CSV file with filtering by split, acquisition type, and MRI classification.

    Args:
        csv_path: Path to CSV file
        base_dir: Base directory to prepend to relative paths
        split: Data split to filter for ('train', 'val', or 'test')
        acquisition_types: List of acquisition types to include (default: ['3D'])
                          Set to None to include all types, or pass specific list like ['3D', '2D']
        mri_classifications: List of MRI classifications to include (e.g., ['T1', 'T2', 'FLAIR'])
                            Set to None to include all classifications
        filter_4d: If True, filters out 4D images (with time dimension) using 'dimensions' column
        log_filtered_path: Optional path to save list of filtered 4D images (CSV format)

    Returns:
        List of absolute image paths

    CSV Format:
        The CSV should have the following columns:
        - relative_path: Relative path to the image file
        - mr_acquisition_type: Type of MR acquisition ('3D' or '2D')
        - split: Data split ('train', 'val', or 'test')
        - MRI_classification (optional): MRI classification type ('T1', 'T2', 'FLAIR', etc.)
        - dimensions (optional): Image dimensions as tuple string, e.g., "(256, 256, 128)"

    Example CSV:
        relative_path,mr_acquisition_type,split,MRI_classification,dimensions
        images/scan001.nii.gz,3D,train,T1,"(256, 256, 128)"
        images/scan002.nii.gz,2D,train,T2,"(256, 256, 128)"
        images/scan003.nii.gz,3D,val,FLAIR,"(256, 256, 128, 10)"
        images/scan004.nii.gz,3D,test,T1,"(256, 256, 128)"

    Example:
        >>> # Load 3D training images only (filtering out 4D) with logging
        >>> train_paths = load_image_paths_from_csv(
        ...     'data.csv',
        ...     base_dir='/data/mri',
        ...     split='train',
        ...     filter_4d=True,
        ...     log_filtered_path='./model/filtered_4d_train.csv'
        ... )
        >>>
        >>> # Load T1 and T2 training images only
        >>> train_paths = load_image_paths_from_csv(
        ...     'data.csv',
        ...     base_dir='/data/mri',
        ...     split='train',
        ...     mri_classifications=['T1', 'T2']
        ... )
        >>>
        >>> # Load validation images of all types (including 4D)
        >>> val_paths = load_image_paths_from_csv(
        ...     'data.csv',
        ...     base_dir='/data/mri',
        ...     split='val',
        ...     acquisition_types=None,
        ...     filter_4d=False
        ... )
    """
    csv_path = Path(csv_path)
    base_dir = Path(base_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_columns = ["relative_path", "mr_acquisition_type", "split"]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"CSV missing required columns: {missing_columns}. "
            f"Found columns: {list(df.columns)}"
        )

    # Filter by split
    df_filtered = df[df["split"] == split].copy()

    if len(df_filtered) == 0:
        raise ValueError(
            f"No images found for split '{split}'. "
            f"Available splits: {df['split'].unique().tolist()}"
        )

    # Filter by acquisition type if specified
    # acquisition_types=None means include all types (no filtering)
    # acquisition_types=['3D'] is the default (3D only)
    if acquisition_types is not None:
        df_filtered = df_filtered[
            df_filtered["mr_acquisition_type"].isin(acquisition_types)
        ]

        if len(df_filtered) == 0:
            raise ValueError(
                f"No images found for split '{split}' with acquisition types {acquisition_types}. "
                f"Available acquisition types: {df[df['split'] == split]['mr_acquisition_type'].unique().tolist()}"
            )
    else:
        # Include all acquisition types
        acquisition_types = df_filtered["mr_acquisition_type"].unique().tolist()

    # Filter by MRI classification if specified
    if mri_classifications is not None and "MRI_classification" in df_filtered.columns:
        df_filtered = df_filtered[
            df_filtered["MRI_classification"].isin(mri_classifications)
        ]

        if len(df_filtered) == 0:
            raise ValueError(
                f"No images found for split '{split}' with MRI classifications {mri_classifications}. "
                f"Available MRI classifications: {df[df['split'] == split]['MRI_classification'].unique().tolist()}"
            )

        print(f"Filtered by MRI classifications: {mri_classifications}")
    elif mri_classifications is not None and "MRI_classification" not in df_filtered.columns:
        print(
            "Warning: 'MRI_classification' column not found in CSV. "
            "Cannot filter by MRI classification. Continuing with all MRI types."
        )

    # Filter out 4D images if requested
    if filter_4d and "dimensions" in df_filtered.columns:
        initial_count = len(df_filtered)

        def is_3d_image(dim_str):
            """Check if dimensions string represents a 3D image (not 4D)."""
            if pd.isna(dim_str):
                # If dimensions column is missing for this row, assume it's okay (will be caught later)
                return True

            try:
                # Parse dimension string - could be "(256, 256, 128)" or similar
                dim_str = str(dim_str).strip()
                # Remove parentheses and split by comma
                dim_str = dim_str.replace("(", "").replace(")", "").strip()
                dims = [int(d.strip()) for d in dim_str.split(",") if d.strip()]
                # Return True if exactly 3 dimensions (3D image)
                return len(dims) == 3
            except Exception as e:
                # If parsing fails, log warning and assume it's okay
                print(f"Warning: Could not parse dimensions '{dim_str}': {e}")
                return True

        # Identify 4D images before filtering
        is_3d_mask = df_filtered["dimensions"].apply(is_3d_image)
        filtered_4d_images = df_filtered[~is_3d_mask].copy()

        # Apply filter
        df_filtered = df_filtered[is_3d_mask].copy()

        filtered_count = len(filtered_4d_images)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} 4D images (with time dimension)")

            # Save filtered images to log file if requested
            if log_filtered_path is not None:
                log_path = Path(log_filtered_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)

                # Add reason column
                filtered_4d_images["filter_reason"] = "4D image (time dimension)"

                # Save to CSV
                filtered_4d_images.to_csv(log_path, index=False)
                print(f"Saved list of filtered 4D images to: {log_path}")

        if len(df_filtered) == 0:
            raise ValueError(
                f"No 3D images found for split '{split}' after filtering out 4D images. "
                f"All {initial_count} images had 4 dimensions."
            )
    elif filter_4d and "dimensions" not in df_filtered.columns:
        print(
            "Warning: 'dimensions' column not found in CSV. Cannot filter 4D images. "
            "Consider adding a 'dimensions' column to your CSV for better filtering."
        )

    # Convert relative paths to absolute paths
    image_paths = []
    missing_files = []
    total_in_csv = len(df_filtered)

    for rel_path in df_filtered["relative_path"]:
        abs_path = base_dir / rel_path
        if not abs_path.exists():
            missing_files.append(abs_path)
        else:
            image_paths.append(abs_path)

    # Report missing files
    if len(missing_files) > 0:
        print(f"Warning: {len(missing_files)}/{total_in_csv} files not found on disk")
        # Show first few missing files
        max_show = 5
        for missing_path in missing_files[:max_show]:
            print(f"  - Missing: {missing_path}")
        if len(missing_files) > max_show:
            print(f"  ... and {len(missing_files) - max_show} more")

    if len(image_paths) == 0:
        raise ValueError(
            f"No valid image files found for split '{split}'. "
            f"Check that files exist in {base_dir}"
        )

    # Build summary message
    summary = f"✓ Loaded {len(image_paths)}/{total_in_csv} {split} images"
    summary += f" (acquisition types: {acquisition_types}"
    if mri_classifications is not None:
        summary += f", MRI classifications: {mri_classifications}"
    summary += ")"
    print(summary)

    return image_paths


def get_image_paths(
    image_dir=None,
    csv_file=None,
    base_dir=None,
    split="train",
    model_dir=None,
    mri_classifications=None,
    acquisition_types=["3D"],
    filter_4d=True,
):
    """
    Get image paths from either directory or CSV file.

    Args:
        image_dir: Directory containing images (mutually exclusive with csv_file)
        csv_file: CSV file with image metadata (mutually exclusive with image_dir)
        base_dir: Base directory for relative paths in CSV (required if csv_file is provided)
        split: Data split for CSV ('train', 'val', or 'test')
        model_dir: Model directory for saving filtered images log (optional)
        mri_classifications: List of MRI classifications to include (e.g., ['T1', 'T2', 'FLAIR'])
                           Only applicable when using csv_file
        acquisition_types: List of acquisition types to include (default: ['3D'])
                          Set to None to include all types. Only applicable when using csv_file
        filter_4d: If True, filters out 4D images (default: True)
                  Only applicable when using csv_file

    Returns:
        List of image paths
    """
    if csv_file is not None:
        if base_dir is None:
            raise ValueError("--base_dir is required when using --csv_file")

        # Set up log path for filtered 4D images if model_dir is provided
        log_filtered_path = None
        if model_dir is not None and filter_4d:
            log_filtered_path = os.path.join(
                model_dir, f"filtered_4d_images_{split}.csv"
            )

        return load_image_paths_from_csv(
            csv_file,
            base_dir,
            split=split,
            acquisition_types=acquisition_types,
            mri_classifications=mri_classifications,
            filter_4d=filter_4d,
            log_filtered_path=log_filtered_path,
        )
    elif image_dir is not None:
        # Get all .nii.gz files from directory
        image_dir = Path(image_dir)
        return sorted([str(p) for p in image_dir.glob("*.nii.gz")])
    else:
        raise ValueError("Either --hr_image_dir or --csv_file must be provided")


# =============================================================================
# Model Checkpoint Management
# =============================================================================

def find_latest_checkpoint(model_dir):
    """Find the latest checkpoint in the model directory."""
    checkpoints = glob.glob(os.path.join(model_dir, "mshved_*_epoch_*.pth"))
    checkpoints += glob.glob(os.path.join(model_dir, "mshved_epoch_*.pth"))
    # Deduplicate (the first pattern may also match the second)
    checkpoints = list(set(checkpoints))
    if not checkpoints:
        return None

    # Sort by epoch number
    def get_epoch(path):
        try:
            return int(path.split('epoch_')[-1].split('.pth')[0])
        except (ValueError, IndexError):
            return -1

    checkpoints.sort(key=get_epoch)
    return checkpoints[-1] if checkpoints else None


def save_model_checkpoint(filepath, model, optimizer, epoch, loss, val_loss=None,
                          model_type="mshved", model_config=None, scheduler_state_dict=None,
                          val_metrics=None, training_config=None, global_step=None):
    """
    Save model checkpoint with complete training state.

    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Training loss
        val_loss: Validation loss (optional)
        model_type: Type of model (default: "mshved")
        model_config: Model architecture configuration
        scheduler_state_dict: Learning rate scheduler state (optional)
        val_metrics: Dictionary of validation metrics (optional)
        training_config: Full training configuration for reproducibility (optional)
        global_step: Global training step count (optional)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_loss': val_loss,
        'model_type': model_type,
        'model_config': model_config or {},
    }

    if scheduler_state_dict:
        checkpoint['scheduler_state_dict'] = scheduler_state_dict

    if val_metrics:
        checkpoint['val_metrics'] = val_metrics

    if training_config:
        checkpoint['training_config'] = training_config

    if global_step is not None:
        checkpoint['global_step'] = global_step

    torch.save(checkpoint, filepath)


def save_training_config(model_dir, args, n_train_samples, n_val_samples):
    """Save training configuration to JSON."""
    import json

    config = {
        # Dataset info
        'n_train_samples': n_train_samples,
        'n_val_samples': n_val_samples,

        # Training parameters (common to all scripts)
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'seed': args.seed,

        # Output configuration
        'output_shape': args.output_shape,
        'atlas_res': args.atlas_res,

        # Resolution parameters
        'min_resolution': args.min_resolution,
        'max_res_aniso': args.max_res_aniso,
        'randomise_res': not args.no_randomise_res,

        # Artifact probabilities
        'prob_motion': args.prob_motion,
        'prob_spike': args.prob_spike,
        'prob_aliasing': args.prob_aliasing,
        'prob_bias_field': args.prob_bias_field,
        'prob_noise': args.prob_noise,
        'apply_intensity_aug': not args.no_intensity_aug,
        'orientation_dropout_prob': args.orientation_dropout_prob,
        'min_orientations': args.min_orientations,
        'drop_orientations': args.drop_orientations,

        # Common loss config
        'perceptual_weight': args.perceptual_weight,
        'use_perceptual': args.use_perceptual,
        'perceptual_network': args.perceptual_network,

        # Training optimization
        'mixed_precision': args.mixed_precision,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'max_grad_norm': args.max_grad_norm,

        # Checkpointing
        'save_interval': args.save_interval,
        'val_interval': args.val_interval,

        'num_orientations': 3,
        'num_scales': args.num_scales,
        'final_activation': args.final_activation,
        'use_prior': True,
        'decoder_upsample_mode': args.decoder_upsample_mode,
        'recon_loss_type': args.recon_loss_type,
        'recon_weight': args.recon_weight,
        'kl_weight': args.kl_weight,
        'ssim_weight': args.ssim_weight,
        'orientation_weight': args.orientation_weight,
        'use_ssim': args.use_ssim,
        'is_fake_3d': getattr(args, 'is_fake_3d', False),
        'reconstruct_orientations': not getattr(args, 'no_reconstruct_orientations', False),
        'balanced_orientation_combos': getattr(args, 'balanced_orientation_combos', False),
        'init_filters': getattr(args, 'init_filters', 32),
        'blocks_down': getattr(args, 'blocks_down', [1, 2, 2, 4]),
        'blocks_up': getattr(args, 'blocks_up', [1, 1, 1]),
        'num_groups': getattr(args, 'num_groups', 8),
    }

    config_path = os.path.join(model_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Training configuration saved to: {config_path}")


# =============================================================================
# Evaluation Metrics
# =============================================================================

def calculate_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    use_perceptual: bool = False,
    perceptual_network: str = 'alex',
    is_fake_3d: bool = True
):
    """
    Calculate evaluation metrics.

    Args:
        pred: Predicted images (B, C, D, H, W)
        target: Target images (B, C, D, H, W)
        max_val: Maximum pixel value for PSNR calculation (default: 1.0)
        use_perceptual: Whether to compute MONAI perceptual loss metric
        perceptual_network: Network backbone for perceptual loss
        is_fake_3d: Use 2.5D slice-based processing for perceptual loss

    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # MAE (L1)
        mae = torch.abs(pred - target).mean().item()

        # MSE (L2)
        mse = ((pred - target) ** 2).mean().item()

        # RMSE
        rmse = torch.sqrt(torch.tensor(mse)).item()

        # PSNR (Peak Signal-to-Noise Ratio)
        if mse > 0:
            psnr = 10 * torch.log10(torch.tensor(max_val**2 / mse)).item()
        else:
            psnr = float("inf")

        # R² (Coefficient of Determination)
        target_mean = target.mean()
        ss_tot = ((target - target_mean) ** 2).sum().item()
        ss_res = ((target - pred) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # SSIM (Structural Similarity Index Measure)
        # MONAI's SSIMLoss returns 1 - SSIM, so we need to invert it
        # SSIM ranges from -1 to 1, where 1 is perfect similarity
        ssim_loss_fn = MonaiSSIM(spatial_dims=3, data_range=max_val)
        ssim_loss = ssim_loss_fn(pred, target).item()
        ssim = 1 - ssim_loss  # Convert from loss to similarity

        # Perceptual loss (optional)
        if use_perceptual:
            try:
                perceptual_fn = PerceptualLoss(
                    spatial_dims=3,
                    network_type=perceptual_network,
                    is_fake_3d=is_fake_3d,
                    pretrained=True
                )
                perceptual_fn.to(pred.device)
                perceptual_loss = perceptual_fn(pred, target).item()
            except Exception as e:
                warnings.warn(f"Perceptual loss computation failed: {e}")
                perceptual_loss = float('nan')
        else:
            perceptual_loss = None

        # Build metrics dictionary
        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "psnr": psnr,
            "r2": r2,
            "ssim": ssim,
        }

        # Add perceptual loss if computed
        if perceptual_loss is not None:
            metrics["perceptual_loss"] = perceptual_loss

        return metrics