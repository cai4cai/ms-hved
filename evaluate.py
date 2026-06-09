"""
Calculate Metrics for Pre-Generated Model Predictions

This script computes evaluation metrics between saved model predictions and ground truth.
It expects predictions and ground truth to be saved in subject directories.

Directory Structure:
    test_dir/
    ├── subject001/
    │   ├── HR_groundtruth.nii.gz
    │   └── {prediction_name}.nii.gz
    ├── subject002/
    │   ├── HR_groundtruth.nii.gz
    │   └── {prediction_name}.nii.gz
    └── ...

Usage:
    python evaluate.py \
        --test_dir /path/to/test \
        --prediction_name model1_sr.nii.gz \
        --output_csv results.csv \
        --output_json results.json
"""

import argparse
import csv
import json
import logging
import os
from typing import Dict, Any, List, Optional, Sequence

import numpy as np
import torch
import nibabel as nib
from monai.metrics import MultiScaleSSIMMetric
from tqdm import tqdm

# Import metrics from project
from src.utils import calculate_metrics


STACK_MASK_CANDIDATES = {
    "axial": [
        "axial_upsampled_fov_mask.nii.gz",
        "axial_fov_mask.nii.gz",
        "stack_0_axial_fov_mask.nii.gz",
    ],
    "coronal": [
        "coronal_upsampled_fov_mask.nii.gz",
        "coronal_fov_mask.nii.gz",
        "stack_1_coronal_fov_mask.nii.gz",
    ],
    "sagittal": [
        "sagittal_upsampled_fov_mask.nii.gz",
        "sagittal_fov_mask.nii.gz",
        "stack_2_sagittal_fov_mask.nii.gz",
    ],
}


def load_nifti_volume(file_path: str) -> np.ndarray:
    """
    Load and normalize a NIfTI volume.

    Args:
        file_path: Path to .nii or .nii.gz file

    Returns:
        Normalized numpy array (values in [0, 1])
    """
    nii = nib.load(file_path)
    volume = nii.get_fdata()

    # Normalize to [0, 1]
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    return volume


def find_subject_pairs(test_dir: str, prediction_name: str, gt_name: str = "HR_groundtruth.nii.gz") -> List[tuple]:
    """
    Find all subject directories with both ground truth and predictions.

    Args:
        test_dir: Root directory containing subject subdirectories
        prediction_name: Name of prediction file (e.g., "model1_sr.nii.gz")
        gt_name: Name of ground truth file (default: "HR_groundtruth.nii.gz")

    Returns:
        List of tuples: [(subject_id, gt_path, pred_path), ...]
    """
    pairs = []

    # Iterate through subdirectories
    for subject_dir in sorted(os.listdir(test_dir)):
        subject_path = os.path.join(test_dir, subject_dir)

        if not os.path.isdir(subject_path):
            continue

        gt_path = os.path.join(subject_path, gt_name)
        pred_path = os.path.join(subject_path, prediction_name)

        # Check both files exist
        if os.path.exists(gt_path) and os.path.exists(pred_path):
            pairs.append((subject_dir, gt_path, pred_path))
        else:
            missing = []
            if not os.path.exists(gt_path):
                missing.append(gt_name)
            if not os.path.exists(pred_path):
                missing.append(prediction_name)
            logging.warning(f"Skipping {subject_dir}: missing {', '.join(missing)}")

    return pairs


def center_crop_to_match(volume1: np.ndarray, volume2: np.ndarray) -> tuple:
    """
    Center crop both volumes to their minimum overlapping size.

    Args:
        volume1: First volume (e.g., ground truth)
        volume2: Second volume (e.g., prediction)

    Returns:
        Tuple of (cropped_volume1, cropped_volume2)
    """
    shape1 = np.array(volume1.shape)
    shape2 = np.array(volume2.shape)

    # Determine minimum shape along each dimension
    min_shape = np.minimum(shape1, shape2)

    # Calculate crop indices for volume1
    start1 = (shape1 - min_shape) // 2
    end1 = start1 + min_shape

    # Calculate crop indices for volume2
    start2 = (shape2 - min_shape) // 2
    end2 = start2 + min_shape

    # Perform center crop
    cropped1 = volume1[start1[0]:end1[0], start1[1]:end1[1], start1[2]:end1[2]]
    cropped2 = volume2[start2[0]:end2[0], start2[1]:end2[1], start2[2]:end2[2]]

    return cropped1, cropped2


def center_crop_volume(volume: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Center crop a volume to target_shape."""
    shape = np.array(volume.shape)
    target_shape = np.array(target_shape)
    start = (shape - target_shape) // 2
    end = start + target_shape
    return volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]


def center_crop_to_common(volumes: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Center crop all volumes to their minimum common spatial shape."""
    min_shape = np.min([np.array(volume.shape) for volume in volumes], axis=0)
    return [center_crop_volume(volume, min_shape) for volume in volumes]


def resolve_stack_mask_name(subject_dir: str, stack: str) -> str:
    """Return the first existing mask filename for a stack in a subject directory."""
    for filename in STACK_MASK_CANDIDATES[stack]:
        if os.path.exists(os.path.join(subject_dir, filename)):
            return filename
    raise FileNotFoundError(
        f"No FOV mask found for stack '{stack}' in {subject_dir}. "
        f"Tried: {', '.join(STACK_MASK_CANDIDATES[stack])}"
    )


def load_eval_mask(
    subject_dir: str,
    mask_names: Optional[Sequence[str]] = None,
    eval_stacks: Optional[Sequence[str]] = None,
    mask_mode: str = "missing",
    mask_threshold: float = 0.5,
) -> Optional[np.ndarray]:
    """
    Load and combine optional evaluation masks for a subject.

    FOV masks are usually encoded as 1=missing, 0=valid. When multiple masks
    are provided, valid regions are combined with a union so that evaluation
    covers any voxel supported by at least one stack used by the method.
    """
    names = list(mask_names or [])

    for stack in eval_stacks or []:
        names.append(resolve_stack_mask_name(subject_dir, stack))

    if not names:
        return None

    valid_mask = None
    for name in names:
        mask_path = name if os.path.isabs(name) else os.path.join(subject_dir, name)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        mask_volume = nib.load(mask_path).get_fdata()
        if mask_volume.ndim != 3:
            raise ValueError(f"Expected 3D mask at {mask_path}, got shape {mask_volume.shape}")

        if mask_mode == "missing":
            current_valid = mask_volume <= mask_threshold
        elif mask_mode == "valid":
            current_valid = mask_volume > mask_threshold
        else:
            raise ValueError(f"Unknown mask_mode: {mask_mode}")

        valid_mask = current_valid if valid_mask is None else np.logical_or(valid_mask, current_valid)

    return valid_mask.astype(bool)


def mask_bounding_box(mask: np.ndarray) -> tuple:
    """Return slicing tuple for the tight bounding box around a boolean mask."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Evaluation mask has no valid voxels")
    start = coords.min(axis=0)
    end = coords.max(axis=0) + 1
    return tuple(slice(start[axis], end[axis]) for axis in range(mask.ndim))


def compute_ncc(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Normalized Cross-Correlation (NCC).
    
    Args:
        pred: Prediction volume
        gt: Ground truth volume
    
    Returns:
        NCC value (higher is better, range: [-1, 1])
    """
    # Flatten arrays
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # Center the data
    pred_centered = pred_flat - np.mean(pred_flat)
    gt_centered = gt_flat - np.mean(gt_flat)
    
    # Compute NCC
    numerator = np.sum(pred_centered * gt_centered)
    denominator = np.sqrt(np.sum(pred_centered**2) * np.sum(gt_centered**2))
    
    ncc = numerator / (denominator + 1e-8)
    
    return float(ncc)


def compute_ms_ssim(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute MONAI multi-scale structural similarity for 3D volumes."""
    with torch.no_grad():
        ms_ssim_metric = MultiScaleSSIMMetric(spatial_dims=3, data_range=max_val)
        return float(ms_ssim_metric(pred, target).item())


def compute_masked_metrics(
    pred_volume: np.ndarray,
    gt_volume: np.ndarray,
    eval_mask: np.ndarray,
    use_perceptual: bool = False,
    perceptual_network: str = 'alex',
    is_fake_3d: bool = True,
) -> Dict[str, float]:
    """Compute metrics inside eval_mask only."""
    valid_voxels = int(eval_mask.sum())
    if valid_voxels == 0:
        raise ValueError("Evaluation mask has no valid voxels")

    pred_values = pred_volume[eval_mask]
    gt_values = gt_volume[eval_mask]
    diff = pred_values - gt_values

    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    psnr = float(10 * np.log10(1.0 / mse)) if mse > 0 else float("inf")

    target_mean = np.mean(gt_values)
    ss_tot = float(np.sum((gt_values - target_mean) ** 2))
    ss_res = float(np.sum(diff ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
        "r2": r2,
    }

    # SSIM/perceptual are spatial metrics, so compute them on the mask bbox
    # after making outside-mask voxels identical to ground truth.
    ssim_pred = pred_volume.copy()
    ssim_pred[~eval_mask] = gt_volume[~eval_mask]
    bbox = mask_bounding_box(eval_mask)
    gt_tensor = torch.from_numpy(gt_volume[bbox]).float().unsqueeze(0).unsqueeze(0)
    pred_tensor = torch.from_numpy(ssim_pred[bbox]).float().unsqueeze(0).unsqueeze(0)
    spatial_metrics = calculate_metrics(
        pred_tensor,
        gt_tensor,
        max_val=1.0,
        use_perceptual=use_perceptual,
        perceptual_network=perceptual_network,
        is_fake_3d=is_fake_3d,
    )
    metrics["ssim"] = spatial_metrics["ssim"]
    # metrics["ms_ssim"] = compute_ms_ssim(pred_tensor, gt_tensor, max_val=1.0)
    if "perceptual_loss" in spatial_metrics:
        metrics["perceptual_loss"] = spatial_metrics["perceptual_loss"]

    metrics["ncc"] = compute_ncc(pred_values, gt_values)
    metrics["coverage_fraction"] = float(valid_voxels / eval_mask.size)
    metrics["mask_voxels"] = float(valid_voxels)

    return metrics


def compute_metrics_for_pair(
    gt_path: str,
    pred_path: str,
    device: str = 'cuda',
    use_perceptual: bool = False,
    perceptual_network: str = 'alex',
    is_fake_3d: bool = True,
    mask_names: Optional[Sequence[str]] = None,
    eval_stacks: Optional[Sequence[str]] = None,
    mask_mode: str = "missing",
    mask_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all metrics for a prediction-ground truth pair.
    
    Args:
        gt_path: Path to ground truth volume
        pred_path: Path to prediction volume
        device: Computation device
        use_perceptual: Whether to compute perceptual loss
        perceptual_network: Network for perceptual loss
        is_fake_3d: Use 2.5D processing for perceptual loss
    
    Returns:
        Dictionary of metric values
    """
    # Load volumes
    gt_volume = load_nifti_volume(gt_path)
    pred_volume = load_nifti_volume(pred_path)
    eval_mask = load_eval_mask(
        subject_dir=os.path.dirname(gt_path),
        mask_names=mask_names,
        eval_stacks=eval_stacks,
        mask_mode=mask_mode,
        mask_threshold=mask_threshold,
    )

    # Handle shape mismatch
    shapes = [gt_volume.shape, pred_volume.shape]
    if eval_mask is not None:
        shapes.append(eval_mask.shape)

    if len(set(shapes)) > 1:
        logging.warning(
            f"Shape mismatch: GT {gt_volume.shape}, Pred {pred_volume.shape}"
            f"{', Mask ' + str(eval_mask.shape) if eval_mask is not None else ''}. "
            f"Applying center crop."
        )
        cropped = center_crop_to_common(
            [gt_volume, pred_volume] if eval_mask is None else [gt_volume, pred_volume, eval_mask]
        )
        gt_volume, pred_volume = cropped[0], cropped[1]
        if eval_mask is not None:
            eval_mask = cropped[2].astype(bool)
        logging.info(f"Cropped to shape: {gt_volume.shape}")

    if eval_mask is not None:
        return compute_masked_metrics(
            pred_volume=pred_volume,
            gt_volume=gt_volume,
            eval_mask=eval_mask,
            use_perceptual=use_perceptual,
            perceptual_network=perceptual_network,
            is_fake_3d=is_fake_3d,
        )

    # Convert to tensors
    gt_tensor = torch.from_numpy(gt_volume).float().unsqueeze(0).unsqueeze(0)
    pred_tensor = torch.from_numpy(pred_volume).float().unsqueeze(0).unsqueeze(0)

    # Compute standard metrics
    metrics = calculate_metrics(
        pred_tensor,
        gt_tensor,
        max_val=1.0,
        use_perceptual=use_perceptual,
        perceptual_network=perceptual_network,
        is_fake_3d=is_fake_3d
    )
    # metrics["ms_ssim"] = compute_ms_ssim(pred_tensor, gt_tensor, max_val=1.0)
    
    # Compute NCC
    metrics['ncc'] = compute_ncc(pred_volume, gt_volume)

    return metrics


def aggregate_metrics(volume_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Compute aggregate statistics across all volumes.

    Args:
        volume_results: List of per-volume metric dictionaries

    Returns:
        Dictionary of {metric_name: {mean, std, median, min, max}, ...}
    """
    if not volume_results:
        return {}

    # Get all metric names from first result
    metric_names = list(volume_results[0].keys())

    aggregates = {}
    for metric_name in metric_names:
        values = [r[metric_name] for r in volume_results]
        aggregates[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    return aggregates


def save_csv_results(
    volume_results: List[Dict[str, float]],
    subject_ids: List[str],
    aggregates: Dict[str, Dict[str, float]],
    output_path: str
):
    """
    Save per-volume and aggregate results to CSV.

    Args:
        volume_results: List of per-volume metric dictionaries
        subject_ids: List of subject IDs
        aggregates: Aggregate statistics
        output_path: Path to save CSV file
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Per-volume section
        writer.writerow(['# Per-Volume Metrics'])

        # Get metric names from first result
        if volume_results:
            metric_names = list(volume_results[0].keys())
            headers = ['subject_id'] + metric_names
            writer.writerow(headers)

            for subject_id, result in zip(subject_ids, volume_results):
                row = [subject_id] + [f"{result[m]:.6f}" for m in metric_names]
                writer.writerow(row)

        # Aggregate section
        writer.writerow([])
        writer.writerow(['# Aggregate Statistics'])
        writer.writerow(['metric', 'mean', 'std', 'median', 'min', 'max'])

        for metric_name, stats in aggregates.items():
            writer.writerow([
                metric_name,
                f"{stats['mean']:.6f}",
                f"{stats['std']:.6f}",
                f"{stats['median']:.6f}",
                f"{stats['min']:.6f}",
                f"{stats['max']:.6f}"
            ])

    logging.info(f"CSV results saved to: {output_path}")


def save_json_results(
    volume_results: List[Dict[str, float]],
    subject_ids: List[str],
    aggregates: Dict[str, Dict[str, float]],
    output_path: str,
    metadata: Dict[str, Any]
):
    """
    Save structured JSON results.

    Args:
        volume_results: List of per-volume metric dictionaries
        subject_ids: List of subject IDs
        aggregates: Aggregate statistics
        output_path: Path to save JSON file
        metadata: Additional metadata to include
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_data = {
        'metadata': metadata,
        'per_volume_results': [
            {
                'subject_id': subject_id,
                'metrics': result
            }
            for subject_id, result in zip(subject_ids, volume_results)
        ],
        'aggregate_statistics': aggregates
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logging.info(f"JSON results saved to: {output_path}")


def print_summary(aggregates: Dict[str, Dict[str, float]], num_volumes: int):
    """Print formatted summary to console."""
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    print(f"Number of volumes evaluated: {num_volumes}")
    print("\n" + "-" * 80)
    print(f"{'Metric':<15} {'Mean ± Std':<25} {'Range':<30}")
    print("-" * 80)

    for metric_name, stats in aggregates.items():
        mean_std = f"{stats['mean']:.6f} ± {stats['std']:.6f}"
        range_str = f"[{stats['min']:.6f} - {stats['max']:.6f}]"
        print(f"{metric_name:<15} {mean_std:<25} {range_str:<30}")

    print("=" * 80 + "\n")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate metrics between saved predictions and ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate model predictions
  python evaluate.py \
    --test_dir /data/test \
    --prediction_name model1_sr.nii.gz \
    --output_csv model1_metrics.csv \
    --output_json model1_metrics.json

  # Evaluate only regions supported by the axial stack
  python evaluate.py \
    --test_dir /data/test \
    --prediction_name eclare_prediction.nii.gz \
    --output_csv eclare_masked_metrics.csv \
    --eval_stacks axial
"""
    )

    # Input/Output arguments
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory containing subject subdirectories')
    parser.add_argument('--prediction_name', type=str, required=True,
                        help='Name of prediction file in each subject directory')
    parser.add_argument('--gt_name', type=str, default='HR_groundtruth.nii.gz',
                        help='Name of ground truth file (default: HR_groundtruth.nii.gz)')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to save CSV results')
    parser.add_argument('--output_json', type=str,
                        help='Path to save JSON results (optional)')

    # Optional mask-aware evaluation
    parser.add_argument('--mask_names', type=str, nargs='+',
                        help='Optional FOV/support mask filename(s) in each subject directory. '
                             'Valid regions are unioned across masks.')
    parser.add_argument('--eval_stacks', type=str, nargs='+',
                        choices=['axial', 'coronal', 'sagittal'],
                        help='Optional stack names used by the method. Resolves standard FOV mask '
                             'filenames and evaluates only supported regions.')
    parser.add_argument('--mask_mode', type=str, default='missing',
                        choices=['missing', 'valid'],
                        help='Mask encoding: missing means 1=missing/0=valid; valid means 1=valid/0=missing')
    parser.add_argument('--mask_threshold', type=float, default=0.5,
                        help='Threshold used to binarize mask values')

    # Perceptual loss arguments
    parser.add_argument('--use_perceptual', action='store_true',
                        help='Compute MONAI perceptual loss metric')
    parser.add_argument('--perceptual_network', type=str, default='alex',
                        choices=['alex', 'vgg', 'squeeze', 'radimagenet', 'medicalnet', 'resnet50'],
                        help='Network backbone for perceptual loss (default: alex)')
    parser.add_argument('--perceptual_fake_3d', action='store_true', default=True,
                        help='Use 2.5D slice-based processing for perceptual loss')

    # Computation arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for computation')

    # Output control
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-volume results')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("=" * 80)
    logging.info("Metrics Calculation Script")
    logging.info("=" * 80)
    logging.info(f"Test directory: {args.test_dir}")
    logging.info(f"Prediction name: {args.prediction_name}")
    logging.info(f"Ground truth name: {args.gt_name}")
    logging.info(f"Device: {args.device}")
    if args.mask_names or args.eval_stacks:
        logging.info(f"Mask names: {args.mask_names}")
        logging.info(f"Eval stacks: {args.eval_stacks}")
        logging.info(f"Mask mode: {args.mask_mode}, threshold: {args.mask_threshold}")
    else:
        logging.info("Mask-aware evaluation: disabled (full-volume metrics)")

    # Find all subject pairs
    logging.info("\nSearching for subject pairs...")
    subject_pairs = find_subject_pairs(args.test_dir, args.prediction_name, args.gt_name)

    if not subject_pairs:
        logging.error("No valid subject pairs found!")
        return

    logging.info(f"Found {len(subject_pairs)} subject pairs\n")

    # Compute metrics for each pair
    volume_results = []
    subject_ids = []

    for subject_id, gt_path, pred_path in tqdm(subject_pairs, desc="Computing metrics"):
        try:
            metrics = compute_metrics_for_pair(
                gt_path=gt_path,
                pred_path=pred_path,
                device=args.device,
                use_perceptual=args.use_perceptual,
                perceptual_network=args.perceptual_network,
                is_fake_3d=args.perceptual_fake_3d,
                mask_names=args.mask_names,
                eval_stacks=args.eval_stacks,
                mask_mode=args.mask_mode,
                mask_threshold=args.mask_threshold,
            )

            volume_results.append(metrics)
            subject_ids.append(subject_id)

            if args.verbose:
                logging.info(f"\n{subject_id}:")
                for metric_name, value in metrics.items():
                    logging.info(f"  {metric_name}: {value:.6f}")

        except Exception as e:
            logging.error(f"Failed to process {subject_id}: {str(e)}")
            continue

    if not volume_results:
        logging.error("No results computed successfully!")
        return

    # Compute aggregate statistics
    logging.info("\nComputing aggregate statistics...")
    aggregates = aggregate_metrics(volume_results)

    # Save CSV results
    logging.info(f"\nSaving results...")
    save_csv_results(
        volume_results=volume_results,
        subject_ids=subject_ids,
        aggregates=aggregates,
        output_path=args.output_csv
    )

    # Save JSON results if requested
    if args.output_json:
        metadata = {
            'test_dir': args.test_dir,
            'prediction_name': args.prediction_name,
            'gt_name': args.gt_name,
            'num_volumes': len(volume_results),
            'device': args.device,
            'use_perceptual': args.use_perceptual,
            'perceptual_network': args.perceptual_network if args.use_perceptual else None,
            'perceptual_fake_3d': args.perceptual_fake_3d if args.use_perceptual else None,
            'mask_names': args.mask_names,
            'eval_stacks': args.eval_stacks,
            'mask_mode': args.mask_mode,
            'mask_threshold': args.mask_threshold,
        }
        save_json_results(
            volume_results=volume_results,
            subject_ids=subject_ids,
            aggregates=aggregates,
            output_path=args.output_json,
            metadata=metadata
        )

    # Print summary
    print_summary(aggregates, len(volume_results))

    logging.info("=" * 80)
    logging.info("COMPUTATION COMPLETE")
    logging.info("=" * 80)


if __name__ == '__main__':
    main()
