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

import os
import argparse
import csv
import json
import logging
from typing import Dict, Any, List

import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

# Import metrics from project
from src.utils import calculate_metrics


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


def compute_metrics_for_pair(
    gt_path: str,
    pred_path: str,
    device: str = 'cuda',
    use_perceptual: bool = False,
    perceptual_network: str = 'alex',
    is_fake_3d: bool = True
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

    # Handle shape mismatch
    if gt_volume.shape != pred_volume.shape:
        logging.warning(
            f"Shape mismatch: GT {gt_volume.shape} vs Pred {pred_volume.shape}. "
            f"Applying center crop."
        )
        gt_volume, pred_volume = center_crop_to_match(gt_volume, pred_volume)
        logging.info(f"Cropped to shape: {gt_volume.shape}")

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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
                is_fake_3d=args.perceptual_fake_3d
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
            'perceptual_fake_3d': args.perceptual_fake_3d if args.use_perceptual else None
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