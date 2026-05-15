"""Batch naive MRI reconstruction.

Walks a directory of subjects. For each subject's `processed/` folder, finds
matched scan + FOV mask pairs (e.g. `axial.nii.gz` + `axial_fov_mask.nii.gz`,
or `cor.nii.gz` + `cor_fov_mask.nii.gz`) and saves two mean reconstructions:

    naive_reconstruction.nii.gz       plain mean across stacks
    naive_reconstruction_fov.nii.gz   FOV-mask-weighted mean

Usage:
    python naive_rec.py /path/to/subjects_dir
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

MASK_SUFFIX = "_fov_mask.nii.gz"
SCAN_SUFFIX = ".nii.gz"


def load_volume(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata()


def save_reconstruction(volume: np.ndarray, reference_path: Path, output_path: Path) -> Path:
    reference_img = nib.load(str(reference_path))
    header = reference_img.header.copy()
    header.set_data_dtype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(volume.astype(np.float32), reference_img.affine, header=header)
    nib.save(img, str(output_path))
    return output_path


def fov_mask_to_confidence(
    fov_mask: np.ndarray,
    smooth_fov: bool = True,
    fov_attenuation: float = 1.0,
    fov_transition_width: float = 3.0,
) -> np.ndarray:
    """Convert 1=missing FOV masks into contribution weights."""
    hard_missing = np.clip(fov_mask.astype(np.float32), 0.0, 1.0)
    hard_valid = 1.0 - hard_missing
    missing = hard_missing
    if smooth_fov and fov_transition_width > 0:
        missing = gaussian_filter(missing, sigma=fov_transition_width, mode="nearest")
        missing = np.clip(missing, 0.0, 1.0)
    return (1.0 - missing * fov_attenuation) * hard_valid


def naive_mean(
    scans: list[np.ndarray],
    fov_masks: list[np.ndarray] | None = None,
    smooth_fov: bool = True,
    fov_attenuation: float = 1.0,
    fov_transition_width: float = 3.0,
) -> np.ndarray:
    stack = np.stack(scans, axis=0)
    if fov_masks is None:
        return stack.mean(axis=0)
    weights = np.stack(
        [
            fov_mask_to_confidence(m, smooth_fov, fov_attenuation, fov_transition_width)
            for m in fov_masks
        ],
        axis=0,
    )
    weight_sum = weights.sum(axis=0)
    weighted_sum = (stack * weights).sum(axis=0)
    return np.divide(
        weighted_sum,
        weight_sum,
        out=np.zeros_like(weighted_sum),
        where=weight_sum > 0,
    )


def find_stack_pairs(processed_dir: Path) -> list[tuple[Path, Path]]:
    """Find (scan, mask) path pairs based on `*_fov_mask.nii.gz` filenames."""
    pairs = []
    for mask_path in sorted(processed_dir.glob(f"*{MASK_SUFFIX}")):
        name = mask_path.name.removesuffix(MASK_SUFFIX)
        scan_path = processed_dir / f"{name}{SCAN_SUFFIX}"
        if scan_path.exists():
            pairs.append((scan_path, mask_path))
    return pairs


def process_scan_dir(
    processed: Path,
    label: str,
    smooth_fov: bool = True,
    fov_attenuation: float = 1.0,
    fov_transition_width: float = 3.0,
    overwrite: bool = False,
) -> str:
    pairs = find_stack_pairs(processed)
    if len(pairs) < 2:
        return f"skip (<2 stacks): {label}"

    scan_paths, mask_paths = zip(*pairs)

    out_plain = processed / "naive_reconstruction.nii.gz"
    out_fov = processed / "naive_reconstruction_fov.nii.gz"
    if not overwrite and out_plain.exists() and out_fov.exists():
        return f"skip (exists): {label}"

    scans = [load_volume(p) for p in scan_paths]
    masks = [load_volume(p) for p in mask_paths]

    shapes = {s.shape for s in scans} | {m.shape for m in masks}
    if len(shapes) > 1:
        return f"skip (shape mismatch {shapes}): {label}"

    mean_plain = naive_mean(scans)
    mean_fov = naive_mean(
        scans,
        fov_masks=masks,
        smooth_fov=smooth_fov,
        fov_attenuation=fov_attenuation,
        fov_transition_width=fov_transition_width,
    )

    save_reconstruction(mean_plain, scan_paths[0], out_plain)
    save_reconstruction(mean_fov, scan_paths[0], out_fov)
    stacks = ", ".join(p.name.removesuffix(SCAN_SUFFIX) for p in scan_paths)
    return f"done [{stacks}]: {label}"


def process_subject(
    subject_dir: Path,
    smooth_fov: bool = True,
    fov_attenuation: float = 1.0,
    fov_transition_width: float = 3.0,
    overwrite: bool = False,
) -> list[str]:
    processed = subject_dir / "processed"
    if not processed.is_dir():
        return [f"skip (no processed/): {subject_dir.name}"]

    sequence_dirs = sorted(p for p in processed.iterdir() if p.is_dir())
    if sequence_dirs:
        return [
            process_scan_dir(
                sequence_dir,
                f"{subject_dir.name}/{sequence_dir.name}",
                smooth_fov=smooth_fov,
                fov_attenuation=fov_attenuation,
                fov_transition_width=fov_transition_width,
                overwrite=overwrite,
            )
            for sequence_dir in sequence_dirs
        ]

    return [
        process_scan_dir(
            processed,
            subject_dir.name,
            smooth_fov=smooth_fov,
            fov_attenuation=fov_attenuation,
            fov_transition_width=fov_transition_width,
            overwrite=overwrite,
        )
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch naive MRI reconstruction")
    parser.add_argument("root", type=Path, help="Directory containing subject subfolders")
    parser.add_argument("--no-smooth-fov", action="store_true")
    parser.add_argument("--fov-attenuation", type=float, default=1.0)
    parser.add_argument("--fov-transition-width", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if outputs already exist.")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"not a directory: {args.root}")
    if not 0.0 <= args.fov_attenuation <= 1.0:
        parser.error("--fov-attenuation must be in [0, 1]")
    if args.fov_transition_width < 0:
        parser.error("--fov-transition-width must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    subjects = sorted(p for p in args.root.iterdir() if p.is_dir())
    for subject in subjects:
        results = process_subject(
            subject,
            smooth_fov=not args.no_smooth_fov,
            fov_attenuation=args.fov_attenuation,
            fov_transition_width=args.fov_transition_width,
            overwrite=args.overwrite,
        )
        for result in results:
            print(result, flush=True)


if __name__ == "__main__":
    main()
