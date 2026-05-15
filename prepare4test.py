#!/usr/bin/env python3
"""MRI preprocessing with FOV mask generation.

After registration, each stack's geometry is compared against the HR reference
grid to produce a binary FOV mask (1=missing, 0=valid) that captures which
HR voxels fall outside each stack's native field of view.

These masks are saved alongside the preprocessed stacks and can be passed
to the MS-HVED model at inference time for precision-weighted fusion.

Pipeline: resample → register → compute FOV masks

Usage:
  # Basic — 3 stacks + external fixed reference
  python prepare4test.py -i ax.nii.gz cor.nii.gz sag.nii.gz \\
      -o out/ --fixed-path hr.nii.gz -v

  # With edge trimming
  python prepare4test.py -i ax.nii.gz cor.nii.gz sag.nii.gz \\
      -o out/ --fixed-path hr.nii.gz --trim-first 2 --trim-last 2 -v

  # Skip FOV mask generation (behaves like original prepare4test.py)
  python prepare4test.py -i ax.nii.gz cor.nii.gz sag.nii.gz \\
      -o out/ --fixed-path hr.nii.gz --no-fov-masks -v

  # Use SimpleITK for the initial spacing/orientation resampling
  python prepare4test.py -i ax.nii.gz cor.nii.gz sag.nii.gz \\
      -o out/ --fixed-path hr.nii.gz --resampler simpleitk -v
"""

import argparse
import os
import tempfile
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
import torch
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Spacing, Orientation
import warnings
warnings.filterwarnings("ignore")

from src.data import resample_with_fov_mask, affine_resample_3d


# ========== RESAMPLING ==========

def resample_monai(path: str, resolution: list, orientation: str = "RAS", interp: str = "trilinear"):
    """Resample NIfTI to target resolution and orientation using MONAI."""
    transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes=orientation),
        Spacing(pixdim=resolution, mode=interp),
    ])
    data = transform(path)
    affine = data.affine.cpu().numpy() if hasattr(data, 'affine') else nib.load(path).affine
    return data.squeeze(0).cpu().numpy(), affine, nib.load(path).header


def resample_nilearn(path: str, resolution: list, orientation: str = "RAS",
                     interp: str = "trilinear"):
    """Resample NIfTI to target resolution and orientation using nilearn."""
    from nilearn import image
    from nilearn.image import reorder_img

    if orientation != "RAS":
        warnings.warn(
            f"nilearn backend always reorders to RAS; requested '{orientation}' will be ignored.",
            UserWarning,
            stacklevel=3,
        )

    interpolation = 'nearest' if interp == 'nearest' else 'continuous'
    reorder_interpolation = 'nearest' if interp == 'nearest' else 'linear'
    resampled_img = image.resample_img(
        str(path),
        target_affine=np.eye(3) * resolution[0],
        interpolation=interpolation,
    )
    reoriented_img = reorder_img(resampled_img, resample=reorder_interpolation)
    data = reoriented_img.get_fdata().astype(np.float32)
    return data, reoriented_img.affine, reoriented_img.header


def _sitk_interpolator(interp: str):
    """Map this script's interpolation names to SimpleITK constants."""
    import SimpleITK as sitk

    if interp == "nearest":
        return sitk.sitkNearestNeighbor
    if interp in {"bilinear", "trilinear"}:
        return sitk.sitkLinear
    raise ValueError(f"Unsupported SimpleITK interpolation mode: {interp}")


def _sitk_image_to_ras_affine(img) -> np.ndarray:
    """Build a nibabel RAS affine from a SimpleITK image.

    SimpleITK stores physical coordinates in LPS. NIfTI/nibabel affines are
    conventionally interpreted in RAS, so both the direction matrix and origin
    need an LPS -> RAS conversion.
    """
    direction_lps = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
    origin_lps = np.asarray(img.GetOrigin(), dtype=np.float64)
    lps_to_ras = np.diag([-1.0, -1.0, 1.0])

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = lps_to_ras @ direction_lps @ np.diag(spacing)
    affine[:3, 3] = lps_to_ras @ origin_lps
    return affine


def resample_simpleitk(path: str, resolution: list, orientation: str = "RAS",
                       interp: str = "trilinear"):
    """Resample NIfTI to target resolution/orientation using SimpleITK."""
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    if img.GetDimension() != 3:
        raise ValueError(f"SimpleITK backend expects a 3D image: {path}")

    if orientation:
        orient = sitk.DICOMOrientImageFilter()
        orient.SetDesiredCoordinateOrientation(orientation.upper())
        img = orient.Execute(img)

    in_spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
    in_size = np.asarray(img.GetSize(), dtype=np.int64)
    out_spacing = np.asarray(resolution, dtype=np.float64)
    out_size = np.maximum(np.round(in_size * in_spacing / out_spacing), 1).astype(np.int64)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(v) for v in out_size])
    resampler.SetOutputSpacing(tuple(float(v) for v in out_spacing))
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(_sitk_interpolator(interp))
    resampler.SetDefaultPixelValue(0.0)
    out_img = resampler.Execute(img)

    # SimpleITK arrays are z, y, x; nibabel expects x, y, z for the affine.
    data = np.transpose(sitk.GetArrayFromImage(out_img), (2, 1, 0)).astype(np.float32)
    affine = _sitk_image_to_ras_affine(out_img)
    header = nib.load(path).header.copy()
    header.set_data_shape(data.shape)
    header.set_data_dtype(np.float32)
    header.set_zooms(tuple(float(v) for v in out_spacing))
    return data, affine, header


def resample(path: str, resolution: list, orientation: str = "RAS",
             interp: str = "trilinear", backend: str = "monai"):
    """Resample NIfTI using the chosen backend."""
    if backend == "nilearn":
        return resample_nilearn(path, resolution, orientation, interp)
    if backend == "simpleitk":
        return resample_simpleitk(path, resolution, orientation, interp)
    return resample_monai(path, resolution, orientation, interp)


def save_nifti(data: np.ndarray, affine: np.ndarray, header, path: str):
    """Save array as NIfTI."""
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine, header), path)


# ========== REGISTRATION ==========

def register_to_fixed(fixed_path: str, moving_path: str,
                      transform_type: str = "Rigid",
                      metric: str = "gc") -> tuple:
    """Register moving image to fixed using ANTs."""
    fixed = ants.image_read(fixed_path)
    moving = ants.image_read(moving_path)

    result = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=transform_type,
        aff_metric=metric,
    )

    out_path = moving_path.replace('.nii', '_registered.nii')
    result['warpedmovout'].to_filename(out_path)
    return out_path, result['fwdtransforms']


def _ras_spacing(path: str) -> tuple:
    """Return voxel spacing reordered to RAS world-axis order."""
    from nibabel.orientations import io_orientation
    img = nib.load(path)
    ornt = io_orientation(img.affine)
    zooms = np.array(img.header.get_zooms()[:3])
    ras = np.ones(3)
    for vox_ax, (world_ax, _) in enumerate(ornt):
        ras[int(world_ax)] = zooms[vox_ax]
    return tuple(ras)


def warp_to_orig_res(orig_path, fixed_ants, transforms, out_path):
    """Apply pre-computed ANTs transforms to *orig_path* at its native voxel spacing."""
    orig = ants.image_read(orig_path)
    ras_sp = _ras_spacing(orig_path)
    ref = ants.resample_image(fixed_ants, ras_sp, use_voxels=False, interp_type=0)
    warped = ants.apply_transforms(
        fixed=ref, moving=orig, transformlist=transforms, interpolator='linear',
    )
    warped.to_filename(out_path)


def apply_registration_to_support(support_path: str, fixed_path: str,
                                  transforms: list | None, out_path: str) -> str:
    """Apply image registration transforms to a binary support mask."""
    if not transforms:
        nib.save(nib.load(support_path), out_path)
        return out_path

    fixed = ants.image_read(fixed_path)
    moving = ants.image_read(support_path)
    warped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=transforms,
        interpolator='nearestNeighbor',
    )
    warped.to_filename(out_path)
    return out_path


# ========== EDGE TRIMMING ==========

def trim_slice_edges(registered_path, orig_path, out_path, n_first=0, n_last=0):
    """Zero out edge slices along the acquisition axis."""
    if n_first == 0 and n_last == 0:
        nib.save(nib.load(registered_path), out_path)
        return

    ras_sp = np.array(_ras_spacing(orig_path))
    acq_axis = int(np.argmax(ras_sp))

    img = nib.load(registered_path)
    data = img.get_fdata().astype(np.float32)
    n_slices = data.shape[acq_axis]

    has_content = np.array([
        np.any(np.take(data, i, axis=acq_axis) != 0)
        for i in range(n_slices)
    ])
    content_idx = np.where(has_content)[0]

    if len(content_idx) == 0:
        nib.save(nib.Nifti1Image(data, img.affine, img.header), out_path)
        return

    head_start = int(content_idx[0])
    head_end = int(content_idx[-1])

    def _zero(axis, idx_from, idx_to):
        slc = [slice(None)] * 3
        slc[axis] = slice(idx_from, idx_to)
        data[tuple(slc)] = 0.0

    if n_first > 0:
        _zero(acq_axis, head_start, head_start + n_first)
    if n_last > 0:
        _zero(acq_axis, head_end - n_last + 1, head_end + 1)

    nib.save(nib.Nifti1Image(data, img.affine, img.header), out_path)


# ========== FOV MASK GENERATION ==========

def create_support_mask(input_path: str, output_path: str) -> str:
    """Create a binary valid-support image in the input image geometry."""
    img = nib.load(input_path)
    data = np.ones(img.shape[:3], dtype=np.float32)
    header = img.header.copy()
    header.set_data_shape(data.shape)
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data, img.affine, header), output_path)
    return output_path


def support_to_fov_mask(support_path: str, reference_path: str) -> np.ndarray:
    """Convert a registered 1=valid support image into a 1=missing FOV mask."""
    support_img = nib.load(support_path)
    reference_img = nib.load(reference_path)
    support = support_img.get_fdata(dtype=np.float32)

    if (
        support.shape[:3] != reference_img.shape[:3]
        or not np.allclose(support_img.affine, reference_img.affine, atol=1e-4)
    ):
        support_t = torch.from_numpy(support).unsqueeze(0)
        support = affine_resample_3d(
            support_t,
            torch.from_numpy(support_img.affine.astype(np.float32)),
            torch.from_numpy(reference_img.affine.astype(np.float32)),
            reference_img.shape[:3],
            mode="nearest",
        )[0].numpy()

    valid = (support > 0.5).astype(np.float32)
    return 1.0 - valid

def compute_fov_mask(stack_path: str, original_input_path: str,
                     hr_reference_path: str, verbose: bool = False) -> np.ndarray:
    """Compute FOV mask for a registered stack against the HR reference grid.

    The mask is computed by resampling an all-ones volume from the stack's
    native (pre-registration) geometry to the HR reference grid. Voxels in
    the HR grid that fall outside the stack's original FOV become 0; after
    inversion, the FOV mask is 1=missing, 0=valid.

    This captures the obliqueness and limited through-plane coverage of each
    acquisition stack.

    Args:
        stack_path: Path to the registered (preprocessed) stack — used only
                    for shape reference.
        original_input_path: Path to the original input stack (before any
                    resampling/registration). The native affine from this
                    file encodes the acquisition geometry.
        hr_reference_path: Path to the HR reference volume. Defines the
                    target grid (shape + affine).
        verbose: Print diagnostic info.

    Returns:
        FOV mask as numpy array with the HR reference shape. 1=missing, 0=valid.
    """
    # Load original stack geometry
    orig_img = nib.load(original_input_path)
    orig_data = orig_img.get_fdata(dtype=np.float32)
    orig_affine = orig_img.affine.astype(np.float32)

    # Load HR reference geometry
    hr_img = nib.load(hr_reference_path)
    hr_shape = hr_img.shape[:3]
    hr_affine = hr_img.affine.astype(np.float32)

    if verbose:
        orig_spacing = np.sqrt((orig_affine[:3, :3] ** 2).sum(axis=0))
        print(f"    Native shape: {orig_data.shape[:3]}, "
              f"spacing: [{orig_spacing[0]:.2f}, {orig_spacing[1]:.2f}, {orig_spacing[2]:.2f}] mm")
        print(f"    HR grid shape: {hr_shape}")

    # Convert to torch
    if orig_data.ndim == 3:
        lr_volume = torch.from_numpy(orig_data).unsqueeze(0)  # (1, D, H, W)
    else:
        lr_volume = torch.from_numpy(orig_data[..., 0]).unsqueeze(0)

    lr_affine_t = torch.from_numpy(orig_affine)
    hr_affine_t = torch.from_numpy(hr_affine)

    # Compute FOV mask via dummy mask trick
    _, fov_mask = resample_with_fov_mask(
        lr_volume, lr_affine_t, hr_affine_t, hr_shape, mode="bilinear"
    )

    fov_mask_np = fov_mask[0].numpy()  # (D, H, W)

    if verbose:
        n_missing = (fov_mask_np > 0.5).sum()
        n_total = fov_mask_np.size
        print(f"    FOV mask: {n_missing}/{n_total} missing ({100 * n_missing / n_total:.1f}%)")

    return fov_mask_np


# ========== PIPELINE ==========

def run_pipeline(
    inputs: list,
    output_dir: str,
    do_resample: bool = True,
    do_register: bool = True,
    do_fov_masks: bool = True,
    resolution: list = None,
    orientation: str = "RAS",
    interp: str = "trilinear",
    resampler: str = "monai",
    fixed_idx: int = None,
    fixed_path: str = None,
    resample_fixed: bool = False,
    transform_type: str = "Rigid",
    metric: str = "gc",
    trim_first: int = 0,
    trim_last: int = 0,
    save_original_res: bool = False,
    verbose: bool = False,
):
    """Run preprocessing pipeline: resample → register → (trim edges) → FOV masks."""
    resolution = resolution or [1.0, 1.0, 1.0]
    iso_resolution = [1.0, 1.0, 1.0]
    os.makedirs(output_dir, exist_ok=True)

    current = list(inputs)
    original_inputs = list(inputs)
    temp_dir = tempfile.mkdtemp()
    support_paths = []
    if do_fov_masks:
        for i, path in enumerate(original_inputs):
            support_paths.append(
                create_support_mask(path, os.path.join(temp_dir, f"support_{i}.nii.gz"))
            )

    # Stage 1: Resample inputs
    if do_resample:
        if verbose:
            print(f"=== Resampling (backend: {resampler}) ===")
        resampled = []
        resampled_support = []
        for i, path in enumerate(current):
            if verbose:
                print(f"  {Path(path).name}")
            data, affine, header = resample(path, resolution, orientation, interp, resampler)
            out = os.path.join(temp_dir, f"resampled_{i}.nii.gz")
            save_nifti(data, affine, header, out)
            resampled.append(out)
            if do_fov_masks:
                support_data, support_affine, support_header = resample(
                    support_paths[i], resolution, orientation, "nearest", resampler
                )
                support_out = os.path.join(temp_dir, f"support_resampled_{i}.nii.gz")
                save_nifti(
                    (support_data > 0.5).astype(np.float32),
                    support_affine,
                    support_header,
                    support_out,
                )
                resampled_support.append(support_out)
        current = resampled
        if do_fov_masks:
            support_paths = resampled_support

    # Stage 2: Register
    fwd_transforms = [None] * len(inputs)
    reg_fixed_ants = None
    hr_reference_for_fov = None  # Path to the HR reference used for FOV mask computation

    if do_register:
        if fixed_path is not None:
            fixed_stem = Path(fixed_path).stem.replace('.nii', '')
            if resample_fixed:
                if verbose:
                    print("=== Resampling fixed volume to 1 mm isotropic ===")
                data, affine, header = resample(
                    fixed_path, iso_resolution, orientation, interp, resampler
                )
                fixed_tmp = os.path.join(temp_dir, "fixed_resampled.nii.gz")
                save_nifti(data, affine, header, fixed_tmp)
                reg_fixed = fixed_tmp
                fixed_out = os.path.join(output_dir, f"{fixed_stem}.nii.gz")
                save_nifti(data, affine, header, fixed_out)
                hr_reference_for_fov = fixed_out
                if verbose:
                    print(f"  Saved resampled fixed: {Path(fixed_out).name}")
            else:
                reg_fixed = fixed_path
                fixed_out = os.path.join(output_dir, f"{fixed_stem}.nii.gz")
                nib.save(nib.load(fixed_path), fixed_out)
                hr_reference_for_fov = fixed_out
                if verbose:
                    print(f"  Saved fixed: {Path(fixed_out).name}")
            if verbose:
                print(f"=== Registering to external fixed: {Path(reg_fixed).name} ===")
        else:
            idx = fixed_idx if fixed_idx is not None else 0
            reg_fixed = current[idx]
            hr_reference_for_fov = reg_fixed
            if verbose:
                print(f"=== Registering to input image index {idx} ===")

        reg_fixed_ants = ants.image_read(reg_fixed)

        registered = []
        registered_support = []
        for i, path in enumerate(current):
            if fixed_path is None and i == (fixed_idx if fixed_idx is not None else 0):
                registered.append(path)
                if do_fov_masks:
                    registered_support.append(support_paths[i])
            else:
                if verbose:
                    print(f"  {Path(path).name} -> fixed")
                reg_path, transforms = register_to_fixed(reg_fixed, path, transform_type, metric)
                fwd_transforms[i] = transforms
                registered.append(reg_path)
                if do_fov_masks:
                    support_out = os.path.join(temp_dir, f"support_registered_{i}.nii.gz")
                    apply_registration_to_support(
                        support_paths[i], reg_fixed, transforms, support_out
                    )
                    registered_support.append(support_out)
        current = registered
        if do_fov_masks:
            support_paths = registered_support
    else:
        # No registration — use first input as reference for FOV masks
        hr_reference_for_fov = current[0]

    # Stage 3: Save outputs (with optional edge trimming)
    do_trim = trim_first > 0 or trim_last > 0
    if verbose:
        if do_trim:
            print(f"=== Trimming edges (first={trim_first}, last={trim_last}) and saving ===")
        else:
            print("=== Saving outputs ===")
    final_outputs = []
    final_support_outputs = []
    for i, (orig, curr) in enumerate(zip(inputs, current)):
        stem = Path(orig).stem.replace('.nii', '')
        out_path = os.path.join(output_dir, f"{stem}.nii.gz")
        if do_trim:
            trim_slice_edges(curr, orig, out_path, trim_first, trim_last)
        else:
            nib.save(nib.load(curr), out_path)
        if do_fov_masks:
            support_out = os.path.join(temp_dir, f"support_final_{i}.nii.gz")
            if do_trim:
                trim_slice_edges(support_paths[i], orig, support_out, trim_first, trim_last)
            else:
                nib.save(nib.load(support_paths[i]), support_out)
            final_support_outputs.append(support_out)
        if verbose:
            print(f"  Saved: {out_path}")
        final_outputs.append(out_path)

    # Stage 4: Compute FOV masks
    if do_fov_masks and hr_reference_for_fov is not None:
        if verbose:
            print("=== Computing FOV masks ===")
        for orig, support_path in zip(original_inputs, final_support_outputs):
            stem = Path(orig).stem.replace('.nii', '')
            if verbose:
                print(f"  {stem}:")
            fov_mask = support_to_fov_mask(support_path, hr_reference_for_fov)
            mask_path = os.path.join(output_dir, f"{stem}_fov_mask.nii.gz")
            hr_affine = nib.load(hr_reference_for_fov).affine
            nib.save(nib.Nifti1Image(fov_mask, hr_affine), mask_path)
            if verbose:
                n_missing = int((fov_mask > 0.5).sum())
                n_total = fov_mask.size
                print(f"    FOV mask: {n_missing}/{n_total} missing ({100 * n_missing / n_total:.1f}%)")
                print(f"    Saved: {mask_path}")

    # Optional: re-apply transforms to original-resolution inputs
    if save_original_res and do_register:
        if verbose:
            print("=== Saving registered stacks at original resolution ===")
        for orig, transforms, out_path in zip(original_inputs, fwd_transforms, final_outputs):
            stem = Path(orig).stem.replace('.nii', '')
            orig_res_path = os.path.join(output_dir, f"{stem}_orig_res.nii.gz")
            if transforms is not None:
                if verbose:
                    print(f"  {Path(orig).name} -> {Path(orig_res_path).name}")
                warp_to_orig_res(orig, reg_fixed_ants, transforms, orig_res_path)
            else:
                orig_ants = ants.image_read(orig)
                registered_ants = ants.image_read(out_path)
                warped_orig_res = ants.resample_image(
                    registered_ants, orig_ants.spacing, use_voxels=False, interp_type=0
                )
                warped_orig_res.to_filename(orig_res_path)
                if verbose:
                    print(f"  {Path(orig).name} (fixed) -> {Path(orig_res_path).name}")

    print(f"Done: {len(inputs)} images -> {output_dir}")


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(
        description="MRI preprocessing with FOV mask generation: resample -> register -> FOV masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with FOV masks
  python prepare4test.py -i ax.nii.gz cor.nii.gz sag.nii.gz -o out/ --fixed-path hr.nii.gz -v

  # With edge trimming
  python prepare4test.py -i ax.nii.gz cor.nii.gz sag.nii.gz -o out/ --fixed-path hr.nii.gz --trim-first 2 --trim-last 2 -v

  # Skip FOV mask generation
  python prepare4test.py -i ax.nii.gz cor.nii.gz sag.nii.gz -o out/ --fixed-path hr.nii.gz --no-fov-masks -v

  # Without external fixed (uses first input as reference)
  python prepare4test.py -i ax.nii.gz cor.nii.gz sag.nii.gz -o out/ -v

  # Save at original resolution too
  python prepare4test.py -i ax.nii.gz cor.nii.gz sag.nii.gz -o out/ --fixed-path hr.nii.gz --save-original-res -v
        """
    )
    parser.add_argument("-i", "--inputs", nargs="+", required=True,
                        help="Input NIfTI files (low-res stacks)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output directory")
    parser.add_argument("-r", "--resolution", nargs=3, type=float, default=[1.0, 1.0, 1.0],
                        metavar=("X", "Y", "Z"), help="Target voxel resolution in mm (default: 1 1 1)")
    parser.add_argument("--orientation", default="RAS",
                        help="Target orientation (default: RAS)")
    parser.add_argument("--interp", choices=["trilinear", "bilinear", "nearest"],
                        default="trilinear",
                        help="Interpolation mode for MONAI/SimpleITK (default: trilinear)")
    parser.add_argument("--resampler", choices=["monai", "nilearn", "simpleitk"], default="monai",
                        help="Resampling backend (default: monai)")
    parser.add_argument("--fixed", type=int, default=None,
                        help="Index of input image to use as fixed for registration (default: 0)")
    parser.add_argument("--fixed-path", type=str, default=None,
                        help="Path to an external 3D volume to use as fixed for registration")
    parser.add_argument("--resample-fixed", action="store_true",
                        help="Resample --fixed-path to 1 mm isotropic before registration")
    parser.add_argument("--transform", choices=["Rigid", "Affine"], default="Rigid",
                        help="ANTs registration type (default: Rigid)")
    parser.add_argument("--metric", choices=["mattes", "meansquares", "gc"], default="gc",
                        help="ANTs registration metric (default: gc)")
    parser.add_argument("--trim-first", type=int, default=0, metavar="N",
                        help="Remove the first N slices along each stack's acquisition axis")
    parser.add_argument("--trim-last", type=int, default=0, metavar="N",
                        help="Remove the last N slices along each stack's acquisition axis")
    parser.add_argument("--no-resample", action="store_true", help="Skip resampling stage")
    parser.add_argument("--no-register", action="store_true", help="Skip registration stage")
    parser.add_argument("--no-fov-masks", action="store_true", help="Skip FOV mask generation")
    parser.add_argument("--save-original-res", action="store_true",
                        help="Also save each registered stack at its original voxel spacing")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    for p in args.inputs:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Not found: {p}")
    if args.fixed_path and not os.path.exists(args.fixed_path):
        raise FileNotFoundError(f"Fixed image not found: {args.fixed_path}")
    if args.fixed is not None and args.fixed_path is not None:
        raise ValueError("Specify either --fixed (index) or --fixed-path, not both.")

    run_pipeline(
        inputs=args.inputs,
        output_dir=args.output,
        do_resample=not args.no_resample,
        do_register=not args.no_register,
        do_fov_masks=not args.no_fov_masks,
        resolution=args.resolution,
        orientation=args.orientation,
        interp=args.interp,
        resampler=args.resampler,
        fixed_idx=args.fixed,
        fixed_path=args.fixed_path,
        resample_fixed=args.resample_fixed,
        transform_type=args.transform,
        metric=args.metric,
        trim_first=args.trim_first,
        trim_last=args.trim_last,
        save_original_res=args.save_original_res,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
