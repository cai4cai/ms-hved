#!/usr/bin/env python3
import argparse
import os
import tempfile
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Spacing, Orientation
import warnings
warnings.filterwarnings("ignore")


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


def resample_nilearn(path: str, resolution: list, orientation: str = "RAS"):
    """Resample NIfTI to target resolution and orientation using nilearn.

    Note: nilearn's reorder_img always reorders to RAS; non-RAS orientations
    are not supported and a warning is issued if one is requested.
    """
    from nilearn import image
    from nilearn.image import reorder_img

    if orientation != "RAS":
        warnings.warn(
            f"nilearn backend always reorders to RAS; requested '{orientation}' will be ignored.",
            UserWarning,
            stacklevel=3,
        )

    resampled_img = image.resample_img(
        str(path),
        target_affine=np.eye(3) * resolution[0],  # assumes isotropic
        interpolation='continuous',
    )
    reoriented_img = reorder_img(resampled_img, resample='linear')
    data = reoriented_img.get_fdata().astype(np.float32)
    return data, reoriented_img.affine, reoriented_img.header


def resample(path: str, resolution: list, orientation: str = "RAS",
             interp: str = "trilinear", backend: str = "monai"):
    """Resample NIfTI using the chosen backend ('monai' or 'nilearn')."""
    if backend == "nilearn":
        return resample_nilearn(path, resolution, orientation)
    return resample_monai(path, resolution, orientation, interp)


def save_nifti(data: np.ndarray, affine: np.ndarray, header, path: str):
    """Save array as NIfTI."""
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine, header), path)


# ========== REGISTRATION ==========

def register_to_fixed(fixed_path: str, moving_path: str,
                      transform_type: str = "Rigid",
                      metric: str = "gc") -> tuple:
    """Register moving image to fixed using ANTs.

    Returns (warped_path, fwd_transforms) so the caller can optionally
    apply the same transforms to the original-resolution input.
    """
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
    """Return voxel spacing reordered to RAS world-axis order (x=R, y=A, z=S).

    ANTs' resample_image interprets the spacing tuple relative to the image's
    own voxel axes.  fixed_ants is in RAS orientation after MONAI resampling,
    so we must supply spacing in (R, A, S) order — not in the original image's
    voxel-index order — to avoid assigning thick-slice spacing to the wrong axis.
    """
    from nibabel.orientations import io_orientation
    img = nib.load(path)
    ornt = io_orientation(img.affine)   # ornt[i, 0] = RAS axis for voxel axis i
    zooms = np.array(img.header.get_zooms()[:3])
    ras = np.ones(3)
    for vox_ax, (world_ax, _) in enumerate(ornt):
        ras[int(world_ax)] = zooms[vox_ax]
    return tuple(ras)


def warp_to_orig_res(
    orig_path: str,
    fixed_ants,
    transforms: list,
    out_path: str,
) -> None:
    """Apply pre-computed ANTs transforms to *orig_path* at its native voxel spacing.

    A reference grid is built that covers the fixed image's physical FOV but
    uses the original image's spacing expressed in RAS world-axis order.
    apply_transforms then samples the native-resolution input at native
    intervals, so the high-res in-plane axis is preserved for all orientations
    (axial, coronal, sagittal).
    """
    orig = ants.image_read(orig_path)

    # Spacing in RAS order so it aligns with fixed_ants' axes (also RAS).
    ras_sp = _ras_spacing(orig_path)
    ref = ants.resample_image(fixed_ants, ras_sp, use_voxels=False, interp_type=0)

    warped = ants.apply_transforms(
        fixed=ref,
        moving=orig,
        transformlist=transforms,
        interpolator='linear',
    )
    warped.to_filename(out_path)


# ========== EDGE TRIMMING ==========

def trim_slice_edges(
    registered_path: str,
    orig_path: str,
    out_path: str,
    n_first: int = 0,
    n_last: int = 0,
) -> None:
    """Zero out edge slices along the acquisition axis without changing the matrix size.

    The acquisition axis — the thick-slice direction — is identified as the
    RAS axis with the largest voxel size in *orig_path*.  The first and last
    non-zero slices along that axis are found to locate where the head begins
    and ends.  Then *n_first* slices inward from the head start, and *n_last*
    slices inward from the head end, are set to zero.

    The volume shape and affine are preserved so that the output matrix size
    matches the registered volume exactly — important for voxel-aligned
    similarity metrics during reconstruction.

    Parameters
    ----------
    registered_path : path to the registered (RAS, isotropic) NIfTI.
    orig_path       : path to the original (pre-resampling) input; used only
                      to determine which axis is the thick-slice direction.
    out_path        : where to write the masked image.
    n_first         : number of slices to zero out from where the head starts.
    n_last          : number of slices to zero out before the head ends.
    """
    if n_first == 0 and n_last == 0:
        nib.save(nib.load(registered_path), out_path)
        return

    # Identify the acquisition (thick-slice) axis in RAS space
    ras_sp = np.array(_ras_spacing(orig_path))
    acq_axis = int(np.argmax(ras_sp))

    img = nib.load(registered_path)
    data = img.get_fdata().astype(np.float32)

    n_slices = data.shape[acq_axis]

    # Find the first and last slice that contain any non-zero (head) content.
    # After registration, voxels outside the original acquisition FOV are 0,
    # so this reliably identifies where the head actually starts and ends.
    has_content = np.array([
        np.any(np.take(data, i, axis=acq_axis) != 0)
        for i in range(n_slices)
    ])
    content_idx = np.where(has_content)[0]

    if len(content_idx) == 0:
        nib.save(nib.Nifti1Image(data, img.affine, img.header), out_path)
        return

    head_start = int(content_idx[0])
    head_end   = int(content_idx[-1])

    # Zero out n_first slices from head_start and n_last slices from head_end.
    # Matrix size and affine are unchanged.
    def _zero(axis, idx_from, idx_to):
        slc = [slice(None)] * 3
        slc[axis] = slice(idx_from, idx_to)
        data[tuple(slc)] = 0.0

    if n_first > 0:
        _zero(acq_axis, head_start, head_start + n_first)
    if n_last > 0:
        _zero(acq_axis, head_end - n_last + 1, head_end + 1)

    nib.save(nib.Nifti1Image(data, img.affine, img.header), out_path)


# ========== PIPELINE ==========

def run_pipeline(
    inputs: list,
    output_dir: str,
    do_resample: bool = True,
    do_register: bool = True,
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
    """Run preprocessing pipeline: resample → register → (trim edges)."""
    resolution = resolution or [1.0, 1.0, 1.0]
    iso_resolution = [1.0, 1.0, 1.0]
    os.makedirs(output_dir, exist_ok=True)

    current = list(inputs)
    original_inputs = list(inputs)
    temp_dir = tempfile.mkdtemp()

    # Stage 1: Resample inputs
    if do_resample:
        if verbose:
            print(f"=== Resampling (backend: {resampler}) ===")
        resampled = []
        for i, path in enumerate(current):
            if verbose:
                print(f"  {Path(path).name}")
            data, affine, header = resample(path, resolution, orientation, interp, resampler)
            out = os.path.join(temp_dir, f"resampled_{i}.nii.gz")
            save_nifti(data, affine, header, out)
            resampled.append(out)
        current = resampled

    # Stage 2: Register
    fwd_transforms = [None] * len(inputs)
    reg_fixed_ants = None

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
                # Save resampled fixed to output directory
                fixed_out = os.path.join(output_dir, f"{fixed_stem}.nii.gz")
                save_nifti(data, affine, header, fixed_out)
                if verbose:
                    print(f"  Saved resampled fixed: {Path(fixed_out).name}")
            else:
                reg_fixed = fixed_path
                # Copy original fixed to output directory
                fixed_out = os.path.join(output_dir, f"{fixed_stem}.nii.gz")
                nib.save(nib.load(fixed_path), fixed_out)
                if verbose:
                    print(f"  Saved fixed: {Path(fixed_out).name}")
            if verbose:
                print(f"=== Registering to external fixed: {Path(reg_fixed).name} ===")
        else:
            idx = fixed_idx if fixed_idx is not None else 0
            reg_fixed = current[idx]
            if verbose:
                print(f"=== Registering to input image index {idx} ===")

        reg_fixed_ants = ants.image_read(reg_fixed)

        registered = []
        for i, path in enumerate(current):
            if fixed_path is None and i == (fixed_idx if fixed_idx is not None else 0):
                registered.append(path)
            else:
                if verbose:
                    print(f"  {Path(path).name} → fixed")
                reg_path, transforms = register_to_fixed(reg_fixed, path, transform_type, metric)
                fwd_transforms[i] = transforms
                registered.append(reg_path)
        current = registered

    # Save outputs (with optional edge trimming along each stack's acquisition axis)
    do_trim = trim_first > 0 or trim_last > 0
    if verbose:
        if do_trim:
            print(f"=== Trimming edges (first={trim_first}, last={trim_last}) and saving ===")
        else:
            print("=== Saving outputs ===")
    final_outputs = []
    for orig, curr in zip(inputs, current):
        stem = Path(orig).stem.replace('.nii', '')
        out_path = os.path.join(output_dir, f"{stem}.nii.gz")
        if do_trim:
            trim_slice_edges(curr, orig, out_path, trim_first, trim_last)
        else:
            nib.save(nib.load(curr), out_path)
        if verbose:
            print(f"  Saved: {out_path}")
        final_outputs.append(out_path)

    # Optional: re-apply transforms to original-resolution inputs and save
    if save_original_res and do_register:
        if verbose:
            print("=== Saving registered stacks at original resolution ===")
        for orig, transforms, out_path in zip(original_inputs, fwd_transforms, final_outputs):
            stem = Path(orig).stem.replace('.nii', '')
            orig_res_path = os.path.join(output_dir, f"{stem}_orig_res.nii.gz")
            if transforms is not None:
                if verbose:
                    print(f"  {Path(orig).name} → {Path(orig_res_path).name}")
                warp_to_orig_res(orig, reg_fixed_ants, transforms, orig_res_path)
            else:
                # Fixed image: resample its registered output back to native spacing
                orig_ants = ants.image_read(orig)
                registered_ants = ants.image_read(out_path)
                warped_orig_res = ants.resample_image(
                    registered_ants, orig_ants.spacing, use_voxels=False, interp_type=0
                )
                warped_orig_res.to_filename(orig_res_path)
                if verbose:
                    print(f"  {Path(orig).name} (fixed) → {Path(orig_res_path).name}")

    print(f"✓ Processed {len(inputs)} images → {output_dir}")


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(
        description="MRI preprocessing: resample → register",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline, fixed image is input index 0
  python prepare4test.py -i a.nii.gz b.nii.gz c.nii.gz -o out/

  # Use a separate 3D groundtruth as fixed reference
  python prepare4test.py -i a.nii.gz b.nii.gz -o out/ --fixed-path gt.nii.gz

  # Use external fixed, resampled to 1 mm isotropic first
  python prepare4test.py -i a.nii.gz b.nii.gz -o out/ --fixed-path gt.nii.gz --resample-fixed

  # Use nilearn backend for resampling
  python prepare4test.py -i a.nii.gz b.nii.gz -o out/ --resampler nilearn

  # Only resample, skip registration
  python prepare4test.py -i a.nii.gz b.nii.gz -o out/ --no-register

  # Save registered stacks also at their original (native) voxel spacing
  python prepare4test.py -i a.nii.gz b.nii.gz -o out/ --save-original-res

  # Custom resolution, affine registration
  python prepare4test.py -i a.nii.gz b.nii.gz -o out/ -r 0.5 0.5 0.5 --transform Affine

  # Trim 2 slices from both ends of each stack's acquisition axis
  python prepare4test.py -i ax.nii.gz cor.nii.gz sag.nii.gz -o out/ --trim-first 2 --trim-last 2

  # Trim only the last 3 slices
  python prepare4test.py -i ax.nii.gz cor.nii.gz sag.nii.gz -o out/ --trim-last 3
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
                        default="trilinear", help="Interpolation mode for MONAI (default: trilinear)")
    parser.add_argument("--resampler", choices=["monai", "nilearn"], default="monai",
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
                        help="Remove the first N slices along each stack's acquisition "
                             "(thick-slice) axis after registration (default: 0)")
    parser.add_argument("--trim-last", type=int, default=0, metavar="N",
                        help="Remove the last N slices along each stack's acquisition "
                             "(thick-slice) axis after registration (default: 0)")
    parser.add_argument("--no-resample", action="store_true", help="Skip resampling stage")
    parser.add_argument("--no-register", action="store_true", help="Skip registration stage")
    parser.add_argument("--save-original-res", action="store_true",
                        help="Also save each registered stack at its original voxel spacing "
                             "(<stem>_orig_res.nii.gz), by re-applying ANTs transforms to the "
                             "native-resolution input")
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
