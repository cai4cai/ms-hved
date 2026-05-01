"""MS-HVED Inference with FOV mask support.

Accepts pre-computed FOV masks (from prepare4test.py) and passes them to the model's fusion mechanism.
FOV masks tell the Product of Gaussians fusion which voxels are missing
in each orientation stack, so it can zero out their precision contribution.

Usage:
  # Single case with FOV masks
  python test.py --input_stacks ax.nii.gz cor.nii.gz sag.nii.gz \\
      --fov_masks ax_fov_mask.nii.gz cor_fov_mask.nii.gz sag_fov_mask.nii.gz \\
      --output sr_output.nii.gz --model checkpoint.pth

  # Single case without FOV masks
  python test.py --input_stacks ax.nii.gz cor.nii.gz sag.nii.gz \\
      --output sr_output.nii.gz --model checkpoint.pth

  # Folder mode (auto-discovers *_fov_mask.nii.gz next to each stack)
  python test.py --input_stacks_root /subjects --output_root /out \\
      --model checkpoint.pth

  # Generate 5 stochastic samples from the latent space
  python test.py --input_stacks ax.nii.gz cor.nii.gz sag.nii.gz \\
      --output sr_output.nii.gz --model checkpoint.pth --num_samples 5
  # Saves: sr_output_sample1.nii.gz ... sr_output_sample5.nii.gz
"""

import os
import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import gc

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
)

import torch.nn.functional as F

from src import MSHVED
from src.utils import (
    pad_to_multiple_of_32,
    unpad_volume,
)


def cuda_cleanup():
    """Best-effort GPU memory cleanup between cases."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def get_resolution_from_affine(affine: np.ndarray) -> np.ndarray:
    """Extract voxel resolution [x,y,z] in mm from NIfTI affine matrix."""
    res_x = np.linalg.norm(affine[:3, 0])
    res_y = np.linalg.norm(affine[:3, 1])
    res_z = np.linalg.norm(affine[:3, 2])
    return np.array([res_x, res_y, res_z])


def is_anisotropic(resolution: np.ndarray, threshold: float = 0.1) -> bool:
    """Check if resolution is anisotropic (non-cubic voxels)."""
    res_range = resolution.max() - resolution.min()
    return res_range > threshold


def create_isotropic_affine(target_res: list, shape: tuple, original_affine: np.ndarray) -> np.ndarray:
    """Create affine matrix for isotropic space, preserving orientation from original."""
    rotation = original_affine[:3, :3]
    u = rotation[:, 0] / np.linalg.norm(rotation[:, 0])
    v = rotation[:, 1] / np.linalg.norm(rotation[:, 1])
    w = rotation[:, 2] / np.linalg.norm(rotation[:, 2])

    new_affine = np.eye(4)
    new_affine[:3, 0] = u * target_res[0]
    new_affine[:3, 1] = v * target_res[1]
    new_affine[:3, 2] = w * target_res[2]
    new_affine[:3, 3] = original_affine[:3, 3]

    return new_affine


def load_mshved_from_checkpoint(checkpoint_path, device="cuda"):
    """Load MS-HVED model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get('model_config', {})

    model_architecture = model_config.get('model_architecture', 'mshved')
    print(f"Detected architecture: {model_architecture}")
    print(f"Model Configuration:")
    for key, value in model_config.items():
        print(f"  - {key}: {value}")

    use_prior = model_config.get('use_prior', True)
    share_encoder = model_config.get('share_encoder', False)
    share_decoder = model_config.get('share_decoder', False)
    in_channels = model_config.get('in_channels', 1)
    out_channels = model_config.get('out_channels', 1)


    num_orientations = model_config.get('num_orientations', 3)
    num_scales = model_config.get('num_scales', 4)
    init_filters = model_config.get('init_filters', 32)
    blocks_down = tuple(model_config.get('blocks_down', [1, 2, 2, 4]))
    blocks_up = tuple(model_config.get('blocks_up', [1, 1, 1]))
    reconstruct_orientations = model_config.get('reconstruct_orientations', False)
    decoder_upsample_mode = model_config.get('decoder_upsample_mode', 'trilinear')
    final_activation = model_config.get('final_activation', 'sigmoid')  
    global_residual = model_config.get('global_residual', False)
    smooth_fov = model_config.get('smooth_fov', True)
    fov_attenuation = model_config.get('fov_attenuation', 1.0)
    fov_transition_width = model_config.get('fov_transition_width', 3.0)

    params = {
        'num_orientations': num_orientations,
        'in_channels': in_channels,
        'out_channels': out_channels,
        'num_scales': num_scales,
        'share_encoder': share_encoder,
        'share_decoder': share_decoder,
        'use_prior': use_prior,
        'upsample_mode': decoder_upsample_mode,
        'reconstruct_orientations': reconstruct_orientations,
        'final_activation': final_activation,
        'global_residual': global_residual,
        'smooth_fov': smooth_fov,
        'fov_attenuation': fov_attenuation,
        'fov_transition_width': fov_transition_width,
    }

    model = MSHVED(
        init_filters=init_filters,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
        **params
    )

    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    return model, checkpoint


def create_inference_transforms(target_res=[1.0, 1.0, 1.0]):
    """Create MONAI preprocessing transforms for inference."""
    return Compose([
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        Spacingd(keys=["image"], pixdim=target_res, mode="bilinear"),
    ])


def load_orthogonal_stacks_from_files(stack_paths, target_res=[1.0, 1.0, 1.0]):
    """
    Load 3 pre-existing orthogonal LR stacks from files.

    IMPORTANT: Stack order must match training configuration:
        - Stack 0 (Axial): High-res in D/depth axis, low-res in H,W
        - Stack 1 (Coronal): High-res in H/height axis, low-res in D,W
        - Stack 2 (Sagittal): High-res in W/width axis, low-res in D,H

    Args:
        stack_paths: List of 3 file paths [axial, coronal, sagittal].
                    Can contain None for missing orientations.
        target_res: Target resolution [x, y, z] in mm

    Returns:
        Tuple of (lr_stacks_tensors, metadata_dict)
    """
    print("  Loading pre-existing orthogonal LR stacks...")
    print("  Stack order: [Axial, Coronal, Sagittal]")

    metadata = {
        'affine_original': None,
        'affine_isotropic': None,
        'resolution_original': None,
        'shape_isotropic': None,
        'is_anisotropic': False,
    }

    for stack_path in stack_paths:
        if stack_path is not None:
            try:
                original_img = nib.load(stack_path)
                metadata['affine_original'] = original_img.affine.copy()
                metadata['resolution_original'] = get_resolution_from_affine(original_img.affine)
                metadata['is_anisotropic'] = is_anisotropic(metadata['resolution_original'])

                if metadata['is_anisotropic']:
                    print(f"  Detected anisotropic resolution: {metadata['resolution_original']} mm")
                    print(f"  Resampling to isotropic: {target_res} mm")
                else:
                    print(f"  Input resolution: {metadata['resolution_original']} mm (already isotropic)")
            except Exception as e:
                print(f"  Warning: Could not load metadata from {stack_path}: {e}")
                metadata['affine_original'] = np.diag([target_res[0], target_res[1], target_res[2], 1.0])
                metadata['resolution_original'] = np.array(target_res)
            break

    if metadata['affine_original'] is None:
        metadata['affine_original'] = np.diag([target_res[0], target_res[1], target_res[2], 1.0])
        metadata['resolution_original'] = np.array(target_res)

    transforms = create_inference_transforms(target_res)
    lr_stacks_tensors = [None, None, None]
    reference_shape = None

    orientation_mapping = [
        ("Axial", "Stack 0", "High-res in D axis"),
        ("Coronal", "Stack 1", "High-res in H axis"),
        ("Sagittal", "Stack 2", "High-res in W axis"),
    ]

    for i, stack_path in enumerate(stack_paths):
        orientation, stack_num, description = orientation_mapping[i]

        if stack_path is None:
            print(f"    - {stack_num} ({orientation}): [MISSING - will use dummy stack]")
            continue

        print(f"    - {stack_num} ({orientation}): {stack_path}")
        print(f"      {description}")

        data_dict = {"image": stack_path}
        data = transforms(data_dict)
        volume = data["image"]

        if isinstance(volume, torch.Tensor):
            volume_np = volume.cpu().numpy()
        else:
            volume_np = np.array(volume)

        if volume_np.ndim == 4 and volume_np.shape[0] == 1:
            volume_np = volume_np[0]

        if reference_shape is None:
            reference_shape = volume_np.shape

        volume_np = (volume_np - volume_np.min()) / (volume_np.max() - volume_np.min() + 1e-8)
        lr_stacks_tensors[i] = torch.from_numpy(volume_np).float().unsqueeze(0)

    if reference_shape is None:
        raise ValueError("No valid stacks provided - at least one stack is required!")

    for i, stack in enumerate(lr_stacks_tensors):
        if stack is None:
            dummy = torch.zeros((1,) + reference_shape, dtype=torch.float32)
            lr_stacks_tensors[i] = dummy
            print(f"    Created dummy stack for {orientation_mapping[i][0]}: shape {dummy.shape}")

    metadata['shape_isotropic'] = lr_stacks_tensors[0].squeeze().shape
    metadata['affine_isotropic'] = create_isotropic_affine(
        target_res, metadata['shape_isotropic'], metadata['affine_original']
    )

    print(f"    Final stack shapes: {[s.shape for s in lr_stacks_tensors]}")
    return lr_stacks_tensors, metadata


def load_fov_masks(fov_mask_paths, target_shape, device="cuda"):
    """Load FOV mask NIfTI files and prepare them for the model.

    Each mask is loaded, resized to match the target shape if needed,
    and returned as a list of tensors (B, 1, D, H, W) ready for the model.

    Args:
        fov_mask_paths: List of 3 paths (can contain None for missing orientations).
        target_shape: Expected spatial shape (D, H, W) after padding.
        device: Torch device.

    Returns:
        List of 3 tensors, each (1, 1, D, H, W). Missing masks are all-zeros.
    """
    fov_masks = []
    for i, mask_path in enumerate(fov_mask_paths):
        label = ["Axial", "Coronal", "Sagittal"][i]
        if mask_path is not None and os.path.exists(mask_path):
            mask_img = nib.load(mask_path)
            mask_np = mask_img.get_fdata(dtype=np.float32)

            # Binarize (threshold at 0.5)
            mask_np = (mask_np > 0.5).astype(np.float32)

            n_missing = mask_np.sum()
            n_total = mask_np.size
            print(f"    {label} FOV mask: {mask_path}")
            print(f"      {int(n_missing)}/{n_total} missing ({100 * n_missing / n_total:.1f}%)")

            fov_masks.append(torch.from_numpy(mask_np).float())
        else:
            print(f"    {label} FOV mask: not provided (using all-zeros)")
            fov_masks.append(None)

    return fov_masks


def _sample_output_path(base_path, sample_idx):
    """Generate output path for a specific sample.

    Given 'output/sr.nii.gz' and sample_idx=2, returns 'output/sr_sample2.nii.gz'.
    """
    p = Path(base_path)
    stem = p.name.replace('.nii.gz', '').replace('.nii', '')
    suffix = '.nii.gz' if p.name.endswith('.nii.gz') else '.nii'
    return str(p.parent / f"{stem}_sample{sample_idx}{suffix}")


def predict_single_volume(
    model,
    output_path,
    device="cuda",
    input_stack_paths=None,
    fov_mask_paths=None,
    target_res=[1.0, 1.0, 1.0],
    orientation_mask=None,
    save_reconstructions=False,
    reconstruction_dir=None,
    num_samples=1,
):
    """
    Run MS-HVED inference on a single volume with optional FOV masks.

    Args:
        model: Trained MS-HVED model
        output_path: Path to save output
        device: 'cuda' or 'cpu'
        input_stack_paths: List of 3 pre-existing stack paths [axial, coronal, sagittal]
        fov_mask_paths: Optional list of 3 FOV mask paths [axial, coronal, sagittal].
                       Each mask is (D, H, W) with 1=missing, 0=valid.
        target_res: Target resolution [x, y, z] in mm
        orientation_mask: Optional binary mask [1/0, 1/0, 1/0]
        save_reconstructions: Whether to save reconstructed orientations
        reconstruction_dir: Directory to save reconstructed orientations
        num_samples: Number of stochastic samples to draw from the latent space.
                    When > 1, each sample is saved as <output>_sample{i}.nii.gz.
    """
    for i, path in enumerate(input_stack_paths):
        print(f"  {['Axial', 'Coronal', 'Sagittal'][i]}: {path}")

    # Load stacks
    lr_stacks, metadata = load_orthogonal_stacks_from_files(input_stack_paths, target_res)
    affine = metadata['affine_isotropic']

    # Pad stacks to multiple of 32
    original_shape = lr_stacks[0].squeeze().shape
    lr_stacks_padded = []

    for stack in lr_stacks:
        stack_np = stack.squeeze().cpu().numpy()
        padded, pad_before, orig_shape = pad_to_multiple_of_32(stack_np)
        lr_stacks_padded.append(
            torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0)
        )

    padded_shape = lr_stacks_padded[0].shape[2:]  # (D_pad, H_pad, W_pad)

    # Load and pad FOV masks
    fov_masks_tensor = None
    if fov_mask_paths is not None:
        print("  Loading FOV masks:")
        raw_masks = load_fov_masks(fov_mask_paths, original_shape, device)

        fov_masks_padded = []
        for i, mask in enumerate(raw_masks):
            if mask is not None:
                mask_np = mask.numpy()
                # Pad mask the same way as stacks
                mask_padded, _, _ = pad_to_multiple_of_32(mask_np)
                fov_masks_padded.append(
                    torch.from_numpy(mask_padded).float().unsqueeze(0).unsqueeze(0).to(device)
                )
            else:
                # All-zeros mask (nothing missing)
                fov_masks_padded.append(
                    torch.zeros(1, 1, *padded_shape, device=device)
                )
        fov_masks_tensor = fov_masks_padded

    # Move stacks to device
    lr_stacks_padded = [stack.to(device) for stack in lr_stacks_padded]

    # Create orientation mask tensor
    if orientation_mask is not None:
        orientation_mask_tensor = torch.tensor(
            orientation_mask, dtype=torch.bool, device=device
        ).unsqueeze(0)
        present_orientations = [i for i, m in enumerate(orientation_mask) if m == 1]
        print(f"  Using orientation mask: {orientation_mask}")
        print(f"  Present orientations: {[['Axial', 'Coronal', 'Sagittal'][i] for i in present_orientations]}")
    else:
        orientation_mask_tensor = None
        print(f"  Using all 3 orientations")

    if fov_masks_tensor is not None:
        print(f"  FOV masks: enabled (precision-weighted fusion)")
    else:
        print(f"  FOV masks: not provided (uniform precision)")

    if num_samples > 1:
        print(f"  Generating {num_samples} stochastic samples from latent space")

    # Run inference
    try:
        model.eval()
        with torch.no_grad():
            # Encode once — shared across all samples
            encoder_outputs = model.encode(lr_stacks_padded, orientation_mask_tensor)

            # Compute reference for global residual (shared across samples)
            reference = None
            if model.global_residual:
                reference = model._compute_reference(lr_stacks_padded, orientation_mask_tensor)

            for sample_idx in range(num_samples):
                if num_samples > 1:
                    print(f"  Sample {sample_idx + 1}/{num_samples}...")

                # Fuse and sample from the latent space
                if num_samples > 1:
                    # Enable stochastic sampling by temporarily putting the
                    # sampler in train mode (GaussianSampler checks self.training)
                    model.fusion.sampler.train()
                    latent_samples, posteriors = model.fuse(
                        encoder_outputs, orientation_mask_tensor, fov_masks_tensor,
                        deterministic=False,
                    )
                    model.fusion.sampler.eval()
                else:
                    # Single sample: deterministic (return posterior mean)
                    latent_samples, posteriors = model.fuse(
                        encoder_outputs, orientation_mask_tensor, fov_masks_tensor,
                        deterministic=True,
                    )

                # Decode
                if model.reconstruct_orientations:
                    sr_output, orientation_outputs = model.decode(latent_samples, encoder_outputs)
                else:
                    sr_output = model.decode(latent_samples, encoder_outputs)
                    orientation_outputs = []

                # Global residual
                if reference is not None:
                    if reference.shape[2:] != sr_output.shape[2:]:
                        ref = F.interpolate(reference, size=sr_output.shape[2:], mode='trilinear', align_corners=False)
                    else:
                        ref = reference
                    sr_output = torch.clamp(sr_output + ref, 0.0, 1.0)

                    for j in range(len(orientation_outputs)):
                        if orientation_outputs[j].shape[2:] != ref.shape[2:]:
                            ref_resized = F.interpolate(ref, size=orientation_outputs[j].shape[2:], mode='trilinear', align_corners=False)
                        else:
                            ref_resized = ref
                        orientation_outputs[j] = torch.clamp(orientation_outputs[j] + ref_resized, 0.0, 1.0)

                # Convert SR output back to numpy
                sr_np = sr_output.squeeze().cpu().numpy()
                sr_np = unpad_volume(sr_np, pad_before, orig_shape)
                sr_np = np.clip(sr_np, 0, 1)

                # Determine save path
                if num_samples > 1:
                    save_path = _sample_output_path(output_path, sample_idx + 1)
                else:
                    save_path = output_path

                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                out_nii = nib.Nifti1Image(sr_np, affine)
                nib.save(out_nii, save_path)

                output_res = get_resolution_from_affine(affine)
                print(f"  SR output saved to: {save_path}")
                print(f"    Output shape: {sr_np.shape}")
                print(f"    Output resolution: [{output_res[0]:.2f}, {output_res[1]:.2f}, {output_res[2]:.2f}] mm")
                print(f"    Output range: [{sr_np.min():.4f}, {sr_np.max():.4f}]")

                # Save reconstructed orientations if requested (only for first sample)
                if save_reconstructions and sample_idx == 0:
                    if len(orientation_outputs) > 0:
                        print(f"\n  Saving reconstructed orientations...")

                        if reconstruction_dir is None:
                            reconstruction_dir = os.path.dirname(output_path) or "."
                        os.makedirs(reconstruction_dir, exist_ok=True)

                        base_name = os.path.splitext(os.path.basename(output_path))[0]
                        if base_name.endswith('.nii'):
                            base_name = base_name[:-4]

                        orientation_names = ['axial', 'coronal', 'sagittal']
                        for j, recon in enumerate(orientation_outputs):
                            recon_np = recon.squeeze().cpu().numpy()
                            recon_np = unpad_volume(recon_np, pad_before, orig_shape)
                            recon_np = np.clip(recon_np, 0, 1)

                            recon_path = os.path.join(reconstruction_dir, f"{base_name}_recon_{orientation_names[j]}.nii.gz")
                            recon_nii = nib.Nifti1Image(recon_np, affine)
                            nib.save(recon_nii, recon_path)
                            print(f"    - {orientation_names[j]}: {recon_path}")
                    else:
                        print(f"  Note: Model was not trained with orientation reconstruction, skipping...")

    finally:
        try:
            del lr_stacks, lr_stacks_padded
            if fov_masks_tensor is not None:
                del fov_masks_tensor
            if 'encoder_outputs' in locals(): del encoder_outputs
            if 'sr_output' in locals(): del sr_output
            if 'orientation_outputs' in locals(): del orientation_outputs
            if 'orientation_mask_tensor' in locals(): del orientation_mask_tensor
        except Exception:
            pass
        cuda_cleanup()


def predict_batch(
    output_paths,
    model_path,
    target_res=[1.0, 1.0, 1.0],
    device="cuda",
    input_stack_paths=None,
    fov_mask_paths=None,
    orientation_mask=None,
    save_reconstructions=False,
    reconstruction_dir=None,
    num_samples=1,
):
    """Process a single case."""
    print("=" * 80)
    print("MS-HVED Inference with FOV Masks")
    print("=" * 80)

    model, checkpoint = load_mshved_from_checkpoint(model_path, device=device)

    print(f"\nInference settings:")
    print(f"  Device: {device}")
    print(f"  Target resolution: {target_res} mm")
    print(f"  FOV masks: {'provided' if fov_mask_paths else 'not provided'}")
    if num_samples > 1:
        print(f"  Latent samples: {num_samples}")

    print(f"\nProcessing 1 set of stacks...\n")

    try:
        predict_single_volume(
            model=model,
            output_path=output_paths[0] if isinstance(output_paths, list) else output_paths,
            target_res=target_res,
            device=device,
            input_stack_paths=input_stack_paths,
            fov_mask_paths=fov_mask_paths,
            orientation_mask=orientation_mask,
            save_reconstructions=save_reconstructions,
            reconstruction_dir=reconstruction_dir,
            num_samples=num_samples,
        )
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Inference complete!")
    print("=" * 80)


def _find_fov_mask(stack_path):
    """Try to find a FOV mask file next to a stack file.

    Looks for <stem>_fov_mask.nii.gz in the same directory.
    Returns the path if found, None otherwise.
    """
    if stack_path is None:
        return None
    p = Path(stack_path)
    stem = p.name.replace('.nii.gz', '').replace('.nii', '')
    candidate = p.parent / f"{stem}_fov_mask.nii.gz"
    if candidate.exists():
        return str(candidate)
    return None


def predict_folder(
    input_stacks_root: str,
    output_root: str,
    model_path: str,
    target_res=[1.0, 1.0, 1.0],
    device="cuda",
    orientation_mask=None,
    pattern_ax="axial_upsampled.nii.gz",
    pattern_cor="coronal_upsampled.nii.gz",
    pattern_sag="sagittal_upsampled.nii.gz",
    pattern_fov_ax=None,
    pattern_fov_cor=None,
    pattern_fov_sag=None,
    output_name="mshved_prediction.nii.gz",
    skip_existing=False,
    fail_fast=False,
    save_reconstructions=False,
    reconstruction_dir=None,
    num_samples=1,
):
    """
    Batch inference over a directory of subject folders.

    Each subject folder is expected to contain:
      - axial_upsampled.nii.gz (+ optional axial_upsampled_fov_mask.nii.gz)
      - coronal_upsampled.nii.gz (+ optional coronal_upsampled_fov_mask.nii.gz)
      - sagittal_upsampled.nii.gz (+ optional sagittal_upsampled_fov_mask.nii.gz)

    FOV masks are auto-discovered: for each stack <stem>.nii.gz, the script
    looks for <stem>_fov_mask.nii.gz in the same directory.
    """
    stacks_root = Path(input_stacks_root)
    if not stacks_root.exists() or not stacks_root.is_dir():
        raise ValueError(f"--input_stacks_root is not a directory: {stacks_root}")

    out_root = Path(output_root) if output_root else stacks_root
    out_root.mkdir(parents=True, exist_ok=True)

    if orientation_mask is None:
        orientation_mask = [1, 1, 1]

    model, checkpoint = load_mshved_from_checkpoint(model_path, device=device)

    subject_dirs = sorted([p for p in stacks_root.iterdir() if p.is_dir()])
    print(f"\nFound {len(subject_dirs)} subject folders in: {stacks_root}\n")

    for i, subj_dir in enumerate(tqdm(subject_dirs, desc="Processing subjects"), start=1):
        subj_id = subj_dir.name

        ax = subj_dir / pattern_ax
        cor = subj_dir / pattern_cor
        sag = subj_dir / pattern_sag

        stack_paths = [None, None, None]
        file_candidates = [ax, cor, sag]

        missing_required = []
        for idx, (m, p) in enumerate(zip(orientation_mask, file_candidates)):
            if m == 1:
                if p.exists():
                    stack_paths[idx] = str(p)
                else:
                    missing_required.append(str(p))
            else:
                stack_paths[idx] = None

        out_path = out_root / subj_id / output_name if output_root else subj_dir / output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and out_path.exists():
            print(f"\n[{i}/{len(subject_dirs)}] {subj_id} -> skipping (exists): {out_path}")
            continue

        if missing_required:
            msg = f"\n[{i}/{len(subject_dirs)}] {subj_id} -> skipping (missing required): {missing_required}"
            if fail_fast:
                raise FileNotFoundError(msg)
            print(msg)
            continue

        # Discover FOV masks: use explicit patterns if provided, else auto-discover
        fov_patterns = [pattern_fov_ax, pattern_fov_cor, pattern_fov_sag]
        fov_mask_paths = []
        for sp, fov_pat in zip(stack_paths, fov_patterns):
            if fov_pat is not None and sp is not None:
                candidate = subj_dir / fov_pat
                fov_mask_paths.append(str(candidate) if candidate.exists() else None)
            else:
                fov_mask_paths.append(_find_fov_mask(sp))
        has_any_mask = any(p is not None for p in fov_mask_paths)

        print(f"\n[{i}/{len(subject_dirs)}] {subj_id}"
              f" {'(with FOV masks)' if has_any_mask else '(no FOV masks)'}")

        try:
            predict_single_volume(
                model=model,
                output_path=str(out_path),
                device=device,
                input_stack_paths=stack_paths,
                fov_mask_paths=fov_mask_paths if has_any_mask else None,
                target_res=target_res,
                orientation_mask=orientation_mask,
                save_reconstructions=save_reconstructions,
                reconstruction_dir=reconstruction_dir,
                num_samples=num_samples,
            )
        except Exception as e:
            print(f"  ERROR on {subj_id}: {e}")
            import traceback
            traceback.print_exc()
            if fail_fast:
                raise

    print("\nAll subjects done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MS-HVED Inference with FOV Mask Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single case with FOV masks
  python test.py --input_stacks ax.nii.gz cor.nii.gz sag.nii.gz \\
      --fov_masks ax_fov_mask.nii.gz cor_fov_mask.nii.gz sag_fov_mask.nii.gz \\
      --output sr.nii.gz --model checkpoint.pth

  # Single case without FOV masks (same as original test.py)
  python test.py --input_stacks ax.nii.gz cor.nii.gz sag.nii.gz \\
      --output sr.nii.gz --model checkpoint.pth

  # Folder mode (auto-discovers *_fov_mask.nii.gz next to each stack)
  python test.py --input_stacks_root /subjects --output_root /out \\
      --model checkpoint.pth

  # With missing orientations
  python test.py --input_stacks ax.nii.gz cor.nii.gz \\
      --fov_masks ax_fov_mask.nii.gz cor_fov_mask.nii.gz \\
      --orientation_mask 1 1 0 --output sr.nii.gz --model checkpoint.pth
        """
    )

    # Input/output
    parser.add_argument("--input_stacks", type=str, nargs='+', default=None,
                       help="Orthogonal LR stack files (1-3 stacks) in order: axial, coronal, sagittal. "
                            "If fewer than 3 stacks, you MUST also specify --orientation_mask.")
    parser.add_argument("--fov_masks", type=str, nargs='+', default=None,
                       help="FOV mask NIfTI files matching the input stacks (same order). "
                            "1=missing, 0=valid. Generated by prepare4test.py.")
    parser.add_argument("--output", type=str, required=False,
                       help="Output image file or directory")

    # Model
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint (.pth file)")

    # Preprocessing
    parser.add_argument("--target_res", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                       help="Target resolution in mm (e.g., 1.0 1.0 1.0)")

    # Inference
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device: cuda or cpu")

    # Orientation handling
    parser.add_argument("--orientation_mask", type=int, nargs=3, default=None,
                       help="Binary mask indicating which orientations are present (e.g., 1 1 0)")
    parser.add_argument("--save_reconstructions", action="store_true",
                       help="Save reconstructed orientation outputs")
    parser.add_argument("--reconstruction_dir", type=str, default=None,
                       help="Directory to save reconstructed orientations")

    # Folder mode
    parser.add_argument("--input_stacks_root", type=str, default=None,
                    help="Root dir containing subject subfolders of orthogonal stacks.")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Where to save outputs for folder mode.")
    parser.add_argument("--pattern_ax", type=str, default="axial_upsampled.nii.gz")
    parser.add_argument("--pattern_cor", type=str, default="coronal_upsampled.nii.gz")
    parser.add_argument("--pattern_sag", type=str, default="sagittal_upsampled.nii.gz")
    parser.add_argument("--pattern_fov_ax", type=str, default=None,
                        help="FOV mask filename for axial stack. Default: auto-discover via <stack_stem>_fov_mask.nii.gz")
    parser.add_argument("--pattern_fov_cor", type=str, default=None,
                        help="FOV mask filename for coronal stack. Default: auto-discover via <stack_stem>_fov_mask.nii.gz")
    parser.add_argument("--pattern_fov_sag", type=str, default=None,
                        help="FOV mask filename for sagittal stack. Default: auto-discover via <stack_stem>_fov_mask.nii.gz")
    parser.add_argument("--output_name", type=str, default="mshved_prediction.nii.gz")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")

    # Multi-sample inference
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of stochastic samples to draw from the latent space. "
                            "Each sample produces a separate output file (<name>_sample{i}.nii.gz). "
                            "Default: 1 (deterministic, returns posterior mean).")

    # FOV fusion overrides (override checkpoint values at inference time)
    parser.add_argument("--fov_attenuation", type=float, default=None,
                       help="Override FOV precision attenuation (0=none, 1=full). Default: use checkpoint value.")
    parser.add_argument("--no_smooth_fov", action="store_true",
                       help="Disable FOV mask smoothing (use hard binary boundaries)")
    parser.add_argument("--fov_transition_width", type=float, default=None,
                       help="Override FOV smoothing sigma in voxels. Default: use checkpoint value.")

    # Global residual override (overrides checkpoint value at inference time)
    parser.add_argument("--global_residual", action="store_true",
                       help="Force-enable global residual at inference (overrides checkpoint). "
                            "Adds the reference (mean of input stacks) to the decoder output and clamps to [0,1].")
    parser.add_argument("--no_global_residual", action="store_true",
                       help="Force-disable global residual at inference (overrides checkpoint). "
                            "Note: if the checkpoint was trained with global_residual=True, the decoder's "
                            "final activation is 'none', so disabling residual may produce unbounded outputs.")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.global_residual and args.no_global_residual:
        raise ValueError("--global_residual and --no_global_residual are mutually exclusive.")

    # Store FOV overrides for post-load application
    _fov_overrides = {}
    if args.no_smooth_fov:
        _fov_overrides['smooth_fov'] = False
    if args.fov_attenuation is not None:
        _fov_overrides['fov_attenuation'] = args.fov_attenuation
    if args.fov_transition_width is not None:
        _fov_overrides['fov_transition_width'] = args.fov_transition_width

    # Resolve global_residual override (None = use checkpoint value)
    _global_residual_override = None
    if args.no_global_residual:
        _global_residual_override = False
    elif args.global_residual:
        _global_residual_override = True

    # Monkey-patch load function to apply overrides
    _orig_load = load_mshved_from_checkpoint
    def _load_with_overrides(checkpoint_path, device="cuda"):
        model, checkpoint = _orig_load(checkpoint_path, device)
        if _fov_overrides:
            fusion_module = model.fusion.fusion  # MultiScaleFusion -> ProductOfGaussians
            for k, v in _fov_overrides.items():
                setattr(fusion_module, k, v)
            print(f"  FOV fusion overrides applied: {_fov_overrides}")
        if _global_residual_override is not None:
            model.global_residual = _global_residual_override
            print(f"  global_residual override: {'enabled' if _global_residual_override else 'disabled'}")
        return model, checkpoint
    load_mshved_from_checkpoint = _load_with_overrides

    # ---- Mode 0: Folder mode ----
    if getattr(args, "input_stacks_root", None):
        if args.orientation_mask is None:
            args.orientation_mask = [1, 1, 1]

        predict_folder(
            input_stacks_root=args.input_stacks_root,
            output_root=getattr(args, "output_root", None),
            model_path=args.model,
            target_res=args.target_res,
            device=args.device,
            orientation_mask=args.orientation_mask,
            pattern_ax=getattr(args, "pattern_ax", "axial_upsampled.nii.gz"),
            pattern_cor=getattr(args, "pattern_cor", "coronal_upsampled.nii.gz"),
            pattern_sag=getattr(args, "pattern_sag", "sagittal_upsampled.nii.gz"),
            pattern_fov_ax=getattr(args, "pattern_fov_ax", None),
            pattern_fov_cor=getattr(args, "pattern_fov_cor", None),
            pattern_fov_sag=getattr(args, "pattern_fov_sag", None),
            output_name=getattr(args, "output_name", "mshved_prediction.nii.gz"),
            skip_existing=getattr(args, "skip_existing", False),
            fail_fast=getattr(args, "fail_fast", False),
            save_reconstructions=args.save_reconstructions,
            reconstruction_dir=args.reconstruction_dir,
            num_samples=args.num_samples,
        )
        raise SystemExit(0)

    # ---- Mode 1: Single case ----
    if args.input_stacks:
        num_stacks = len(args.input_stacks)
        if not (1 <= num_stacks <= 3):
            raise ValueError(f"Expected 1-3 input stacks, got {num_stacks}")

        for i, stack_path in enumerate(args.input_stacks):
            if not Path(stack_path).exists():
                raise ValueError(f"Provided stack {i+1} not found: {stack_path}")

        # Validate orientation mask
        if args.orientation_mask is None:
            if num_stacks == 3:
                args.orientation_mask = [1, 1, 1]
            else:
                raise ValueError(
                    f"When providing fewer than 3 stacks ({num_stacks} provided), "
                    "you MUST specify --orientation_mask to indicate which orientations are present.\n"
                    "Example: For axial+coronal, use: --orientation_mask 1 1 0"
                )
        else:
            if len(args.orientation_mask) != 3:
                raise ValueError(f"--orientation_mask must have 3 values. Got: {args.orientation_mask}")
            if any(v not in (0, 1) for v in args.orientation_mask):
                raise ValueError(f"--orientation_mask values must be 0 or 1. Got: {args.orientation_mask}")

        num_present = sum(args.orientation_mask)
        if num_present == 0:
            raise ValueError("orientation_mask cannot be all zeros.")

        # Build full [ax, cor, sag] list with None placeholders
        if num_stacks < 3:
            if num_present != num_stacks:
                raise ValueError(
                    f"Orientation mask indicates {num_present} present orientations, "
                    f"but {num_stacks} stacks were provided. These must match!"
                )

            input_stack_paths = [None, None, None]
            stack_idx = 0
            for i, present in enumerate(args.orientation_mask):
                if present == 1:
                    input_stack_paths[i] = args.input_stacks[stack_idx]
                    stack_idx += 1
        else:
            input_stack_paths = list(args.input_stacks)
            for i, present in enumerate(args.orientation_mask):
                if present == 0:
                    input_stack_paths[i] = None

        # Build FOV mask paths (same logic: align with orientation mask)
        fov_mask_paths = None
        if args.fov_masks:
            num_masks = len(args.fov_masks)
            if num_masks != num_stacks:
                raise ValueError(
                    f"Number of FOV masks ({num_masks}) must match number of input stacks ({num_stacks})"
                )
            for mask_path in args.fov_masks:
                if not Path(mask_path).exists():
                    raise ValueError(f"FOV mask not found: {mask_path}")

            if num_stacks < 3:
                fov_mask_paths = [None, None, None]
                mask_idx = 0
                for i, present in enumerate(args.orientation_mask):
                    if present == 1:
                        fov_mask_paths[i] = args.fov_masks[mask_idx]
                        mask_idx += 1
            else:
                fov_mask_paths = list(args.fov_masks)
                for i, present in enumerate(args.orientation_mask):
                    if present == 0:
                        fov_mask_paths[i] = None

        names = ["Axial", "Coronal", "Sagittal"]
        used = [n for n, p in zip(names, input_stack_paths) if p is not None]
        print(f"\nStack mode: using {len(used)}/3 orientations -> {', '.join(used)}")
        print(f"   orientation_mask = {args.orientation_mask}")
        if fov_mask_paths:
            print(f"   FOV masks: provided")
        else:
            print(f"   FOV masks: not provided")

        predict_batch(
            output_paths=args.output,
            model_path=args.model,
            target_res=args.target_res,
            device=args.device,
            input_stack_paths=input_stack_paths,
            fov_mask_paths=fov_mask_paths,
            orientation_mask=args.orientation_mask,
            save_reconstructions=args.save_reconstructions,
            reconstruction_dir=args.reconstruction_dir,
            num_samples=args.num_samples,
        )
