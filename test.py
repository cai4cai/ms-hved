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
    """
    Extract voxel resolution [x,y,z] in mm from NIfTI affine matrix.

    Args:
        affine: 4x4 affine transformation matrix

    Returns:
        Array of [res_x, res_y, res_z] in mm
    """
    res_x = np.linalg.norm(affine[:3, 0])
    res_y = np.linalg.norm(affine[:3, 1])
    res_z = np.linalg.norm(affine[:3, 2])
    return np.array([res_x, res_y, res_z])

def is_anisotropic(resolution: np.ndarray, threshold: float = 0.1) -> bool:
    """
    Check if resolution is anisotropic (non-cubic voxels).

    Args:
        resolution: Array of [res_x, res_y, res_z] in mm
        threshold: Maximum allowed difference (mm) to consider isotropic

    Returns:
        True if anisotropic, False if isotropic
    """
    res_range = resolution.max() - resolution.min()
    return res_range > threshold

def create_isotropic_affine(target_res: list, shape: tuple, original_affine: np.ndarray) -> np.ndarray:
    """
    Create affine matrix for isotropic space, preserving orientation from original.

    Args:
        target_res: Isotropic resolution [x,y,z] in mm
        shape: Shape of isotropic volume (D,H,W)
        original_affine: Original affine to preserve orientation

    Returns:
        4x4 affine matrix for isotropic space
    """
    # Extract rotation/orientation from original affine (normalized)
    rotation = original_affine[:3, :3]
    u = rotation[:, 0] / np.linalg.norm(rotation[:, 0])
    v = rotation[:, 1] / np.linalg.norm(rotation[:, 1])
    w = rotation[:, 2] / np.linalg.norm(rotation[:, 2])

    # Create new affine with isotropic scaling
    new_affine = np.eye(4)
    new_affine[:3, 0] = u * target_res[0]
    new_affine[:3, 1] = v * target_res[1]
    new_affine[:3, 2] = w * target_res[2]
    new_affine[:3, 3] = original_affine[:3, 3]  # Preserve translation

    return new_affine

def load_mshved_from_checkpoint(checkpoint_path, device="cuda"):
    """
    Load MS-HVED model from checkpoint.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get('model_config', {})

    # Detect architecture (default to 'uhved' for backward compatibility)
    model_architecture = model_config.get('model_architecture', 'mshved')
    print(f"Detected architecture: {model_architecture}")
    print(f"Model Configuration:")
    for key, value in model_config.items():
        print(f"  - {key}: {value}")

    # Extract common parameters
    num_orientations = model_config.get('num_orientations', 3)
    num_scales = model_config.get('num_scales', 4)
    reconstruct_orientations = model_config.get('reconstruct_orientations', False)
    use_prior = model_config.get('use_prior', True)
    use_encoder_outputs_as_skip = model_config.get('use_encoder_outputs_as_skip', False)
    decoder_upsample_mode = model_config.get('decoder_upsample_mode', 'trilinear')
    final_activation = model_config.get('final_activation', 'sigmoid')
    share_encoder = model_config.get('share_encoder', False)
    share_decoder = model_config.get('share_decoder', False)
    in_channels = model_config.get('in_channels', 1)
    out_channels = model_config.get('out_channels', 1)
    use_norm = model_config.get('use_norm', False)
    global_residual = model_config.get('global_residual', True)

    print(f"Model configuration:")
    print(f"  - Architecture: {model_architecture}")
    print(f"  - num_orientations: {num_orientations}")
    print(f"  - num_scales: {num_scales}")

    # Common creation parameters
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
        'use_norm': use_norm,
        'global_residual': global_residual
    }

    init_filters = model_config.get('init_filters', 32)
    blocks_down = tuple(model_config.get('blocks_down', [1, 2, 2, 4]))
    blocks_up = tuple(model_config.get('blocks_up', [1, 1, 1]))
    num_groups = model_config.get('num_groups', 8)

    print(f"  - init_filters: {init_filters}")
    print(f"  - blocks_down: {blocks_down}")
    print(f"  - blocks_up: {blocks_up}")
    print(f"  - num_groups: {num_groups}")
    print(f"  - global_residual: {global_residual}")
    print(f"  - use_norm: {use_norm}")

    model = MSHVED(
        init_filters=init_filters,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
        num_groups=num_groups,
        **params
    )

    # Remap checkpoint keys for backward compatibility
    state_dict = checkpoint['model_state_dict']
    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    return model, checkpoint

def create_inference_transforms(target_res=[1.0, 1.0, 1.0]):
    """
    Create MONAI preprocessing transforms for inference.

    Args:
        target_res: Target resolution [x, y, z] in mm

    Returns:
        MONAI Compose transform
    """
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
        stack_paths: List of 3 file paths [axial, coronal, sagittal] - ORDER MATTERS!
                    Can contain None for missing orientations.
        target_res: Target resolution [x, y, z] in mm

    Returns:
        Tuple of (lr_stacks_tensors, metadata_dict)

        metadata_dict contains:
        - 'affine_original': Original affine from first stack
        - 'affine_isotropic': Affine for isotropic resampled space
        - 'resolution_original': Original resolution [x,y,z] in mm
        - 'shape_isotropic': Shape after resampling (D,H,W)
        - 'is_anisotropic': Boolean flag
    """
    print("  Loading pre-existing orthogonal LR stacks...")
    print("  ⚠️  Order: [Axial, Coronal, Sagittal]")

    # Initialize metadata dictionary
    metadata = {
        'affine_original': None,
        'affine_isotropic': None,
        'resolution_original': None,
        'shape_isotropic': None,
        'is_anisotropic': False,
    }

    # BEFORE transforms: Load first valid stack to get original metadata
    for stack_path in stack_paths:
        if stack_path is not None:
            try:
                original_img = nib.load(stack_path)
                metadata['affine_original'] = original_img.affine.copy()
                metadata['resolution_original'] = get_resolution_from_affine(original_img.affine)
                metadata['is_anisotropic'] = is_anisotropic(metadata['resolution_original'])

                if metadata['is_anisotropic']:
                    print(f"  ⚠️  Detected anisotropic resolution: {metadata['resolution_original']} mm")
                    print(f"      Resampling to isotropic: {target_res} mm")
                else:
                    print(f"  ℹ️  Input resolution: {metadata['resolution_original']} mm (already isotropic)")
            except Exception as e:
                print(f"  Warning: Could not load metadata from {stack_path}: {e}")
                # Fallback to default
                metadata['affine_original'] = np.diag([target_res[0], target_res[1], target_res[2], 1.0])
                metadata['resolution_original'] = np.array(target_res)
            break

    # If no valid stacks found, use default
    if metadata['affine_original'] is None:
        metadata['affine_original'] = np.diag([target_res[0], target_res[1], target_res[2], 1.0])
        metadata['resolution_original'] = np.array(target_res)

    transforms = create_inference_transforms(target_res)
    lr_stacks_tensors = [None, None, None]  # Initialize with placeholders
    reference_shape = None

    orientation_mapping = [
        ("Axial", "Stack 0", "High-res in D axis"),
        ("Coronal", "Stack 1", "High-res in H axis"),
        ("Sagittal", "Stack 2", "High-res in W axis"),
    ]

    for i, stack_path in enumerate(stack_paths):
        orientation, stack_num, description = orientation_mapping[i]

        if stack_path is None:
            # Missing orientation - create dummy stack later
            print(f"    - {stack_num} ({orientation}): [MISSING - will use dummy stack]")
            # Keep as None, will create dummy after we know the shape
            continue

        print(f"    - {stack_num} ({orientation}): {stack_path}")
        print(f"      └─ {description}")

        # Load and preprocess with dictionary-based transforms
        data_dict = {"image": stack_path}
        data = transforms(data_dict)

        # Extract volume (dictionary-based transforms return dict)
        volume = data["image"]

        # Convert to numpy if tensor
        if isinstance(volume, torch.Tensor):
            volume_np = volume.cpu().numpy()
        else:
            volume_np = np.array(volume)

        # Remove channel dimension for processing (C, D, H, W) -> (D, H, W)
        if volume_np.ndim == 4 and volume_np.shape[0] == 1:
            volume_np = volume_np[0]

        # Store reference shape for creating dummy stacks
        if reference_shape is None:
            reference_shape = volume_np.shape

        # Normalize to [0, 1]
        volume_np = (volume_np - volume_np.min()) / (volume_np.max() - volume_np.min() + 1e-8)

        # Add to list as tensor with channel dimension
        lr_stacks_tensors[i] = torch.from_numpy(volume_np).float().unsqueeze(0)

    # Create dummy stacks for missing orientations
    if reference_shape is None:
        raise ValueError("No valid stacks provided - at least one stack is required!")

    for i, stack in enumerate(lr_stacks_tensors):
        if stack is None:
            # Create zero-filled dummy stack with same shape as reference
            dummy = torch.zeros((1,) + reference_shape, dtype=torch.float32)
            lr_stacks_tensors[i] = dummy
            print(f"    Created dummy stack for {orientation_mapping[i][0]}: shape {dummy.shape}")

    # AFTER resampling: Create isotropic affine and finalize metadata
    metadata['shape_isotropic'] = lr_stacks_tensors[0].squeeze().shape  # (D,H,W)
    metadata['affine_isotropic'] = create_isotropic_affine(
        target_res,
        metadata['shape_isotropic'],
        metadata['affine_original']
    )

    print(f"    Final stack shapes: {[s.shape for s in lr_stacks_tensors]}")
    return lr_stacks_tensors, metadata

def predict_single_volume(
    model,
    output_path,
    device="cuda",
    input_stack_paths=None,
    target_res=[1.0, 1.0, 1.0],
    orientation_mask=None,
    save_reconstructions=False,
    reconstruction_dir=None,
):
    """
    Run MS-HVED inference on a single volume.

    Args:
        model: Trained MS-HVED model
        output_path: Path to save output
        device: 'cuda' or 'cpu'
        input_stack_paths: Optional list of 3 pre-existing stack paths [axial, coronal, sagittal]
        orientation_mask: Optional binary mask [1/0, 1/0, 1/0] indicating which orientations are present

        save_reconstructions: Whether to save reconstructed orientations
        reconstruction_dir: Directory to save reconstructed orientations
    """
    for i, path in enumerate(input_stack_paths):
        print(f"  {['Axial', 'Coronal', 'Sagittal'][i]}: {path}")


    # Mode 1: Load 3 pre-existing orthogonal LR stacks WITH METADATA
    lr_stacks, metadata = load_orthogonal_stacks_from_files(input_stack_paths, target_res)
    # Use isotropic affine for output (matches resampled data)
    affine = metadata['affine_isotropic']

    # Pad to multiple of 32 if needed
    original_shape = lr_stacks[0].squeeze().shape
    lr_stacks_padded = []

    for stack in lr_stacks:
        stack_np = stack.squeeze().cpu().numpy()
        padded, pad_before, orig_shape = pad_to_multiple_of_32(stack_np)
        lr_stacks_padded.append(
            torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        )

    # Move to device
    lr_stacks_padded = [stack.to(device) for stack in lr_stacks_padded]
    # Create orientation mask tensor
    if orientation_mask is not None:
        # Convert to boolean tensor
        orientation_mask_tensor = torch.tensor(orientation_mask, dtype=torch.bool, device=device).unsqueeze(0)  # (1, 3)
        present_orientations = [i for i, m in enumerate(orientation_mask) if m == 1]
        print(f"  Using orientation mask: {orientation_mask}")
        print(f"  Present orientations: {[['Axial', 'Coronal', 'Sagittal'][i] for i in present_orientations]}")
    else:
        # All orientations present
        orientation_mask_tensor = None
        print(f"  Using all 3 orientations")

    # Run inference
    try: 
        model.eval()
        with torch.no_grad():
            print("  Running standard inference...")
            outputs = model(lr_stacks_padded, orientation_mask=orientation_mask_tensor)

            # Extract outputs
            sr_output = outputs['sr_output']
            orientation_outputs = outputs.get('orientation_outputs', [])

        # Convert SR output back to numpy
        sr_output = sr_output.squeeze().cpu().numpy()  # (D, H, W)
        print(f"  SR output shape before unpad: {sr_output.shape}")

        # Unpad to original shape
        sr_output = unpad_volume(sr_output, pad_before, orig_shape)
        print(f"  SR output shape after unpad: {sr_output.shape}")

        # Denormalize (keep in 0-1 range, scale by original max)
        sr_output = np.clip(sr_output, 0, 1)

        # Save SR output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        out_nii = nib.Nifti1Image(sr_output, affine)
        nib.save(out_nii, output_path)

        # Log output information
        output_res = get_resolution_from_affine(affine)
        print(f"  ✓ SR output saved to: {output_path}")
        print(f"    Output shape: {sr_output.shape}")
        print(f"    Output resolution: [{output_res[0]:.2f}, {output_res[1]:.2f}, {output_res[2]:.2f}] mm")
        print(f"    Output range: [{sr_output.min():.4f}, {sr_output.max():.4f}]")

        # Save reconstructed orientations if requested
        if save_reconstructions and len(orientation_outputs) > 0:
            print(f"\n  Saving reconstructed orientations...")

            # Determine output directory
            if reconstruction_dir is None:
                reconstruction_dir = os.path.dirname(output_path) or "."
            os.makedirs(reconstruction_dir, exist_ok=True)

            # Get base filename
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            if base_name.endswith('.nii'):
                base_name = base_name[:-4]  # Remove .nii from .nii.gz

            orientation_names = ['axial', 'coronal', 'sagittal']
            for i, recon in enumerate(orientation_outputs):
                # Convert to numpy and unpad
                recon_np = recon.squeeze().cpu().numpy()
                recon_np = unpad_volume(recon_np, pad_before, orig_shape)
                recon_np = np.clip(recon_np, 0, 1)

                # Save
                recon_path = os.path.join(reconstruction_dir, f"{base_name}_recon_{orientation_names[i]}.nii.gz")
                recon_nii = nib.Nifti1Image(recon_np, affine)
                nib.save(recon_nii, recon_path)
                print(f"    - {orientation_names[i]}: {recon_path}")
        elif save_reconstructions and len(orientation_outputs) == 0:
            print(f"  Note: Model was not trained with orientation reconstruction, skipping...")
    
    finally:
        try:
            del lr_stacks, lr_stacks_padded
            if 'outputs' in locals(): del outputs
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
    orientation_mask=None,
    save_reconstructions=False,
    reconstruction_dir=None,
):
    """Process multiple volumes in batch."""
    print("=" * 80)
    print("MS-HVED Inference - Orthogonal Stack Super-Resolution")
    print("=" * 80)

    # Load model
    model, checkpoint = load_mshved_from_checkpoint(model_path, device=device)

    print(f"\nInference settings:")
    print(f"  Mode: {'Pre-existing stacks' if input_stack_paths else 'Generate from HR volume'}")
    print(f"  Device: {device}")
    print(f"  Target resolution: {target_res} mm")

    if input_stack_paths:
        print(f"\nProcessing 1 set of stacks...\n")

    # Process volume(s)
    if input_stack_paths:
        # Single case: process one set of 3 stacks
        print(f"[1/1]")
        try:
            predict_single_volume(
                model=model,
                output_path=output_paths[0] if isinstance(output_paths, list) else output_paths,
                target_res=target_res,
                device=device,
                input_stack_paths=input_stack_paths,
                orientation_mask=orientation_mask,
                save_reconstructions=save_reconstructions,
                reconstruction_dir=reconstruction_dir,
            )
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Inference complete!")
    print("=" * 80)

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
    output_name="mshved_prediction.nii.gz",
    skip_existing=False,
    fail_fast=False,
    save_reconstructions=False,
    reconstruction_dir=None,
):
    """
    Batch inference over a directory of subject folders.

    Each subject folder is expected to contain:
      - axial_upsampled.nii.gz
      - coronal_upsampled.nii.gz
      - sagittal_upsampled.nii.gz

    orientation_mask controls which orientations are USED, even if files exist.
    Missing masked-in files -> skip subject (or fail_fast).
    Masked-out orientations are ignored (passed as None to the loader).
    """
    stacks_root = Path(input_stacks_root)
    if not stacks_root.exists() or not stacks_root.is_dir():
        raise ValueError(f"--input_stacks_root is not a directory: {stacks_root}")

    out_root = Path(output_root) if output_root else stacks_root
    out_root.mkdir(parents=True, exist_ok=True)

    # Default: use all 3
    if orientation_mask is None:
        orientation_mask = [1, 1, 1]

    # Load model
    model, checkpoint = load_mshved_from_checkpoint(model_path, device=device)

    subject_dirs = sorted([p for p in stacks_root.iterdir() if p.is_dir()])
    print(f"\nFound {len(subject_dirs)} subject folders in: {stacks_root}\n")

    for i, subj_dir in enumerate(tqdm(subject_dirs, desc="Generating LR stacks"), start=1):
        subj_id = subj_dir.name

        ax = subj_dir / pattern_ax
        cor = subj_dir / pattern_cor
        sag = subj_dir / pattern_sag

        # Build input_stack_paths with None placeholders to match your existing loader
        # Index mapping: [Axial, Coronal, Sagittal]
        stack_paths = [None, None, None]
        file_candidates = [ax, cor, sag]

        # Enforce mask: if mask=0, pass None even if file exists
        # If mask=1, require file exists
        missing_required = []
        for idx, (m, p) in enumerate(zip(orientation_mask, file_candidates)):
            if m == 1:
                if p.exists():
                    stack_paths[idx] = str(p)
                else:
                    missing_required.append(str(p))
            else:
                stack_paths[idx] = None

        # Decide output path
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

        print(f"\n[{i}/{len(subject_dirs)}] {subj_id}")
        try:
            predict_single_volume(
                model=model,
                output_path=str(out_path),
                device=device,
                input_stack_paths=stack_paths,
                target_res=target_res,
                orientation_mask=orientation_mask,
                save_reconstructions=save_reconstructions,
                reconstruction_dir=reconstruction_dir,
            )
        except Exception as e:
            print(f"  ✗ ERROR on {subj_id}: {e}")
            import traceback
            traceback.print_exc()
            if fail_fast:
                raise

    print("\nAll subjects done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MS-HVED Inference with Orthogonal Stacks")

    # Input/output arguments
    parser.add_argument("--input_stacks", type=str, nargs='+', default=None,
                       help="Orthogonal LR stack files (1-3 stacks). Provide in order: axial, coronal, sagittal. "
                            "If fewer than 3 stacks, you MUST also specify --orientation_mask to indicate which orientations are present. "
                            "Example: For axial+coronal only, use '--input_stacks axial.nii.gz coronal.nii.gz --orientation_mask 1 1 0'")
    parser.add_argument("--output", type=str, required=False,
                       help="Output image file or directory")

    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint (.pth file)")

    # Preprocessing arguments
    parser.add_argument("--target_res", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                       help="Target resolution in mm (e.g., 1.0 1.0 1.0)")

    # Inference arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device: cuda or cpu")

    # Orientation handling arguments
    parser.add_argument("--orientation_mask", type=int, nargs=3, default=None,
                       help="Binary mask indicating which orientations are present (e.g., 1 1 0 for axial+coronal only). "
                            "Use this to handle missing modalities. Default: all present (1 1 1)")
    parser.add_argument("--save_reconstructions", action="store_true",
                       help="Save reconstructed orientation outputs (if model was trained with orientation reconstruction)")
    parser.add_argument("--reconstruction_dir", type=str, default=None,
                       help="Directory to save reconstructed orientations (default: same as output directory)")

    parser.add_argument("--input_stacks_root", type=str, default=None,
                    help="Root dir containing subject subfolders of orthogonal stacks.")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Where to save outputs for folder mode. If omitted, saves into each subject folder.")
    parser.add_argument("--pattern_ax", type=str, default="axial_upsampled.nii.gz")
    parser.add_argument("--pattern_cor", type=str, default="coronal_upsampled.nii.gz")
    parser.add_argument("--pattern_sag", type=str, default="sagittal_upsampled.nii.gz")
    parser.add_argument("--output_name", type=str, default="mshved_prediction.nii.gz")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # -------------------------------
    # Mode 0: Folder mode (batch over subject directories)
    # -------------------------------
    if getattr(args, "input_stacks_root", None):
        # Default mask = use all orientations
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
            output_name=getattr(args, "output_name", "mshved_prediction.nii.gz"),
            skip_existing=getattr(args, "skip_existing", False),
            fail_fast=getattr(args, "fail_fast", False),
            save_reconstructions=args.save_reconstructions,
            reconstruction_dir=args.reconstruction_dir,
        )
        raise SystemExit(0)

    # -------------------------------
    # Mode 1: Pre-existing orthogonal stacks (single case)
    # -------------------------------
    if args.input_stacks:
        num_stacks = len(args.input_stacks)
        if not (1 <= num_stacks <= 3):
            raise ValueError(f"Expected 1-3 input stacks, got {num_stacks}")

        # Validate provided stack files exist
        for i, stack_path in enumerate(args.input_stacks):
            if not Path(stack_path).exists():
                raise ValueError(f"Provided stack {i+1} not found: {stack_path}")

        # Normalize / validate orientation mask
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
                raise ValueError(f"--orientation_mask must have 3 values (ax cor sag). Got: {args.orientation_mask}")
            if any(v not in (0, 1) for v in args.orientation_mask):
                raise ValueError(f"--orientation_mask values must be 0 or 1. Got: {args.orientation_mask}")

        num_present = sum(args.orientation_mask)
        if num_present == 0:
            raise ValueError("orientation_mask cannot be all zeros.")

        # Build full [ax, cor, sag] list with None placeholders
        # If 3 stacks provided, we assume order is [ax, cor, sag] and still allow mask to disable any.
        if num_stacks < 3:
            if num_present != num_stacks:
                raise ValueError(
                    f"Orientation mask indicates {num_present} present orientations, "
                    f"but {num_stacks} stacks were provided. These must match!\n"
                    "Tip: stacks are passed in the same order as the 1s in --orientation_mask "
                    "(axial, coronal, sagittal)."
                )

            input_stack_paths = [None, None, None]
            stack_idx = 0
            for i, present in enumerate(args.orientation_mask):
                if present == 1:
                    input_stack_paths[i] = args.input_stacks[stack_idx]
                    stack_idx += 1
        else:
            input_stack_paths = list(args.input_stacks)  # [ax, cor, sag]
            for i, present in enumerate(args.orientation_mask):
                if present == 0:
                    input_stack_paths[i] = None

        # Optional logging
        names = ["Axial", "Coronal", "Sagittal"]
        used = [n for n, p in zip(names, input_stack_paths) if p is not None]
        print(f"\n📦 Stack mode: using {len(used)}/3 orientations -> {', '.join(used)}")
        print(f"   orientation_mask = {args.orientation_mask}")

        # Run inference (single case)
        predict_batch(
            output_paths=args.output,  # if predict_batch expects list, change to [args.output]
            model_path=args.model,
            target_res=args.target_res,
            device=args.device,
            input_stack_paths=input_stack_paths,
            orientation_mask=args.orientation_mask,
            save_reconstructions=args.save_reconstructions,
            reconstruction_dir=args.reconstruction_dir,
        )
        raise SystemExit(0)

    # -------------------------------
    # If no valid mode was chosen
    # -------------------------------
    raise ValueError(
        "No valid inference mode selected. Use one of:\n"
        "  - --input_stacks_root <dir>\n"
        "  - --input_stacks <1-3 paths> (and possibly --orientation_mask)\n"
    )