"""MS-HVED memory-efficient sliding-window inference.

Drop-in alternative to test.py that runs the *entire* encode -> fuse -> decode
pipeline under a sliding window instead of on the whole volume at once.

Usage mirrors test.py:
  python infer.py --input_stacks ax.nii.gz cor.nii.gz sag.nii.gz \\
      --fov_masks ax_fov.nii.gz cor_fov.nii.gz sag_fov.nii.gz \\
      --output sr.nii.gz --model checkpoint.pth \\
      --roi_size 96 96 96 --overlap 0.5 --amp

  python infer.py --input_stacks_root /subjects --output_root /out \\
      --model checkpoint.pth --roi_size 128 128 128

Notes / limitations vs test.py:
  - Designed for the deterministic SR output (posterior mean). --num_samples > 1
    is supported but each window samples latent noise independently, which can
    produce visible seams between tiles; prefer the full-volume test.py for
    stochastic ensembles unless memory forbids it.
  - --save_reconstructions (orientation reconstructions) is intentionally not
    produced here to keep per-window memory minimal.
"""

import os
import argparse
from pathlib import Path

import time
import contextlib

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm

from monai.inferers import sliding_window_inference

try:
    import psutil  # optional, for host-RAM tracking
    _PROC = psutil.Process(os.getpid())
except Exception:  # pragma: no cover
    psutil = None
    _PROC = None

# Reuse all the (well-tested) IO / preprocessing helpers from test.py.
from test import (
    cuda_cleanup,
    get_resolution_from_affine,
    load_mshved_from_checkpoint,
    load_orthogonal_stacks_from_files,
    load_fov_masks,
    _sample_output_path,
    _find_fov_mask,
)

ORIENTATION_NAMES = ["Axial", "Coronal", "Sagittal"]


def _fmt_bytes(n):
    """Human-readable byte count."""
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:6.1f} {unit}"
        n /= 1024.0
    return f"{n:6.1f} PB"


def _scale_output(sr_np, output_scale):
    """Rescale a [0,1] prediction to the requested output scale / dtype.

    'unit'    -> float32 in [0, 1] (default, matches test.py)
    'uint8'   -> uint8 in [0, 255]  (8-bit, smallest files, viewer-friendly)
    'float255'-> float32 in [0, 255] (0-255 range but keeps sub-integer precision)
    """
    if output_scale == "unit":
        return sr_np.astype(np.float32)
    if output_scale == "uint8":
        return np.clip(np.rint(sr_np * 255.0), 0, 255).astype(np.uint8)
    if output_scale == "float255":
        return (sr_np * 255.0).astype(np.float32)
    raise ValueError(f"Unknown output_scale: {output_scale}")


def _host_ram_bytes():
    """Resident set size of this process, or None if psutil is unavailable."""
    if _PROC is None:
        return None
    try:
        return _PROC.memory_info().rss
    except Exception:
        return None


class MemoryProfiler:
    """Tracks peak GPU and host-RAM usage across an inference run.

    GPU peak comes from torch.cuda's allocator stats (max_memory_allocated =
    tensors actually in use; max_memory_reserved = cached by the caching
    allocator). For sliding-window inference the peak is the per-tile working
    set, since only one window's activations are live at a time — that is the
    number that tells you whether chunking is paying off.

    Optionally measures the per-window peak by resetting the allocator's peak
    counter around each predictor call and keeping the maximum.
    """

    def __init__(self, device, enabled=True, per_window=False):
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        self.enabled = enabled
        self.per_window = per_window and self.is_cuda
        self.window_calls = 0
        self.window_peak_alloc = 0  # max over windows of per-window peak alloc
        self.t0 = None
        self.host_start = None

    def reset(self):
        if not self.enabled:
            return
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
        self.window_calls = 0
        self.window_peak_alloc = 0
        self.host_start = _host_ram_bytes()
        self.t0 = time.perf_counter()

    def wrap_predictor(self, predictor):
        """Optionally instrument a predictor to record per-window peak memory."""
        if not (self.enabled and self.per_window):
            if self.enabled:
                # Still count windows even without per-window peak tracking.
                def counting(window):
                    self.window_calls += 1
                    return predictor(window)
                return counting
            return predictor

        def profiled(window):
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            out = predictor(window)
            torch.cuda.synchronize(self.device)
            peak = torch.cuda.max_memory_allocated(self.device)
            self.window_peak_alloc = max(self.window_peak_alloc, peak)
            self.window_calls += 1
            return out

        return profiled

    def report(self, tag=""):
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.t0 if self.t0 is not None else 0.0
        prefix = f"  [mem]{(' ' + tag) if tag else ''}"
        print(f"{prefix} wall time: {elapsed:.1f}s  windows: {self.window_calls}")
        if self.is_cuda:
            peak_alloc = torch.cuda.max_memory_allocated(self.device)
            peak_resv = torch.cuda.max_memory_reserved(self.device)
            print(f"{prefix} GPU peak allocated: {_fmt_bytes(peak_alloc)}   "
                  f"reserved: {_fmt_bytes(peak_resv)}")
            if self.per_window and self.window_peak_alloc:
                print(f"{prefix} GPU peak per-window allocated: "
                      f"{_fmt_bytes(self.window_peak_alloc)}")
        host_now = _host_ram_bytes()
        if host_now is not None:
            delta = host_now - (self.host_start or host_now)
            print(f"{prefix} host RAM: {_fmt_bytes(host_now)} "
                  f"(+{_fmt_bytes(delta)} this run)")
        elif self.is_cuda:
            print(f"{prefix} (install psutil for host-RAM tracking)")


def _make_predictor(model, base_orientation_mask, has_fov, num_orientations,
                    amp=False, amp_dtype=torch.bfloat16, deterministic=True):
    """Build the per-window predictor closure for sliding_window_inference.

    The windowed input tensor is the channel-wise concatenation
        [stack_0, stack_1, stack_2, (fov_0, fov_1, fov_2)]
    so MONAI crops all of them with identical window coordinates. Inside, we
    split the channels back into the list-of-tensors layout the model expects.
    """
    def predictor(window):
        # window: (sw_batch, C, d, h, w) with C = num_orientations (+ num_orientations if fov)
        b = window.shape[0]
        stacks = [window[:, i:i + 1] for i in range(num_orientations)]

        fov_masks = None
        if has_fov:
            fov_masks = [window[:, num_orientations + i:num_orientations + i + 1]
                         for i in range(num_orientations)]

        # Expand the (1, num_orientations) mask to the current window batch size.
        om = None
        if base_orientation_mask is not None:
            om = base_orientation_mask.to(window.device)
            if om.shape[0] != b:
                om = om.expand(b, -1).contiguous()

        autocast_ctx = (
            torch.autocast(device_type=window.device.type, dtype=amp_dtype)
            if amp else torch.autocast(device_type=window.device.type, enabled=False)
        )
        with autocast_ctx:
            out = model(
                stacks,
                orientation_mask=om,
                fov_masks=fov_masks,
                deterministic=deterministic,
            )
        # Aggregation in MONAI is done in fp32; cast back from any autocast dtype.
        return out["sr_output"].float()

    return predictor


def _build_packed_input(lr_stacks, fov_full, num_orientations, device):
    """Concatenate stacks (+ fov masks) into one (1, C, D, H, W) tensor on `device`."""
    chans = []
    for s in lr_stacks:
        # load_orthogonal_stacks_from_files returns each stack as (1, D, H, W)
        v = s.squeeze()
        chans.append(v.unsqueeze(0).unsqueeze(0))  # (1,1,D,H,W)

    if fov_full is not None:
        for m in fov_full:
            chans.append(m.unsqueeze(0).unsqueeze(0))  # (1,1,D,H,W)

    packed = torch.cat(chans, dim=1).float().to(device)
    return packed


def _prepare_fov_full(fov_mask_paths, target_shape, device):
    """Load FOV masks and resize each to `target_shape` (D,H,W). Missing -> zeros.

    Returns a list of `num_orientations` CPU tensors, each (D, H, W), or None if
    no masks were provided. Kept on CPU so they pack alongside the (CPU) stacks;
    the assembled tensor is moved to `device` once in _build_packed_input.
    """
    if fov_mask_paths is None:
        return None

    raw = load_fov_masks(fov_mask_paths, target_shape, device)  # list of (D,H,W) or None
    out = []
    for m in raw:
        if m is None:
            out.append(torch.zeros(target_shape, dtype=torch.float32))
            continue
        mt = m.float().cpu()
        if tuple(mt.shape) != tuple(target_shape):
            mt = F.interpolate(
                mt.unsqueeze(0).unsqueeze(0), size=tuple(target_shape),
                mode="nearest",
            ).squeeze(0).squeeze(0)
            mt = (mt > 0.5).float()
        out.append(mt)
    return out


def predict_single_volume_chunked(
    model,
    output_path,
    device="cuda",
    input_stack_paths=None,
    fov_mask_paths=None,
    target_res=(1.0, 1.0, 1.0),
    orientation_mask=None,
    num_samples=1,
    roi_size=(96, 96, 96),
    overlap=0.5,
    sw_batch_size=1,
    sw_mode="gaussian",
    amp=False,
    amp_dtype=torch.bfloat16,
    cpu_output=True,
    profile=True,
    profile_windows=False,
    output_scale="unit",
):
    """Sliding-window MS-HVED inference for a single set of orthogonal stacks."""
    num_orientations = getattr(model, "num_orientations", 3)
    profiler = MemoryProfiler(device, enabled=profile, per_window=profile_windows)

    for i, path in enumerate(input_stack_paths):
        print(f"  {ORIENTATION_NAMES[i]}: {path}")

    # --- Load + preprocess stacks (resample to isotropic, min-max normalize) ---
    lr_stacks, metadata = load_orthogonal_stacks_from_files(input_stack_paths, list(target_res))
    affine = metadata["affine_isotropic"]
    vol_shape = tuple(lr_stacks[0].squeeze().shape)  # (D, H, W)

    # --- FOV masks (resized to the stack grid; not pre-padded — windows are exact ROI) ---
    has_fov = fov_mask_paths is not None
    fov_full = None
    if has_fov:
        print("  Loading FOV masks:")
        fov_full = _prepare_fov_full(fov_mask_paths, vol_shape, device)

    # --- Orientation mask (batched form: (1, num_orientations)) ---
    base_mask = None
    if orientation_mask is not None:
        base_mask = torch.tensor(orientation_mask, dtype=torch.bool, device=device).unsqueeze(0)
        present = [ORIENTATION_NAMES[i] for i, m in enumerate(orientation_mask) if m == 1]
        print(f"  Orientation mask: {orientation_mask} -> {', '.join(present)}")
    else:
        print(f"  Using all {num_orientations} orientations")

    print(f"  FOV masks: {'enabled (precision-weighted fusion)' if has_fov else 'not provided'}")
    print(f"  Sliding window: roi={tuple(roi_size)} overlap={overlap} "
          f"mode={sw_mode} sw_batch={sw_batch_size} amp={amp}")
    print(f"  Volume shape: {vol_shape}  output accumulator on {'CPU' if cpu_output else 'GPU'}")

    # ROI must be divisible by 32 (model has 4 scales -> 2^? downsampling; matches pad_to_multiple_of_32).
    for d, name in zip(roi_size, ("D", "H", "W")):
        if d % 32 != 0:
            raise ValueError(f"--roi_size {name}={d} must be a multiple of 32.")

    packed = _build_packed_input(lr_stacks, fov_full, num_orientations, device)

    out_device = torch.device("cpu") if cpu_output else torch.device(device)

    model.eval()
    n = max(1, num_samples)
    if n > 1:
        print(f"  WARNING: --num_samples={n} with tiling samples each window's latent "
              f"noise independently; expect seams between tiles.")

    try:
        with torch.no_grad():
            for sample_idx in range(n):
                deterministic = (n == 1)
                if deterministic:
                    model.fusion.sampler.eval()
                else:
                    # GaussianSampler draws noise only when in train mode.
                    model.fusion.sampler.train()
                    print(f"  Sample {sample_idx + 1}/{n}...")

                predictor = _make_predictor(
                    model, base_mask, has_fov, num_orientations,
                    amp=amp, amp_dtype=amp_dtype, deterministic=deterministic,
                )
                profiler.reset()
                predictor = profiler.wrap_predictor(predictor)

                sr = sliding_window_inference(
                    inputs=packed,
                    roi_size=tuple(roi_size),
                    sw_batch_size=sw_batch_size,
                    predictor=predictor,
                    overlap=overlap,
                    mode=sw_mode,
                    padding_mode="constant",
                    cval=0.0,
                    sw_device=torch.device(device),
                    device=out_device,
                    progress=True,
                )
                model.fusion.sampler.eval()

                sr_np = sr.squeeze().cpu().numpy().astype(np.float32)
                sr_np = np.clip(sr_np, 0, 1)
                sr_np = _scale_output(sr_np, output_scale)

                save_path = _sample_output_path(output_path, sample_idx + 1) if n > 1 else output_path
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                nib.save(nib.Nifti1Image(sr_np, affine), save_path)

                out_res = get_resolution_from_affine(affine)
                print(f"  SR output saved to: {save_path}")
                print(f"    shape={sr_np.shape}  res=[{out_res[0]:.2f},{out_res[1]:.2f},{out_res[2]:.2f}] mm"
                      f"  range=[{sr_np.min():.4f},{sr_np.max():.4f}]")

                profiler.report(tag=f"sample {sample_idx + 1}/{n}" if n > 1 else "")

                del sr
                cuda_cleanup()
    finally:
        del packed, lr_stacks
        if fov_full is not None:
            del fov_full
        cuda_cleanup()


def predict_folder_chunked(
    input_stacks_root,
    output_root,
    model_path,
    target_res=(1.0, 1.0, 1.0),
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
    **chunk_kwargs,
):
    """Batch sliding-window inference over a directory of subject folders."""
    stacks_root = Path(input_stacks_root)
    if not stacks_root.is_dir():
        raise ValueError(f"--input_stacks_root is not a directory: {stacks_root}")

    out_root = Path(output_root) if output_root else stacks_root
    out_root.mkdir(parents=True, exist_ok=True)

    if orientation_mask is None:
        orientation_mask = [1, 1, 1]

    model, _ = load_mshved_from_checkpoint(model_path, device=device)

    subject_dirs = sorted([p for p in stacks_root.iterdir() if p.is_dir()])
    print(f"\nFound {len(subject_dirs)} subject folders in: {stacks_root}\n")

    for i, subj_dir in enumerate(tqdm(subject_dirs, desc="Subjects"), start=1):
        subj_id = subj_dir.name
        candidates = [subj_dir / pattern_ax, subj_dir / pattern_cor, subj_dir / pattern_sag]

        stack_paths = [None, None, None]
        missing_required = []
        for idx, (m, p) in enumerate(zip(orientation_mask, candidates)):
            if m == 1:
                if p.exists():
                    stack_paths[idx] = str(p)
                else:
                    missing_required.append(str(p))

        out_path = (out_root / subj_id / output_name) if output_root else (subj_dir / output_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and out_path.exists():
            print(f"\n[{i}/{len(subject_dirs)}] {subj_id} -> skip (exists)")
            continue
        if missing_required:
            msg = f"\n[{i}/{len(subject_dirs)}] {subj_id} -> skip (missing): {missing_required}"
            if fail_fast:
                raise FileNotFoundError(msg)
            print(msg)
            continue

        fov_patterns = [pattern_fov_ax, pattern_fov_cor, pattern_fov_sag]
        fov_mask_paths = []
        for sp, fov_pat in zip(stack_paths, fov_patterns):
            if fov_pat is not None and sp is not None:
                cand = subj_dir / fov_pat
                fov_mask_paths.append(str(cand) if cand.exists() else None)
            else:
                fov_mask_paths.append(_find_fov_mask(sp))
        has_any = any(p is not None for p in fov_mask_paths)

        print(f"\n[{i}/{len(subject_dirs)}] {subj_id} {'(with FOV)' if has_any else '(no FOV)'}")
        try:
            predict_single_volume_chunked(
                model=model,
                output_path=str(out_path),
                device=device,
                input_stack_paths=stack_paths,
                fov_mask_paths=fov_mask_paths if has_any else None,
                target_res=target_res,
                orientation_mask=orientation_mask,
                **chunk_kwargs,
            )
        except Exception as e:
            print(f"  ERROR on {subj_id}: {e}")
            import traceback
            traceback.print_exc()
            if fail_fast:
                raise

    print("\nAll subjects done.")


def _resolve_stack_and_mask_lists(input_stacks, fov_masks, orientation_mask):
    """Mirror test.py's logic for mapping provided stacks -> [ax, cor, sag] slots."""
    num_stacks = len(input_stacks)
    if not (1 <= num_stacks <= 3):
        raise ValueError(f"Expected 1-3 input stacks, got {num_stacks}")
    for i, sp in enumerate(input_stacks):
        if not Path(sp).exists():
            raise ValueError(f"Provided stack {i + 1} not found: {sp}")

    if orientation_mask is None:
        if num_stacks == 3:
            orientation_mask = [1, 1, 1]
        else:
            raise ValueError(
                "When providing fewer than 3 stacks you MUST specify --orientation_mask "
                "(e.g. axial+coronal -> --orientation_mask 1 1 0)."
            )
    if len(orientation_mask) != 3 or any(v not in (0, 1) for v in orientation_mask):
        raise ValueError(f"--orientation_mask must be three 0/1 values. Got: {orientation_mask}")
    if sum(orientation_mask) == 0:
        raise ValueError("orientation_mask cannot be all zeros.")

    if num_stacks < 3:
        if sum(orientation_mask) != num_stacks:
            raise ValueError(
                f"orientation_mask has {sum(orientation_mask)} present orientations "
                f"but {num_stacks} stacks were provided; they must match."
            )
        stack_paths = [None, None, None]
        k = 0
        for i, present in enumerate(orientation_mask):
            if present:
                stack_paths[i] = input_stacks[k]
                k += 1
    else:
        stack_paths = list(input_stacks)
        for i, present in enumerate(orientation_mask):
            if not present:
                stack_paths[i] = None

    fov_mask_paths = None
    if fov_masks:
        if len(fov_masks) != num_stacks:
            raise ValueError(
                f"Number of FOV masks ({len(fov_masks)}) must match input stacks ({num_stacks})."
            )
        for mp in fov_masks:
            if not Path(mp).exists():
                raise ValueError(f"FOV mask not found: {mp}")
        if num_stacks < 3:
            fov_mask_paths = [None, None, None]
            k = 0
            for i, present in enumerate(orientation_mask):
                if present:
                    fov_mask_paths[i] = fov_masks[k]
                    k += 1
        else:
            fov_mask_paths = list(fov_masks)
            for i, present in enumerate(orientation_mask):
                if not present:
                    fov_mask_paths[i] = None

    return stack_paths, fov_mask_paths, orientation_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MS-HVED memory-efficient sliding-window inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Input/output (mirrors test.py)
    parser.add_argument("--input_stacks", type=str, nargs="+", default=None)
    parser.add_argument("--fov_masks", type=str, nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--target_res", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--orientation_mask", type=int, nargs=3, default=None)
    parser.add_argument("--num_samples", type=int, default=1)

    # Sliding-window / memory controls
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 96, 96],
                        help="Tile size (each dim must be a multiple of 32). Smaller = less GPU memory.")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Fractional window overlap (0-1). Higher hides seams better but is slower.")
    parser.add_argument("--sw_batch_size", type=int, default=1,
                        help="Windows per forward pass. 1 = lowest memory.")
    parser.add_argument("--sw_mode", type=str, default="gaussian", choices=["gaussian", "constant"],
                        help="Window blend weighting. 'gaussian' downweights tile edges (recommended).")
    parser.add_argument("--amp", action="store_true",
                        help="Autocast convolutions (fusion still runs fp32 internally).")
    parser.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--gpu_output", action="store_true",
                        help="Keep the output accumulator on GPU (default: CPU, lower GPU memory).")
    parser.add_argument("--output_scale", type=str, default="unit",
                        choices=["unit", "uint8", "float255"],
                        help="Output intensity scale: 'unit' = float [0,1] (default); "
                             "'uint8' = 8-bit [0,255] (smallest files); "
                             "'float255' = float [0,255] (0-255 range, full precision).")

    # Memory profiling
    parser.add_argument("--no_profile", action="store_true",
                        help="Disable peak GPU/host memory + timing reporting (on by default).")
    parser.add_argument("--profile_windows", action="store_true",
                        help="Also measure per-window peak GPU allocation (resets allocator "
                             "stats around each tile; slightly slower due to extra syncs).")

    # FOV fusion overrides (parity with test.py)
    parser.add_argument("--fov_attenuation", type=float, default=None)
    parser.add_argument("--no_smooth_fov", action="store_true")
    parser.add_argument("--fov_transition_width", type=float, default=None)
    parser.add_argument("--global_residual", action="store_true")
    parser.add_argument("--no_global_residual", action="store_true")

    # Folder mode
    parser.add_argument("--input_stacks_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--pattern_ax", type=str, default="axial_upsampled.nii.gz")
    parser.add_argument("--pattern_cor", type=str, default="coronal_upsampled.nii.gz")
    parser.add_argument("--pattern_sag", type=str, default="sagittal_upsampled.nii.gz")
    parser.add_argument("--pattern_fov_ax", type=str, default=None)
    parser.add_argument("--pattern_fov_cor", type=str, default=None)
    parser.add_argument("--pattern_fov_sag", type=str, default=None)
    parser.add_argument("--output_name", type=str, default="mshved_prediction.nii.gz")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    if args.global_residual and args.no_global_residual:
        raise ValueError("--global_residual and --no_global_residual are mutually exclusive.")

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    # FOV / residual overrides applied after load (same scheme as test.py).
    fov_overrides = {}
    if args.no_smooth_fov:
        fov_overrides["smooth_fov"] = False
    if args.fov_attenuation is not None:
        fov_overrides["fov_attenuation"] = args.fov_attenuation
    if args.fov_transition_width is not None:
        fov_overrides["fov_transition_width"] = args.fov_transition_width
    residual_override = True if args.global_residual else (False if args.no_global_residual else None)

    def load_with_overrides(checkpoint_path, device="cuda"):
        model, checkpoint = load_mshved_from_checkpoint(checkpoint_path, device)
        if fov_overrides:
            fusion_module = model.fusion.fusion  # MultiScaleFusion -> ProductOfGaussians
            for k, v in fov_overrides.items():
                setattr(fusion_module, k, v)
            print(f"  FOV fusion overrides applied: {fov_overrides}")
        if residual_override is not None:
            model.global_residual = residual_override
            print(f"  global_residual override: {'enabled' if residual_override else 'disabled'}")
        return model, checkpoint

    chunk_kwargs = dict(
        num_samples=args.num_samples,
        roi_size=tuple(args.roi_size),
        overlap=args.overlap,
        sw_batch_size=args.sw_batch_size,
        sw_mode=args.sw_mode,
        amp=args.amp,
        amp_dtype=amp_dtype,
        cpu_output=not args.gpu_output,
        profile=not args.no_profile,
        profile_windows=args.profile_windows,
        output_scale=args.output_scale,
    )

    # ---- Folder mode ----
    if args.input_stacks_root:
        if args.orientation_mask is None:
            args.orientation_mask = [1, 1, 1]

        # predict_folder_chunked resolves load_mshved_from_checkpoint from this
        # module's globals; rebind it so folder mode also applies the overrides.
        globals()["load_mshved_from_checkpoint"] = load_with_overrides

        predict_folder_chunked(
            input_stacks_root=args.input_stacks_root,
            output_root=args.output_root,
            model_path=args.model,
            target_res=tuple(args.target_res),
            device=args.device,
            orientation_mask=args.orientation_mask,
            pattern_ax=args.pattern_ax,
            pattern_cor=args.pattern_cor,
            pattern_sag=args.pattern_sag,
            pattern_fov_ax=args.pattern_fov_ax,
            pattern_fov_cor=args.pattern_fov_cor,
            pattern_fov_sag=args.pattern_fov_sag,
            output_name=args.output_name,
            skip_existing=args.skip_existing,
            fail_fast=args.fail_fast,
            **chunk_kwargs,
        )
        raise SystemExit(0)

    # ---- Single case ----
    if args.input_stacks:
        if args.output is None:
            raise ValueError("--output is required in single-case mode.")
        stack_paths, fov_mask_paths, orientation_mask = _resolve_stack_and_mask_lists(
            args.input_stacks, args.fov_masks, args.orientation_mask
        )

        used = [n for n, p in zip(ORIENTATION_NAMES, stack_paths) if p is not None]
        print("=" * 80)
        print("MS-HVED Sliding-Window Inference")
        print("=" * 80)
        print(f"Using {len(used)}/3 orientations -> {', '.join(used)}  mask={orientation_mask}")

        model, _ = load_with_overrides(args.model, device=args.device)

        predict_single_volume_chunked(
            model=model,
            output_path=args.output,
            device=args.device,
            input_stack_paths=stack_paths,
            fov_mask_paths=fov_mask_paths,
            target_res=tuple(args.target_res),
            orientation_mask=orientation_mask,
            **chunk_kwargs,
        )
        print("\n" + "=" * 80)
        print("Inference complete!")
        print("=" * 80)
        raise SystemExit(0)

    parser.error("Provide either --input_stacks (single case) or --input_stacks_root (folder mode).")
