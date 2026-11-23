"""Optional CUDA implementation of deterministic Product-of-Gaussians fusion."""

import torch
import triton
import triton.language as tl


@triton.jit
def _contribution(mu_ptr, logvar_ptr, conf_ptr, present_ptr, b, i, v, valid,
                  slot: tl.constexpr, ms: tl.constexpr, ls: tl.constexpr,
                  fs: tl.constexpr, has_conf: tl.constexpr, has_present: tl.constexpr,
                  eps: tl.constexpr):
    mu = tl.load(mu_ptr + b * ms + i, valid, 0).to(tl.float32)
    lv = tl.load(logvar_ptr + b * ls + i, valid, 0).to(tl.float32)
    lv = tl.minimum(tl.maximum(lv, -10.0), 10.0)
    p = 1.0 / (tl.exp(lv) + eps)
    if has_conf:
        p *= tl.load(conf_ptr + b * fs + v, valid, 0).to(tl.float32)
    if has_present:
        p *= tl.load(present_ptr + b * 3 + slot, valid, 0).to(tl.float32)
    return p, p * mu


@triton.jit
def _fuse(mu0, mu1, mu2, lv0, lv1, lv2, f0, f1, f2, present, out, out_lv,
          n: tl.constexpr, sample_size: tl.constexpr, voxels: tl.constexpr,
          ms0: tl.constexpr, ms1: tl.constexpr, ms2: tl.constexpr,
          ls0: tl.constexpr, ls1: tl.constexpr, ls2: tl.constexpr,
          fs0: tl.constexpr, fs1: tl.constexpr, fs2: tl.constexpr,
          has_f0: tl.constexpr, has_f1: tl.constexpr, has_f2: tl.constexpr,
          has_present: tl.constexpr, prior_precision: tl.constexpr,
          prior_weighted_mean: tl.constexpr, eps: tl.constexpr,
          return_logvar: tl.constexpr, block: tl.constexpr):
    x = tl.program_id(0) * block + tl.arange(0, block)
    b, i, v = x // sample_size, x % sample_size, x % voxels
    valid = x < n
    p0, w0 = _contribution(mu0, lv0, f0, present, b, i, v, valid, 0, ms0, ls0, fs0, has_f0, has_present, eps)
    p1, w1 = _contribution(mu1, lv1, f1, present, b, i, v, valid, 1, ms1, ls1, fs1, has_f1, has_present, eps)
    p2, w2 = _contribution(mu2, lv2, f2, present, b, i, v, valid, 2, ms2, ls2, fs2, has_f2, has_present, eps)
    var = 1.0 / (((p0 + p1) + p2) + prior_precision + eps)
    mean = (((w0 + w1) + w2) + prior_weighted_mean) * var
    tl.store(out + x, tl.minimum(tl.maximum(mean, -10.0), 10.0), valid)
    if return_logvar:
        logvar = tl.log(var + eps)
        tl.store(out_lv + x, tl.minimum(tl.maximum(logvar, -10.0), 10.0), valid)


def fuse_gaussians(
    mus, logvars, orientation_mask, fov_masks, confidence_fn, *,
    prior_precision, prior_weighted_mean, eps, return_logvar,
):
    """Return None for unsupported inputs so the caller can use eager PyTorch."""
    if tuple(mus) != (0, 1, 2) or tuple(logvars) != (0, 1, 2):
        return None
    mu = tuple(mus.values())
    lv = tuple(logvars.values())
    ref = mu[0]
    # The kernel uses 32-bit element offsets.
    if (not ref.is_cuda or ref.ndim != 5 or not 0 < ref.numel() < 2**31
            or ref.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        return None
    if any(t.shape != ref.shape or t.device != ref.device or t.dtype != ref.dtype
           or not t[0].is_contiguous()
           or (ref.shape[0] - 1) * t.stride(0) + ref[0].numel() >= 2**31
           for t in (*mu, *lv)):
        return None
    if orientation_mask is not None and (
        orientation_mask.shape != (ref.shape[0], 3) or orientation_mask.device != ref.device
        or not orientation_mask.is_contiguous()
    ):
        return None
    fov = tuple(fov_masks.get(k) if fov_masks is not None else None for k in range(3))
    if any(m is not None and (m.shape != (ref.shape[0], 1, *ref.shape[2:]) or m.device != ref.device)
           for m in fov):
        return None
    conf = tuple(confidence_fn(m) if m is not None else ref for m in fov)
    if any(not t[0].is_contiguous() for t in conf):
        return None

    mean = torch.empty(ref.shape, dtype=ref.dtype, device=ref.device)
    out_logvar = torch.empty_like(mean) if return_logvar else mean
    block = 256
    _fuse[(triton.cdiv(mean.numel(), block),)](
        *mu, *lv, *conf, orientation_mask if orientation_mask is not None else ref,
        mean, out_logvar, mean.numel(), ref[0].numel(), ref[0, 0].numel(),
        *(t.stride(0) for t in mu), *(t.stride(0) for t in lv), *(t.stride(0) for t in conf),
        *(m is not None for m in fov), orientation_mask is not None,
        prior_precision, prior_weighted_mean, eps, return_logvar, block,
        enable_fp_fusion=False,
    )
    return mean, out_logvar if return_logvar else None
