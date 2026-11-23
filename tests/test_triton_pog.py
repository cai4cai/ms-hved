import importlib.util
import unittest

import torch

from src.fusion import ProductOfGaussians


class ProductOfGaussiansTest(unittest.TestCase):
    def test_eager_mean_only(self):
        mus = {k: torch.ones(1, 2, 2, 3, 4) * k for k in range(3)}
        logvars = {k: torch.zeros_like(mu) for k, mu in mus.items()}
        pog = ProductOfGaussians().eval()
        mean, logvar = pog(mus, logvars)
        mean_only, skipped = pog(mus, logvars, return_logvar=False)
        torch.testing.assert_close(mean, mean_only, rtol=0, atol=0)
        self.assertIsNotNone(logvar)
        self.assertIsNone(skipped)


@unittest.skipUnless(torch.cuda.is_available() and torch.version.cuda is not None
                     and importlib.util.find_spec("triton") is not None,
                     "requires NVIDIA CUDA and Triton")
class TritonProductOfGaussiansTest(unittest.TestCase):
    @torch.inference_mode()
    def test_matches_eager_with_optional_masks(self):
        torch.manual_seed(42)
        dtypes = [torch.float32, torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        for dtype in dtypes:
            for smooth in (False, True):
                with self.subTest(dtype=dtype, smooth=smooth):
                    # Means are channel slices, as in the encoder (batch stride has a gap).
                    packed = [torch.randn(2, 10, 5, 7, 9, device="cuda", dtype=dtype) for _ in range(3)]
                    mus = {k: t[:, :5] for k, t in enumerate(packed)}
                    logvars = {k: t[:, 5:].clamp(-5, 5) for k, t in enumerate(packed)}
                    masks = {k: (t[:, :1] > 0).float() for k, t in enumerate(packed)}
                    present = torch.tensor([[True, True, True], [True, False, True]], device="cuda")
                    for mask in masks.values():
                        mask[:, :, 0] = 1
                    stock = ProductOfGaussians(smooth_fov=smooth).eval()
                    accelerated = ProductOfGaussians(smooth_fov=smooth).eval()
                    accelerated.enable_triton()
                    kernel = accelerated._triton_fuse

                    def require_kernel(*args, **kwargs):
                        result = kernel(*args, **kwargs)
                        self.assertIsNotNone(result, "supported test inputs fell back to eager")
                        return result

                    accelerated._triton_fuse = require_kernel
                    if dtype == torch.float32:
                        tol = dict(rtol=1e-5, atol=2e-6)
                    elif dtype == torch.float16:
                        tol = dict(rtol=0.001, atol=5e-5)
                    else:
                        tol = dict(rtol=0.008, atol=2e-5)
                    for fov, orientation in ((None, None), (masks, present)):
                        mean, logvar = stock(mus, logvars, orientation, fov)
                        result = accelerated(mus, logvars, orientation, fov)
                        torch.testing.assert_close(result, (mean, logvar), **tol)
                        mean_only, skipped = accelerated(mus, logvars, orientation, fov, return_logvar=False)
                        torch.testing.assert_close(mean_only, mean, **tol)
                        self.assertIsNone(skipped)
                        if not smooth and fov is not None:
                            self.assertTrue(torch.all(mean_only[:, :, 0] == 0).item())

    @torch.inference_mode()
    def test_falls_back_for_single_encoded_orientation(self):
        mus = {2: torch.randn(1, 4, 5, 6, 7, device="cuda")}
        logvars = {2: torch.zeros_like(mus[2])}
        expected = ProductOfGaussians().eval()(mus, logvars)
        accelerated = ProductOfGaussians().eval()
        accelerated.enable_triton()
        torch.testing.assert_close(accelerated(mus, logvars), expected)


if __name__ == "__main__":
    unittest.main()
