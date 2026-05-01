# MS-HVED++: Multi-Scale Hierarchical Variational Encoder-Decoder for Brain MRI Super-Resolution

A deep learning framework for isotropic super-resolution of brain MRI from multiple anisotropic orthogonal acquisitions. MS-HVED++ fuses axial, coronal, and sagittal low-resolution stacks via a Product-of-Gaussians (PoG) posterior in a multi-scale latent space to produce high-resolution isotropic outputs.

Building on the original MS-HVED, this version introduces a normalization-free architecture with spectral regularization, consistency training across degradation variations, FOV-aware fusion with obliqueness simulation.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- MONAI
- NiBabel
- NumPy
- (Optional) Weights & Biases for experiment tracking

Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

### Basic Training

```bash
python train.py \
  --hr_image_dir /path/to/hr/images \
  --val_image_dir /path/to/val/images \
  --model_dir ./models \
  --epochs 100 \
  --batch_size 2 \
  --learning_rate 1e-4
```

### Full-Featured Training

```bash
python train.py \
  --hr_image_dir /path/to/hr/images \
  --val_image_dir /path/to/val/images \
  --model_dir ./models \
  --epochs 200 \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --output_shape 128 128 128 \
  --recon_loss_type charbonnier \
  --recon_weight 0.4 \
  --kl_weight 0.1 \
  --use_perceptual --perceptual_weight 0.1 \
  --use_ssim --ssim_weight 0.1 \
  --orientation_weight 0.4 \
  --num_variations 2 \
  --consistency_weight 0.2 \
  --latent_consistency_weight 0.05 \
  --enable_obliqueness \
  --prob_obliqueness 0.5 \
  --obliqueness_range 15.0 \
  --mixed_precision fp16 \
  --use_wandb --wandb_project mshved
```

### Key Training Arguments

**Architecture:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--num_scales` | 4 | Number of hierarchical scales |
| `--init_filters` | 32 | Initial convolution filters |
| `--blocks_down` | 1 2 2 4 | Encoder blocks per scale |
| `--blocks_up` | 1 1 1 | Decoder blocks per scale |
| `--final_activation` | clamp | Output activation: clamp, sigmoid, tanh, none |
| `--decoder_upsample_mode` | trilinear | Upsample: trilinear or transpose |
| `--no_reconstruct_orientations` | | Disable auxiliary orientation reconstruction |
| `--no_global_residual` | | Disable global residual learning |

**Loss Weights:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--recon_loss_type` | charbonnier | Reconstruction loss: l1, l2, charbonnier |
| `--recon_weight` | 0.4 | Reconstruction loss weight |
| `--kl_weight` | 0.1 | KL divergence weight (linearly annealed) |
| `--perceptual_weight` | 0.1 | Perceptual loss weight (requires `--use_perceptual`) |
| `--ssim_weight` | 0.1 | SSIM loss weight (requires `--use_ssim`) |
| `--orientation_weight` | 0.4 | Orientation reconstruction weight |
| `--perceptual_network` | alex | Perceptual backbone: alex, vgg, radimagenet, medicalnet |

**Consistency Training:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--num_variations` | 1 | Degradation variations per image (>1 enables consistency) |
| `--consistency_weight` | 0.2 | Output consistency loss weight |
| `--latent_consistency_weight` | 0.05 | Latent consistency (symmetric KL) weight |
| `--consistency_warmup_steps` | 5000 | Linear warmup steps for consistency losses |

**MRI Artifact Simulation:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--prob_motion` | 0.5 | Motion ghosting probability |
| `--prob_spike` | 0.5 | K-space spike probability |
| `--prob_aliasing` | 0.02 | Aliasing probability |
| `--prob_bias_field` | 0.5 | B1 bias field probability |
| `--prob_noise` | 0.8 | Gaussian noise probability |
| `--fov_augmentation_prob` | 0.7 | FOV slice drop probability |
| `--no_intensity_aug` | | Disable gamma/clip augmentation |

**Obliqueness Simulation:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--enable_obliqueness` | off | Enable affine-based obliqueness simulation |
| `--prob_obliqueness` | 0.5 | Per-stack probability of oblique rotation |
| `--obliqueness_range` | 15.0 | Maximum rotation angle in degrees |

**Resolution:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--atlas_res` | 1.0 1.0 1.0 | HR voxel resolution in mm |
| `--min_resolution` | 1.0 1.0 1.0 | Minimum LR resolution |
| `--max_res_aniso` | 9.0 9.0 9.0 | Maximum through-plane resolution |
| `--no_randomise_res` | | Disable random resolution sampling |

## Orientation Dropout

MS-HVED++ can handle missing orientations at inference by training with orientation dropout.

### Random Dropout

```bash
# 50% chance to drop 1-2 orientations per sample, keep at least 1
python train.py \
  --model_dir ./models \
  --orientation_dropout_prob 0.5 \
  --min_orientations 1
```

### Deterministic Dropout

```bash
# Always drop axial (index 0), train with coronal + sagittal only
python train.py --model_dir ./models --drop_orientations 0

# Orientation indices: 0=Axial, 1=Coronal, 2=Sagittal
```

## Inference

### Standard Inference

```bash
python test.py \
  --input_stacks axial.nii.gz coronal.nii.gz sagittal.nii.gz \
  --model checkpoint.pth \
  --output sr_output.nii.gz
```

### FOV-Aware Inference

When obliqueness-trained models are used on real clinical data, FOV masks improve fusion by telling the model which voxels fall outside each stack's field of view:

```bash
python test.py \
  --input_stacks axial.nii.gz coronal.nii.gz sagittal.nii.gz \
  --fov_masks axial_fov_mask.nii.gz coronal_fov_mask.nii.gz sagittal_fov_mask.nii.gz \
  --model checkpoint.pth \
  --output sr_output.nii.gz
```

### Multi-Sample Inference

Draw multiple stochastic samples from the learned posterior for uncertainty estimation:

```bash
python test.py \
  --input_stacks axial.nii.gz coronal.nii.gz sagittal.nii.gz \
  --model checkpoint.pth \
  --output sr_output.nii.gz \
  --num_samples 5
# Produces: sr_output_sample1.nii.gz ... sr_output_sample5.nii.gz
```

### Missing Orientations

```bash
# Inference with axial + coronal only
python test.py \
  --input_stacks axial.nii.gz coronal.nii.gz \
  --orientation_mask 1 1 0 \
  --model checkpoint.pth \
  --output sr_output.nii.gz
```

### Batch Inference (Folder Mode)

```bash
python test_fov.py \
  --input_stacks_root /path/to/subjects \
  --model checkpoint.pth \
  --output_root /path/to/results \
  --num_samples 5
```

In folder mode, FOV masks are auto-discovered: for each stack `<stem>.nii.gz`, the script looks for `<stem>_fov_mask.nii.gz` in the same directory.

## Preparing Test Data

Resample input volumes into orthogonal low-resolution stacks:

```bash
python prepare4test.py \
  --input /path/to/input.nii.gz \
  --output_dir /path/to/output
```

This produces `axial_upsampled.nii.gz`, `coronal_upsampled.nii.gz`, `sagittal_upsampled.nii.gz`, and their corresponding `*_fov_mask.nii.gz` files.

### Standalone FOV Mask Generation

Generate FOV masks for existing clinical LR scans using their native NIfTI headers:

```bash
python generate_fov_masks.py \
  --lr_scans axial.nii.gz coronal.nii.gz sagittal.nii.gz \
  --hr_reference hr.nii.gz \
  --output_dir ./fov_masks
```

## Evaluation

```bash
python evaluate.py \
  --test_dir /path/to/test \
  --prediction_name sr_output.nii.gz \
  --output_csv results.csv
```

## Project Structure

```
ms-hved++/
├── train.py                 # Training script
├── test.py                  # Standard inference
├── evaluate.py              # Metric computation (PSNR, SSIM, MAE, etc.)
├── prepare4test.py          # Test data preprocessing
├── src/
│   ├── mshved.py            # MS-HVED++ model (encode → fuse → decode)
│   ├── encoder.py           # Multi-modal encoder with spectral normalization
│   ├── decoder.py           # SR decoder and multi-output orientation decoder
│   ├── blocks.py            # Spectral-normed residual blocks (no normalization)
│   ├── fusion.py            # Product-of-Gaussians fusion with FOV mask support
│   ├── losses.py            # Loss functions (recon, KL, perceptual, SSIM, consistency)
│   ├── data.py              # Data pipeline with MRI artifact simulation
│   └── utils.py             # Utilities (padding, metrics, etc.)
└── requirements.txt
```

## Acknowledgements

This work uses the [IXI dataset](https://brain-development.org/ixi-dataset/), which is available under a CC BY-SA 3.0 license.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mshved2025,
  title     = {},
  author    = {},
  journal   = {},
  year      = {2025}
}
```

## License

<!-- Add license information here -->
