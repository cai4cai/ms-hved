# [MS-HVED](https://arxiv_link): A Sequence Agnostic Multi-Scale Hierarchical Variational Encoder-Decoder for Brain MRI Super-Resolution

A deep learning framework for isotropic super-resolution of brain MRI from multiple anisotropic orthogonal acquisitions. MS-HVED fuses axial, coronal, and sagittal low-resolution stacks via a Product-of-Gaussians (PoG) posterior in a multi-scale latent space to produce high-resolution isotropic outputs.

## Architecture

<p align="center">
  <img src="assets/architecture.png" width="90%" />
</p>

## Qualitative Results

<p align="center">
  <img src="assets/qualitative_results.png" width="90%" />
</p>

## Pre-trained Model 

| Model | Training Data | Link |
|-------|--------------|------|
| MS-HVED | IXI T1 + IXI T2 + MEN| [Download](https://huggingface.co/marshallhamzah/ms-hved) |

## Try it out

[HF-Space](https://marshallhamzah-ms-hved.hf.space/)

## Study Platform
[Github repo](https://github.com/Marshall-mk/review-app)

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


## Orientation Dropout

MS-HVED can handle missing orientations at inference by training with orientation dropout.

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

### Preprocessing
```bash
# 3 stacks, no external reference — axial (index 0) used as registration target
python prepare4test.py -i axial.nii.gz coronal.nii.gz sagittal.nii.gz -o out/ -v

# Choose which input is the registration target
python prepare4test.py -i axial.nii.gz coronal.nii.gz sagittal.nii.gz -o out/ --fixed 0 -v

```

### Standard Inference

```bash
python test.py \
  --input_stacks axial.nii.gz coronal.nii.gz sagittal.nii.gz \
  --model checkpoint.pth \
  --output sr_output.nii.gz
```

### FOV-Aware Inference

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

## Acknowledgements

This work was supported by King's College London through the King's Doctoral College Africa Studentship. LG-FM acknowledges funding from the EPSRC Centre for Doctoral Training in Smart Medical Imaging  [grant number EP/S022104/1].

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mshved2026,
  title     = {},
  author    = {},
  journal   = {},
  year      = {2026}
}
```
