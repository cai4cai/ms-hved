"""
Training script for MS-HVED Super-Resolution with Orthogonal Stacks

This script trains the MS-HVED model using three orthogonal low-resolution stacks
generated from high-resolution volumes. Each stack has high resolution in one
orientation (axial, coronal, sagittal).

"""

import os
import argparse
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
from typing import List
import wandb

# Accelerate for easy multi-GPU training
from accelerate import Accelerator
from accelerate.utils import set_seed

# Transformers imports for scheduler
from transformers import get_cosine_schedule_with_warmup

# MONAI imports
from monai.data import DataLoader

# Import our modules
from src import MSHVEDLoss, create_mshved
from src.data import HRLRDataGenerator, create_dataset
from src.utils import (
    save_model_checkpoint,
    get_image_paths,
    save_training_config,
    find_latest_checkpoint,
    calculate_metrics,
)


def build_model_config(
    num_orientations: int,
    num_scales: int,
    output_shape: tuple,
    reconstruct_orientations: bool,
    decoder_upsample_mode: str,
    final_activation: str,
    init_filters: int = None,
    blocks_down: tuple = None,
    blocks_up: tuple = None,
    global_residual: bool = False,
    smooth_fov: bool = True,
    fov_attenuation: float = 1.0,
    fov_transition_width: float = 3.0,
    use_kaiming_init: bool = False,
) -> dict:
    """Build model configuration dictionary for checkpoint saving."""
    config = {
        "num_orientations": num_orientations,
        "num_scales": num_scales,
        "output_shape": output_shape,
        "reconstruct_orientations": reconstruct_orientations,
        "decoder_upsample_mode": decoder_upsample_mode,
        "final_activation": final_activation,
        "init_filters": init_filters,
        "blocks_down": list(blocks_down),
        "blocks_up": list(blocks_up),
        "global_residual": global_residual,
        "smooth_fov": smooth_fov,
        "fov_attenuation": fov_attenuation,
        "fov_transition_width": fov_transition_width,
        "use_kaiming_init": use_kaiming_init,
    }

    return config


def train_mshved_model(
    # Data sources and outputs
    hr_image_paths: List[str],
    model_dir: str,
    val_image_paths: List[str] = None,
    checkpoint: str = None,

    # Training schedule and runtime
    epochs: int = 100,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    save_interval: int = 10,
    val_interval: int = 1,
    mixed_precision: str = "no",
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    seed: int = 42,

    # Data generation and augmentation
    output_shape: tuple = (128, 128, 128),
    atlas_res: list = [1.0, 1.0, 1.0],
    min_resolution: list = [1.0, 1.0, 1.0],
    max_res_aniso: list = [9.0, 9.0, 9.0],
    randomise_res: bool = True,
    prob_motion: float = 0.2,
    prob_spike: float = 0.05,
    prob_aliasing: float = 0.1,
    prob_bias_field: float = 0.5,
    prob_noise: float = 0.8,
    fov_augmentation_prob: float = 0.7,
    apply_intensity_aug: bool = True,
    orientation_dropout_prob: float = 0.0,
    min_orientations: int = 1,
    drop_orientations: list = None,
    balanced_orientation_combos: bool = False,
    num_variations: int = 1,
    upsample_mode: str = "trilinear",
    enable_obliqueness: bool = False,
    prob_obliqueness: float = 0.5,
    obliqueness_range: float = 15.0,

    # Data loading
    num_workers: int = None,
    use_cache: bool = False,

    # Model architecture
    num_scales: int = 4,
    init_filters: int = 32,
    blocks_down: list = None,
    blocks_up: list = None,
    reconstruct_orientations: bool = True,
    final_activation: str = "clamp",
    decoder_upsample_mode: str = "trilinear",
    global_residual: bool = False,
    smooth_fov: bool = True,
    fov_attenuation: float = 1.0,
    fov_transition_width: float = 3.0,
    use_kaiming_init: bool = False,

    # Loss configuration
    recon_loss_type: str = "charbonnier",
    recon_weight: float = 0.4,
    kl_weight: float = 0.1,
    perceptual_weight: float = 0.1,
    ssim_weight: float = 0.1,
    orientation_weight: float = 0.4,
    use_perceptual: bool = False,
    use_ssim: bool = True,
    perceptual_network: str = 'alex',
    is_fake_3d: bool = False,

    # Consistency loss
    consistency_weight: float = 0.2,
    latent_consistency_weight: float = 0.05,
    consistency_warmup_steps: int = 5000,

    # Experiment tracking
    use_wandb: bool = False,
    wandb_project: str = "mshved",
    wandb_entity: str = None,
    wandb_run_name: str = None,
):
    """
    Train MS-HVED model with orthogonal LR stacks

    Args:
        Data sources and outputs:
        hr_image_paths: List of paths to high-resolution images
        model_dir: Directory to save trained models
        val_image_paths: Optional list of validation image paths
        checkpoint: Optional checkpoint to resume from

        Training schedule and runtime:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: 'cuda' or 'cpu'
        save_interval: Save checkpoint every N epochs
        val_interval: Run validation every N epochs
        mixed_precision: Mixed precision training ('no', 'fp16', 'bf16')
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping (0 to disable)
        seed: Random seed for reproducibility

        Data generation and augmentation:
        output_shape: Output volume shape
        atlas_res: Physical resolution of input HR images [x, y, z] in mm
        min_resolution: Minimum resolution for randomization
        max_res_aniso: Maximum anisotropic resolution
        randomise_res: Whether to randomize resolution
        prob_motion: Probability of motion artifacts
        prob_spike: Probability of k-space spikes
        prob_aliasing: Probability of aliasing artifacts
        prob_bias_field: Probability of bias field
        prob_noise: Probability of noise
        fov_augmentation_prob: Probability of FOV augmentation
        apply_intensity_aug: Whether to apply intensity augmentation
        orientation_dropout_prob: Probability of applying orientation dropout (0.0-1.0)
        min_orientations: Minimum number of orientations to keep after dropout (1-3)
        drop_orientations: Specific orientations to drop (0=Axial, 1=Coronal, 2=Sagittal). If specified, always drops these.
        balanced_orientation_combos: Whether to use balanced orientation mask combos per epoch. If True, will cycle through a balanced set of orientation dropout combinations each epoch to ensure all views are learned.
        num_variations: Number of variations to generate per sample for consistency training (default=1, set >1 to enable)
        upsample_mode: Upsampling strategy for data generator ('trilinear' or 'nearest')
        enable_obliqueness: Whether to enable obliqueness augmentation
        prob_obliqueness: Probability of applying obliqueness augmentation
        obliqueness_range: Maximum angle in degrees for obliqueness augmentation

        Data loading:
        num_workers: Number of data loading workers
        use_cache: Whether to use CacheDataset

        Model architecture:
        num_scales: Number of hierarchical scales
        init_filters: Number of filters in the first scale of the model (doubled at each subsequent scale)
        blocks_down: List of number of blocks at each downsampling scale (length should match num_scales)
        blocks_up: List of number of blocks at each upsampling scale (length should match num_scales)
        reconstruct_orientations: Whether to include auxiliary decoders for reconstructing input orientations
        final_activation: Final activation function ('tanh', 'sigmoid', or 'none')
        decoder_upsample_mode: Decoder upsampling strategy ('trilinear', 'transpose', or 'pixelshuffle')
        global_residual: Whether to use a global residual connection from input to output
        smooth_fov: Whether to apply smooth FOV attenuation
        fov_attenuation: Strength of FOV attenuation (higher = stronger attenuation)
        fov_transition_width: Width of transition zone for smooth FOV attenuation in mm
        use_kaiming_init: Whether to use Kaiming initialization for model weights

        Loss configuration:
        recon_loss_type: Type of reconstruction loss ('l1', 'l2', or 'charbonnier')
        recon_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence loss
        perceptual_weight: Weight for perceptual loss
        ssim_weight: Weight for SSIM loss
        orientation_weight: Weight for orientation reconstruction loss
        use_perceptual: Whether to use perceptual loss
        use_ssim: Whether to use SSIM loss
        perceptual_network: MONAI network for perceptual loss ('alex', 'vgg', 'squeeze', 'radimagenet', 'medicalnet', 'resnet50')
        is_fake_3d: Use 2.5D (fake 3D) mode for perceptual loss (False = full 3D, True = 2.5D slices)

        Consistency loss:
        consistency_weight: Weight for output consistency loss between variations
        latent_consistency_weight: Weight for latent consistency loss between variations
        consistency_warmup_steps: Number of steps to warm up consistency losses (start at 0 weight and linearly increase)

        Experiment tracking:
        use_wandb: Whether to use Weights & Biases for tracking
        wandb_project: W&B project name
        wandb_entity: W&B entity/team name
        wandb_run_name: W&B run name
    """
    # Initialize Accelerator for multi-GPU training
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb" if use_wandb else None,
    )

    # Set seed for reproducibility
    set_seed(seed)

    # Only print from main process
    if accelerator.is_main_process:
        print("=" * 80)
        print("Training MS-HVED with Orthogonal LR Stacks")
        print("=" * 80)
        print(f"Distributed training: {accelerator.num_processes} process(es)")
        print(f"Mixed precision: {mixed_precision}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    # Create model directory
    os.makedirs(model_dir, exist_ok=True)

    # Auto-detect optimal num_workers if not provided
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        if device == "cuda" and torch.cuda.is_available():
            num_workers = min(4, max(cpu_count // 2, 1))
        elif cpu_count >= 4:
            num_workers = 2
        else:
            num_workers = 0

    # Enable pin_memory for faster GPU transfer
    pin_memory = False  # Accelerate handles this

    if accelerator.is_main_process:
        print(
            f"DataLoader settings: num_workers={num_workers}, pin_memory={pin_memory}, use_cache={use_cache}"
        )

    # Create data generator for orthogonal stacks
    if accelerator.is_main_process:
        print(f"Input HR image resolution: {atlas_res} mm")
        print(f"Resolution randomization: {min_resolution} to {max_res_aniso} mm")
        print(f"Generating 3 orthogonal LR stacks per HR volume (RAS-oriented)")
        print(f"  - Stack 0 (Axial): High in-plane (R,A), Low through-plane (S)")
        print(f"  - Stack 1 (Coronal): High in-plane (R,S), Low through-plane (A)")
        print(f"  - Stack 2 (Sagittal): High in-plane (A,S), Low through-plane (R)")
        if orientation_dropout_prob > 0.0:
            print(f"orientation dropout enabled: {orientation_dropout_prob:.2f} probability, min {min_orientations} orientations")
            print(f"  → Training will randomly drop views to simulate missing data")
        if balanced_orientation_combos:
            print("Balanced orientation combos enabled per epoch")

    generator = HRLRDataGenerator(
        atlas_res=atlas_res,
        target_res=[1.0, 1.0, 1.0],
        output_shape=list(output_shape),
        min_resolution=min_resolution,
        max_res_aniso=max_res_aniso,
        randomise_res=randomise_res,
        prob_motion=prob_motion,
        prob_spike=prob_spike,
        prob_aliasing=prob_aliasing,
        prob_bias_field=prob_bias_field,
        prob_noise=prob_noise,
        fov_augmentation_prob=fov_augmentation_prob,
        apply_intensity_aug=apply_intensity_aug,
        clip_to_unit_range=True,
        orientation_dropout_prob=orientation_dropout_prob,
        min_orientations=min_orientations,
        drop_orientations=drop_orientations,
        upsample_mode=upsample_mode,
        return_intermediate=False,
        enable_obliqueness=enable_obliqueness,
        prob_obliqueness=prob_obliqueness,
        obliqueness_range=obliqueness_range,
    )

    # Create dataset
    dataset = create_dataset(
        image_paths=hr_image_paths,
        generator=generator,
        target_shape=list(output_shape),
        target_spacing=atlas_res,
        use_cache=use_cache,
        return_resolution=True,
        is_training=True,
        balanced_orientation_combos=balanced_orientation_combos,
        num_variations=num_variations,
    )

    # Create DataLoader
    from src.data import multi_variation_collate_fn
    dataloader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    if num_variations > 1:
        dataloader_kwargs['collate_fn'] = multi_variation_collate_fn
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    if accelerator.is_main_process:
        print(f"Training dataset: {len(dataset)} images")

    # Create validation dataset if provided
    val_dataloader = None
    if val_image_paths:
        val_dataset = create_dataset(
            image_paths=val_image_paths,
            generator=generator,
            target_shape=list(output_shape),
            target_spacing=atlas_res,
            use_cache=use_cache,
            return_resolution=True,
            is_training=False,
            balanced_orientation_combos=False,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
        if accelerator.is_main_process:
            print(f"Validation dataset: {len(val_dataset)} images")

    # Auto-detect checkpoint
    start_epoch = 0
    checkpoint_data = None

    if checkpoint is None:
        checkpoint = find_latest_checkpoint(model_dir)
        if checkpoint and accelerator.is_main_process:
            print(f"Auto-detected checkpoint: {checkpoint}")

    # Load checkpoint if available
    if checkpoint and os.path.exists(checkpoint):
        if accelerator.is_main_process:
            print(f"Loading checkpoint from {checkpoint}")
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        start_epoch = checkpoint_data.get("epoch", 0) + 1
        if accelerator.is_main_process:
            print(f"Resuming training from epoch {start_epoch}")

    # Create MS-HVED model
    if accelerator.is_main_process:
        print(f"Creating MS-HVED model:")
        print(f"  - Number of orientations: 3 (orthogonal stacks)")
        print(f"  - Number of scales: {num_scales}")
        print(f"  - Final activation: {final_activation}")
        print(f"  - Init filters: {init_filters}")
        print(f"  - Blocks down: {blocks_down}")
        print(f"  - Blocks up: {blocks_up}")

    # Other parameters for architectures
    other_params = {
        'num_orientations': 3,
        'in_channels': 1,
        'out_channels': 1,
        'num_scales': num_scales,
        'share_encoder': False,
        'share_decoder': False,
        'use_prior': True,
        'reconstruct_orientations': reconstruct_orientations,
        'final_activation': final_activation,
        'upsample_mode': decoder_upsample_mode,
        'global_residual': global_residual,
        'smooth_fov': smooth_fov,
        'fov_attenuation': fov_attenuation,
        'fov_transition_width': fov_transition_width,
        'use_kaiming_init': use_kaiming_init,
    }

    model = create_mshved(
        config='default',
        init_filters=init_filters,
        blocks_down=tuple(blocks_down),
        blocks_up=tuple(blocks_up),
        **other_params
    )

    # Load checkpoint weights if available
    if checkpoint_data is not None:
        model.load_state_dict(checkpoint_data["model_state_dict"])
        if accelerator.is_main_process:
            print(f"✓ Loaded model weights from checkpoint")

    # Optimizer and loss - Add weight decay to prevent unbounded weight growth
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    criterion = MSHVEDLoss(
        recon_loss_type=recon_loss_type,
        recon_weight=recon_weight,
        kl_weight=kl_weight,
        perceptual_weight=perceptual_weight,
        ssim_weight=ssim_weight,
        orientation_weight=orientation_weight,
        use_perceptual=use_perceptual,
        use_ssim=use_ssim,
        perceptual_network=perceptual_network,
        is_fake_3d=is_fake_3d,
        consistency_weight=consistency_weight if num_variations > 1 else 0.0,
        latent_consistency_weight=latent_consistency_weight if num_variations > 1 else 0.0,
        consistency_warmup_steps=consistency_warmup_steps,
    )

    if accelerator.is_main_process:
        cons_str = f", consistency={consistency_weight}, latent_consistency={latent_consistency_weight}" if num_variations > 1 else ""
        print(f"Loss weights: recon={recon_weight}, kl={kl_weight}, perceptual={perceptual_weight}, orientation={orientation_weight}{cons_str}")
        if num_variations > 1:
            print(f"Multi-variation consistency training: {num_variations} variations per sample, warmup={consistency_warmup_steps} steps")

    # Calculate total training steps for scheduler
    num_steps = len(dataloader) * epochs
    warmup_steps = int(0.05 * num_steps)  # 5% warmup

    # Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_steps,
    )

    if accelerator.is_main_process:
        print(f"Using Cosine LR schedule with warmup ({warmup_steps} warmup steps, {num_steps} total steps)")
        if max_grad_norm > 0:
            print(f"Gradient clipping enabled: max_norm={max_grad_norm}")
        else:
            print(f"Gradient clipping disabled")

    # Load optimizer and scheduler state if resuming
    if checkpoint_data is not None:
        if "optimizer_state_dict" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

    # Prepare everything with accelerator
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    # Track best validation loss for saving best model
    best_val_loss = float('inf')
    if checkpoint_data is not None and 'val_loss' in checkpoint_data and checkpoint_data['val_loss'] is not None:
        best_val_loss = checkpoint_data['val_loss']
        if accelerator.is_main_process:
            print(f"Best validation loss from checkpoint: {best_val_loss:.4f}")

    # Initialize Weights & Biases if enabled
    if use_wandb and accelerator.is_main_process:
        wandb_config = {
            "num_orientations": 3,
            "num_scales": num_scales,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "output_shape": output_shape,
            "kl_weight": kl_weight,
            "perceptual_weight": perceptual_weight,
            "orientation_weight": orientation_weight,
            "balanced_orientation_combos": balanced_orientation_combos,
            "reconstruct_orientations": reconstruct_orientations,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "n_train_samples": len(hr_image_paths),
            "n_val_samples": len(val_image_paths) if val_image_paths else 0,
            "init_filters": init_filters,
            "blocks_down": blocks_down,
            "blocks_up": blocks_up,
            "num_variations": num_variations,
            "consistency_weight": consistency_weight if num_variations > 1 else 0.0,
            "latent_consistency_weight": latent_consistency_weight if num_variations > 1 else 0.0,
            "consistency_warmup_steps": consistency_warmup_steps,
        }

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=wandb_config,
            resume="allow" if checkpoint else False,
        )
        wandb.watch(model, criterion, log="all", log_freq=100)
        print(f"Weights & Biases initialized: {wandb.run.name}")

    # Setup CSV logging
    csv_file = None
    csv_writer = None
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"training_log_{timestamp}.csv"
        csv_path = os.path.join(model_dir, csv_filename)

        csv_headers = ["epoch", "train_loss", "train_recon", "train_kl", "train_ssim", "train_perceptual", "train_orientation",
                        "train_output_consistency", "train_latent_consistency",
                        "learning_rate", "epoch_time"]
        if val_dataloader:
            csv_headers.extend([
                "val_loss", "val_recon", "val_kl", "val_ssim_loss", "val_perceptual", "val_orientation",
                "val_mae", "val_mse", "val_rmse", "val_psnr", "val_r2", "val_ssim", "validation_time"
            ])

        csv_file = open(csv_path, mode='a', newline='')
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
        csv_writer.writeheader()
        csv_file.flush()
        print(f"Logging training metrics to: {csv_path}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        if hasattr(dataloader.dataset, "set_epoch"):
            dataloader.dataset.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_ssim_loss = 0.0
        epoch_perceptual_loss = 0.0
        epoch_orientation_loss = 0.0
        epoch_output_consistency_loss = 0.0
        epoch_latent_consistency_loss = 0.0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            disable=not accelerator.is_main_process,
            leave=False,  
            dynamic_ncols=True,  
        )

        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch depending on mode
            if num_variations > 1:
                # Multi-variation mode: batch_data = (variations_data, hr_target)
                variations_data, target_img = batch_data
                target_img = target_img.float()

                # Variation 1 (with gradients): use variation index based on batch_idx parity
                # to symmetrize gradient signal across variations
                v1_idx = batch_idx % num_variations
                v2_idx = (batch_idx + 1) % num_variations

                v1_lr, v1_res, v1_thick, v1_mask, v1_fov = variations_data[v1_idx]
                v1_orientations = [m.float() for m in v1_lr]

                with accelerator.accumulate(model):
                    # Forward variation 1 (with gradients)
                    outputs_v1 = model(v1_orientations, orientation_mask=v1_mask, fov_masks=v1_fov)

                    # Forward variation 2 (detached — saves memory)
                    v2_lr, v2_res, v2_thick, v2_mask, v2_fov = variations_data[v2_idx]
                    v2_orientations = [m.float() for m in v2_lr]
                    with torch.no_grad():
                        outputs_v2 = model(v2_orientations, orientation_mask=v2_mask, fov_masks=v2_fov)

                    # Detach posteriors from variation 2
                    posteriors_v2_detached = [
                        (mu.detach(), lv.detach()) for mu, lv in outputs_v2['posteriors']
                    ]

                    # Compute loss: standard losses on v1 + consistency between v1 and v2
                    losses = criterion(
                        sr_output=outputs_v1['sr_output'],
                        sr_target=target_img,
                        posteriors=outputs_v1['posteriors'],
                        orientation_outputs=outputs_v1.get('orientation_outputs'),
                        orientation_targets=v1_orientations,
                        sr_outputs_other=[outputs_v2['sr_output'].detach()],
                        posteriors_other=[posteriors_v2_detached],
                        return_components=True
                    )

                    loss = losses['total']

                    # Backward pass
                    optimizer.zero_grad()
                    accelerator.backward(loss)

                    if max_grad_norm > 0:
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                    optimizer.step()
                    scheduler.step()

            else:
                # Single-variation mode (original behavior)
                lr_stacks_list, target_img, resolutions_list, thicknesses_list, orientation_mask, fov_masks = batch_data
                orientations = lr_stacks_list

                orientations = [m.float() for m in orientations]
                target_img = target_img.float()

                with accelerator.accumulate(model):
                    outputs = model(orientations, orientation_mask=orientation_mask, fov_masks=fov_masks)

                    losses = criterion(
                        sr_output=outputs['sr_output'],
                        sr_target=target_img,
                        posteriors=outputs['posteriors'],
                        orientation_outputs=outputs.get('orientation_outputs'),
                        orientation_targets=orientations,
                        return_components=True
                    )

                    loss = losses['total']

                    optimizer.zero_grad()
                    accelerator.backward(loss)

                    if max_grad_norm > 0:
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                    optimizer.step()
                    scheduler.step()

            epoch_loss += loss.item()

            epoch_recon_loss += losses['reconstruction'].item()
            epoch_kl_loss += losses['kl'].item()
            epoch_ssim_loss += losses['ssim'].item() if 'ssim' in losses else 0.0
            epoch_perceptual_loss += losses['perceptual'].item() if 'perceptual' in losses else 0.0
            epoch_orientation_loss += losses['orientation'].item() if 'orientation' in losses else 0.0
            epoch_output_consistency_loss += losses.get('output_consistency', torch.tensor(0.0)).item()
            epoch_latent_consistency_loss += losses.get('latent_consistency', torch.tensor(0.0)).item()

            # Update progress bar with loss and memory
            current_lr = optimizer.param_groups[0]['lr']
            postfix_dict = {
                "loss": f"{loss.item():.4f}",
                "recon": f"{losses['reconstruction'].item():.4f}",
                "kl": f"{losses['kl'].item():.4f}",
                "ssim": f"{losses['ssim'].item():.4f}" if 'ssim' in losses else "N/A",
                "perceptual": f"{losses['perceptual'].item():.4f}" if 'perceptual' in losses else "N/A",
                "orientation": f"{losses['orientation'].item():.4f}" if 'orientation' in losses else "N/A",
                "lr": f"{current_lr:.2e}"
            }
            if num_variations > 1:
                postfix_dict["o_cons"] = f"{losses.get('output_consistency', torch.tensor(0.0)).item():.4f}"
                postfix_dict["l_cons"] = f"{losses.get('latent_consistency', torch.tensor(0.0)).item():.4f}"

            pbar.set_postfix(postfix_dict)


        avg_loss = epoch_loss / len(dataloader)
        avg_recon = epoch_recon_loss / len(dataloader)
        avg_kl = epoch_kl_loss / len(dataloader)
        avg_ssim = epoch_ssim_loss / len(dataloader) if epoch_ssim_loss > 0 else 0.0
        avg_perceptual = epoch_perceptual_loss / len(dataloader) if epoch_perceptual_loss > 0 else 0.0
        avg_orientation = epoch_orientation_loss / len(dataloader) if epoch_orientation_loss > 0 else 0.0
        avg_output_consistency = epoch_output_consistency_loss / len(dataloader) if epoch_output_consistency_loss > 0 else 0.0
        avg_latent_consistency = epoch_latent_consistency_loss / len(dataloader) if epoch_latent_consistency_loss > 0 else 0.0

        # Validation
        val_loss = None
        val_metrics = None
        validation_time = 0.0

        if val_dataloader and (epoch + 1) % val_interval == 0:
            val_start_time = time.time()
            model.eval()
            val_epoch_loss = 0.0
            metrics_sum = {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "psnr": 0.0, "r2": 0.0, "ssim": 0.0}
            num_val_batches = 0
            val_recon_losses = 0.0
            val_kl_losses = 0.0
            val_ssim_losses = 0.0
            val_perceptual_losses = 0.0
            val_orientation_losses = 0.0

            with torch.no_grad():
                for val_batch_data in val_dataloader:
                    lr_stacks_list, target_img, _, _, orientation_mask, fov_masks = val_batch_data
                    orientations = [m.float() for m in lr_stacks_list]
                    target_img = target_img.float()

                    # Pass orientation_mask and fov_masks to the model's fusion mechanism
                    outputs = model(orientations, orientation_mask=orientation_mask, fov_masks=fov_masks)

                    losses = criterion(
                        sr_output=outputs['sr_output'],
                        sr_target=target_img,
                        posteriors=outputs['posteriors'],
                        orientation_outputs=outputs.get('orientation_outputs'),
                        orientation_targets=orientations,
                        return_components=True
                    )

                    val_epoch_loss += losses['total'].item()
                    val_recon_losses += losses['reconstruction'].item()
                    val_kl_losses += losses['kl'].item()
                    val_ssim_losses += losses['ssim'].item()
                    val_perceptual_losses += losses['perceptual'].item()
                    val_orientation_losses += losses['orientation'].item()

                    # Calculate metrics
                    batch_metrics = calculate_metrics(outputs['sr_output'], target_img, max_val=1.0)
                    for key in metrics_sum:
                        metrics_sum[key] += batch_metrics[key]
                    num_val_batches += 1

            val_loss = val_epoch_loss / num_val_batches
            val_metrics = {k: v / num_val_batches for k, v in metrics_sum.items()}
            val_loss_components = {
                'val_recon': val_recon_losses / num_val_batches,
                'val_kl': val_kl_losses / num_val_batches,
                'val_ssim_loss': val_ssim_losses / num_val_batches,
                'val_perceptual': val_perceptual_losses / num_val_batches,
                'val_orientation': val_orientation_losses / num_val_batches,
            }
            validation_time = time.time() - val_start_time

            epoch_time = time.time() - epoch_start_time
            # Print validation results 
            cons_str = f", OutCons: {avg_output_consistency:.4f}, LatCons: {avg_latent_consistency:.4f}" if num_variations > 1 else ""
            accelerator.print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f} "
                f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.6f}, SSIM: {avg_ssim:.4f}, Percep: {avg_perceptual:.4f}, Orient: {avg_orientation:.4f}{cons_str})"
            )
            accelerator.print(
                f"  Val Loss: {val_loss:.4f} "
                f"(Recon: {val_loss_components['val_recon']:.4f}, KL: {val_loss_components['val_kl']:.6f}, "
                f"SSIM: {val_loss_components['val_ssim_loss']:.4f}, Percep: {val_loss_components['val_perceptual']:.4f}, "
                f"Orient: {val_loss_components['val_orientation']:.4f})"
            )
            accelerator.print(
                f"  Val Metrics - MAE: {val_metrics['mae']:.4f} | RMSE: {val_metrics['rmse']:.4f} | "
                f"PSNR: {val_metrics['psnr']:.2f} dB | SSIM: {val_metrics['ssim']:.4f} | "
                f"R²: {val_metrics['r2']:.4f} | LR: {current_lr:.2e}"
            )
        else:
            epoch_time = time.time() - epoch_start_time
            # Print training summary 
            cons_str = f", OutCons: {avg_output_consistency:.4f}, LatCons: {avg_latent_consistency:.4f}" if num_variations > 1 else ""
            accelerator.print(
                f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} "
                f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.6f}, SSIM: {avg_ssim:.4f}, Percep: {avg_perceptual:.4f}, orientation: {avg_orientation:.4f}{cons_str}) - LR: {current_lr:.2e}"
            )

        # Log to CSV
        if accelerator.is_main_process and csv_writer is not None:
            log_data = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_recon": avg_recon,
                "train_kl": avg_kl,
                "train_ssim": avg_ssim,
                "train_perceptual": avg_perceptual,
                "train_orientation": avg_orientation,
                "train_output_consistency": avg_output_consistency,
                "train_latent_consistency": avg_latent_consistency,
                "learning_rate": current_lr,
                "epoch_time": epoch_time,
            }
            if val_loss is not None and val_metrics is not None:
                log_data.update({
                    "val_loss": val_loss,
                    "val_recon": val_loss_components['val_recon'],
                    "val_kl": val_loss_components['val_kl'],
                    "val_ssim_loss": val_loss_components['val_ssim_loss'],
                    "val_perceptual": val_loss_components['val_perceptual'],
                    "val_orientation": val_loss_components['val_orientation'],
                    "val_mae": val_metrics['mae'],
                    "val_mse": val_metrics['mse'],
                    "val_rmse": val_metrics['rmse'],
                    "val_psnr": val_metrics['psnr'],
                    "val_r2": val_metrics['r2'],
                    "val_ssim": val_metrics['ssim'],
                    "validation_time": validation_time,
                })
            csv_writer.writerow(log_data)
            csv_file.flush()

        # Log to W&B
        if use_wandb and accelerator.is_main_process:
            wandb_log_data = {
                "epoch": epoch + 1,
                "train/loss": avg_loss,
                "train/reconstruction": avg_recon,
                "train/kl": avg_kl,
                "train/ssim": avg_ssim,
                "train/perceptual": avg_perceptual,
                "train/orientation": avg_orientation,
                "train/output_consistency": avg_output_consistency,
                "train/latent_consistency": avg_latent_consistency,
                "train/learning_rate": current_lr,
            }
            if val_loss is not None and val_metrics is not None:
                wandb_log_data.update({
                    "val/loss": val_loss,
                    "val/recon": val_loss_components['val_recon'],
                    "val/kl": val_loss_components['val_kl'],
                    "val/ssim_loss": val_loss_components['val_ssim_loss'],
                    "val/perceptual": val_loss_components['val_perceptual'],
                    "val/orientation": val_loss_components['val_orientation'],
                    "val/mae": val_metrics['mae'],
                    "val/mse": val_metrics['mse'],
                    "val/rmse": val_metrics['rmse'],
                    "val/psnr": val_metrics['psnr'],
                    "val/r2": val_metrics['r2'],
                    "val/ssim": val_metrics['ssim'],
                })
            wandb.log(wandb_log_data)

        # Save best model if validation loss improved
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                best_model_path = os.path.join(model_dir, "mshved_orthogonal_best.pth")

                model_config = build_model_config(
                    num_orientations=3,
                    num_scales=num_scales,
                    output_shape=output_shape,
                    reconstruct_orientations=reconstruct_orientations,
                    decoder_upsample_mode=decoder_upsample_mode,
                    final_activation=final_activation,
                    init_filters=init_filters,
                    blocks_down=tuple(blocks_down) if blocks_down else None,
                    blocks_up=tuple(blocks_up) if blocks_up else None,
                    global_residual=global_residual,
                    smooth_fov=smooth_fov,
                    fov_attenuation=fov_attenuation,
                    fov_transition_width=fov_transition_width,
                    use_kaiming_init=use_kaiming_init,
                )

                training_config = {
                    "learning_rate": learning_rate,
                    "kl_weight": kl_weight,
                    "recon_weight": recon_weight,
                    "ssim_weight": ssim_weight,
                    "perceptual_weight": perceptual_weight,
                    "perceptual_network": perceptual_network,
                    "is_fake_3d": is_fake_3d,
                    "orientation_weight": orientation_weight,
                    "balanced_orientation_combos": balanced_orientation_combos,
                }

                unwrapped_model = accelerator.unwrap_model(model)
                save_model_checkpoint(
                    filepath=best_model_path,
                    model=unwrapped_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=avg_loss,
                    val_loss=val_loss,
                    model_type="mshved",
                    model_config=model_config,
                    scheduler_state_dict=scheduler.state_dict(),
                    val_metrics=val_metrics,
                    training_config=training_config,
                )
                accelerator.print(f"✓ Saved best model (val_loss: {val_loss:.4f}): {best_model_path}")

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint_path = os.path.join(
                    model_dir, f"mshved_orthogonal_epoch_{epoch + 1:04d}.pth"
                )

                model_config = build_model_config(
                    num_orientations=3,
                    num_scales=num_scales,
                    output_shape=output_shape,
                    reconstruct_orientations=reconstruct_orientations,
                    decoder_upsample_mode=decoder_upsample_mode,
                    final_activation=final_activation,
                    init_filters=init_filters,
                    blocks_down=tuple(blocks_down) if blocks_down else None,
                    blocks_up=tuple(blocks_up) if blocks_up else None,
                    global_residual=global_residual,
                    smooth_fov=smooth_fov,
                    fov_attenuation=fov_attenuation,
                    fov_transition_width=fov_transition_width,
                    use_kaiming_init=use_kaiming_init,
                )

                training_config = {
                    "learning_rate": learning_rate,
                    "kl_weight": kl_weight,
                    "recon_weight": recon_weight,
                    "ssim_weight": ssim_weight,
                    "perceptual_weight": perceptual_weight,
                    "perceptual_network": perceptual_network,
                    "is_fake_3d": is_fake_3d,
                    "orientation_weight": orientation_weight,
                    "balanced_orientation_combos": balanced_orientation_combos,
                }

                unwrapped_model = accelerator.unwrap_model(model)
                save_model_checkpoint(
                    filepath=checkpoint_path,
                    model=unwrapped_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=avg_loss,
                    val_loss=val_loss,
                    model_type="mshved",
                    model_config=model_config,
                    scheduler_state_dict=scheduler.state_dict(),
                    val_metrics=val_metrics,
                    training_config=training_config,
                )
                accelerator.print(f"Saved checkpoint: {checkpoint_path}")

    # Close CSV file
    if accelerator.is_main_process and csv_file is not None:
        csv_file.close()

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = os.path.join(model_dir, "mshved_orthogonal_final.pth")
        model_config = build_model_config(
            num_orientations=3,
            num_scales=num_scales,
            output_shape=output_shape,
            reconstruct_orientations=reconstruct_orientations,
            decoder_upsample_mode=decoder_upsample_mode,
            final_activation=final_activation,
            init_filters=init_filters,
            blocks_down=tuple(blocks_down) if blocks_down else None,
            blocks_up=tuple(blocks_up) if blocks_up else None,
            global_residual=global_residual,
            smooth_fov=smooth_fov,
            fov_attenuation=fov_attenuation,
            fov_transition_width=fov_transition_width,
            use_kaiming_init=use_kaiming_init,
        )

        training_config = {
            "learning_rate": learning_rate,
            "kl_weight": kl_weight,
            "recon_weight": recon_weight,
            "ssim_weight": ssim_weight,
            "perceptual_weight": perceptual_weight,
            "perceptual_network": perceptual_network,
            "is_fake_3d": is_fake_3d,
            "orientation_weight": orientation_weight,
            "balanced_orientation_combos": balanced_orientation_combos,
        }

        unwrapped_model = accelerator.unwrap_model(model)
        save_model_checkpoint(
            filepath=final_path,
            model=unwrapped_model,
            optimizer=optimizer,
            epoch=epochs - 1,
            loss=avg_loss,
            val_loss=val_loss,
            model_type="mshved",
            model_config=model_config,
            scheduler_state_dict=scheduler.state_dict(),
            val_metrics=val_metrics,
            training_config=training_config,
        )
        print(f"Training complete! Final model saved to: {final_path}")
        if best_val_loss < float('inf'):
            print(f"Best validation loss: {best_val_loss:.4f}")


        if use_wandb:
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
                description="Final trained MS-HVED model with orthogonal stacks",
            )
            artifact.add_file(final_path)
            wandb.log_artifact(artifact)
            wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    # Set multiprocessing start method
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Train MS-HVED with Orthogonal LR Stacks")

    # Training data arguments
    parser.add_argument("--hr_image_dir", type=str, default=None, help="Directory containing HR images")
    parser.add_argument("--csv_file", type=str, nargs="+", default=None,
                        help="CSV file(s) with image metadata. Pass multiple paths to "
                             "combine datasets, e.g. --csv_file a.csv b.csv")
    parser.add_argument("--base_dir", type=str, nargs="+", default=None,
                        help="Base director(ies) for CSV relative paths. Must have the "
                             "same count as --csv_file (one base_dir per CSV).")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save models")
    parser.add_argument("--val_image_dir", type=str, default=None, help="Validation images directory")
    parser.add_argument("--mri_classes", type=str, nargs="+", default=None,
                       help="MRI classifications to include (e.g., T1 T2 FLAIR). Only for CSV mode")
    parser.add_argument("--acquisition_types", type=str, nargs="+", default=["3D"],
                       help="Acquisition types to include (e.g., 3D 2D). Use 'all' for all types. Default: 3D only")
    parser.add_argument("--no_filter_4d", action="store_true",
                       help="Don't filter out 4D images (with time dimension)")

    # Model parameters
    parser.add_argument("--num_scales", type=int, default=4, help="Number of hierarchical scales")
    parser.add_argument("--final_activation", type=str, default="clamp", choices=["clamp", "sigmoid", "tanh", "none"],
                        help="Final activation function")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping (0 to disable)")
    parser.add_argument("--decoder_upsample_mode", type=str, default="trilinear",
                        choices=["trilinear", "transpose", "pixelshuffle"],
                        help="Decoder upsampling strategy: trilinear (interpolation+conv), "
                             "transpose (transposed conv), or pixelshuffle (sub-pixel conv)")
    parser.add_argument("--init_filters", type=int, default=32,
                        help="Initial filter count (default: 32)")
    parser.add_argument("--blocks_down", type=int, nargs='+', default=[1, 2, 2, 4],
                        help="Residual blocks per encoder scale")
    parser.add_argument("--blocks_up", type=int, nargs='+', default=[1, 1, 1],
                        help="Residual blocks per decoder scale")
    parser.add_argument("--no_reconstruct_orientations", action="store_true",
                        help="Disable orientation reconstruction (SR decoder only, for ablation studies)")
    parser.add_argument("--global_residual", action="store_true",
                        help="Enable global residual learning (network predicts residual instead of full output)")
    parser.add_argument("--kaiming_init", action="store_true",
                        help="Enable explicit Kaiming init on Conv3d/ConvTranspose3d weights")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_shape", type=int, nargs=3, default=[128, 128, 128], help="Output shape")

    # Loss parameters and weights
    parser.add_argument("--recon_loss_type", type=str, default="charbonnier", choices=["l1", "l2", "charbonnier"],
                        help="Reconstruction loss type")
    parser.add_argument("--recon_weight", type=float, default=0.4, help="Reconstruction loss weight")
    parser.add_argument("--kl_weight", type=float, default=0.1, help="KL divergence weight")
    parser.add_argument("--perceptual_weight", type=float, default=0.1, help="Perceptual loss weight")
    parser.add_argument("--ssim_weight", type=float, default=0.1, help="SSIM loss weight")
    parser.add_argument("--orientation_weight", type=float, default=0.4, help="orientation reconstruction weight")
    parser.add_argument("--use_perceptual", action="store_true", help="Use perceptual loss")
    parser.add_argument("--use_ssim", action="store_true", help="Use SSIM loss")
    parser.add_argument("--perceptual_network", type=str, default="alex",
                        choices=["alex", "vgg", "squeeze", "radimagenet", "medicalnet", "resnet50"],
                        help="Perceptual loss network (MONAI). Options: alex (default, fast LPIPS), "
                             "vgg, squeeze, radimagenet (RadImageNet ResNet-50), "
                             "medicalnet (MedicalNet ResNet-10), resnet50")
    parser.add_argument("--is_fake_3d", action="store_true",
                        help="Use 2.5D (fake 3D) mode for perceptual loss (faster, lower memory)")

    # Consistency training parameters
    parser.add_argument("--num_variations", type=int, default=1,
                        help="Number of degradation variations per HR volume for consistency training. "
                             "1 = standard training (default), 2+ = multi-variation consistency training")
    parser.add_argument("--consistency_weight", type=float, default=0.2,
                        help="Weight for output-space consistency loss between variations (default: 0.2)")
    parser.add_argument("--latent_consistency_weight", type=float, default=0.05,
                        help="Weight for latent-space consistency loss between PoG posteriors (default: 0.05)")
    parser.add_argument("--consistency_warmup_steps", type=int, default=5000,
                        help="Number of steps to linearly warm up consistency losses (default: 5000)")

    # Data generation parameters
    parser.add_argument("--atlas_res", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="HR resolution")
    parser.add_argument("--min_resolution", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Min resolution")
    parser.add_argument("--max_res_aniso", type=float, nargs=3, default=[9.0, 9.0, 9.0], help="Max aniso resolution")
    parser.add_argument("--no_randomise_res", action="store_true", help="Disable resolution randomization")
    parser.add_argument("--prob_motion", type=float, default=0.5, help="Probability of motion artifacts")
    parser.add_argument("--prob_spike", type=float, default=0.5, help="Probability of k-space spikes")
    parser.add_argument("--prob_aliasing", type=float, default=0.02, help="Probability of aliasing")
    parser.add_argument("--prob_bias_field", type=float, default=0.5, help="Probability of bias field")
    parser.add_argument("--prob_noise", type=float, default=0.8, help="Probability of noise")
    parser.add_argument("--fov_augmentation_prob", type=float, default=0.7, help="Probability of FOV augmentation")
    parser.add_argument("--no_intensity_aug", action="store_true", help="Disable intensity augmentation")

    # orientation dropout (for robust training with missing views)
    parser.add_argument("--orientation_dropout_prob", type=float, default=0.0,
                        help="Probability of applying orientation dropout (0.0-1.0). "
                             "Randomly drops 1-2 orthogonal views to simulate missing data during inference. "
                             "Default: 0.0 (no dropout)")
    parser.add_argument("--min_orientations", type=int, default=1,
                        help="Minimum number of orientations to keep after dropout (1-3). "
                             "Default: 1 (allows training with single views)")
    parser.add_argument("--drop_orientations", type=int, nargs="+", default=None,
                        choices=[0, 1, 2],
                        help="Specific orientations to drop (0=Axial, 1=Coronal, 2=Sagittal). "
                             "If specified, these orientations will ALWAYS be dropped. "
                             "Mutually exclusive with random orientation_dropout_prob.")
    parser.add_argument(
        "--balanced_orientation_combos",
        action="store_true",
        help="Use balanced orientation mask combos per epoch for training.",
    )
    parser.add_argument("--upsample_mode", type=str, default="trilinear",
                        choices=["nearest", "trilinear", "nearest-exact"],
                        help="Interpolation mode for FFT upsample recovery (default: nearest)")

    # Obliqueness simulation
    parser.add_argument("--enable_obliqueness", action="store_true",
                        help="Enable oblique slice simulation (tilted LR stacks with affine resampling)")
    parser.add_argument("--prob_obliqueness", type=float, default=0.5,
                        help="Probability of applying obliqueness per stack (when enabled)")
    parser.add_argument("--obliqueness_range", type=float, default=15.0,
                        help="Maximum rotation angle in degrees per axis for obliqueness")

    # FOV fusion parameters
    parser.add_argument("--smooth_fov", action="store_true", default=True,
                        help="Smooth FOV mask boundaries with Gaussian kernel (default: True)")
    parser.add_argument("--no_smooth_fov", action="store_true",
                        help="Disable FOV mask smoothing (use hard binary boundaries)")
    parser.add_argument("--fov_attenuation", type=float, default=1.0,
                        help="Precision suppression for out-of-FOV voxels (0=none, 1=full). Default: 1.0")
    parser.add_argument("--fov_transition_width", type=float, default=3.0,
                        help="Gaussian sigma in voxels for FOV boundary smoothing. Default: 3.0")

    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--save_interval", type=int, default=10, help="Save every N epochs")
    parser.add_argument("--val_interval", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--use_cache", action="store_true", help="Use MONAI CacheDataset")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="mshved", help="W&B project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Process acquisition_types: convert "all" to None for the function
    acquisition_types = args.acquisition_types
    if acquisition_types and len(acquisition_types) == 1 and acquisition_types[0].lower() == "all":
        acquisition_types = None

    # Validate --csv_file / --base_dir pairing
    if args.csv_file is not None:
        if args.base_dir is None:
            raise ValueError("--base_dir is required when using --csv_file")
        if len(args.csv_file) != len(args.base_dir):
            raise ValueError(
                f"Number of --csv_file entries ({len(args.csv_file)}) must match "
                f"number of --base_dir entries ({len(args.base_dir)}). "
                f"Got csv_file={args.csv_file}, base_dir={args.base_dir}"
            )

    # Validate orientation dropout settings
    if args.drop_orientations is not None and len(args.drop_orientations) > 0:
        if args.orientation_dropout_prob > 0.0:
            print("WARNING: Both --drop_orientations and --orientation_dropout_prob specified. "
                  "Using deterministic dropout (--drop_orientations) and ignoring probability.")
        if len(args.drop_orientations) >= 3:
            raise ValueError("Cannot drop all 3 orientations. At least one must remain.")

    # Validate loss weights when orientation reconstruction is disabled
    if args.no_reconstruct_orientations:
        if args.orientation_weight > 0.0:
            print("\n" + "="*80)
            print("WARNING: Orientation reconstruction disabled but orientation_weight is non-zero")
            print(f"  Current orientation_weight: {args.orientation_weight}")
            print("  Orientation loss will be 0.0 regardless of this weight.")
            print("  Recommendation: Set --orientation_weight 0.0 for cleaner metrics.")
            print("="*80 + "\n")

    # Get image paths
    hr_image_paths = get_image_paths(
        image_dir=args.hr_image_dir,
        csv_file=args.csv_file,
        base_dir=args.base_dir,
        split="train",
        model_dir=args.model_dir,
        mri_classifications=args.mri_classes,
        acquisition_types=acquisition_types,
        filter_4d=not args.no_filter_4d,
    )

    val_image_paths = None
    if args.val_image_dir or args.csv_file:
        val_image_paths = get_image_paths(
            image_dir=args.val_image_dir,
            csv_file=args.csv_file,
            base_dir=args.base_dir,
            split="val",
            model_dir=args.model_dir,
            mri_classifications=args.mri_classes,
            acquisition_types=acquisition_types,
            filter_4d=not args.no_filter_4d,
        )

    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)

    # Save configuration
    save_training_config(
        model_dir=args.model_dir,
        args=args,
        n_train_samples=len(hr_image_paths),
        n_val_samples=len(val_image_paths) if val_image_paths else 0,
    )

    # Prepare architecture-specific parameters
    blocks_down = args.blocks_down if hasattr(args, 'blocks_down') else [1, 2, 2, 4]
    blocks_up = args.blocks_up if hasattr(args, 'blocks_up') else [1, 1, 1]

    # Train model
    train_mshved_model(
        # Data sources and outputs
        hr_image_paths=hr_image_paths,
        model_dir=args.model_dir,
        val_image_paths=val_image_paths,
        checkpoint=args.checkpoint,

        # Training schedule and runtime
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        save_interval=args.save_interval,
        val_interval=args.val_interval,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,

        # Data generation and augmentation
        output_shape=tuple(args.output_shape),
        atlas_res=args.atlas_res,
        min_resolution=args.min_resolution,
        max_res_aniso=args.max_res_aniso,
        randomise_res=not args.no_randomise_res,
        prob_motion=args.prob_motion,
        prob_spike=args.prob_spike,
        prob_aliasing=args.prob_aliasing,
        prob_bias_field=args.prob_bias_field,
        prob_noise=args.prob_noise,
        fov_augmentation_prob=args.fov_augmentation_prob,
        apply_intensity_aug=not args.no_intensity_aug,
        orientation_dropout_prob=args.orientation_dropout_prob,
        min_orientations=args.min_orientations,
        drop_orientations=args.drop_orientations,
        balanced_orientation_combos=args.balanced_orientation_combos,
        num_variations=args.num_variations,
        upsample_mode=args.upsample_mode,
        enable_obliqueness=args.enable_obliqueness,
        prob_obliqueness=args.prob_obliqueness,
        obliqueness_range=args.obliqueness_range,

        # Data loading
        num_workers=args.num_workers,
        use_cache=args.use_cache,

        # Model architecture
        num_scales=args.num_scales,
        init_filters=args.init_filters,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
        reconstruct_orientations=not args.no_reconstruct_orientations,
        decoder_upsample_mode=args.decoder_upsample_mode,
        global_residual=args.global_residual,
        smooth_fov=not args.no_smooth_fov,
        fov_attenuation=args.fov_attenuation,
        fov_transition_width=args.fov_transition_width,
        use_kaiming_init=args.kaiming_init,

        # Loss configuration
        recon_loss_type=args.recon_loss_type,
        recon_weight=args.recon_weight,
        kl_weight=args.kl_weight,
        perceptual_weight=args.perceptual_weight,
        ssim_weight=args.ssim_weight,
        orientation_weight=args.orientation_weight,
        use_perceptual=args.use_perceptual,
        use_ssim=args.use_ssim,
        perceptual_network=args.perceptual_network,
        is_fake_3d=args.is_fake_3d,

        # Consistency loss
        consistency_weight=args.consistency_weight,
        latent_consistency_weight=args.latent_consistency_weight,
        consistency_warmup_steps=args.consistency_warmup_steps,

        # Experiment tracking
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )
