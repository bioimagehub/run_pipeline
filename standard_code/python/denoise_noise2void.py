"""
Noise2Void Denoising Module for BIPHUB Pipeline
Self-supervised learning-based denoising that learns from noisy images only

Author: BIPHUB
Date: 2025-01-06
"""

import numpy as np
import bioimage_pipeline_utils as rp
from pathlib import Path
import argparse
import logging
from typing import Optional, Tuple
import warnings

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_noise2void_model(
    training_images: list[np.ndarray],
    model_name: str = "n2v_model",
    train_steps: int = 100,
    train_epochs: int = 100,
    batch_size: int = 128,
    patch_size: int = 64,
    n2v_perc_pix: float = 0.198,
    save_dir: str = "models"
) -> 'N2V':
    """
    Train a Noise2Void model on provided training images.
    
    Args:
        training_images: List of 2D numpy arrays for training
        model_name: Name for the saved model
        train_steps: Training steps per epoch
        train_epochs: Number of training epochs
        batch_size: Batch size for training
        patch_size: Size of training patches
        n2v_perc_pix: Percentage of pixels to mask for N2V
        save_dir: Directory to save trained model
        
    Returns:
        Trained N2V model
    """
    try:
        from n2v.models import N2VConfig, N2V
        from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
    except ImportError:
        raise ImportError(
            "Noise2Void (n2v) not installed. Install with: pip install n2v\n"
            "Note: This also requires TensorFlow 2.x"
        )
    
    logger.info(f"Training Noise2Void model with {len(training_images)} images")
    logger.info(f"Training parameters: epochs={train_epochs}, steps={train_steps}, patch_size={patch_size}")
    
    # Prepare training data - stack images along first axis
    X = np.stack(training_images, axis=0)
    
    # Add channel dimension if needed (N2V expects YXC format)
    if X.ndim == 3:
        X = X[..., np.newaxis]
    
    logger.info(f"Training data shape: {X.shape}")
    
    # Create data generator
    datagen = N2V_DataGenerator()
    
    # Generate training patches
    logger.info("Generating training patches...")
    patches = datagen.generate_patches_from_list(
        [X[i, ..., 0] for i in range(X.shape[0])],
        shape=(patch_size, patch_size)
    )
    
    # Split into training and validation (90/10 split)
    n_train = int(0.9 * len(patches))
    X_train = patches[:n_train]
    X_val = patches[n_train:]
    
    logger.info(f"Training patches: {len(X_train)}, Validation patches: {len(X_val)}")
    
    # Configure model
    config = N2VConfig(
        X_train,
        unet_kern_size=3,
        train_steps_per_epoch=train_steps,
        train_epochs=train_epochs,
        train_batch_size=batch_size,
        n2v_perc_pix=n2v_perc_pix,
        n2v_patch_shape=(patch_size, patch_size),
        n2v_manipulator='uniform_withCP',
        n2v_neighborhood_radius=5
    )
    
    # Create and train model
    model = N2V(config, model_name, basedir=save_dir)
    
    logger.info("Starting training...")
    history = model.train(X_train, X_val)
    
    logger.info(f"Training complete! Model saved to {save_dir}/{model_name}")
    
    return model


def denoise_image_n2v(
    image_data: np.ndarray,
    model: 'N2V',
    normalize: bool = True
) -> np.ndarray:
    """
    Denoise a single 2D image using trained Noise2Void model.
    
    Args:
        image_data: 2D numpy array
        model: Trained N2V model
        normalize: Whether to normalize input to [0, 1] range
        
    Returns:
        Denoised 2D numpy array
    """
    # Store original dtype and range
    original_dtype = image_data.dtype
    original_min = image_data.min()
    original_max = image_data.max()
    
    # Normalize to [0, 1] if requested
    if normalize:
        normalized = (image_data - original_min) / (original_max - original_min)
        normalized = normalized.astype(np.float32)
    else:
        normalized = image_data.astype(np.float32)
    
    # Predict (N2V expects YX input, returns YXC)
    denoised = model.predict(normalized, axes='YX')
    
    # Remove channel dimension if present
    if denoised.ndim == 3 and denoised.shape[-1] == 1:
        denoised = denoised[..., 0]
    
    # Scale back to original range
    if normalize:
        denoised = denoised * (original_max - original_min) + original_min
    
    denoised = denoised.astype(original_dtype)
    
    return denoised


def denoise_stack_n2v(
    image_path: str,
    output_path: str,
    model_path: str,
    model_name: str = "n2v_model",
    normalize: bool = True
):
    """
    Denoise a multi-dimensional image stack using Noise2Void.
    
    Args:
        image_path: Path to input image
        output_path: Path to save denoised image
        model_path: Directory containing trained model
        model_name: Name of the model
        normalize: Whether to normalize input
    """
    try:
        from n2v.models import N2V
    except ImportError:
        raise ImportError("Noise2Void (n2v) not installed. Install with: pip install n2v")
    
    logger.info(f"Loading model from: {model_path}/{model_name}")
    model = N2V(config=None, name=model_name, basedir=model_path)
    
    logger.info(f"Loading image: {image_path}")
    img = rp.load_tczyx_image(image_path)
    
    T, C, Z, Y, X = img.shape
    logger.info(f"Image dimensions: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
    
    # Create output array
    denoised_stack = np.zeros_like(img.data)
    
    total_slices = T * C * Z
    processed = 0
    
    logger.info(f"Processing {total_slices} slices...")
    
    for t in range(T):
        for c in range(C):
            for z in range(Z):
                # Get 2D slice
                slice_2d = img.data[t, c, z, :, :]
                
                # Denoise
                denoised_2d = denoise_image_n2v(slice_2d, model, normalize)
                
                # Store result
                denoised_stack[t, c, z, :, :] = denoised_2d
                
                processed += 1
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{total_slices} slices ({processed/total_slices*100:.1f}%)")
    
    # Save result
    logger.info(f"Saving denoised image: {output_path}")
    rp.save_tczyx_image(denoised_stack, output_path)
    logger.info("Done!")


def main():
    """Command line interface for Noise2Void denoising."""
    parser = argparse.ArgumentParser(
        description="Noise2Void denoising for bioimage data (self-supervised learning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Train Noise2Void model
  environment: uv@3.11:denoise
  commands:
  - python
  - '%REPO%/standard_code/python/denoise_noise2void.py'
  - --mode: train
  - --input-search-pattern: '%YAML%/training_data/**/*.tif'
  - --model-dir: '%YAML%/models'
  - --model-name: my_n2v_model
  - --train-epochs: 100
  - --train-steps: 100

- name: Denoise images with trained model
  environment: uv@3.11:denoise
  commands:
  - python
  - '%REPO%/standard_code/python/denoise_noise2void.py'
  - --mode: denoise
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/denoised_output'
  - --model-dir: '%YAML%/models'
  - --model-name: my_n2v_model

- name: Train and denoise in one step
  environment: uv@3.11:denoise
  commands:
  - python
  - '%REPO%/standard_code/python/denoise_noise2void.py'
  - --mode: train_and_denoise
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/denoised_output'
  - --model-dir: '%YAML%/models'
  - --model-name: auto_trained
  - --train-epochs: 50
"""
    )
    
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['train', 'denoise', 'train_and_denoise'],
                        help='Operation mode: train model, denoise with existing model, or both')
    parser.add_argument('--input-search-pattern', type=str, required=True,
                        help='Glob pattern for input images (e.g., "data/**/*.tif")')
    parser.add_argument('--output-folder', type=str,
                        help='Output folder for denoised images (required for denoise mode)')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save/load model (default: "models")')
    parser.add_argument('--model-name', type=str, default='n2v_model',
                        help='Model name (default: "n2v_model")')
    parser.add_argument('--train-epochs', type=int, default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--train-steps', type=int, default=100,
                        help='Training steps per epoch (default: 100)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training batch size (default: 128)')
    parser.add_argument('--patch-size', type=int, default=64,
                        help='Training patch size (default: 64)')
    parser.add_argument('--n2v-perc-pix', type=float, default=0.198,
                        help='Percentage of pixels to mask (default: 0.198)')
    parser.add_argument('--suffix', type=str, default='_n2v_denoised',
                        help='Suffix for output files (default: "_n2v_denoised")')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Disable input normalization')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['denoise', 'train_and_denoise'] and not args.output_folder:
        parser.error("--output-folder is required for denoise mode")
    
    # Get input files
    input_files = list(Path().glob(args.input_search_pattern))
    
    if not input_files:
        logger.error(f"No files found matching pattern: {args.input_search_pattern}")
        return
    
    logger.info(f"Found {len(input_files)} files")
    
    model = None
    
    # Training phase
    if args.mode in ['train', 'train_and_denoise']:
        logger.info("=" * 70)
        logger.info("TRAINING PHASE")
        logger.info("=" * 70)
        
        # Load training images (use first image or all images for training)
        training_images = []
        
        # Load a subset of images for training (max 10 to avoid memory issues)
        train_files = input_files[:min(10, len(input_files))]
        logger.info(f"Loading {len(train_files)} images for training...")
        
        for train_file in train_files:
            logger.info(f"Loading: {train_file.name}")
            img = rp.load_tczyx_image(str(train_file))
            
            # Extract 2D slices (use middle Z and first T, C)
            T, C, Z, Y, X = img.shape
            
            # Sample multiple slices if available
            z_indices = [Z // 4, Z // 2, 3 * Z // 4] if Z >= 4 else [Z // 2]
            
            for z in z_indices:
                for c in range(C):
                    slice_2d = img.data[0, c, z, :, :]
                    training_images.append(slice_2d)
        
        logger.info(f"Collected {len(training_images)} training slices")
        
        # Train model
        model = train_noise2void_model(
            training_images,
            model_name=args.model_name,
            train_steps=args.train_steps,
            train_epochs=args.train_epochs,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            n2v_perc_pix=args.n2v_perc_pix,
            save_dir=args.model_dir
        )
        
        if args.mode == 'train':
            logger.info("Training complete! Use --mode denoise to apply the model.")
            return
    
    # Denoising phase
    if args.mode in ['denoise', 'train_and_denoise']:
        logger.info("=" * 70)
        logger.info("DENOISING PHASE")
        logger.info("=" * 70)
        
        # Create output folder
        output_folder = Path(args.output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing {len(input_files)} files...")
        
        # Process each file
        for idx, input_file in enumerate(input_files, 1):
            logger.info(f"\n[{idx}/{len(input_files)}] Processing: {input_file.name}")
            
            # Generate output path
            output_path = output_folder / f"{input_file.stem}{args.suffix}{input_file.suffix}"
            
            try:
                denoise_stack_n2v(
                    str(input_file),
                    str(output_path),
                    model_path=args.model_dir,
                    model_name=args.model_name,
                    normalize=not args.no_normalize
                )
            except Exception as e:
                logger.error(f"Error processing {input_file.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        logger.info(f"\nAll done! Processed {len(input_files)} files")


if __name__ == "__main__":
    main()
