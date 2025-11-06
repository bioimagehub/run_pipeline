"""
Simple Bilateral Denoising Module for BIPHUB Pipeline
Drop-in replacement for Cellpose 3 denoise models

Author: BIPHUB
Date: 2025-01-06
"""

import cv2
import numpy as np
import bioimage_pipeline_utils as rp
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def denoise_image(image_data: np.ndarray, 
                  filter_size: int = 9,
                  sigma_color: int = 75,
                  sigma_space: int = 75) -> np.ndarray:
    """
    Denoise a single 2D image using bilateral filter.
    
    Args:
        image_data: 2D numpy array
        filter_size: Diameter of pixel neighborhood (increase for more smoothing)
        sigma_color: Filter sigma in color space (increase for more smoothing)
        sigma_space: Filter sigma in coordinate space
        
    Returns:
        Denoised 2D numpy array
    """
    # Store original dtype and range
    original_dtype = image_data.dtype
    original_min = image_data.min()
    original_max = image_data.max()
    
    # Normalize to 8-bit for bilateral filter
    if original_dtype != np.uint8:
        normalized = ((image_data - original_min) / (original_max - original_min) * 255)
        normalized = normalized.astype(np.uint8)
    else:
        normalized = image_data
    
    # Apply bilateral filter
    denoised = cv2.bilateralFilter(normalized, filter_size, sigma_color, sigma_space)
    
    # Scale back to original range
    if original_dtype != np.uint8:
        denoised = (denoised.astype(np.float32) / 255.0) * (original_max - original_min) + original_min
        denoised = denoised.astype(original_dtype)
    
    return denoised


def denoise_stack(image_path: str,
                  output_path: str,
                  filter_size: int = 9,
                  sigma_color: int = 75,
                  sigma_space: int = 75,
                  process_z: bool = True,
                  process_t: bool = True,
                  process_c: bool = True):
    """
    Denoise a multi-dimensional image stack.
    
    Args:
        image_path: Path to input image
        output_path: Path to save denoised image
        filter_size: Bilateral filter diameter
        sigma_color: Color space sigma
        sigma_space: Coordinate space sigma
        process_z: Denoise each Z slice
        process_t: Denoise each timepoint
        process_c: Denoise each channel
    """
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
                denoised_2d = denoise_image(slice_2d, filter_size, sigma_color, sigma_space)
                
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
    """Command line interface for bilateral denoising."""
    parser = argparse.ArgumentParser(
        description="Bilateral filter denoising for bioimage data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Denoise images (bilateral filter)
  environment: uv@3.11:denoise
  commands:
  - python
  - '%REPO%/standard_code/python/bilateral_denoise.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/denoised_output'
  - --filter-size: 9
  - --sigma-color: 75
  - --sigma-space: 75

- name: Denoise with stronger smoothing
  environment: uv@3.11:denoise
  commands:
  - python
  - '%REPO%/standard_code/python/bilateral_denoise.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/denoised_strong'
  - --filter-size: 15
  - --sigma-color: 100
  - --sigma-space: 100
"""
    )
    
    parser.add_argument('--input-search-pattern', type=str, required=True,
                        help='Glob pattern for input images (e.g., "data/**/*.tif")')
    parser.add_argument('--output-folder', type=str, required=True,
                        help='Output folder for denoised images')
    parser.add_argument('--filter-size', type=int, default=9,
                        help='Bilateral filter diameter (default: 9, increase for more smoothing)')
    parser.add_argument('--sigma-color', type=int, default=75,
                        help='Color space sigma (default: 75, increase for more smoothing)')
    parser.add_argument('--sigma-space', type=int, default=75,
                        help='Coordinate space sigma (default: 75)')
    parser.add_argument('--suffix', type=str, default='_denoised',
                        help='Suffix for output files (default: "_denoised")')
    
    args = parser.parse_args()
    
    # Create output folder
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get input files
    input_files = list(Path().glob(args.input_search_pattern))
    
    if not input_files:
        logger.error(f"No files found matching pattern: {args.input_search_pattern}")
        return
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Process each file
    for idx, input_file in enumerate(input_files, 1):
        logger.info(f"\n[{idx}/{len(input_files)}] Processing: {input_file.name}")
        
        # Generate output path
        output_path = output_folder / f"{input_file.stem}{args.suffix}{input_file.suffix}"
        
        try:
            denoise_stack(
                str(input_file),
                str(output_path),
                filter_size=args.filter_size,
                sigma_color=args.sigma_color,
                sigma_space=args.sigma_space
            )
        except Exception as e:
            logger.error(f"Error processing {input_file.name}: {e}")
            continue
    
    logger.info(f"\nAll done! Processed {len(input_files)} files")


if __name__ == "__main__":
    main()
