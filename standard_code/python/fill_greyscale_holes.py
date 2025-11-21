"""
Fill Holes in Greyscale Images

Fills dark regions (holes) in greyscale images while preserving intensity information.
Perfect for filling nucleoli inside nuclei or other dark internal structures.

Uses morphological reconstruction (flood-fill from borders) - equivalent to MATLAB's imfill.

Author: BIPHUB , University of Oslo
Written by: Øyvind Ødegård Fougner
License: MIT
"""

from __future__ import annotations
import os
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Literal
import logging
from skimage.morphology import reconstruction
from tqdm import tqdm

# Local imports
import bioimage_pipeline_utils as rp


def greyscale_fill_holes_2d(image: np.ndarray) -> np.ndarray:
    """
    Fill holes in 2D greyscale image using morphological reconstruction.
    
    This is the Python equivalent of MATLAB's imfill(image, 'holes').
    Works by flood-filling from image borders - holes (regions not connected
    to borders) remain unfilled and are then inverted back.
    
    Perfect for filling dark nucleoli inside bright nuclei.
    
    Parameters
    ----------
    image : np.ndarray
        2D greyscale image (YX) with holes to fill.
        Dark regions (holes) will be filled with surrounding intensities.
        
    Returns
    -------
    filled : np.ndarray
        Image with holes filled, preserving greyscale intensities.
        
    Examples
    --------
    >>> # Fill nucleoli in nucleus image
    >>> nucleus = load_image("nucleus.tif")
    >>> filled = greyscale_fill_holes_2d(nucleus)
    
    Notes
    -----
    The algorithm:
    1. Inverts the image (dark holes become bright peaks)
    2. Seeds from image borders
    3. Flood-fills inward (cannot reach holes)
    4. Inverts back (filled peaks become filled valleys)
    
    References
    ----------
    MATLAB's imfill: http://www.ece.northwestern.edu/CSEL/local-apps/matlabhelp/toolbox/images/morph13.html
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    
    # Handle edge case: empty or constant image
    if image.size == 0 or np.all(image == image.flat[0]):
        return image.copy()
    
    # Invert the image (holes become peaks)
    image_max = image.max()
    inverted = image_max - image
    
    # Create seed: keep border, clear interior
    seed = inverted.copy()
    seed[1:-1, 1:-1] = inverted.min()
    
    # Morphological reconstruction: flood fill from borders
    # Border regions are connected, holes are isolated
    reconstructed = reconstruction(seed, inverted, method='dilation')
    
    # Invert back to get filled result
    filled = image_max - reconstructed
    
    return filled


def greyscale_fill_holes_3d(image: np.ndarray) -> np.ndarray:
    """
    Fill holes in 3D greyscale image (Z-stack) using morphological reconstruction.
    
    Extends the 2D hole filling to 3D volumes. Useful for filling nucleoli
    in confocal Z-stacks of nuclei.
    
    Parameters
    ----------
    image : np.ndarray
        3D greyscale image (ZYX) with holes to fill.
        Dark regions (holes) will be filled with surrounding intensities.
        
    Returns
    -------
    filled : np.ndarray
        Volume with holes filled, preserving greyscale intensities.
        
    Examples
    --------
    >>> # Fill nucleoli in 3D nucleus Z-stack
    >>> nucleus_stack = load_image("nucleus_zstack.tif")  # ZYX
    >>> filled = greyscale_fill_holes_3d(nucleus_stack)
    
    Notes
    -----
    Uses 3D morphological reconstruction to fill holes that are isolated
    in all three dimensions.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {image.shape}")
    
    # Handle edge case: empty or constant image
    if image.size == 0 or np.all(image == image.flat[0]):
        return image.copy()
    
    # Invert the image (holes become peaks)
    image_max = image.max()
    inverted = image_max - image
    
    # Create seed: keep border, clear interior volume
    seed = inverted.copy()
    seed[1:-1, 1:-1, 1:-1] = inverted.min()
    
    # Morphological reconstruction in 3D
    reconstructed = reconstruction(seed, inverted, method='dilation')
    
    # Invert back to get filled result
    filled = image_max - reconstructed
    
    return filled


def fill_holes_auto(image: np.ndarray) -> np.ndarray:
    """
    Automatically detect dimensionality and fill holes appropriately.
    
    Parameters
    ----------
    image : np.ndarray
        2D (YX) or 3D (ZYX) greyscale image with holes to fill.
        
    Returns
    -------
    filled : np.ndarray
        Image with holes filled.
        
    Raises
    ------
    ValueError
        If image is not 2D or 3D.
    """
    if image.ndim == 2:
        return greyscale_fill_holes_2d(image)
    elif image.ndim == 3:
        return greyscale_fill_holes_3d(image)
    else:
        raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D with shape {image.shape}")


def process_single_image(
    input_path: str,
    output_path: str,
    channels: Optional[list[int]] = None,
    mode_3d: bool = False
) -> None:
    """
    Process a single image file: load, fill holes, save.
    
    Parameters
    ----------
    input_path : str
        Path to input image file.
    output_path : str
        Path where filled image will be saved.
    channels : list[int], optional
        List of channel indices to process. If None, processes all channels.
    mode_3d : bool
        If True, treat each Z-stack as a 3D volume for hole filling.
        If False, process each Z-slice independently as 2D.
        
    Notes
    -----
    For multi-channel, multi-timepoint, or Z-stack images:
    - Processes each timepoint independently
    - Can process each channel independently or all channels
    - Can process Z-slices as 2D or entire Z-stack as 3D
    """
    logging.info(f"Loading image: {Path(input_path).name}")
    img = rp.load_tczyx_image(input_path)
    
    T, C, Z, Y, X = img.shape
    logging.info(f"  Shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
    
    # Determine which channels to process
    if channels is None:
        channels_to_process = list(range(C))
    else:
        channels_to_process = [c for c in channels if 0 <= c < C]
        if not channels_to_process:
            raise ValueError(f"No valid channels found. Image has {C} channels, requested {channels}")
    
    logging.info(f"  Processing channels: {channels_to_process}")
    logging.info(f"  Mode: {'3D (Z-stack as volume)' if mode_3d else '2D (slice-by-slice)'}")
    
    # Create output array (copy to preserve non-processed channels)
    output_data = img.data.copy()
    
    # Process each timepoint and channel
    total_operations = T * len(channels_to_process)
    
    with tqdm(total=total_operations, desc="Filling holes") as pbar:
        for t in range(T):
            for c in channels_to_process:
                # Extract the data for this T,C
                if mode_3d and Z > 1:
                    # Process entire Z-stack as 3D volume
                    volume = img.data[t, c, :, :, :]  # ZYX
                    filled_volume = greyscale_fill_holes_3d(volume)
                    output_data[t, c, :, :, :] = filled_volume
                else:
                    # Process each Z-slice as 2D
                    for z in range(Z):
                        slice_2d = img.data[t, c, z, :, :]  # YX
                        filled_slice = greyscale_fill_holes_2d(slice_2d)
                        output_data[t, c, z, :, :] = filled_slice
                
                pbar.update(1)
    
    # Save result
    logging.info(f"Saving filled image to: {Path(output_path).name}")
    rp.save_tczyx_image(output_data, output_path)
    logging.info("  Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Fill holes in greyscale images while preserving intensity information',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Fill nucleoli in nucleus images
  environment: uv@3.11:fill-greyscale-holes
  commands:
  - python
  - '%REPO%/standard_code/python/fill_greyscale_holes.py'
  - --input-search-pattern
  - '%YAML%/input_data/**/*nucleus*.tif'
  - --output-folder
  - '%YAML%/output_data'

- name: Fill holes in 3D Z-stacks (volumetric)
  environment: uv@3.11:fill-greyscale-holes
  commands:
  - python
  - '%REPO%/standard_code/python/fill_greyscale_holes.py'
  - --input-search-pattern
  - '%YAML%/input_data/**/*.tif'
  - --output-folder
  - '%YAML%/output_data'
  - --mode-3d

- name: Fill holes in specific channel only
  environment: uv@3.11:fill-greyscale-holes
  commands:
  - python
  - '%REPO%/standard_code/python/fill_greyscale_holes.py'
  - --input-search-pattern
  - '%YAML%/input_data/**/*.tif'
  - --output-folder
  - '%YAML%/output_data'
  - --channels
  - '0'

Description:
  Fills dark regions (holes) in greyscale images while preserving intensity.
  Perfect for filling nucleoli inside nuclei or similar dark internal structures.
  
  Uses morphological reconstruction (flood-fill from borders) - equivalent to:
    MATLAB: imcomplement(imfill(imcomplement(image), 'holes'))
  
Algorithm:
  1. Inverts image (dark holes become bright peaks)
  2. Seeds from image borders
  3. Flood-fills inward (cannot reach isolated holes)
  4. Inverts back (filled peaks become filled valleys)
  
Use Cases:
  - Fill nucleoli in nucleus images
  - Fill vacuoles in cell bodies
  - Fill any dark internal structures in bright objects
  
2D vs 3D Mode:
  - 2D mode (default): Processes each Z-slice independently
  - 3D mode (--mode-3d): Treats each Z-stack as a single 3D volume
    Use 3D mode when holes extend through multiple Z-slices

Notes:
  - Input should be greyscale (not binary masks)
  - For binary masks, use scipy.ndimage.binary_fill_holes instead
  - Preserves original intensities (unlike binary fill which returns 0/1)
  - Multi-channel images: specify --channels or processes all by default
        """
    )
    
    parser.add_argument(
        '--input-search-pattern',
        type=str,
        required=True,
        help='Glob pattern for input images (e.g., "data/**/*.tif")'
    )
    
    parser.add_argument(
        '--output-folder',
        type=str,
        required=True,
        help='Output folder for filled images'
    )
    
    parser.add_argument(
        '--suffix',
        type=str,
        default='_filled',
        help='Suffix to add to output filenames (default: "_filled")'
    )
    
    parser.add_argument(
        '--channels',
        type=int,
        nargs='+',
        default=None,
        help='Channel indices to process (0-based). If not specified, processes all channels. Example: --channels 0 2'
    )
    
    parser.add_argument(
        '--mode-3d',
        action='store_true',
        help='Process Z-stacks as 3D volumes instead of individual 2D slices. Use when holes span multiple Z-slices.'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Find input files
    logging.info(f"Searching for files: {args.input_search_pattern}")
    input_files = rp.get_files_to_process2(args.input_search_pattern, True)
    
    if not input_files:
        raise FileNotFoundError(f"No files found matching pattern: {args.input_search_pattern}")
    
    logging.info(f"Found {len(input_files)} files to process")
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    logging.info(f"Output folder: {args.output_folder}")
    
    # Process each file
    for i, input_path in enumerate(input_files, 1):
        logging.info(f"\n{'='*70}")
        logging.info(f"Processing file {i}/{len(input_files)}")
        
        # Generate output path
        input_name = Path(input_path).stem
        output_name = f"{input_name}{args.suffix}.tif"
        output_path = os.path.join(args.output_folder, output_name)
        
        try:
            process_single_image(
                input_path=input_path,
                output_path=output_path,
                channels=args.channels,
                mode_3d=args.mode_3d
            )
        except Exception as e:
            logging.error(f"Error processing {Path(input_path).name}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            continue
    
    logging.info(f"\n{'='*70}")
    logging.info("Processing complete!")
    logging.info(f"Processed {len(input_files)} files")
    logging.info(f"Output saved to: {args.output_folder}")


if __name__ == "__main__":
    main()
