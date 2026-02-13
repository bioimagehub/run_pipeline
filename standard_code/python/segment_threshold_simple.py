"""
Simple threshold-based segmentation pipeline.

This script provides a straightforward segmentation workflow:
1. Optional Gaussian blur preprocessing
2. Threshold application (various methods available)
3. Fill holes in binary masks
4. Remove objects touching edges
5. Save labeled masks and ImageJ ROIs

No caching, no complex parameter logic - just clean, simple segmentation.
"""

from __future__ import annotations
import os
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.filters import (
    threshold_otsu, threshold_yen, threshold_li, threshold_triangle,
    threshold_mean, threshold_minimum, threshold_isodata,
    threshold_niblack, threshold_sauvola
)
from skimage.measure import label
from skimage.morphology import remove_small_objects

# Local imports
import bioimage_pipeline_utils as rp

# Configure logging
logger = logging.getLogger(__name__)


# Available threshold methods
THRESHOLD_METHODS = {
    "otsu": threshold_otsu,
    "yen": threshold_yen,
    "li": threshold_li,
    "triangle": threshold_triangle,
    "mean": threshold_mean,
    "minimum": threshold_minimum,
    "isodata": threshold_isodata,
    "niblack": lambda img: threshold_niblack(img, window_size=25),
    "sauvola": lambda img: threshold_sauvola(img, window_size=25)
}


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to each 2D plane in a 5D TCZYX image."""
    if sigma <= 0:
        return image
    
    t, c, z, y, x = image.shape
    blurred = np.empty_like(image)
    
    for ti in range(t):
        for ci in range(c):
            for zi in range(z):
                blurred[ti, ci, zi] = gaussian_filter(image[ti, ci, zi], sigma=sigma)
    
    return blurred


def apply_threshold_simple(image: np.ndarray, method: str, channel: int = 0) -> np.ndarray:
    """Apply threshold to specified channel and return labeled mask (5D TCZYX)."""
    if method not in THRESHOLD_METHODS:
        raise ValueError(f"Unsupported method: {method}. Available: {list(THRESHOLD_METHODS.keys())}")
    
    threshold_fn = THRESHOLD_METHODS[method]
    t, c, z, y, x = image.shape
    mask = np.zeros((t, c, z, y, x), dtype=np.uint16)
    
    for ti in range(t):
        for zi in range(z):
            plane = image[ti, channel, zi]
            try:
                thresh_value = threshold_fn(plane)
                binary = plane > thresh_value
                labeled = label(binary)
                mask[ti, channel, zi] = labeled.astype(np.uint16)
            except Exception as e:
                logger.warning(f"Threshold failed at T={ti}, Z={zi}: {e}")
                continue
    
    return mask


def fill_holes_in_mask(mask: np.ndarray) -> np.ndarray:
    """Fill holes in each labeled region independently."""
    filled = np.copy(mask)
    t, c, z, y, x = mask.shape
    
    for ti in range(t):
        for ci in range(c):
            for zi in range(z):
                plane = mask[ti, ci, zi]
                for label_id in np.unique(plane):
                    if label_id == 0:
                        continue
                    region_mask = plane == label_id
                    filled_region = binary_fill_holes(region_mask)
                    holes = filled_region & ~region_mask
                    filled[ti, ci, zi][holes] = label_id
    
    return filled


def remove_edge_objects(mask: np.ndarray, remove_xy: bool = True, remove_z: bool = False) -> np.ndarray:
    """Remove objects touching image edges."""
    cleaned = np.zeros_like(mask)
    t, c, z, y, x = mask.shape
    
    for ti in range(t):
        for ci in range(c):
            for zi in range(z):
                plane = mask[ti, ci, zi]
                for label_id in np.unique(plane):
                    if label_id == 0:
                        continue
                    
                    region_mask = plane == label_id
                    
                    # Check XY edges
                    touches_xy = (
                        np.any(region_mask[0, :]) or np.any(region_mask[-1, :]) or
                        np.any(region_mask[:, 0]) or np.any(region_mask[:, -1])
                    )
                    
                    if remove_xy and touches_xy:
                        continue
                    
                    # Check Z edges (across all z-planes for this object)
                    if remove_z and (zi == 0 or zi == z - 1):
                        if np.any(region_mask):
                            continue
                    
                    cleaned[ti, ci, zi][region_mask] = label_id
    
    return cleaned


def remove_small_objects_from_mask(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove objects smaller than min_size pixels."""
    if min_size <= 0:
        return mask
    
    cleaned = np.zeros_like(mask)
    t, c, z, y, x = mask.shape
    
    for ti in range(t):
        for ci in range(c):
            for zi in range(z):
                plane = mask[ti, ci, zi]
                for label_id in np.unique(plane):
                    if label_id == 0:
                        continue
                    
                    region_mask = plane == label_id
                    if np.sum(region_mask) >= min_size:
                        cleaned[ti, ci, zi][region_mask] = label_id
    
    return cleaned


def remove_large_objects_from_mask(mask: np.ndarray, max_size: float) -> np.ndarray:
    """Remove objects larger than max_size pixels."""
    if max_size >= float('inf'):
        return mask
    
    cleaned = np.zeros_like(mask)
    t, c, z, y, x = mask.shape
    
    for ti in range(t):
        for ci in range(c):
            for zi in range(z):
                plane = mask[ti, ci, zi]
                for label_id in np.unique(plane):
                    if label_id == 0:
                        continue
                    
                    region_mask = plane == label_id
                    if np.sum(region_mask) <= max_size:
                        cleaned[ti, ci, zi][region_mask] = label_id
    
    return cleaned


def process_image(
    input_path: str,
    output_folder: str,
    channel: int = 0,
    threshold_method: str = "li",
    gaussian_sigma: float = 0,
    fill_holes: bool = True,
    remove_xy_edges: bool = True,
    remove_z_edges: bool = False,
    min_size: int = 0,
    max_size: float = float('inf')
) -> None:
    """Process a single image file."""
    logger.info(f"Processing: {input_path}")
    
    # Load image
    img = rp.load_tczyx_image(input_path)
    image_data = img.data
    
    # Optional Gaussian blur
    if gaussian_sigma > 0:
        logger.info(f"  Applying Gaussian blur (sigma={gaussian_sigma})...")
        image_data = apply_gaussian_blur(image_data, gaussian_sigma)
    
    # Apply threshold
    logger.info(f"  Applying {threshold_method} threshold to channel {channel}...")
    mask = apply_threshold_simple(image_data, threshold_method, channel)
    
    # Fill holes
    if fill_holes:
        logger.info("  Filling holes...")
        mask = fill_holes_in_mask(mask)
    
    # Remove small objects
    if min_size > 0:
        logger.info(f"  Removing objects < {min_size} pixels...")
        mask = remove_small_objects_from_mask(mask, min_size)
    
    # Remove large objects
    if max_size < float('inf'):
        logger.info(f"  Removing objects > {max_size} pixels...")
        mask = remove_large_objects_from_mask(mask, max_size)

    # Remove edge objects
    if remove_xy_edges or remove_z_edges:
        logger.info(f"  Removing edge objects (XY={remove_xy_edges}, Z={remove_z_edges})...")
        mask = remove_edge_objects(mask, remove_xy_edges, remove_z_edges)
    
    # Prepare output paths
    input_name = Path(input_path).stem
    mask_path = os.path.join(output_folder, f"{input_name}_mask.tif")
    roi_path = os.path.join(output_folder, f"{input_name}_rois.zip")
    
    # Save mask (ImageJ-compatible TIFF)
    logger.info(f"  Saving mask to {mask_path}...")
    rp.save_mask(mask, mask_path, as_binary=True)
    
    # Generate and save ROIs
    logger.info(f"  Generating ROIs...")
    rois = rp.mask_to_rois(mask)
    if rois:
        from roifile import roiwrite
        if os.path.exists(roi_path):
            os.remove(roi_path)
        roiwrite(roi_path, rois)
        logger.info(f"  Saved {len(rois)} ROIs to {roi_path}")
    else:
        logger.info("  No ROIs generated (empty mask)")
    
    logger.info(f"  ✓ Done")


def process_folder(
    input_pattern: str,
    output_folder: str,
    channel: int,
    threshold_method: str,
    gaussian_sigma: float,
    fill_holes: bool,
    remove_xy_edges: bool,
    remove_z_edges: bool,
    min_size: int,
    max_size: float,
    no_parallel: bool
) -> None:
    """Process multiple files matching the input pattern."""
    # Find files
    search_subfolders = '**' in input_pattern
    files = rp.get_files_to_process2(input_pattern, search_subfolders=search_subfolders)
    
    if not files:
        logger.warning(f"No files found matching: {input_pattern}")
        return
    
    logger.info(f"Found {len(files)} file(s) to process")
    os.makedirs(output_folder, exist_ok=True)
    
    def process_one(file_path):
        try:
            process_image(
                file_path, output_folder, channel, threshold_method,
                gaussian_sigma, fill_holes, remove_xy_edges, remove_z_edges, min_size, max_size
            )
        except Exception as e:
            logger.error(f"ERROR processing {file_path}: {e}")
    
    if not no_parallel and len(files) > 1:
        from joblib import Parallel, delayed
        Parallel(n_jobs=-1)(delayed(process_one)(f) for f in tqdm(files, desc="Processing"))
    else:
        for file_path in files:
            process_one(file_path)


# =============================================================================
# Example YAML configuration for run_pipeline.exe
# =============================================================================
"""
Example YAML config for run_pipeline.exe:

---
run:
  - name: Simple threshold segmentation
    environment: uv@3.11:segment-threshold
    commands:
      - python
      - '%REPO%/standard_code/python/segment_threshold_simple.py'
      - --input-pattern: '%YAML%/input_data/**/*.tif'
      - --output-folder: '%YAML%/output_masks'
      - --channel: 0
      - --method: li
      - --gaussian-sigma: 0
      - --fill-holes
      - --remove-xy-edges
      - --min-size: 100
      - --max-size: 10000
      - --no-parallel

  - name: With Gaussian blur preprocessing
    environment: uv@3.11:segment-threshold
    commands:
      - python
      - '%REPO%/standard_code/python/segment_threshold_simple.py'
      - --input-pattern: '%YAML%/input_data/**/*.tif'
      - --output-folder: '%YAML%/output_masks'
      - --channel: 1
      - --method: otsu
      - --gaussian-sigma: 2.0
      - --fill-holes
      - --remove-xy-edges
      - --remove-z-edges
      - --min-size: 500
      - --max-size: inf

Notes:
  - Available threshold methods: otsu, yen, li, triangle, mean, minimum, isodata, niblack, sauvola
  - Set --gaussian-sigma to 0 to skip blurring
  - Parallel processing is enabled by default, use --no-parallel to disable
  - Use --verbose to see processing logs (default: only warnings and errors)
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple threshold-based segmentation with optional Gaussian blur",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available threshold methods:
  otsu, yen, li, triangle, mean, minimum, isodata, niblack, sauvola

Example usage:
  python segment_threshold_simple.py --input-pattern "data/*.tif" \\
      --output-folder "output" --channel 0 --method li --fill-holes
        """
    )
    
    parser.add_argument(
        "--input-pattern", required=True,
        help="Input file pattern (supports wildcards, use '**' for recursive)"
    )
    parser.add_argument(
        "--output-folder", required=True,
        help="Output folder for masks and ROIs"
    )
    parser.add_argument(
        "--channel", type=int, default=0,
        help="Channel to segment (default: 0)"
    )
    parser.add_argument(
        "--method", type=str, default="li",
        choices=list(THRESHOLD_METHODS.keys()),
        help="Threshold method (default: li)"
    )
    parser.add_argument(
        "--gaussian-sigma", type=float, default=0,
        help="Gaussian blur sigma for preprocessing (0 = no blur, default: 0)"
    )
    parser.add_argument(
        "--fill-holes", action="store_true",
        help="Fill holes in segmented objects"
    )
    parser.add_argument(
        "--remove-xy-edges", action="store_true",
        help="Remove objects touching XY edges"
    )
    parser.add_argument(
        "--remove-z-edges", action="store_true",
        help="Remove objects touching Z edges"
    )
    parser.add_argument(
        "--min-size", type=int, default=0,
        help="Minimum object size in pixels (0 = no filtering, default: 0)"
    )
    parser.add_argument(
        "--max-size", type=str, default="inf",
        help="Maximum object size in pixels ('inf' = no filtering, default: inf)"
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Disable parallel processing (parallel is enabled by default)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging output"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(message)s'
    )
    
    # Parse max_size (handle 'inf' string)
    max_size = float('inf') if args.max_size.lower() == 'inf' else float(args.max_size)
    
    process_folder(
        input_pattern=args.input_pattern,
        output_folder=args.output_folder,
        channel=args.channel,
        threshold_method=args.method,
        gaussian_sigma=args.gaussian_sigma,
        fill_holes=args.fill_holes,
        remove_xy_edges=args.remove_xy_edges,
        remove_z_edges=args.remove_z_edges,
        min_size=args.min_size,
        max_size=max_size,
        no_parallel=args.no_parallel
    )
    
    logger.info("\n✓ All processing complete!")
