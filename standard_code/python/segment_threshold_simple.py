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
    """Apply automatic threshold method to specified channel and return labeled mask (5D TCZYX)."""
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


def apply_numeric_threshold(image: np.ndarray, min_val: float, max_val: float, channel: int = 0) -> np.ndarray:
    """Apply numeric threshold to specified channel and return labeled mask (5D TCZYX).
    
    Args:
        image: Input image (5D TCZYX)
        min_val: Minimum threshold value (inclusive)
        max_val: Maximum threshold value (inclusive). Use float('inf') for no upper limit.
        channel: Channel to threshold
    
    Returns:
        Labeled mask with connected components
    """
    t, c, z, y, x = image.shape
    mask = np.zeros((t, c, z, y, x), dtype=np.uint16)
    
    for ti in range(t):
        for zi in range(z):
            plane = image[ti, channel, zi]
            try:
                # Create binary mask where min_val <= pixel <= max_val
                if max_val >= float('inf'):
                    binary = plane >= min_val
                else:
                    binary = (plane >= min_val) & (plane <= max_val)
                
                labeled = label(binary)
                mask[ti, channel, zi] = labeled.astype(np.uint16)
            except Exception as e:
                logger.warning(f"Numeric threshold failed at T={ti}, Z={zi}: {e}")
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
    threshold_method: str = None,
    threshold_min: float = None,
    threshold_max: float = None,
    gaussian_sigma: float = 0,
    fill_holes: bool = True,
    remove_xy_edges: bool = True,
    remove_z_edges: bool = False,
    min_size: int = 0,
    max_size: float = float('inf'),
    save_rois: bool = False,
) -> None:
    """Process a single image file."""
    logger.info(f"Processing: {input_path}")
    
    # Prepare output paths early (needed for early exits)
    input_name = Path(input_path).stem
    mask_path = os.path.join(output_folder, f"{input_name}_mask.tif")
    roi_path = os.path.join(output_folder, f"{input_name}_rois.zip")
    failed_mask_path_basename = os.path.join(output_folder, f"{input_name}")

    # Load image
    img = rp.load_tczyx_image(input_path)
    image_data = img.data
    
    # Optional Gaussian blur
    if gaussian_sigma > 0:
        logger.info(f"  Applying Gaussian blur (sigma={gaussian_sigma})...")
        image_data = apply_gaussian_blur(image_data, gaussian_sigma)
    
    # Apply threshold - determine which method to use
    if threshold_min is not None and threshold_max is not None:
        # Numeric threshold takes priority
        if threshold_method is not None and threshold_method != "li":
            logger.warning(f"Both --method '{threshold_method}' and --threshold [{threshold_min}, {threshold_max}] specified. Using numeric threshold.")
        logger.info(f"  Applying numeric threshold [{threshold_min}, {threshold_max}] to channel {channel}...")
        mask = apply_numeric_threshold(image_data, threshold_min, threshold_max, channel)
    else:
        # Use automatic threshold method
        if threshold_method is None:
            threshold_method = "li"
        logger.info(f"  Applying {threshold_method} threshold to channel {channel}...")
        mask = apply_threshold_simple(image_data, threshold_method, channel)
    
    # Early exit: threshold removed everything despite non-zero input
    if bool(np.any(image_data > 0)) and not np.any(mask > 0):
        logger.warning("  WARNING: Non-zero input but empty mask after thresholding. Early exit.")
        rp.save_mask(mask, mask_path, as_binary=True)
        return
    
    # Fill holes
    if fill_holes:
        logger.info("  Filling holes...")
        mask = fill_holes_in_mask(mask)
    
    # Fill holes can not delete objects so no early exit here

    # Remove small objects
    if min_size > 0:
        logger.info(f"  Removing objects < {min_size} pixels...")
        mask_tmp = remove_small_objects_from_mask(mask, min_size)
    
        # Early exit: this step emptied mask
        if np.any(mask > 0) and not np.any(mask_tmp > 0):
            logger.warning("  WARNING: min_size filtering emptied mask. Saving *_failed and early exit.")
            rp.save_mask(mask, failed_mask_path_basename + "_failed_rm_small.tif", as_binary=True)  # previous step
            rp.save_mask(mask_tmp, mask_path, as_binary=True) # current (empty) result
            logger.info("  No ROIs generated (empty mask)")
            return

        mask = mask_tmp  # IMPORTANT: keep filtered result


    # Remove large objects
    if max_size < float('inf'):
        logger.info(f"  Removing objects > {max_size} pixels...")
        mask_tmp = remove_large_objects_from_mask(mask, max_size)

        # Early exit: this step emptied mask
        if np.any(mask > 0) and not np.any(mask_tmp > 0):
            logger.warning("  WARNING: max_size filtering emptied mask. Saving *_failed and early exit.")
            rp.save_mask(mask, failed_mask_path_basename + "_failed_rm_large.tif", as_binary=True)  # previous step
            rp.save_mask(mask_tmp, mask_path, as_binary=True) # current (empty) result
            logger.info("  No ROIs generated (empty mask)")
            return

        mask = mask_tmp  # IMPORTANT: keep filtered result

    # Remove edge objects
    if remove_xy_edges or remove_z_edges:
        logger.info(f"  Removing edge objects (XY={remove_xy_edges}, Z={remove_z_edges})...")
        mask_tmp = remove_edge_objects(mask, remove_xy_edges, remove_z_edges)

        # Early exit: this step emptied mask
        if np.any(mask > 0) and not np.any(mask_tmp > 0):
            logger.warning("  WARNING: edge filtering emptied mask. Saving *_failed and early exit.")
            rp.save_mask(mask, failed_mask_path_basename + "_failed_rm_edges.tif", as_binary=True)  # previous step
            rp.save_mask(mask_tmp, mask_path, as_binary=True) # current (empty) result
            logger.info("  No ROIs generated (empty mask)")
            return
        mask = mask_tmp  # IMPORTANT: keep filtered result

        
    # Save mask (ImageJ-compatible TIFF)
    logger.info(f"  Saving mask to {mask_path}...")
    rp.save_mask(mask, mask_path, as_binary=True)
    
    # Generate and save ROIs (optional)
    if save_rois:
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
    else:
        logger.info("  Skipping ROI export (set --save-rois to enable)")
    
    logger.info(f"  ✓ Done")


def process_folder(
    input_pattern: str,
    output_folder: str,
    channel: int,
    threshold_method: str,
    threshold_min: float,
    threshold_max: float,
    gaussian_sigma: float,
    fill_holes: bool,
    remove_xy_edges: bool,
    remove_z_edges: bool,
    min_size: int,
    max_size: float,
    save_rois: bool,
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
                threshold_min, threshold_max, gaussian_sigma, fill_holes, remove_xy_edges, remove_z_edges, min_size, max_size, save_rois
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
  - name: Simple threshold segmentation with automatic method
    environment: uv@3.11:segment-threshold
    commands:
      - python
      - '%REPO%/standard_code/python/segment_threshold_simple.py'
      - --input-search-pattern: '%YAML%/input_data/**/*.tif'
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
      - --input-search-pattern: '%YAML%/input_data/**/*.tif'
      - --output-folder: '%YAML%/output_masks'
      - --channel: 1
      - --method: otsu
      - --gaussian-sigma: 2.0
      - --fill-holes
      - --remove-xy-edges
      - --remove-z-edges
      - --min-size: 500
      - --max-size: inf

  - name: Threshold distance maps with numeric range
    environment: uv@3.11:segment-threshold
    commands:
      - python
      - '%REPO%/standard_code/python/segment_threshold_simple.py'
      - --input-search-pattern: '%YAML%/distance_matrix/**/*.tif'
      - --output-folder: '%YAML%/mask_edge'
      - --threshold: 1 10

  - name: Threshold distance maps for nucleus interior
    environment: uv@3.11:segment-threshold
    commands:
      - python
      - '%REPO%/standard_code/python/segment_threshold_simple.py'
      - --input-search-pattern: '%YAML%/distance_matrix/**/*.tif'
      - --output-folder: '%YAML%/mask_bulk'
      - --threshold: 11 inf

Notes:
  - Available threshold methods: otsu, yen, li, triangle, mean, minimum, isodata, niblack, sauvola
  - Use --method for automatic threshold detection (default: li)
  - Use --threshold for numeric thresholding with min and max values (e.g., '--threshold 1 10' or '--threshold 5 inf')
  - If both --method and --threshold are specified, numeric threshold takes priority with a warning
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

Example usage (automatic method):
  python segment_threshold_simple.py --input-search-pattern "data/*.tif" \\
      --output-folder "output" --channel 0 --method li --fill-holes

Example usage (numeric threshold):
  python segment_threshold_simple.py --input-search-pattern "data/*.tif" \\
      --output-folder "output" --threshold 1 10
        """
    )
    
    parser.add_argument(
        "--input-search-pattern", required=True,
        help="Input file pattern (supports wildcards, use '**' for recursive)"
    )
    parser.add_argument(
        "--output-folder", required=True,
        help="Output folder for masks (and ROIs if --save-rois is set)"
    )
    parser.add_argument(
        "--channel", type=int, default=0,
        help="Channel to segment (default: 0)"
    )
    parser.add_argument(
        "--method", type=str, default=None,
        choices=list(THRESHOLD_METHODS.keys()),
        help="Threshold method (default: li if --threshold not specified)"
    )
    parser.add_argument(
        "--threshold", type=str, default=None,
        help="Numeric threshold range (e.g., '1 10' or '5 inf'). Can be specified as two space-separated values or a string. Takes priority over --method if both specified."
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
        "--save-rois", action="store_true",
        help="Save ImageJ ROIs as .zip files (disabled by default)"
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
    
    # Parse threshold values - handle both "1 10" (from YAML) and 1 10 (CLI)
    threshold_min = None
    threshold_max = None
    if args.threshold is not None:
        # Split the threshold string by whitespace
        threshold_parts = str(args.threshold).strip().split()
        if len(threshold_parts) >= 2:
            try:
                threshold_min = float(threshold_parts[0])
                # Handle 'inf' string or numeric
                threshold_max_str = threshold_parts[1]
                threshold_max = float('inf') if threshold_max_str.lower() == 'inf' else float(threshold_max_str)
            except (ValueError, IndexError) as e:
                logger.error(f"Failed to parse --threshold values: {args.threshold}")
                raise
        else:
            logger.error(f"--threshold expects two values (min max), got: {args.threshold}")
            raise ValueError(f"Invalid threshold format: {args.threshold}")
    
    process_folder(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        channel=args.channel,
        threshold_method=args.method,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        gaussian_sigma=args.gaussian_sigma,
        fill_holes=args.fill_holes,
        remove_xy_edges=args.remove_xy_edges,
        remove_z_edges=args.remove_z_edges,
        min_size=args.min_size,
        max_size=max_size,
        save_rois=args.save_rois,
        no_parallel=args.no_parallel
    )
    
    logger.info("\n✓ All processing complete!")
