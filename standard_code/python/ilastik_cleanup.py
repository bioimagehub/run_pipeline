"""
Ilastik Probability Map Cleanup and Segmentation
=================================================

Python rewrite of debug_segmentation.ijm for processing Ilastik probability maps.

This module processes HDF5 probability maps from Ilastik, performing per-slice
segmentation following ImageJ "stack" conventions (T and Z dimensions processed
independently).

Processing pipeline per XY slice:
1. Thresholding of probability maps
2. Size and edge-based filtering (per XY slice)
3. Iterative watershed segmentation with erosion/dilation
4. Perimeter-based quality control (per XY slice)
5. Convex hull fallback for failed segmentations

The goal is to extract a single, high-quality cell mask per slice that:
- Does not touch XY image edges
- Has area within specified bounds (minsize - maxsize, per XY slice)
- Has perimeter within specified bounds (minperim - maxperim, per XY slice)

Data format: TCZYX dimension order for consistency with pipeline standards.
When "stack" is mentioned (ImageJ convention), always loops over T and Z dimensions.

MIT License - BIPHUB, University of Oslo
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from scipy import ndimage
from skimage import morphology, measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import convex_hull_image
import tifffile
import h5py

try:
    import bioimage_pipeline_utils as rp
    HAS_RP = True
except ImportError:
    HAS_RP = False
    print("Warning: bioimage_pipeline_utils not available, show_image will be disabled")

# Module-level logger
logger = logging.getLogger(__name__)


def load_h5_probability_map(
    h5_path: str,
    dataset_name: str = "/exported_data",
    channel: int = 1,
    threshold_lower: float = 0.5,
    threshold_upper: float = 1.0
) -> np.ndarray:
    """
    Load HDF5 probability map from Ilastik and threshold it.
    
    Args:
        h5_path: Path to HDF5 file
        dataset_name: Name of dataset in HDF5 file
        channel: Which channel to extract (1 = foreground probability in Ilastik)
        threshold_lower: Lower threshold for probability
        threshold_upper: Upper threshold for probability
    
    Returns:
        Binary mask as 5D numpy array (T, C=1, Z, Y, X) in TCZYX order
    """
    logger.info(f"Loading H5 file: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        if dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in {h5_path}")
        
        # Ilastik exports as (T, Z, Y, X, C) by default with axisorder=tzyxc
        data = f[dataset_name][:]
        logger.info(f"H5 data shape: {data.shape}")
        
        # Extract the specified channel and convert to TCZYX
        if data.ndim == 5:  # (T, Z, Y, X, C)
            # Extract channel and rearrange to TCZYX
            prob_map = data[:, :, :, :, channel]  # (T, Z, Y, X)
            prob_map = prob_map[:, np.newaxis, :, :, :]  # (T, C=1, Z, Y, X)
        elif data.ndim == 4:  # (Z, Y, X, C) - no T dimension
            prob_map = data[:, :, :, channel]  # (Z, Y, X)
            prob_map = prob_map[np.newaxis, np.newaxis, :, :, :]  # (T=1, C=1, Z, Y, X)
        elif data.ndim == 3:  # (Z, Y, X) - already single channel, no T
            prob_map = data[np.newaxis, np.newaxis, :, :, :]  # (T=1, C=1, Z, Y, X)
        else:
            raise ValueError(f"Unexpected data dimensions: {data.shape}")
        
        logger.info(f"Probability map shape (TCZYX): {prob_map.shape}, dtype: {prob_map.dtype}")
        logger.info(f"Probability range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")
    
    # Threshold the probability map
    binary_mask = (prob_map >= threshold_lower) & (prob_map <= threshold_upper)
    logger.info(f"Binary mask: {np.sum(binary_mask)} pixels above threshold")
    
    return binary_mask.astype(np.uint8)


def check_on_edge(mask: np.ndarray) -> bool:
    """
    Check if any non-zero pixels touch the XY edges.
    
    Args:
        mask: 2D binary mask (Y, X) - single slice
    
    Returns:
        True if any non-zero pixels are on the edges
    """
    # Check all four edges
    edges = [
        mask[0, :],    # Top
        mask[-1, :],   # Bottom
        mask[:, 0],    # Left
        mask[:, -1]    # Right
    ]
    
    for edge in edges:
        if np.any(edge > 0):
            return True
    
    return False


def get_mask_area(mask: np.ndarray) -> int:
    """Count non-zero pixels in mask."""
    return np.sum(mask > 0)


def get_largest_object(labeled_mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component.
    
    Args:
        labeled_mask: Labeled mask from scipy.ndimage.label
    
    Returns:
        Binary mask with only largest object
    """
    if labeled_mask.max() == 0:
        return labeled_mask
    
    # Count sizes of all labels
    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0  # Ignore background
    
    if len(sizes) <= 1:
        return labeled_mask
    
    # Get largest label
    largest_label = np.argmax(sizes)
    
    # Create binary mask with only largest object
    largest_mask = (labeled_mask == largest_label).astype(np.uint8)
    
    return largest_mask


def validate_object(
    mask: np.ndarray,
    minsize: int,
    maxsize: int,
    minperim: int,
    maxperim: int
) -> Tuple[bool, dict]:
    """
    Validate object against size and perimeter criteria.
    
    Args:
        mask: Binary 2D mask (Y, X) - single slice
        minsize: Minimum area in pixels (per XY slice)
        maxsize: Maximum area in pixels (per XY slice)
        minperim: Minimum perimeter (per XY slice)
        maxperim: Maximum perimeter (per XY slice)
    
    Returns:
        Tuple of (is_valid, metrics_dict)
    """
    metrics = {
        'area': 0,
        'perimeter': 0,
        'on_edge': False,
        'is_empty': False
    }
    
    # Check if empty
    if mask.max() == 0:
        metrics['is_empty'] = True
        logger.debug("Validation failed: Empty mask")
        return False, metrics
    
    # Check edges
    if check_on_edge(mask):
        metrics['on_edge'] = True
        logger.debug("Validation failed: Object touches edge")
        return False, metrics
    
    # Calculate area (2D area for this slice)
    area = get_mask_area(mask)
    metrics['area'] = area
    
    if area < minsize:
        logger.debug(f"Validation failed: area ({area}) < minsize ({minsize})")
        return False, metrics
    
    if area > maxsize:
        logger.debug(f"Validation failed: area ({area}) > maxsize ({maxsize})")
        return False, metrics
    
    # Get perimeter using skimage
    props = measure.regionprops((mask > 0).astype(int))
    if len(props) == 0:
        logger.debug("Validation failed: No region properties found")
        return False, metrics
    
    perimeter = props[0].perimeter
    metrics['perimeter'] = perimeter
    
    if perimeter < minperim:
        logger.debug(f"Validation failed: perimeter ({perimeter:.1f}) < minperim ({minperim})")
        return False, metrics
    
    if perimeter > maxperim:
        logger.debug(f"Validation failed: perimeter ({perimeter:.1f}) > maxperim ({maxperim})")
        return False, metrics
    
    logger.info(f"Validation passed: area={area}, perimeter={perimeter:.1f}")
    return True, metrics


def particle_filter(
    mask: np.ndarray,
    minsize: int,
    maxsize: Optional[int] = None,
    exclude_edges: bool = True
) -> np.ndarray:
    """
    Filter particles by size and optionally exclude edge-touching objects.
    Processes each 2D slice (T,Z) independently.
    
    Args:
        mask: Binary 2D mask (Y, X) - single slice
        minsize: Minimum object size (per XY slice)
        maxsize: Maximum object size (None for no limit, per XY slice)
        exclude_edges: Remove objects touching XY edges
    
    Returns:
        Filtered binary mask (Y, X)
    """
    filtered_mask = np.zeros_like(mask)
    
    if mask.max() == 0:
        return filtered_mask
    
    # Label connected components
    labeled, num_features = ndimage.label(mask)
    
    if num_features == 0:
        return filtered_mask
    
    # Get region properties
    props = measure.regionprops(labeled)
    
    for prop in props:
        label = prop.label
        area = prop.area
        
        # Size filter
        if area < minsize:
            continue
        if maxsize is not None and area > maxsize:
            continue
        
        # Edge filter
        if exclude_edges:
            # Check if this object touches edges
            obj_mask = (labeled == label)
            if (np.any(obj_mask[0, :]) or np.any(obj_mask[-1, :]) or
                np.any(obj_mask[:, 0]) or np.any(obj_mask[:, -1])):
                continue
        
        # Keep this object
        filtered_mask[labeled == label] = 1
    
    return filtered_mask


def erode_preserve_border(mask: np.ndarray) -> np.ndarray:
    """
    Erode mask (simple erosion for 2D slices).
    
    Args:
        mask: Binary 2D mask (Y, X)
    
    Returns:
        Eroded mask (Y, X)
    """
    # Simple erosion for 2D
    eroded = morphology.binary_erosion(mask)
    return eroded.astype(np.uint8)


def apply_convex_hull_per_slice(mask: np.ndarray) -> np.ndarray:
    """
    Apply convex hull to a 2D slice.
    
    Args:
        mask: Binary 2D mask (Y, X)
    
    Returns:
        Mask with convex hull applied (Y, X)
    """
    if mask.max() > 0:
        return convex_hull_image(mask > 0).astype(np.uint8)
    return mask.astype(np.uint8)


def grow_shrink_cycles(
    mask: np.ndarray,
    minsize: int,
    maxsize: int,
    minperim: int,
    maxperim: int,
    max_cycles: int = 10
) -> Tuple[Optional[np.ndarray], int]:
    """
    Try grow-shrink cycles to find a valid mask.
    
    Tests cycles from 1 to max_cycles and returns the first one that passes validation.
    
    Args:
        mask: Binary 2D mask (Y, X)
        minsize: Minimum area (per XY slice)
        maxsize: Maximum area (per XY slice)
        minperim: Minimum perimeter (per XY slice)
        maxperim: Maximum perimeter (per XY slice)
        max_cycles: Maximum number of grow/shrink cycles to try
    
    Returns:
        Tuple of (best_mask or None, num_cycles_used)
    """
    for cycles in range(1, max_cycles + 1):
        test_mask = mask.copy()
        
        # Grow
        for _ in range(cycles):
            test_mask = morphology.binary_dilation(test_mask)
        
        # Shrink
        for _ in range(cycles):
            test_mask = morphology.binary_erosion(test_mask)
        
        # Validate
        is_valid, metrics = validate_object(test_mask, minsize, maxsize, minperim, maxperim)
        
        if is_valid:
            logger.info(f"Grow-shrink validation passed at {cycles} cycles")
            return test_mask.astype(np.uint8), cycles
    
    logger.warning(f"Grow-shrink failed after {max_cycles} cycles")
    return None, max_cycles


def save_imagej_tif(
    mask: np.ndarray,
    output_path: str,
    as_binary: bool = True
) -> None:
    """
    Save mask as ImageJ-compatible TIFF.
    
    Args:
        mask: 5D mask (T, C, Z, Y, X) in TCZYX order
        output_path: Output file path
        as_binary: If True, convert to 0/255 binary, else keep as-is
    """
    if as_binary:
        # Convert to binary (0 or 255) 8-bit mask
        mask_out = (mask > 0).astype(np.uint8) * 255
    else:
        mask_out = mask.astype(np.uint8)
    
    # Remove C dimension (should be 1) and convert to TZYX for ImageJ
    if mask_out.shape[1] == 1:
        mask_out = mask_out[:, 0, :, :, :]  # (T, Z, Y, X)
    
    logger.info(f"Saving mask (TZYX): shape={mask_out.shape}, dtype={mask_out.dtype}, max={mask_out.max()}")
    
    # Save as ImageJ-compatible TIFF
    # ImageJ expects TZYX order for stacks
    tifffile.imwrite(
        output_path,
        mask_out,
        imagej=True,
        metadata={'axes': 'TZYX'},
        compression='deflate'
    )
    
    logger.info(f"Saved: {output_path}")


def fill_temporal_gaps(
    mask: np.ndarray,
    max_time_gap: int = 1,
    require_matching_labels: bool = True
) -> tuple[np.ndarray, int]:
    """
    Fill temporal gaps in mask data based on spatial overlap.
    
    If a pixel is background at time t, but has non-zero values at both t-gap and t+gap,
    fill it with the label from t-gap. Supports multi-gap filling.
    Additionally, fills gaps at the start and end by copying the closest non-zero label.
    
    This is useful when segmentation has failed in just a few frames but succeeded
    in adjacent frames. The function can bridge these gaps to maintain temporal continuity.
    
    Args:
        mask: 5D array in TCZYX order
        max_time_gap: Maximum temporal gap to fill (e.g., 2 means check up to t±2)
        require_matching_labels: If True, only fill if labels match at t-gap and t+gap.
                                If False, fill based on spatial overlap (any labels present).
                                Use True for indexed masks, False for binary masks (default: True).
    
    Returns:
        tuple: (filled_mask, total_pixels_filled)
    """
    fill_count = 0
    filled = mask.copy()
    
    for c in range(filled.shape[1]):
        for z in range(filled.shape[2]):
            if filled.shape[0] < (2 * max_time_gap + 1):  # Need enough time points
                continue
            
            # Iterate through each time point
            for t in range(filled.shape[0]):
                t_current = filled[t, c, z, :, :]
                
                # Only process background pixels
                is_background = (t_current == 0)
                if not np.any(is_background):
                    continue
                
                # Check for labels within max_time_gap frames before and after
                for gap in range(1, max_time_gap + 1):
                    t_before_idx = t - gap
                    t_after_idx = t + gap
                    
                    # Check if indices are valid
                    if t_before_idx < 0 and t_after_idx < filled.shape[0]:
                        # Fill gaps at the start by copying the closest non-zero label
                        t_after = filled[t_after_idx, c, z, :, :]
                        fill_mask = is_background & (t_after > 0)
                        t_current[fill_mask] = t_after[fill_mask]
                        fill_count += np.sum(fill_mask)
                        is_background = (t_current == 0)
                        if not np.any(is_background):
                            break
                    
                    elif t_after_idx >= filled.shape[0] and t_before_idx >= 0:
                        # Fill gaps at the end by copying the closest non-zero label
                        t_before = filled[t_before_idx, c, z, :, :]
                        fill_mask = is_background & (t_before > 0)
                        t_current[fill_mask] = t_before[fill_mask]
                        fill_count += np.sum(fill_mask)
                        is_background = (t_current == 0)
                        if not np.any(is_background):
                            break
                    
                    elif 0 <= t_before_idx < filled.shape[0] and 0 <= t_after_idx < filled.shape[0]:
                        t_before = filled[t_before_idx, c, z, :, :]
                        t_after = filled[t_after_idx, c, z, :, :]
                        
                        # Find pixels that:
                        # 1. Are background at current time
                        # 2. Have non-zero values at both t-gap and t+gap
                        # 3. (Optional) Labels match at t-gap and t+gap
                        has_before = (t_before > 0)
                        has_after = (t_after > 0)
                        
                        if require_matching_labels:
                            # For indexed masks: require matching labels to prevent merging different objects
                            match_mask = (t_before == t_after) & (t_before > 0)
                            fill_mask = is_background & match_mask
                        else:
                            # For binary masks: fill based on spatial overlap
                            fill_mask = is_background & has_before & has_after
                        
                        # Count and fill
                        n_filled = np.sum(fill_mask)
                        if n_filled > 0:
                            # Fill with the label from t-gap
                            t_current[fill_mask] = t_before[fill_mask]
                            filled[t, c, z, :, :] = t_current
                            fill_count += n_filled
                            
                            # Update is_background for next gap iteration
                            is_background = (t_current == 0)
                            if not np.any(is_background):
                                break  # No more background pixels to fill
    
    return filled, fill_count


def process_single_file(
    h5_path: str,
    output_dir: str,
    minsize: int = 24000,
    maxsize: int = 120000,
    minperim: int = 750,
    maxperim: int = 1500,
    max_watershed_rounds: int = 5,
    max_grow_shrink_cycles: int = 10,
    max_time_gap: int = -1,
    debug_all: bool = False,
    debug_failed: bool = False,
    image_path: Optional[str] = None,
    yaml_path: Optional[str] = None
) -> bool:
    """
    Process a single Ilastik probability map HDF5 file.
    
    Processes each T,Z slice independently following ImageJ "stack" convention.
    All size/perimeter criteria are per XY slice.
    
    This is the main processing function that implements the full pipeline:
    1. Load and threshold probability map (TCZYX format)
    2. Loop over T and Z, process each 2D slice:
       a. Fill holes and filter small objects
       b. Get largest connected component
       c. Iterative watershed with erosion/dilation if needed
       d. Validation against size/perimeter criteria
       e. Convex hull fallback if main pipeline fails
       f. Grow-shrink cycles if perimeter is problematic
    3. Fill temporal gaps if max_time_gap > 0
    4. Save result or failure file
    
    Args:
        h5_path: Path to input HDF5 probability map file
        output_dir: Output directory
        minsize: Minimum object area (per XY slice)
        maxsize: Maximum object area (per XY slice)
        minperim: Minimum perimeter (per XY slice)
        maxperim: Maximum perimeter (per XY slice)
        max_watershed_rounds: Maximum watershed iterations
        max_grow_shrink_cycles: Maximum grow/shrink cycles
        max_time_gap: Fill temporal gaps up to this many frames (e.g., 1 or 2). 
                      Default -1 (disabled). Fills gaps where segmentation failed
                      in a few frames but succeeded in adjacent frames.
        debug_all: If True, show all results (both successes and failures) with image overlay
        debug_failed: If True, show only failed results with image overlay
        image_path: Optional path to original image for visualization
        yaml_path: Optional path to metadata YAML file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        basename = os.path.splitext(os.path.basename(h5_path))[0]
        output_path = os.path.join(output_dir, f"{basename}.tif")
        output_fail = os.path.join(output_dir, f"{basename}_fail.tif")
        
        # Check if output already exists
        if os.path.exists(output_path) or os.path.exists(output_fail):
            logger.info(f"Output exists, skipping: {basename}")
            return True
        
        logger.info(f"Processing: {basename}")
        
        # Load real image if provided for visualization
   

        real_image = None
        if image_path and HAS_RP:
            try:
                logger.info(f"Loading real image: {image_path}")
                real_image = rp.load_tczyx_image(image_path)
            except Exception as e:
                logger.warning(f"Could not load real image: {e}")
        
        # Debug: Show real image if available
        # if debug and HAS_RP and real_image is not None:
        #     rp.show_image(image=real_image, title=f"Real Image: {basename}")



        # STEP 1: Load and threshold H5 probability map -> (T, C, Z, Y, X)
        mask_tczyx = load_h5_probability_map(h5_path, channel=1, threshold_lower=0.5)
        
        T, C, Z, Y, X = mask_tczyx.shape
        logger.info(f"Mask shape (TCZYX): {mask_tczyx.shape}")
        

        
        # Output mask in same shape
        output_mask = np.zeros_like(mask_tczyx)
        
        # Track overall success
        all_slices_succeeded = True
        
        # STEP 2: Process each T,Z slice independently
        for t in range(T):
            for z in range(Z):
                slice_label = f"T={t}, Z={z}"
                logger.info(f"Processing {slice_label}")
                
                # Extract 2D slice (Y, X)
                mask = mask_tczyx[t, 0, z, :, :].copy()
                
                area = get_mask_area(mask)
                logger.info(f"  Initial area: {area}")
                
                if area < minsize:
                    logger.warning(f"  Initial area ({area}) < minsize ({minsize}), skipping slice")
                    all_slices_succeeded = False
                    continue
                
                # STEP 2a: Fill holes
                logger.debug(f"  Filling holes...")
                mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
                
                area = get_mask_area(mask)
                logger.debug(f"  Area after fill holes: {area}")
                
                # STEP 2b: Particle filter - remove small objects and edge-touching
                logger.debug(f"  Filtering small particles and edges...")
                mask = particle_filter(mask, minsize=200, exclude_edges=True)
                
                # STEP 2c: Connected components - keep largest
                logger.debug(f"  Finding largest connected component...")
                labeled, num_features = ndimage.label(mask)
                mask = get_largest_object(labeled)
                
                area = get_mask_area(mask)
                logger.debug(f"  Largest object area: {area}")
                
                # STEP 2d: Iterative watershed if needed
                best_mask = mask.copy()
                
                for i in range(max_watershed_rounds):
                    logger.debug(f"  Watershed round {i+1}/{max_watershed_rounds}")
                    
                    area = get_mask_area(best_mask)
                    logger.debug(f"    Current area: {area}")
                    
                    # Check if we're done
                    if not check_on_edge(best_mask) and area < maxsize:
                        logger.debug(f"    Success! No edges and area within bounds at round {i+1}")
                        break
                    
                    # Apply watershed
                    logger.debug("    Applying watershed...")
                    distance = ndimage.distance_transform_edt(best_mask)
                    
                    # Find peaks for watershed seeds
                    local_max = peak_local_max(
                        distance,
                        min_distance=1,
                        threshold_rel=0.5,
                        labels=best_mask
                    )
                    
                    if len(local_max) == 0:
                        logger.warning("    No peaks found for watershed, stopping")
                        break
                    
                    markers = np.zeros_like(best_mask, dtype=int)
                    for idx, peak in enumerate(local_max):
                        markers[tuple(peak)] = idx + 1
                    
                    # Perform watershed
                    labels_ws = watershed(-distance, markers, mask=best_mask)
                    
                    # Filter by size
                    logger.debug("    Filtering watershed result...")
                    best_mask = particle_filter(labels_ws > 0, minsize=minsize, exclude_edges=True)
                    
                    # Erosion rounds (increasing with iteration)
                    for j in range(i):
                        best_mask = erode_preserve_border(best_mask)
                    
                    # Filter again
                    best_mask = particle_filter(best_mask, minsize=200, exclude_edges=True)
                    
                    # Dilate then erode to merge
                    best_mask = morphology.binary_dilation(best_mask)
                    best_mask = morphology.binary_erosion(best_mask)
                    
                    # Get largest component again
                    labeled, _ = ndimage.label(best_mask)
                    best_mask = get_largest_object(labeled)
                    
                    # Dilate back
                    for j in range(i + 1):
                        best_mask = morphology.binary_dilation(best_mask)
                
                # STEP 2e: Final filtering
                logger.debug("  Final particle filter...")
                best_mask = particle_filter(best_mask, minsize=minsize, maxsize=maxsize, exclude_edges=True)
                
                # STEP 2f: Validate
                is_valid, metrics = validate_object(best_mask, minsize, maxsize, minperim, maxperim)
                
                if is_valid:
                    logger.info(f"  {slice_label} validation passed!")
                    output_mask[t, 0, z, :, :] = best_mask
                    continue
                
                logger.warning(f"  {slice_label} validation failed after watershed rounds")
                
                # STEP 2g: Fallback - try convex hull
                logger.debug("  Trying convex hull fallback...")
                fallback_mask = apply_convex_hull_per_slice(best_mask)
                fallback_mask = particle_filter(fallback_mask, minsize=minsize, maxsize=maxsize, exclude_edges=True)
                
                is_valid, metrics = validate_object(fallback_mask, minsize, maxsize, minperim, maxperim)
                
                if is_valid:
                    logger.info(f"  {slice_label} convex hull fallback passed!")
                    output_mask[t, 0, z, :, :] = fallback_mask
                    continue
                
                # STEP 2h: Try grow-shrink cycles if perimeter is the issue
                if metrics.get('perimeter', 0) > maxperim or metrics.get('perimeter', 0) < minperim:
                    logger.debug("  Perimeter out of range, trying grow-shrink cycles...")
                    grow_shrink_mask, cycles = grow_shrink_cycles(
                        fallback_mask, minsize, maxsize, minperim, maxperim, max_grow_shrink_cycles
                    )
                    
                    if grow_shrink_mask is not None:
                        logger.info(f"  {slice_label} grow-shrink passed with {cycles} cycles!")
                        output_mask[t, 0, z, :, :] = grow_shrink_mask
                        continue
                
                # All methods failed for this slice
                logger.error(f"  {slice_label} all methods failed")
                output_mask[t, 0, z, :, :] = fallback_mask
                all_slices_succeeded = False
        
        # STEP 2i: Fill temporal gaps if enabled
        if max_time_gap > 0:
            logger.info(f"Filling temporal gaps (max_gap={max_time_gap})...")
            # For binary masks (0 or 255), we don't require matching labels
            output_mask, temporal_fill_count = fill_temporal_gaps(
                output_mask, 
                max_time_gap, 
                require_matching_labels=False
            )
            logger.info(f"Filled {temporal_fill_count} pixels in temporal gaps")
            
            # If temporal filling recovered some failed slices, update success status
            if temporal_fill_count > 0 and not all_slices_succeeded:
                logger.info("Re-validating after temporal gap filling...")
                # Quick check if we now have complete coverage
                has_content = np.sum(output_mask > 0, axis=(2, 3, 4))  # Sum over Z, Y, X
                all_slices_have_content = np.all(has_content > 0)
                if all_slices_have_content:
                    logger.info("Temporal gap filling recovered all slices!")
                    all_slices_succeeded = True
        
        # STEP 3: Save results
        if all_slices_succeeded:
            logger.info("All slices validated successfully! Saving mask...")
            save_imagej_tif(output_mask, output_path)
            
            # Show visualization if debug_all is enabled
            if debug_all and HAS_RP:
                if real_image is not None:
                    rp.show_image(image=real_image, mask=output_mask, title=f"SUCCESS: {basename}", timer=1.0)
                else:
                    rp.show_image(image=output_mask, mask=output_mask, title=f"SUCCESS: {basename}", timer=1.0)
            
            return True
        else:
            logger.error("Some slices failed validation, saving as fail file")
            save_imagej_tif(output_mask, output_fail)
            
            # Show visualization if debug_all or debug_failed is enabled
            if (debug_all or debug_failed) and HAS_RP:
                if real_image is not None:
                    rp.show_image(image=real_image, mask=output_mask, title=f"FAILED: {basename}", timer=1.0)
                else:
                    rp.show_image(image=output_mask, mask=output_mask, title=f"FAILED: {basename}", timer=1.0)
            
            return False
        
    except Exception as e:
        logger.error(f"Error processing {h5_path}: {e}", exc_info=True)
        return False


def process_files(
    image_pattern: str,
    h5_pattern: str,
    output_folder: Optional[str] = None,
    yaml_pattern: Optional[str] = None,
    minsize: int = 24000,
    maxsize: int = 120000,
    minperim: int = 750,
    maxperim: int = 1500,
    max_watershed_rounds: int = 5,
    max_grow_shrink_cycles: int = 10,
    max_time_gap: int = -1,
    debug_all: bool = False,
    debug_failed: bool = False,
    dry_run: bool = False
) -> None:
    """
    Process multiple files using grouped search patterns.
    
    Uses get_grouped_files_to_process to match images, H5 probability maps,
    and optional YAML metadata files by their common basename.
    
    Args:
        image_pattern: Glob pattern for original images (e.g., 'data/**/*.tif')
        h5_pattern: Glob pattern for H5 probability maps (e.g., 'prob/**/*_Probabilities.h5')
        output_folder: Output directory (default: same as image folder + '_masks')
        yaml_pattern: Optional glob pattern for YAML metadata files
        minsize: Minimum object area
        maxsize: Maximum object area
        minperim: Minimum perimeter
        maxperim: Maximum perimeter
        max_watershed_rounds: Maximum watershed iterations
        max_grow_shrink_cycles: Maximum grow/shrink cycles
        max_time_gap: Fill temporal gaps up to this many frames (default: -1, disabled)
        debug_all: Show all results (both successes and failures) with image overlay
        debug_failed: Show only failed results with image overlay
        dry_run: Only print planned actions without executing
    """
    if not HAS_RP:
        logger.error("bioimage_pipeline_utils is required for this module")
        return
    
    # Build search patterns dictionary
    search_patterns = {
        'image': image_pattern,
        'h5': h5_pattern
    }
    if yaml_pattern:
        search_patterns['yaml'] = yaml_pattern
    
    # Find and group files
    search_subfolders = '**' in image_pattern or '**' in h5_pattern
    grouped_files = rp.get_grouped_files_to_process(search_patterns, search_subfolders)
    
    if not grouped_files:
        logger.error("No matching file groups found")
        return
    
    logger.info(f"Found {len(grouped_files)} file group(s) to process")
    
    # Determine output folder
    if output_folder is None:
        # Use directory of first image file + '_masks'
        first_group = next(iter(grouped_files.values()))
        if 'image' in first_group:
            base_dir = str(Path(first_group['image']).parent)
            output_folder = base_dir + "_masks"
        else:
            output_folder = "output_masks"
    
    logger.info(f"Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    # Dry run - just print plans
    if dry_run:
        print(f"[DRY RUN] Would process {len(grouped_files)} file groups")
        print(f"[DRY RUN] Output folder: {output_folder}")
        for basename, files in grouped_files.items():
            print(f"\n[DRY RUN] Group: {basename}")
            for key, path in files.items():
                print(f"  {key}: {path}")
        return
    
    # Process each file group
    success_count = 0
    fail_count = 0
    
    for basename, files in grouped_files.items():
        # Check if we have required files
        if 'h5' not in files:
            logger.warning(f"Skipping {basename}: no H5 file found")
            continue
        
        h5_path = files['h5']
        image_path = files.get('image', None)
        yaml_path = files.get('yaml', None)
        
        logger.info(f"Processing group: {basename}")
        if image_path:
            logger.info(f"  Image: {image_path}")
        logger.info(f"  H5: {h5_path}")
        if yaml_path:
            logger.info(f"  YAML: {yaml_path}")
        
        success = process_single_file(
            h5_path=h5_path,
            output_dir=output_folder,
            minsize=minsize,
            maxsize=maxsize,
            minperim=minperim,
            maxperim=maxperim,
            max_watershed_rounds=max_watershed_rounds,
            max_grow_shrink_cycles=max_grow_shrink_cycles,
            max_time_gap=max_time_gap,
            debug_all=debug_all,
            debug_failed=debug_failed,
            image_path=image_path,
            yaml_path=yaml_path
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    logger.info(f"Processing complete: {success_count} successful, {fail_count} failed")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Clean up Ilastik probability maps and create cell masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Process Ilastik probability maps with real images
  environment: uv@3.11:ilastik-cleanup
  commands:
  - python
  - '%REPO%/standard_code/python/ilastik_cleanup.py'
  - --image-search-pattern: '%YAML%/images/**/*.tif'
  - --probabilities-search-pattern: '%YAML%/probabilities/**/*_Probabilities.h5'
  - --output-folder: '%YAML%/masks'
  - --debug-all
  
- name: Process with metadata and custom parameters
  environment: uv@3.11:ilastik-cleanup
  commands:
  - python
  - '%REPO%/standard_code/python/ilastik_cleanup.py'
  - --image-search-pattern: '%YAML%/images/**/*.tif'
  - --probabilities-search-pattern: '%YAML%/probabilities/**/*_Probabilities.h5'
  - --yaml-search-pattern: '%YAML%/metadata/**/*_metadata.yaml'
  - --output-folder: '%YAML%/masks'
  - --minsize: 20000
  - --maxsize: 150000
  - --max-time-gap: 2

Examples:
  # Process with image overlay
  python ilastik_cleanup.py \\
    --image-search-pattern "data/images/*.tif" \\
    --probabilities-search-pattern "data/ilastik/*_Probabilities.h5"
  
  # With metadata and debug visualization (all results)
  python ilastik_cleanup.py \\
    --image-search-pattern "data/**/*.tif" \\
    --probabilities-search-pattern "data/**/*_Probabilities.h5" \\
    --yaml-search-pattern "data/**/*_metadata.yaml" \\
    --debug-all
  
  # Show only failed results
  python ilastik_cleanup.py \\
    --image-search-pattern "data/**/*.tif" \\
    --probabilities-search-pattern "data/**/*_Probabilities.h5" \\
    --debug-failed
  
  # Fill temporal gaps (useful when segmentation fails in a few frames)
  python ilastik_cleanup.py \\
    --image-search-pattern "data/**/*.tif" \\
    --probabilities-search-pattern "data/**/*_Probabilities.h5" \\
    --max-time-gap 2
  
  # Dry run to preview file matching
  python ilastik_cleanup.py \\
    --image-search-pattern "data/**/*.tif" \\
    --probabilities-search-pattern "data/**/*.h5" \\
    --dry-run
        """
    )
    
    parser.add_argument(
        "--image-search-pattern",
        type=str,
        required=True,
        help="Search pattern for original images (e.g., 'data/**/*.tif')"
    )
    
    parser.add_argument(
        "--probabilities-search-pattern",
        type=str,
        required=True,
        help="Search pattern for Ilastik H5 probability maps (e.g., 'prob/**/*_Probabilities.h5')"
    )
    
    parser.add_argument(
        "--yaml-search-pattern",
        type=str,
        default=None,
        help="Optional search pattern for metadata YAML files (e.g., 'metadata/**/*_metadata.yaml')"
    )
    
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder (default: image folder + '_masks')"
    )
    
    parser.add_argument(
        "--minsize",
        type=int,
        default=24000,
        help="Minimum object area in pixels per XY slice (default: 24000)"
    )
    parser.add_argument(
        "--maxsize",
        type=int,
        default=120000,
        help="Maximum object area in pixels per XY slice (default: 120000)"
    )
    parser.add_argument(
        "--minperim",
        type=int,
        default=750,
        help="Minimum perimeter per XY slice (default: 750)"
    )
    parser.add_argument(
        "--maxperim",
        type=int,
        default=1500,
        help="Maximum perimeter per XY slice (default: 1500)"
    )
    parser.add_argument(
        "--max-watershed-rounds",
        type=int,
        default=5,
        help="Maximum watershed iterations (default: 5)"
    )
    parser.add_argument(
        "--max-grow-shrink-cycles",
        type=int,
        default=10,
        help="Maximum grow/shrink cycles (default: 10)"
    )
    
    parser.add_argument(
        "--max-time-gap",
        type=int,
        default=-1,
        help="Fill temporal gaps up to this many frames (e.g., 1 or 2). Default -1 (disabled). "
             "Useful when segmentation failed in a few frames but succeeded in adjacent frames."
    )
    
    parser.add_argument(
        "--debug-all",
        action="store_true",
        help="Show all results (both successes and failures) with image overlay for 1 second each"
    )
    
    parser.add_argument(
        "--debug-failed",
        action="store_true",
        help="Show only failed results with image overlay for 1 second each"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned file groupings without executing"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Process files
    process_files(
        image_pattern=args.image_search_pattern,
        h5_pattern=args.probabilities_search_pattern,
        output_folder=args.output_folder,
        yaml_pattern=args.yaml_search_pattern,
        minsize=args.minsize,
        maxsize=args.maxsize,
        minperim=args.minperim,
        maxperim=args.maxperim,
        max_watershed_rounds=args.max_watershed_rounds,
        max_grow_shrink_cycles=args.max_grow_shrink_cycles,
        max_time_gap=args.max_time_gap,
        debug_all=args.debug_all,
        debug_failed=args.debug_failed,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
