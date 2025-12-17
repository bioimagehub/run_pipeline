"""

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
import bioimage_pipeline_utils as rp

# Module-level logger
logger = logging.getLogger(__name__)


def load_h5_probability_map(
    h5_path: str,
    dataset_name: str = "/exported_data",
    channel: int = 1,
    threshold_lower: float = 0.3,
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


def _save_failed_file(h5_path: str, output_dir: str, output_mask: Optional[np.ndarray] = None) -> None:
    """
    Save a _failed file when segmentation fails.
    
    Attempts to save original probability map from H5, falls back to processed mask.
    
    Args:
        h5_path: Path to original H5 file
        output_dir: Output directory for _failed file
        output_mask: Optional processed mask to save as fallback
    """
    basename = os.path.splitext(os.path.basename(h5_path))[0]
    
    # Build output filename
    if basename.endswith("_Probabilities"):
        base_without_prob = basename[:-len("_Probabilities")]
        output_fail = os.path.join(output_dir, f"{base_without_prob}_Probabilities_failed.tif")
    else:
        output_fail = os.path.join(output_dir, f"{basename}_Probabilities_failed.tif")
    
    try:
        # Try to reload and save original probability map
        with h5py.File(h5_path, 'r') as f:
            dataset_name = "/exported_data"
            if dataset_name in f:
                data = f[dataset_name][:]
                
                # Convert to TCZYX
                if data.ndim == 5:  # (T, Z, Y, X, C)
                    prob_data = data[:, :, :, :, 1]
                    prob_data = prob_data[:, np.newaxis, :, :, :]
                elif data.ndim == 4:  # (Z, Y, X, C)
                    prob_data = data[:, :, :, 1]
                    prob_data = prob_data[np.newaxis, np.newaxis, :, :, :]
                elif data.ndim == 3:  # (Z, Y, X)
                    prob_data = data[np.newaxis, np.newaxis, :, :, :]
                else:
                    raise ValueError(f"Unexpected data shape: {data.shape}")
                
                # Convert probabilities (0-1 float) to 0-255 uint8
                prob_uint8 = (prob_data * 255).astype(np.uint8)
                save_imagej_tif(prob_uint8, output_fail, as_binary=False)
                logger.info(f"Saved failed file (original probabilities): {output_fail}")
                return
    except Exception as e:
        logger.warning(f"Could not save original probabilities: {e}")
    
    # Fallback: save the processed mask if available
    if output_mask is not None:
        try:
            save_imagej_tif(output_mask, output_fail, as_binary=True)
            logger.info(f"Saved failed file (processed mask): {output_fail}")
        except Exception as e:
            logger.error(f"Could not save failed file: {e}")

def fill_holes(mask_tczyx: np.ndarray) -> np.ndarray:
    """Fill holes in each 2D slice independently."""
    T, C, Z, Y, X = mask_tczyx.shape
    output = np.zeros_like(mask_tczyx)
    
    for t in range(T):
        for z in range(Z):
            output[t, 0, z, :, :] = ndimage.binary_fill_holes(
                mask_tczyx[t, 0, z, :, :]
            ).astype(np.uint8)
    
    return output

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
    output_suffix: str = "",
    debug_all: bool = False,
    debug_failed: bool = False,
    image_path: Optional[str] = None,
    yaml_path: Optional[str] = None
) -> bool:
    """
    Process a single Ilastik probability map HDF5 file.
        
    Args:
        h5_path: Path to HDF5 probability map
        output_dir: Output directory for results
        minsize: Minimum object area (per XY slice)
        maxsize: Maximum object area (per XY slice)
        minperim: Minimum perimeter (per XY slice)
        maxperim: Maximum perimeter (per XY slice)
        max_watershed_rounds: Maximum watershed iterations
        max_grow_shrink_cycles: Maximum grow/shrink cycles
        max_time_gap: Fill temporal gaps up to this many frames (-1 = disabled)
        output_suffix: Suffix for successful output filenames
        debug_all: Show all results (successes and failures)
        debug_failed: Show only failed results
        image_path: Optional path to real image for visualization
        yaml_path: Optional path to YAML metadata
    
    Returns:
        True if segmentation succeeded, False otherwise
    """
    basename = os.path.splitext(os.path.basename(h5_path))[0]
    
    # Build output paths
    if output_suffix:
        output_path = os.path.join(output_dir, f"{basename}{output_suffix}.tif")
    else:
        output_path = os.path.join(output_dir, f"{basename}.tif")
    
    if basename.endswith("_Probabilities"):
        base_without_prob = basename[:-len("_Probabilities")]
        output_fail = os.path.join(output_dir, f"{base_without_prob}_Probabilities_failed.tif")
    else:
        output_fail = os.path.join(output_dir, f"{basename}_Probabilities_failed.tif")
    
    # Check if already processed
    if os.path.exists(output_path) or os.path.exists(output_fail):
        logger.info(f"Output exists, skipping: {basename}")
        return True
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing: {basename}")
    logger.info(f"{'='*70}")
    
    # Load real image if provided
    real_image = None
    if image_path:
        try:
            logger.info(f"Loading real image: {image_path}")
            real_image = rp.load_tczyx_image(image_path)
        except Exception as e:
            logger.warning(f"Could not load real image: {e}")
    
    try:
        # Load and threshold probability map
        logger.info("Loading and thresholding probability map...")
        mask_tczyx = load_h5_probability_map(h5_path, channel=1, threshold_lower=0.3, threshold_upper=1.0)
        T, C, Z, Y, X = mask_tczyx.shape
        logger.info(f"Loaded mask shape (TCZYX): {mask_tczyx.shape}")
        
        # make a use connected components analysis for TZYX

        # Fill holes
        mask_tczyx = fill_holes(mask_tczyx)  


        # Prepare output containers
        labeled_tczyx = np.zeros_like(mask_tczyx, dtype=np.int32)

        # 4D connectivity: consider neighbors in all four dimensions (3x3x3x3)
        structure4d = np.ones((3, 3, 3, 3), dtype=np.uint8)

        # Run per-channel (typically C=1)
        total_components = 0
        for c in range(C):
            # Foreground in 4D: (T, Z, Y, X)
            fg_4d = mask_tczyx[:, c, :, :, :].astype(bool)
            labeled_4d, n_comp = ndimage.label(fg_4d, structure=structure4d)
            labeled_tczyx[:, c, :, :, :] = labeled_4d
            total_components += n_comp
            logger.info(f"Channel {c}: found {n_comp} connected component(s)")

        #rp.show_image(real_image, labeled_tczyx, title=f"Processed: {basename}")

        
        for t in range(T):
            for z in range(Z):
                slice_2d = labeled_tczyx[t, 0, z, :, :]
                props = measure.regionprops(slice_2d)
                for prop in props:
                    

                    prop_on_edge = check_on_edge(slice_2d == prop.label)

                    # 1 Remove labelled masks that are too small
                    if prop.area < minsize:
                        logger.debug(f"Removing small object {prop.label} (area={prop.area}) at T={t},Z={z}")
                        slice_2d[slice_2d == prop.label] = 0

                    # Watershed masks that are too large or touches edges (ImageJ-style EDM watershed)
                    elif prop.area > maxsize or prop_on_edge:
                        logger.debug(f"Applying watershed to split large/edge object {prop.label} (area={prop.area}) at T={t},Z={z}")
                        obj_mask = (slice_2d == prop.label).astype(bool)
                        
                        # EDM: Euclidean Distance Map (same as ImageJ)
                        distance = ndimage.distance_transform_edt(obj_mask)
                        
                        # Find maxima of EDM (ImageJ uses tolerance ~0.5)
                        coords = peak_local_max(distance, min_distance=int(np.sqrt(minsize)/2), 
                                               labels=obj_mask, exclude_border=False)
                        
                        if len(coords) > 1:
                            # Create markers from maxima
                            mask_markers = np.zeros(distance.shape, dtype=np.int32)
                            mask_markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
                            
                            # Watershed segmentation from maxima (negative EDM as elevation)
                            labels = watershed(-distance, mask_markers, mask=obj_mask)
                            
                            # Replace original with splits (validate: minsize AND not on edge)
                            slice_2d[obj_mask] = 0
                            next_label = int(slice_2d.max()) + 1
                            for region in measure.regionprops(labels):
                                split_mask = (labels == region.label)
                                # Only keep if: size OK AND doesn't touch edges
                                if region.area >= minsize and not check_on_edge(split_mask):
                                    slice_2d[split_mask] = next_label
                                    next_label += 1
                            logger.debug(f"Watershed split {prop.label} (area={prop.area}) → {len(coords)} parts, kept {next_label - int(slice_2d.max()) - 1} valid at T={t},Z={z}")
                        else:
                            slice_2d[obj_mask] = 0  # Can't split, remove


                labeled_tczyx[t, 0, z, :, :] = slice_2d     
        
        #rp.show_image(real_image, labeled_tczyx, title=f"After: {basename}")
        
        
        # Binary output mask (components merged to 1)
        output_mask = (labeled_tczyx > 0).astype(np.uint8)

        
        # Apply 2D particle filtering per slice
        logger.info("Applying 2D particle filtering per slice...")
        for t in range(T):
            for z in range(Z):
                slice_2d = output_mask[t, 0, z, :, :]
                filtered_slice =  particle_filter(
                    slice_2d,
                    minsize=minsize,
                    maxsize=maxsize,
                    exclude_edges=True

                )
                output_mask[t, 0, z, :, :] = filtered_slice

        rp.show_image(real_image, output_mask, title=f"After: {basename}")




        # Save results
        if np.sum(output_mask) > 0:
            logger.info(f"\n{'='*70}")
            logger.info("✓ SUCCESS: All slices passed validation")
            logger.info(f"{'='*70}")
            save_imagej_tif(output_mask, output_path)
            
            if debug_all:
                if real_image is not None:
                    rp.show_image(image=real_image, mask=output_mask, 
                                 title=f"SUCCESS: {basename}", timer=1.0)
            
            return True
        
        else:
            logger.info(f"\n{'='*70}")
            logger.info("✗ FAILED: Some slices failed validation")
            logger.info(f"{'='*70}")
            _save_failed_file(h5_path, output_dir, output_mask)
            
            if debug_all or debug_failed:
                if real_image is not None:
                    rp.show_image(image=real_image, mask=output_mask, 
                                 title=f"FAILED: {basename}", timer=1.0)
            
            return False
    
    except Exception as e:
        logger.error(f"\n{'='*70}")
        logger.error(f"✗ EXCEPTION: {type(e).__name__}: {e}")
        logger.error(f"{'='*70}")
        logger.error(f"Detailed traceback:", exc_info=True)
        
        # Guarantee file output even on exception
        _save_failed_file(h5_path, output_dir, None)
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
    output_suffix: str = "",
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
        output_suffix: Suffix to add to successful output filenames (default: "")
        debug_all: Show all results (both successes and failures) with image overlay
        debug_failed: Show only failed results with image overlay
        dry_run: Only print planned actions without executing
    """
    
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
            output_suffix=output_suffix,
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
  - '%REPO%/standard_code/python/segment_ilastik_cleanup.py'
  - --image-search-pattern: '%YAML%/images/**/*.tif'
  - --probabilities-search-pattern: '%YAML%/probabilities/**/*_Probabilities.h5'
  - --output-folder: '%YAML%/masks'
  - --debug-all
  
- name: Process with metadata and custom parameters
  environment: uv@3.11:ilastik-cleanup
  commands:
  - python
  - '%REPO%/standard_code/python/segment_ilastik_cleanup.py'
  - --image-search-pattern: '%YAML%/images/**/*.tif'
  - --probabilities-search-pattern: '%YAML%/probabilities/**/*_Probabilities.h5'
  - --yaml-search-pattern: '%YAML%/metadata/**/*_metadata.yaml'
  - --output-folder: '%YAML%/masks'
  - --minsize: 20000
  - --maxsize: 150000
  - --max-time-gap: 2

Examples:
  # Process with image overlay
  python segment_ilastik_cleanup.py \\
    --image-search-pattern "data/images/*.tif" \\
    --probabilities-search-pattern "data/ilastik/*_Probabilities.h5"
  
  # With metadata and debug visualization (all results)
  python segment_ilastik_cleanup.py \\
    --image-search-pattern "data/**/*.tif" \\
    --probabilities-search-pattern "data/**/*_Probabilities.h5" \\
    --yaml-search-pattern "data/**/*_metadata.yaml" \\
    --debug-all
  
  # Show only failed results
  python segment_ilastik_cleanup.py \\
    --image-search-pattern "data/**/*.tif" \\
    --probabilities-search-pattern "data/**/*_Probabilities.h5" \\
    --debug-failed
  
  # Fill temporal gaps (useful when segmentation fails in a few frames)
  python segment_ilastik_cleanup.py \\
    --image-search-pattern "data/**/*.tif" \\
    --probabilities-search-pattern "data/**/*_Probabilities.h5" \\
    --max-time-gap 2
  
  # Dry run to preview file matching
  python segment_ilastik_cleanup.py \\
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
        "--output-suffix",
        type=str,
        default="",
        help="Suffix to add to successful output filenames (default: no suffix). "
             "Example: '_cleaned' or '_processed'. Failed outputs always get '_Probabilities_failed' suffix."
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
        output_suffix=args.output_suffix,
        debug_all=args.debug_all,
        debug_failed=args.debug_failed,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
