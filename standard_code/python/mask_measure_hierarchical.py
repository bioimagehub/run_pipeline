"""
Hierarchical Mask Measurement Module

This module measures objects within other objects using a hierarchical approach.
The first mask defines parent objects (e.g., nuclei), and subsequent masks define
child objects (e.g., spots or structures within nuclei).

Author: BIPHUB - Bioimage Informatics Hub, University of Oslo
License: MIT
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import measure
from typing import List, Tuple, Dict
import logging

# Local imports
import bioimage_pipeline_utils as rp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def measure_region_stats(image_data: np.ndarray, mask: np.ndarray, region_id: int) -> Dict[str, float]:
    """
    Measure statistics for a specific region in the image.
    
    Args:
        image_data: 2D or 3D numpy array containing image intensities
        mask: 2D or 3D numpy array containing the mask (same shape as image_data)
        region_id: The label value to measure
        
    Returns:
        Dictionary containing mean, min, max, std, sum, and area statistics
    """
    region_mask = (mask == region_id)
    if not np.any(region_mask):
        return {
            'mean': np.nan,
            'min': np.nan,
            'max': np.nan,
            'std': np.nan,
            'sum': np.nan,
            'area': 0
        }
    
    region_pixels = image_data[region_mask]
    return {
        'mean': float(np.mean(region_pixels)),
        'min': float(np.min(region_pixels)),
        'max': float(np.max(region_pixels)),
        'std': float(np.std(region_pixels)),
        'sum': float(np.sum(region_pixels)),
        'area': int(np.sum(region_mask))
    }


def measure_hierarchical_masks(
    image_path: str,
    mask_paths: List[str],
    output_folder: str,
    base_name: str,
    channel_index: int = None
) -> None:
    """
    Measure objects in a hierarchical manner where the first mask defines parent objects
    and subsequent masks define child objects within the parents.
    
    Args:
        image_path: Path to the input image
        mask_paths: List of mask paths, first mask is the parent, rest are children
        output_folder: Folder to save output CSV files
        base_name: Base name for output files
        channel_index: Optional specific channel index to measure (0-based). If None, measures all channels.
    """
    logger.info(f"Processing {image_path} with {len(mask_paths)} masks")
    
    # Load the image using rp helper
    img = rp.load_tczyx_image(image_path)
    image_data = img.data  # 5D TCZYX array
    
    # Get the input image basename (without extension) to use as identifier
    input_image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Load all masks
    masks = []
    mask_names = []
    for i, mask_path in enumerate(mask_paths):
        mask_img = rp.load_tczyx_image(mask_path)
        masks.append(mask_img.data)  # 5D TCZYX array
        # Extract mask name from filename (for logging only)
        mask_name = os.path.splitext(os.path.basename(mask_path))[0]
        mask_names.append(mask_name)
        logger.info(f"  Loaded mask {i+1}/{len(mask_paths)}: {mask_name}")
    
    # Get dimensions
    T, C, Z, Y, X = image_data.shape
    
    # Determine which channels to measure
    if channel_index is not None:
        if channel_index < 0 or channel_index >= C:
            raise ValueError(f"Channel index {channel_index} out of range [0, {C-1}]")
        channels_to_measure = [channel_index]
        logger.info(f"Measuring only channel {channel_index}")
    else:
        channels_to_measure = list(range(C))
        logger.info(f"Measuring all {C} channels")
    
    # Process the first mask (parent objects) - measure in the input image
    logger.info(f"Measuring parent objects (mask 1: {mask_names[0]})...")
    parent_mask = masks[0]
    
    # Collect measurements for parent objects
    parent_measurements = []
    
    for t in range(T):
        for z in range(Z):
            # Get 2D slice
            image_slice = image_data[t, :, z, :, :]  # CYX
            parent_slice = parent_mask[t, 0, z, :, :].astype(int)  # YX (assuming masks are single channel)
            
            # Get unique parent object IDs (excluding 0 which is background)
            parent_ids = np.unique(parent_slice)
            parent_ids = parent_ids[parent_ids > 0]
            
            for parent_id in parent_ids:
                # Measure in each channel
                for c in channels_to_measure:
                    channel_data = image_slice[c, :, :]
                    stats = measure_region_stats(channel_data, parent_slice, parent_id)
                    
                    measurement = {
                        'parent_id': int(parent_id),
                        'timepoint': int(t),
                        'z_slice': int(z),
                        'channel': int(c),
                        'image_name': input_image_name,
                        **stats
                    }
                    parent_measurements.append(measurement)
    
    # Save parent measurements
    parent_csv_path = os.path.join(output_folder, f'{base_name}_{mask_names[0]}.csv')
    parent_df = pd.DataFrame(parent_measurements)
    parent_df.to_csv(parent_csv_path, index=False)
    logger.info(f"  Saved parent measurements to {parent_csv_path}")
    
    # Process child masks (if any)
    for child_idx in range(1, len(masks)):
        logger.info(f"Measuring child objects (mask {child_idx+1}: {mask_names[child_idx]})...")
        child_mask = masks[child_idx]
        child_measurements = []
        
        for t in range(T):
            for z in range(Z):
                # Get 2D slices
                image_slice = image_data[t, :, z, :, :]  # CYX
                parent_slice = parent_mask[t, 0, z, :, :].astype(int)  # YX
                child_slice = child_mask[t, 0, z, :, :].astype(int)  # YX
                
                # Get unique parent object IDs
                parent_ids = np.unique(parent_slice)
                parent_ids = parent_ids[parent_ids > 0]
                
                # For each parent object
                for parent_id in parent_ids:
                    # Create a mask for this parent
                    parent_region_mask = (parent_slice == parent_id)
                    
                    # Find child objects within this parent
                    child_within_parent = child_slice * parent_region_mask
                    child_ids_in_parent = np.unique(child_within_parent)
                    child_ids_in_parent = child_ids_in_parent[child_ids_in_parent > 0]
                    
                    # Measure each child object
                    for child_id in child_ids_in_parent:
                        # Create mask for this specific child
                        child_region_mask = (child_slice == child_id)
                        
                        # Measure in each channel
                        for c in channels_to_measure:
                            channel_data = image_slice[c, :, :]
                            stats = measure_region_stats(channel_data, child_slice, child_id)
                            
                            measurement = {
                                'parent_id': int(parent_id),
                                'child_id': int(child_id),
                                'timepoint': int(t),
                                'z_slice': int(z),
                                'channel': int(c),
                                'image_name': input_image_name,
                                **stats
                            }
                            child_measurements.append(measurement)
        
        # Save child measurements
        child_csv_path = os.path.join(output_folder, f'{base_name}_{mask_names[child_idx]}.csv')
        child_df = pd.DataFrame(child_measurements)
        child_df.to_csv(child_csv_path, index=False)
        logger.info(f"  Saved child measurements to {child_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical measurement of objects within objects. "
                    "First mask defines parent objects (e.g., nuclei), "
                    "subsequent masks define child objects within parents."
    )
    parser.add_argument(
        '--input-search-pattern',
        required=True,
        help='Glob pattern for input images, e.g. "folder/*.tif"'
    )
    parser.add_argument(
        '--mask-search-patterns',
        required=True,
        nargs='+',
        help='List of glob patterns for masks. First pattern is parent mask, '
             'rest are child masks. Use * wildcard to match input filename prefix. '
             'Example: "folder/*_nuc.tif" "folder/*_spot1.tif" "folder/*_spot2.tif"'
    )
    parser.add_argument(
        '--output',
        required=False,
        default=None,
        help='Path to output folder. Defaults to the folder of the first mask.'
    )
    parser.add_argument(
        '--search-subfolders',
        action='store_true',
        help='Enable recursive search for files'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing (default: parallel enabled)'
    )
    parser.add_argument(
        '--channel',
        type=int,
        required=False,
        default=None,
        help='Specific channel index to measure (0-based). If not provided, measures all channels.'
    )
    
    args = parser.parse_args()
    
    # Get input image files
    image_files = rp.get_files_to_process2(args.input_search_pattern, args.search_subfolders)
    logger.info(f"Found {len(image_files)} input images")
    
    if len(image_files) == 0:
        logger.error("No input images found matching pattern")
        return
    
    is_batch = len(image_files) > 1
    
    # Set default output folder
    if args.output is None:
        mask_folder = os.path.dirname(args.mask_search_patterns[0])
        output_folder = mask_folder
    else:
        output_folder = args.output
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Created output folder: {output_folder}")
    
    # Extract prefix from input pattern
    import re
    def get_prefix(pattern):
        m = re.search(r'(.*)\*', pattern)
        return m.group(1) if m else ''
    
    input_prefix = get_prefix(args.input_search_pattern)
    
    # Prepare jobs: (image_path, mask_paths, output_folder, base_name)
    jobs = []
    
    for image_path in image_files:
        # Get relative path for filename matching
        rel_img = image_path[len(input_prefix):] if image_path.startswith(input_prefix) else os.path.basename(image_path)
        rel_img_noext = os.path.splitext(rel_img)[0]
        
        # Find all matching masks
        mask_paths = []
        all_masks_found = True
        
        for mask_pattern in args.mask_search_patterns:
            # Replace * with the image filename (without extension)
            specific_mask_pattern = mask_pattern.replace('*', rel_img_noext)
            mask_files = rp.get_files_to_process2(specific_mask_pattern, args.search_subfolders)
            
            if len(mask_files) == 0:
                logger.warning(f"Mask not found for {image_path} using pattern {specific_mask_pattern}, skipping.")
                all_masks_found = False
                break
            if len(mask_files) > 1:
                logger.warning(f"Multiple masks found for {image_path} using pattern {specific_mask_pattern}, skipping.")
                all_masks_found = False
                break
            
            mask_paths.append(mask_files[0])
        
        if not all_masks_found:
            continue
        
        # Use the base name from the input image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        jobs.append((image_path, mask_paths, output_folder, base_name))
    
    logger.info(f"Prepared {len(jobs)} jobs for processing")
    
    # Define processing function
    def process_job(job):
        image_path, mask_paths, out_folder, base_name = job
        try:
            measure_hierarchical_masks(image_path, mask_paths, out_folder, base_name, args.channel)
            return (image_path, None)
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return (image_path, str(e))
    
    # Process jobs
    if not args.no_parallel and is_batch:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        logger.info("Processing in parallel mode")
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_job, job) for job in jobs]
            for f in tqdm(as_completed(futures), total=len(jobs), desc='Measuring'):
                result = f.result()
                if result[1] is not None:
                    logger.error(f"Failed: {result[0]}")
    else:
        logger.info("Processing in sequential mode")
        for job in tqdm(jobs, desc='Measuring'):
            result = process_job(job)
            if result[1] is not None:
                logger.error(f"Failed: {result[0]}")
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
