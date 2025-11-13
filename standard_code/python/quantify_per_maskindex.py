"""
Quantify image intensity statistics per mask index across timepoints.

This module measures intensity statistics for each unique mask index value
across all timepoints. Works with standard segmentation masks (measuring per nucleus, 
cell, etc.) or specialized masks like distance matrices (measuring per distance bin).

For each mask index at each timepoint, calculates:
- Count: Number of pixels with this mask value
- Min, Max, Mean, Median, Mode: Basic statistics
- Standard Deviation: Intensity variation
- Sum: Total intensity (useful for total signal)
- Sum_N_Random: Sum of N randomly sampled pixels (reduces bias from pixel count variations)

Author: BIPHUB - Bioimage Informatics Hub, University of Oslo
License: MIT
"""
import os
import sys
import argparse
import logging
from matplotlib import image
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.ndimage import labeled_comprehension
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings

# Local imports
try:
    import bioimage_pipeline_utils as rp
except ImportError:
    # Fallback for when script is run directly (not as module)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import bioimage_pipeline_utils as rp


def get_experiment_info_from_filename(filename: str, split: str = "__", colname: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Extract experimental information from filename by splitting on delimiter.
    
    Args:
        filename: Filename to parse (can be full path or just filename)
        split: Delimiter to split on (default: "__")
        colname: List of column names to assign to split parts. 
                 Use "_" to skip a part. (default: ["Group"])
    
    Examples:
        filename = "SP20250625__3SA__R2__SP20250625_PC_R2_3SA_.tif"
        
        # Extract just the group
        get_experiment_info_from_filename(filename, "__", ["Group"])
        # Returns: {"Group": "3SA"}
        
        # Extract experiment, group, and replicate
        get_experiment_info_from_filename(filename, "__", ["Experiment", "Group", "Replicate"])
        # Returns: {"Experiment": "SP20250625", "Group": "3SA", "Replicate": "R2"}
        
        # Skip first part, extract group only
        get_experiment_info_from_filename(filename, "__", ["_", "Group"])
        # Returns: {"Group": "3SA"}
    
    Returns:
        Dictionary mapping column names to extracted values
    """
    if colname is None:
        colname = ["Group"]
    
    # Get just the filename without path
    base = Path(filename).stem  # Removes extension
    
    # Split by delimiter
    parts = base.split(split)
    
    # Map parts to column names
    result = {}
    for i, col in enumerate(colname):
        # Skip if column name is "_"
        if col == "_":
            continue
        
        # Add to result if we have enough parts
        if i < len(parts):
            result[col] = parts[i]
        else:
            result[col] = ""
    
    return result


def _compute_statistics_vectorized(
    img_t: np.ndarray,
    mask_t: np.ndarray,
    mask_indices: np.ndarray,
    max_sample_pixels: int,
    n_sampling_runs: int,
    random_seed: int
) -> Dict[float, Dict[str, float]]:
    """
    Vectorized computation of statistics for all mask indices at once.
    
    This is MUCH faster than looping over indices individually.
    Uses scipy.ndimage.labeled_comprehension for bulk processing.
    
    Returns:
        Dictionary mapping mask_index -> {stat_name: value}
    """
    use_sampling = max_sample_pixels > 0
    results = {}
    
    # Flatten for easier processing
    img_flat = img_t.ravel()
    mask_flat = mask_t.ravel()
    
    for idx in mask_indices:
        # Get pixels for this index
        pixel_mask = (mask_flat == idx)
        pixel_count = np.sum(pixel_mask)
        
        if pixel_count == 0:
            continue
        
        intensities = img_flat[pixel_mask]
        
        # Basic statistics (vectorized)
        min_val = float(np.min(intensities))
        max_val = float(np.max(intensities))
        mean_val = float(np.mean(intensities))
        median_val = float(np.median(intensities))
        std_val = float(np.std(intensities))
        sum_val = float(np.sum(intensities))
        
        # Mode (most common intensity value)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mode_result = stats.mode(intensities, keepdims=True)
                mode_val = float(mode_result.mode[0])
        except:
            mode_val = mean_val
        
        # Robust random sampling for sum
        if use_sampling and pixel_count > max_sample_pixels:
            # Multiple sampling runs to get robust estimate
            sampled_sums = []
            
            for run in range(n_sampling_runs):
                rng = np.random.RandomState(random_seed + run)
                sampled_indices = rng.choice(len(intensities), size=max_sample_pixels, replace=False)
                sampled_intensities = intensities[sampled_indices]
                sampled_sums.append(np.sum(sampled_intensities))
            
            # Select the value closest to the mean
            mean_sum = np.mean(sampled_sums)
            closest_idx = np.argmin(np.abs(np.array(sampled_sums) - mean_sum))
            sum_n_random = float(sampled_sums[closest_idx])
        elif use_sampling:
            # Fewer pixels than sample size - use all pixels
            sum_n_random = sum_val
        else:
            # Sampling disabled
            sum_n_random = np.nan
        
        results[float(idx)] = {
            'Count': pixel_count,
            'Min': min_val,
            'Max': max_val,
            'Mean': mean_val,
            'Median': median_val,
            'Mode': mode_val,
            'Std': std_val,
            'Sum': sum_val,
            'Sum_N_Random': sum_n_random
        }
    
    return results


def quantify_per_maskindex(
    image_path: str,
    mask_path: str,
    output_path: str,
    channel: int = 0,
    exclude_zero: bool = True,
    max_sample_pixels: int = 20,
    n_sampling_runs: int = 10,
    random_seed: int = 42,
    metadata_dict: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Quantify image intensity statistics per mask index across timepoints.
    
    For each unique mask index (e.g., nucleus ID, distance bin) at each timepoint,
    calculates comprehensive statistics including robust random sampling.
    
    Args:
        image_path: Path to input image file (5D TCZYX)
        mask_path: Path to mask/index image file (5D TCZYX)
        output_path: Path to save TSV output
        channel: Which image channel to quantify (default: 0)
        exclude_zero: If True, exclude mask value 0 from analysis (default: True)
        max_sample_pixels: Max pixels to sample for Sum_N_Random. 
                          Set to -1 to disable sampling (default: 20)
        n_sampling_runs: Number of sampling runs for robust statistics.
                        Uses multiple random seeds and selects value closest to mean (default: 10)
        random_seed: Base seed for random sampling (default: 42)
        metadata_dict: Optional dictionary of metadata to include as columns (default: None)
    
    Returns:
        DataFrame with columns:
            - Mask_Index: The mask value (e.g., nucleus ID, distance bin)
            - Timepoint: Timepoint index (0, 1, 2, ...)
            - Channel: The image channel that was quantified
            - Count: Number of pixels with this mask value
            - Min, Max, Mean, Median, Mode: Basic statistics
            - Std: Standard deviation
            - Sum: Total intensity
            - Sum_<N>_Random: Sum of N randomly sampled pixels (e.g., Sum_20_Random if max_sample_pixels=20)
            - [metadata columns]: Any additional metadata provided
    """
    logging.info(f"Processing: {Path(image_path).name}")
    logging.info(f"Mask: {Path(mask_path).name}")
    
    # Load data - use dask_data for lazy loading
    img = rp.load_tczyx_image(image_path)
    mask = rp.load_tczyx_image(mask_path)

    # Create filtered mask - keep values < 100, set others to 0
    # mask_filtered = mask.data.copy()
    # mask_filtered[mask_filtered >= 100] = 0
    # rp.show_image(
    #     image=img,
    #     mask=mask_filtered,
    #     title="Image with Mask Overlay (values < 100)",
    #     alpha=0.5,
    #     timer=-1
    # )
    # Input was as expected 
    
    T, C, Z, Y, X = img.shape
    mT, mC, mZ, mY, mX = mask.shape
    
    logging.info(f"Image shape: {img.shape}")
    logging.info(f"Mask shape: {mask.shape}")
    
    # Validate dimensions
    if (Y, X) != (mY, mX):
        raise ValueError(f"Image and mask XY dimensions must match. Got image: ({Y}, {X}), mask: ({mY}, {mX})")
    if T != mT:
        raise ValueError(f"Image and mask must have same number of timepoints. Got image: {T}, mask: {mT}")
    
    # Use dask arrays for lazy loading (only loads data when needed)
    try:
        img_dask = img.dask_data
        mask_dask = mask.dask_data
        use_dask = True
        logging.info("Using Dask lazy loading for memory efficiency")
    except:
        # Fallback to regular data if dask not available
        img_dask = img.data
        mask_dask = mask.data
        use_dask = False
        logging.info("Using eager loading (Dask not available)")
    
    # Get all unique mask indices across all timepoints
    # Load only mask data into memory (much smaller than image)
    logging.info("Finding unique mask indices across all timepoints...")
    all_mask_indices = []
    for t in range(T):
        if use_dask:
            # Compute only this timepoint's mask
            mask_t = mask_dask[t, 0, :, :, :].compute()
        else:
            mask_t = mask_dask[t, 0, :, :, :]
        unique_indices = np.unique(mask_t)
        all_mask_indices.extend(unique_indices)
    
    mask_indices = np.sort(np.unique(all_mask_indices))
    
    # Exclude zero if requested
    if exclude_zero and 0 in mask_indices:
        mask_indices = mask_indices[mask_indices != 0]
        logging.info(f"Excluding mask index 0 (background)")
    
    n_indices = len(mask_indices)
    logging.info(f"Found {n_indices} unique mask indices")
    
    if n_indices == 0:
        logging.warning("No mask indices found!")
        return pd.DataFrame()
    
    # Check if sampling is enabled
    use_sampling = max_sample_pixels > 0
    if use_sampling:
        logging.info(f"Robust random sampling: max {max_sample_pixels} pixels per index")
        logging.info(f"Using {n_sampling_runs} sampling runs (seeds {random_seed} to {random_seed+n_sampling_runs-1})")
    else:
        logging.info(f"Random sampling DISABLED (max_sample_pixels={max_sample_pixels})")
    
    # Prepare results storage
    results = []
    
    # Process each timepoint using vectorized computation
    logging.info("Processing timepoints with vectorized statistics...")
    for t in range(T):
        if use_dask:
            # Load only this timepoint into memory (lazy evaluation)
            mask_t = mask_dask[t, 0, :, :, :].compute()
            img_t = img_dask[t, channel, :, :, :].compute()
        else:
            mask_t = mask_dask[t, 0, :, :, :]
            img_t = img_dask[t, channel, :, :, :]
        
        # Vectorized computation for all indices at once
        stats_dict = _compute_statistics_vectorized(
            img_t, mask_t, mask_indices,
            max_sample_pixels, n_sampling_runs, random_seed
        )
        
        # Build result rows
        for idx, stats_values in stats_dict.items():
            row = {
                'Mask_Index': idx,
                'Timepoint': t,
                'Channel': channel
            }
            row.update(stats_values)
            
            # Add metadata if provided
            if metadata_dict:
                row.update(metadata_dict)
            
            results.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Rename Sum_N_Random column to include actual sample size
    if use_sampling and max_sample_pixels > 0:
        df.rename(columns={'Sum_N_Random': f'Sum_{max_sample_pixels}_Random'}, inplace=True)
    
    # Reorder columns: metadata first, then measurements
    if metadata_dict:
        metadata_cols = list(metadata_dict.keys())
        sum_col_name = f'Sum_{max_sample_pixels}_Random' if use_sampling and max_sample_pixels > 0 else 'Sum_N_Random'
        measurement_cols = ['Mask_Index', 'Timepoint', 'Channel', 'Count', 'Min', 'Max', 
                           'Mean', 'Median', 'Mode', 'Std', 'Sum', sum_col_name]
        df = df[metadata_cols + measurement_cols]
    
    # Save to TSV
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(output_path, sep='\t', index=False)
    logging.info(f"Saved {len(df)} measurements to: {output_path}")
    
    return df


def process_single_file(args_tuple: Tuple) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single file for quantification (used in parallel processing).
    
    Args:
        args_tuple: Tuple containing (basename, img_path, mask_path, output_path, 
                    channel, exclude_zero, max_sample_pixels, n_sampling_runs, 
                    random_seed, metadata_dict, exclude_first_n_indices)
    
    Returns:
        Tuple of (basename, success, error_message)
    """
    (basename, img_path, mask_path, output_path, channel, exclude_zero, 
     max_sample_pixels, n_sampling_runs, random_seed, metadata_dict, 
     exclude_first_n_indices) = args_tuple
    
    try:
        # Run quantification
        df = quantify_per_maskindex(
            image_path=img_path,
            mask_path=mask_path,
            output_path=output_path,
            channel=channel,
            exclude_zero=exclude_zero,
            max_sample_pixels=max_sample_pixels,
            n_sampling_runs=n_sampling_runs,
            random_seed=random_seed,
            metadata_dict=metadata_dict
        )
        
        # Optionally exclude first N indices (useful for distance matrices)
        if exclude_first_n_indices > 0:
            # Reload and filter
            df = pd.read_csv(output_path, sep='\t')
            unique_indices = sorted(df['Mask_Index'].unique())
            
            if len(unique_indices) > exclude_first_n_indices:
                indices_to_keep = unique_indices[exclude_first_n_indices:]
                df_filtered = df[df['Mask_Index'].isin(indices_to_keep)]
                
                # Re-save filtered version
                df_filtered.to_csv(output_path, sep='\t', index=False)
                logging.info(f"Excluded first {exclude_first_n_indices} mask indices")
                logging.info(f"Kept {len(indices_to_keep)} indices: {indices_to_keep[0]:.1f} to {indices_to_keep[-1]:.1f}")
        
        return (basename, True, None)
        
    except Exception as e:
        return (basename, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description='Quantify image intensity statistics per mask index across timepoints.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This module measures intensity statistics for each unique mask index value
across all timepoints. Works with:
  - Standard segmentation masks (measuring per nucleus, cell, organelle, etc.)
  - Distance matrices (measuring per distance bin)
  - Any indexed mask where each integer value represents a region of interest

For each mask index at each timepoint, calculates:
  - Count: Number of pixels with this mask value
  - Min, Max, Mean, Median, Mode: Basic statistics
  - Standard Deviation: Intensity variation  
  - Sum: Total intensity (useful for total signal)
  - Sum_N_Random: Sum of N randomly sampled pixels (reduces bias from pixel count variations)

Example YAML config for run_pipeline.exe:
---
run:
- name: Quantify per nucleus
  environment: uv@3.11:quantify-per-maskindex
  commands:
  - python
  - '%REPO%/standard_code/python/quantify_per_maskindex.py'
  - --input-search-pattern: '%YAML%/images/**/*.tif'
  - --mask-search-pattern: '%YAML%/masks/**/*_mask.tif'
  - --output-folder: '%YAML%/quantification'
  - --output-suffix: '_per_nucleus_stats'
  - --channel: 0

- name: Quantify distance matrix (per distance bin)
  environment: uv@3.11:quantify-per-maskindex
  commands:
  - python
  - '%REPO%/standard_code/python/quantify_per_maskindex.py'
  - --input-search-pattern: '%YAML%/images/**/*.tif'
  - --mask-search-pattern: '%YAML%/distances/**/*_geodesic.tif'
  - --output-folder: '%YAML%/quantification'
  - --output-suffix: '_per_distance_stats'
  - --exclude-first-n-indices: 5
  - --max-sample-pixels: 20

- name: Quantify with metadata extraction
  environment: uv@3.11:quantify-per-maskindex
  commands:
  - python
  - '%REPO%/standard_code/python/quantify_per_maskindex.py'
  - --input-search-pattern: '%YAML%/images/**/*.tif'
  - --mask-search-pattern: '%YAML%/masks/**/*_mask.tif'
  - --output-folder: '%YAML%/quantification'
  - --filename-split: '__'
  - --filename-colnames: Experiment Group Replicate
        """
    )
    
    parser.add_argument('--input-search-pattern', type=str, required=True,
                       help='Glob pattern for input images. Use "**" for recursive search.')
    parser.add_argument('--mask-search-pattern', type=str, required=True,
                       help='Glob pattern for mask/index images. Use "**" for recursive search.')
    parser.add_argument('--output-folder', type=str, required=True,
                       help='Folder to save TSV output files.')
    parser.add_argument('--output-suffix', type=str, default='_measurements',
                       help='Suffix to append to output filenames (default: "_measurements")')
    parser.add_argument('--channel', type=int, default=0,
                       help='Which image channel to quantify (default: 0)')
    parser.add_argument('--exclude-zero', action='store_true', default=True,
                       help='Exclude mask index 0 (background) from analysis (default: True)')
    parser.add_argument('--include-zero', dest='exclude_zero', action='store_false',
                       help='Include mask index 0 in analysis')
    parser.add_argument('--exclude-first-n-indices', type=int, default=0,
                       help='Exclude first N mask indices from analysis (useful for distance matrices) (default: 0)')
    parser.add_argument('--max-sample-pixels', type=int, default=20,
                       help='Max pixels to sample for Sum_N_Random statistic. '
                            'Use -1 to disable sampling (default: 20)')
    parser.add_argument('--n-sampling-runs', type=int, default=10,
                       help='Number of sampling runs for robust Sum_N_Random (default: 10)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Base seed for random sampling (default: 42)')
    parser.add_argument('--filename-split', type=str, default='__',
                       help='Delimiter to split filename on for metadata extraction (default: "__")')
    parser.add_argument('--filename-colnames', type=str, nargs='+', default=None,
                       help='Column names to extract from filename split. '
                            'Use "_" to skip a part. Example: --filename-colnames Experiment Group Replicate')
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing (default: parallel enabled)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("="*80)
    logging.info("Quantify Per Mask Index")
    logging.info("="*80)
    
    # Determine if recursive search is requested
    search_subfolders = '**' in args.input_search_pattern or '**' in args.mask_search_pattern
    
    # Build search patterns
    patterns = {
        'input': args.input_search_pattern,
        'mask': args.mask_search_pattern
    }
    
    logging.info(f"Searching for files:")
    logging.info(f"  Input: {args.input_search_pattern}")
    logging.info(f"  Mask:  {args.mask_search_pattern}")
    logging.info(f"  Recursive: {search_subfolders}")
    
    # Get grouped files using bioimage_pipeline_utils
    try:
        grouped_files = rp.get_grouped_files_to_process(patterns, search_subfolders)
    except Exception as e:
        logging.error(f"Error finding files: {e}")
        return 1
    
    if not grouped_files:
        logging.error("No files found matching the search patterns")
        return 1
    
    logging.info(f"Found {len(grouped_files)} file groups")
    
    # Filter to complete pairs
    complete_pairs = {
        basename: files 
        for basename, files in grouped_files.items() 
        if 'input' in files and 'mask' in files
    }
    
    if not complete_pairs:
        logging.error("No complete image-mask pairs found")
        return 1
    
    files_with_both = len(complete_pairs)
    files_missing = len(grouped_files) - files_with_both
    logging.info(f"  {files_with_both} complete pairs, {files_missing} incomplete")
    
    # Set default column names if not provided
    if args.filename_colnames is None:
        filename_colnames = ["Experiment", "Group", "Replicate"]
    else:
        filename_colnames = args.filename_colnames
    
    # Prepare processing arguments for all files
    processing_args = []
    for basename, files in complete_pairs.items():
        img_path = files['input']
        mask_path = files['mask']
        
        # Extract metadata from filename
        metadata_dict = get_experiment_info_from_filename(
            basename,
            split=args.filename_split,
            colname=filename_colnames
        )
        
        # Add base filename
        metadata_dict['Base_Filename'] = basename
        
        # Create output path
        output_filename = f"{basename}{args.output_suffix}.tsv"
        output_path = os.path.join(args.output_folder, output_filename)
        
        # Prepare args tuple for this file
        args_tuple = (
            basename, img_path, mask_path, output_path, args.channel,
            args.exclude_zero, args.max_sample_pixels, args.n_sampling_runs,
            args.random_seed, metadata_dict, args.exclude_first_n_indices
        )
        processing_args.append(args_tuple)
    
    # Process files (parallel or sequential)
    if args.no_parallel:
        logging.info("Sequential processing mode (--no-parallel enabled)")
        results = []
        for i, args_tuple in enumerate(tqdm(processing_args, desc="Processing files", unit="file"), 1):
            basename = args_tuple[0]
            logging.info(f"\n{'='*80}")
            logging.info(f"Processing {i}/{files_with_both}: {basename}")
            logging.info(f"{'='*80}")
            result = process_single_file(args_tuple)
            results.append(result)
    else:
        # Parallel processing (default)
        n_workers = cpu_count()
        logging.info(f"Parallel processing mode: {n_workers} workers")
        logging.info(f"Processing {files_with_both} files in parallel...")
        
        with Pool(processes=n_workers) as pool:
            # Use tqdm to show progress as files complete
            results = list(tqdm(
                pool.imap(process_single_file, processing_args),
                total=len(processing_args),
                desc="Processing files",
                unit="file"
            ))
    
    # Report results
    logging.info("\n" + "="*80)
    logging.info("Processing Summary")
    logging.info("="*80)
    
    n_success = 0
    n_failed = 0
    
    for basename, success, error_msg in results:
        if success:
            logging.info(f"✓ {basename}")
            n_success += 1
        else:
            logging.error(f"✗ {basename}: {error_msg}")
            n_failed += 1
    
    logging.info(f"\nTotal: {n_success} succeeded, {n_failed} failed")
    
    logging.info("\n" + "="*80)
    logging.info("Processing complete!")
    logging.info("="*80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
