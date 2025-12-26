"""
Swap T and Z dimensions in images.

This utility swaps the time (T) and Z-stack dimensions in 5D TCZYX images.
Useful for correcting dimension ordering issues from different acquisition systems.

Author: BIPHUB - Bioimage Informatics Hub, University of Oslo
License: MIT
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np

# Local imports
try:
    import bioimage_pipeline_utils as rp
except ImportError:
    # Fallback for when script is run directly (not as module)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import bioimage_pipeline_utils as rp
from bioio import BioImage

def swap_t_z_dimensions(
    input_path: str,
    output_path: str
) -> Tuple[bool, Optional[str]]:
    """
    Swap T and Z dimensions in a 5D TCZYX image.
    
    Loads image as TCZYX, transposes to ZCTYX, then back to TCZYX format
    to swap the T and Z dimensions.
    
    Args:
        input_path: Path to input image file (5D TCZYX)
        output_path: Path to save output image file
    
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    try:
        logging.info(f"Loading: {Path(input_path).name}")
        
        # Load image in TCZYX format
        img = rp.load_tczyx_image(input_path)
        T, C, Z, Y, X = img.shape
        
        logging.info(f"  Original shape (TCZYX): {img.shape}")
        
        # Get data (will be 5D TCZYX numpy array)
        data = img.data
        
        # Swap T and Z: TCZYX -> ZCTYX
        data_swapped = np.transpose(data, (2, 1, 0, 3, 4))
        
        # New shape after swap
        new_T, new_C, new_Z, new_Y, new_X = data_swapped.shape
        logging.info(f"  Swapped shape (TCZYX): ({new_Z}, {new_C}, {new_T}, {new_Y}, {new_X})")
        
        # Create BioImage wrapper for the swapped data
        # The new shape is (Z, C, T, Y, X) but we'll interpret it as (T, C, Z, Y, X)
        swapped_img = BioImage(data_swapped, dims="TCZYX")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save using bioimage_pipeline_utils
        logging.info(f"Saving: {Path(output_path).name}")
        rp.save_tczyx_image(swapped_img, output_path)
        
        logging.info(f"✓ Successfully swapped dimensions")
        return (True, None)
        
    except Exception as e:
        error_msg = f"Error swapping dimensions: {str(e)}"
        logging.error(error_msg)
        return (False, error_msg)


def process_single_file(args_tuple: Tuple) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single file for dimension swapping (used in parallel processing).
    
    Args:
        args_tuple: Tuple containing (basename, input_path, output_path)
    
    Returns:
        Tuple of (basename, success, error_message)
    """
    basename, input_path, output_path = args_tuple
    
    success, error_msg = swap_t_z_dimensions(input_path, output_path)
    return (basename, success, error_msg)


def main():
    parser = argparse.ArgumentParser(
        description='Swap T and Z dimensions in 5D TCZYX images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This utility swaps the time (T) and Z-stack dimensions in 5D TCZYX images.
Useful for correcting dimension ordering issues when images from different 
acquisition systems have inconsistent dimension ordering.

Example: Image with shape (37, 3, 1, 512, 512) [TCZYX] becomes 
         (1, 3, 37, 512, 512) [TCZYX]

Example YAML config for run_pipeline.exe:
---
run:
- name: Swap T and Z dimensions
  environment: uv@3.11:swap-dimensions
  commands:
  - python
  - '%REPO%/standard_code/python/swap_dimensions.py'
  - --input-search-pattern: '%YAML%/input/**/*.tif'
  - --output-folder: '%YAML%/output'
  - --output-suffix: '_tz_swapped'
        """
    )
    
    parser.add_argument('--input-search-pattern', type=str, required=True,
                       help='Glob pattern for input images. Use "**" for recursive search.')
    parser.add_argument('--output-folder', type=str, required=True,
                       help='Folder to save output files.')
    parser.add_argument('--output-suffix', type=str, default='_tz_swapped',
                       help='Suffix to append to output filenames (default: "_tz_swapped")')
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
    logging.info("Swap T and Z Dimensions")
    logging.info("="*80)
    
    # Determine if recursive search is requested
    search_subfolders = '**' in args.input_search_pattern
    
    logging.info(f"Input pattern: {args.input_search_pattern}")
    logging.info(f"Recursive: {search_subfolders}")
    
    # Get files using bioimage_pipeline_utils
    try:
        # Build patterns dict (only input pattern needed)
        patterns = {'input': args.input_search_pattern}
        grouped_files = rp.get_grouped_files_to_process(patterns, search_subfolders)
    except Exception as e:
        logging.error(f"Error finding files: {e}")
        return 1
    
    if not grouped_files:
        logging.error("No files found matching the search pattern")
        return 1
    
    # Extract files from grouped structure
    processing_args = []
    for basename, files in grouped_files.items():
        if 'input' not in files:
            continue
        
        input_path = files['input']
        output_filename = f"{basename}{args.output_suffix}.tif"
        output_path = os.path.join(args.output_folder, output_filename)
        
        args_tuple = (basename, input_path, output_path)
        processing_args.append(args_tuple)
    
    n_files = len(processing_args)
    logging.info(f"Found {n_files} files to process")
    
    if n_files == 0:
        logging.error("No files to process")
        return 1
    
    # Process files (parallel or sequential)
    if args.no_parallel:
        logging.info("Sequential processing mode (--no-parallel enabled)")
        results = []
        for i, args_tuple in enumerate(tqdm(processing_args, desc="Processing files", unit="file"), 1):
            basename = args_tuple[0]
            logging.info(f"\n{'='*80}")
            logging.info(f"Processing {i}/{n_files}: {basename}")
            logging.info(f"{'='*80}")
            result = process_single_file(args_tuple)
            results.append(result)
    else:
        # Parallel processing (default)
        n_workers = cpu_count()
        logging.info(f"Parallel processing mode: {n_workers} workers")
        logging.info(f"Processing {n_files} files in parallel...")
        
        with Pool(processes=n_workers) as pool:
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
    
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
