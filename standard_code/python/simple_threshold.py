"""
Simple threshold-based segmentation for distance maps and intensity images.

Supports:
- Automatic threshold methods (Otsu, Li, Yen, Triangle)
- Manual threshold ranges (e.g., 1-10 pixels for distance maps)
- Optional preprocessing (median, mean, gaussian)
- Parallel file processing

MIT License - BIPHUB, University of Oslo
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Literal, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import ndimage
from skimage import filters

import bioimage_pipeline_utils as rp

# Module-level logger
logger = logging.getLogger(__name__)


def parse_threshold_args(threshold_args: list[str]) -> Tuple[
    Literal["otsu", "li", "yen", "triangle", "manual"], 
    Optional[float], 
    Optional[float]
]:
    """
    Parse flexible threshold arguments.
    
    Handles both separate args and space-separated strings from YAML:
    - ["otsu"] or ["1 10"] from YAML quoted strings
    - ["1", "10"] from command line or YAML lists
    
    Args:
        threshold_args: List of 1 or 2 strings from --threshold argument
    
    Returns:
        Tuple of (method, min_val, max_val)
        - For auto methods: ("otsu", None, None)
        - For manual range: ("manual", 1.0, 10.0)
    
    Examples:
        ["otsu"] -> ("otsu", None, None)
        ["li"] -> ("li", None, None)
        ["1", "10"] -> ("manual", 1.0, 10.0)
        ["1 10"] -> ("manual", 1.0, 10.0)  # YAML quoted string
        ["11", "inf"] -> ("manual", 11.0, float('inf'))
        ["11 inf"] -> ("manual", 11.0, float('inf'))  # YAML quoted string
    """
    # Handle case where YAML passes quoted string like '1 10' as single element
    if len(threshold_args) == 1 and ' ' in threshold_args[0]:
        threshold_args = threshold_args[0].split()
    
    if len(threshold_args) == 1:
        method = threshold_args[0].lower()
        if method in ["otsu", "li", "yen", "triangle"]:
            return method, None, None  # type: ignore
        else:
            raise ValueError(
                f"Unknown threshold method: {method}. "
                f"Use 'otsu', 'li', 'yen', 'triangle', or provide min/max values."
            )
    
    elif len(threshold_args) == 2:
        try:
            min_val = float(threshold_args[0])
            max_val = float('inf') if threshold_args[1].lower() == 'inf' else float(threshold_args[1])
            return "manual", min_val, max_val
        except ValueError:
            raise ValueError(
                f"Invalid threshold range: {threshold_args}. "
                f"Expected numbers or 'inf'."
            )
    
    else:
        raise ValueError(f"--threshold takes 1 or 2 arguments, got {len(threshold_args)}")


def apply_preprocessing(
    image: np.ndarray,
    method: Literal["none", "median", "mean", "gaussian"],
    size: int
) -> np.ndarray:
    """
    Apply preprocessing filter to image.
    
    Args:
        image: Input image (any dimensionality)
        method: Filter type
        size: Kernel/filter size (-1 to disable)
    
    Returns:
        Filtered image
    """
    if method == "none" or size <= 0:
        return image
    
    logger.info(f"Applying {method} filter (size={size})")
    
    if method == "median":
        return ndimage.median_filter(image, size=size)
    elif method == "mean":
        return ndimage.uniform_filter(image, size=size)
    elif method == "gaussian":
        sigma = size / 3.0  # Rule of thumb: kernel ~ 3*sigma
        return ndimage.gaussian_filter(image, sigma=sigma)
    else:
        return image


def apply_threshold(
    image: np.ndarray,
    method: Literal["otsu", "li", "yen", "triangle", "manual"],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> np.ndarray:
    """
    Apply threshold to create binary mask.
    
    Args:
        image: Input image (any dimensionality)
        method: Threshold method
        min_val: Minimum threshold value (for manual mode)
        max_val: Maximum threshold value (for manual mode)
    
    Returns:
        Binary mask (0 or 255, uint8)
    """
    if method == "manual":
        if min_val is None or max_val is None:
            raise ValueError("Manual thresholding requires min_val and max_val")
        
        # Create mask where values are between min and max
        mask = (image >= min_val) & (image <= max_val)
        logger.info(f"Manual threshold: {min_val} to {max_val}")
        
    elif method == "otsu":
        threshold = filters.threshold_otsu(image)
        mask = image > threshold
        logger.info(f"Otsu threshold: {threshold:.2f}")
        
    elif method == "li":
        threshold = filters.threshold_li(image)
        mask = image > threshold
        logger.info(f"Li threshold: {threshold:.2f}")
        
    elif method == "yen":
        threshold = filters.threshold_yen(image)
        mask = image > threshold
        logger.info(f"Yen threshold: {threshold:.2f}")
        
    elif method == "triangle":
        threshold = filters.threshold_triangle(image)
        mask = image > threshold
        logger.info(f"Triangle threshold: {threshold:.2f}")
    
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    # Convert to uint8 (0 or 255)
    return (mask.astype(np.uint8) * 255)


def process_single_file(
    input_path: str,
    output_path: str,
    threshold_method: str,
    threshold_min: Optional[float],
    threshold_max: Optional[float],
    preprocess_method: str,
    preprocess_size: int,
    force: bool = False
) -> bool:
    """
    Process a single image file.
    
    Args:
        input_path: Path to input image
        output_path: Path to output mask
        threshold_method: Threshold method name
        threshold_min: Minimum threshold (manual mode)
        threshold_max: Maximum threshold (manual mode)
        preprocess_method: Preprocessing filter type
        preprocess_size: Preprocessing kernel size
        force: Force reprocessing even if output exists
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing: {os.path.basename(input_path)}")
        
        # Check if output exists
        if os.path.exists(output_path) and not force:
            logger.info(f"Output exists, skipping: {output_path}")
            return True
        
        # Load image using bioimage_pipeline_utils
        img = rp.load_tczyx_image(input_path)
        
        # Get dimensions
        T, C, Z, Y, X = img.shape
        logger.info(f"Dimensions: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
        
        # Get the image data (handle all dimensions)
        # For simplicity, process the entire stack as-is
        image_data = img.data
        
        # Preprocessing
        if preprocess_method != "none" and preprocess_size > 0:
            processed = apply_preprocessing(
                image_data,
                method=preprocess_method,  # type: ignore
                size=preprocess_size
            )
        else:
            processed = image_data
        
        # Thresholding
        mask = apply_threshold(
            processed,
            method=threshold_method,  # type: ignore
            min_val=threshold_min,
            max_val=threshold_max
        )
        
        # Save mask using bioimage_pipeline_utils helper
        # This ensures proper TZCYX order for ImageJ compatibility
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        rp.save_mask(
            mask,
            output_path,
            as_binary=False  # Preserve values (255 for binary, or float for distance maps)
        )
        
        logger.info(f"Saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_files(
    input_pattern: str,
    output_folder: Optional[str],
    threshold_args: list[str],
    preprocess_method: str,
    preprocess_size: int,
    no_parallel: bool,
    dry_run: bool,
    force: bool
) -> None:
    """
    Process multiple files with optional parallel execution.
    
    Args:
        input_pattern: Input file search pattern
        output_folder: Output directory
        threshold_args: Threshold arguments to parse
        preprocess_method: Preprocessing filter type
        preprocess_size: Preprocessing kernel size
        no_parallel: Disable parallel processing
        dry_run: Only print planned actions
        force: Force reprocessing
    """
    # Parse threshold arguments
    threshold_method, threshold_min, threshold_max = parse_threshold_args(threshold_args)
    
    logger.info(f"Threshold mode: {threshold_method}")
    if threshold_method == "manual":
        logger.info(f"Range: {threshold_min} to {threshold_max}")
    
    # Find input files
    search_subfolders = '**' in input_pattern
    input_files = rp.get_files_to_process2(input_pattern, search_subfolders=search_subfolders)
    
    if not input_files:
        logger.error(f"No files matched pattern: {input_pattern}")
        return
    
    logger.info(f"Found {len(input_files)} files")
    
    # Determine output folder
    if output_folder is None:
        base_dir = os.path.dirname(input_pattern.replace('**/', '').replace('*', ''))
        output_folder = (base_dir or ".") + "_threshold"
    
    logger.info(f"Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    # Build task list
    tasks = []
    for input_path in input_files:
        # Generate output filename
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{basename}_mask.tif"
        output_path = os.path.join(output_folder, output_filename)
        
        tasks.append((input_path, output_path))
    
    # Dry run
    if dry_run:
        print(f"[DRY RUN] Would process {len(tasks)} files")
        print(f"[DRY RUN] Output folder: {output_folder}")
        print(f"[DRY RUN] Threshold: {threshold_method}", end="")
        if threshold_method == "manual":
            print(f" (range: {threshold_min} to {threshold_max})")
        else:
            print()
        print(f"[DRY RUN] Preprocess: {preprocess_method} (size={preprocess_size})")
        for inp, out in tasks[:5]:  # Show first 5
            print(f"[DRY RUN]   {os.path.basename(inp)} -> {os.path.basename(out)}")
        if len(tasks) > 5:
            print(f"[DRY RUN]   ... and {len(tasks) - 5} more files")
        return
    
    # Process files
    success_count = 0
    fail_count = 0
    
    if no_parallel or len(tasks) == 1:
        # Sequential processing
        logger.info("Processing files sequentially")
        for input_path, output_path in tasks:
            success = process_single_file(
                input_path, output_path,
                threshold_method, threshold_min, threshold_max,
                preprocess_method, preprocess_size,
                force
            )
            if success:
                success_count += 1
            else:
                fail_count += 1
    else:
        # Parallel processing
        cpu_count = os.cpu_count() or 1
        max_workers = max(cpu_count - 1, 1)
        logger.info(f"Processing files in parallel (workers={max_workers})")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for input_path, output_path in tasks:
                future = executor.submit(
                    process_single_file,
                    input_path, output_path,
                    threshold_method, threshold_min, threshold_max,
                    preprocess_method, preprocess_size,
                    force
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    fail_count += 1
    
    logger.info(f"Processing complete: {success_count} succeeded, {fail_count} failed")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Simple threshold-based segmentation for distance maps and intensity images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

1. Distance map threshold (1-10 pixels -> 255):
   python simple_threshold.py --input "distance_maps/**/*.tif" --threshold 1 10

2. Distance map threshold (11 to infinity):
   python simple_threshold.py --input "distance_maps/**/*.tif" --threshold 11 inf

3. Otsu auto-threshold:
   python simple_threshold.py --input "images/**/*.tif" --threshold otsu

4. Li threshold with median preprocessing:
   python simple_threshold.py --input "images/**/*.tif" --threshold li \\
       --preprocess median --preprocess-size 5

Example YAML config for run_pipeline.exe:
---
run:
- name: Threshold distance maps (1-10 pixels)
  environment: uv@3.11:segmentation
  commands:
  - python
  - '%REPO%/standard_code/python/simple_threshold.py'
  - --input-search-pattern: '%YAML%/distance_maps/**/*.tif'
  - --output-folder: '%YAML%/masks'
  - --threshold: 1 10

- name: Auto-threshold with preprocessing
  environment: uv@3.11:segmentation
  commands:
  - python
  - '%REPO%/standard_code/python/simple_threshold.py'
  - --input-search-pattern: '%YAML%/images/**/*.tif'
  - --output-folder: '%YAML%/masks'
  - --threshold: otsu
  - --preprocess: median
  - --preprocess-size: 5
        """
    )
    
    parser.add_argument(
        "--input-search-pattern",
        required=True,
        help="Input file pattern (supports wildcards, use '**' for recursive)"
    )
    
    parser.add_argument(
        "--output-folder",
        help="Output folder (default: input_folder + '_threshold')"
    )
    
    parser.add_argument(
        "--threshold",
        nargs='+',
        default=["otsu"],
        help=(
            "Threshold method: 'otsu'/'li'/'yen'/'triangle' OR min max "
            "(e.g., '1 10' or '11 inf'). Default: otsu"
        )
    )
    
    parser.add_argument(
        "--preprocess",
        choices=["none", "median", "mean", "gaussian"],
        default="none",
        help="Preprocessing filter (default: none)"
    )
    
    parser.add_argument(
        "--preprocess-size",
        type=int,
        default=3,
        help="Preprocessing kernel size, -1 to disable (default: 3)"
    )
    
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Do not use parallel processing"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if output files exist"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without executing"
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit"
    )
    
    args = parser.parse_args()
    
    if args.version:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        try:
            version = version_file.read_text(encoding='utf-8').strip()
        except Exception:
            version = "unknown"
        print(f"simple_threshold.py version: {version}")
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Process files
    process_files(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        threshold_args=args.threshold,
        preprocess_method=args.preprocess,
        preprocess_size=args.preprocess_size,
        no_parallel=args.no_parallel,
        dry_run=args.dry_run,
        force=args.force
    )


if __name__ == "__main__":
    main()
