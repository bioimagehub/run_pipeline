"""
Bleach correction for time-series microscopy images using histogram matching.
Corrects photobleaching artifacts by matching intensity distributions across timepoints.

MIT License - BIPHUB, University of Oslo
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

import bioimage_pipeline_utils as rp

# Module-level logger
logger = logging.getLogger(__name__)


def histogram_correct(
        images: np.ndarray,
        contrast_limits: Tuple[int, int],
        match: str = "first"
) -> np.ndarray:
    """
    Apply histogram matching bleach correction to image time-series.
    
    Args:
        images: Input image array (3D or 4D: TYX or TZYX)
        contrast_limits: Min and max intensity values to clip to
        match: Matching method - "first" (match all to first frame) or 
               "neighbor" (match each to previous frame)
    
    Returns:
        Corrected image array with same shape as input
    """
    # cache image dtype
    dtype = images.dtype

    assert (
            3 <= len(images.shape) <= 4
    ), f"Expected 3d or 4d image stack, instead got {len(images.shape)} dimensions"

    if len(images.shape) == 3:
        k, m, n = images.shape
        z = None
        pixel_size = m * n
    else:
        k, z, m, n = images.shape
        pixel_size = z * m * n

    avail_match_methods = ["first", "neighbor"]
    assert (
        match in avail_match_methods
    ), f"'match' expected to be one of {avail_match_methods}, instead got {match}"

    # flatten the last dimensions and calculate normalized cdf
    images = images.reshape(k, -1)
    values, cdfs = [], []

    for i in range(k):

        if i > 0:
            if match == "first":
                match_ix = 0
            else:
                match_ix = i - 1

            val, ix, cnt = np.unique(images[i, ...].flatten(), return_inverse=True, return_counts=True)
            cdf = np.cumsum(cnt) / pixel_size

            interpolated = np.interp(cdf, cdfs[match_ix], values[match_ix])
            images[i, ...] = interpolated[ix]

        if i == 0 or match == "neighbor":
            val, cnt = np.unique(images[i, ...].flatten(), return_counts=True)
            cdf = np.cumsum(cnt) / pixel_size
            values.append(val)
            cdfs.append(cdf)

    if z is None:
        images = images.reshape(k, m, n)
    else:
        images = images.reshape(k, z, m, n)
    images[images < contrast_limits[0]] = contrast_limits[0]
    images[images > contrast_limits[1]] = contrast_limits[1]
    return images.astype(dtype)


def correct_single_file(
    input_path: str,
    output_path: str,
    contrast_limits: Tuple[int, int],
    match_method: str = "first"
) -> bool:
    """
    Apply bleach correction to a single image file.
    
    Args:
        input_path: Path to input image file
        output_path: Path to output corrected file
        contrast_limits: Min and max intensity values to clip to
        match_method: "first" or "neighbor" matching method
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Correcting: {os.path.basename(input_path)}")
        
        # Load image using standard TCZYX format
        img = rp.load_tczyx_image(input_path)
        data = img.data  # Get numpy array
        
        # Extract metadata for saving
        physical_pixel_sizes = None
        channel_names = None
        try:
            if hasattr(img, 'physical_pixel_sizes'):
                physical_pixel_sizes = img.physical_pixel_sizes
            if hasattr(img, 'channel_names'):
                channel_names = [str(name) for name in img.channel_names]
            logger.info(f"Extracted metadata - Pixel sizes: {physical_pixel_sizes}, Channels: {channel_names}")
        except Exception as e:
            logger.warning(f"Could not extract metadata: {e}")
        
        # Process each channel separately
        T, C, Z, Y, X = data.shape
        logger.info(f"Processing image with shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
        
        corrected_data = np.zeros_like(data)
        
        for c in range(C):
            logger.info(f"Correcting channel {c+1}/{C}")
            
            # Extract channel data (TZYX or TYX depending on Z dimension)
            if Z > 1:
                channel_data = data[:, c, :, :, :]  # TZYX
            else:
                channel_data = data[:, c, 0, :, :]  # TYX (squeeze Z)
            
            # Apply correction
            corrected_channel = histogram_correct(
                channel_data,
                contrast_limits=contrast_limits,
                match=match_method
            )
            
            # Put back into full array
            if Z > 1:
                corrected_data[:, c, :, :, :] = corrected_channel
            else:
                corrected_data[:, c, 0, :, :] = corrected_channel
        
        # Save corrected image with metadata
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        save_kwargs = {}
        if physical_pixel_sizes is not None:
            save_kwargs['physical_pixel_sizes'] = physical_pixel_sizes
        if channel_names is not None:
            save_kwargs['channel_names'] = channel_names
        
        rp.save_tczyx_image(corrected_data, output_path, **save_kwargs)
        logger.info(f"Saved: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to correct {input_path}: {e}")
        return False


def process_files(
    input_pattern: str,
    output_folder: Optional[str] = None,
    contrast_limits: Tuple[int, int] = (0, 65535),
    match_method: str = "first",
    collapse_delimiter: str = "__",
    no_parallel: bool = False,
    output_extension: str = "_bleach_corrected",
    dry_run: bool = False
) -> None:
    """
    Process multiple files matching a pattern.
    
    Args:
        input_pattern: File search pattern (supports ** for recursive)
        output_folder: Output directory (default: input_dir + '_bleach_corrected')
        contrast_limits: Min and max intensity values to clip to
        match_method: "first" or "neighbor" histogram matching method
        collapse_delimiter: Delimiter for collapsing subfolder paths
        no_parallel: Disable parallel processing
        output_extension: Extension to add before .tif
        dry_run: Only print planned actions without executing
    """
    # Find files
    search_subfolders = '**' in input_pattern
    files = rp.get_files_to_process2(input_pattern, search_subfolders=search_subfolders)
    
    if not files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Found {len(files)} file(s) to process")
    
    # Determine base folder
    if '**' in input_pattern:
        base_folder = input_pattern.split('**')[0].rstrip('/\\')
        if not base_folder:
            base_folder = os.getcwd()
        base_folder = os.path.abspath(base_folder)
    else:
        base_folder = str(Path(files[0]).parent)
    
    # Determine output folder
    if output_folder is None:
        output_folder = base_folder + "_bleach_corrected"
    
    logger.info(f"Output folder: {output_folder}")
    
    # Prepare file pairs
    file_pairs = []
    for src in files:
        collapsed = rp.collapse_filename(src, base_folder, collapse_delimiter)
        out_name = os.path.splitext(collapsed)[0] + output_extension + ".tif"
        out_path = os.path.join(output_folder, out_name)
        file_pairs.append((src, out_path))
    
    # Dry run - just print plans
    if dry_run:
        print(f"[DRY RUN] Would process {len(file_pairs)} files")
        print(f"[DRY RUN] Output folder: {output_folder}")
        print(f"[DRY RUN] Contrast limits: {contrast_limits}")
        print(f"[DRY RUN] Match method: {match_method}")
        for src, dst in file_pairs:
            print(f"[DRY RUN] {src} -> {dst}")
        return
    
    # Process files
    if no_parallel or len(file_pairs) == 1:
        # Sequential processing
        for src, dst in file_pairs:
            correct_single_file(src, dst, contrast_limits, match_method)
    else:
        # Parallel processing
        max_workers = min(os.cpu_count() or 4, len(file_pairs))
        logger.info(f"Processing with {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(correct_single_file, src, dst, contrast_limits, match_method): (src, dst)
                for src, dst in file_pairs
            }
            
            for future in as_completed(futures):
                src, dst = futures[future]
                try:
                    success = future.result()
                    if not success:
                        logger.error(f"Failed: {src}")
                except Exception as e:
                    logger.error(f"Exception processing {src}: {e}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bleach correction for time-series microscopy using histogram matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Bleach correction (default settings)
  environment: uv@3.11:bleach-correct
  commands:
  - python
  - '%REPO%/standard_code/python/bleach_correct.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_bleach_corrected'

- name: Bleach correction (neighbor matching, 8-bit)
  environment: uv@3.11:bleach-correct
  commands:
  - python
  - '%REPO%/standard_code/python/bleach_correct.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_bleach_corrected'
  - --match-method: neighbor
  - --contrast-min: 0
  - --contrast-max: 255

- name: Bleach correction (dry run preview)
  environment: uv@3.11:bleach-correct
  commands:
  - python
  - '%REPO%/standard_code/python/bleach_correct.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --dry-run
        """
    )
    
    parser.add_argument(
        "--input-search-pattern",
        type=str,
        required=True,
        help="Input file pattern (supports wildcards, use '**' for recursive search)"
    )
    
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder (default: input_folder + '_bleach_corrected')"
    )
    
    parser.add_argument(
        "--contrast-min",
        type=int,
        default=0,
        help="Minimum intensity value for clipping (default: 0)"
    )
    
    parser.add_argument(
        "--contrast-max",
        type=int,
        default=65535,
        help="Maximum intensity value for clipping (default: 65535 for 16-bit)"
    )
    
    parser.add_argument(
        "--match-method",
        type=str,
        default="first",
        choices=["first", "neighbor"],
        help="Histogram matching method: 'first' (match all to first frame) or 'neighbor' (match to previous frame)"
    )
    
    parser.add_argument(
        "--collapse-delimiter",
        type=str,
        default="__",
        help="Delimiter for collapsing subfolder paths (default: '__')"
    )
    
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (process files sequentially)"
    )
    
    parser.add_argument(
        "--output-file-name-extension",
        type=str,
        default="_bleach_corrected",
        help="Extension to add before .tif (default: '_bleach_corrected')"
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
        print(f"bleach_correct.py version: {version}")
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Build contrast limits tuple
    contrast_limits = (args.contrast_min, args.contrast_max)
    
    # Process files
    process_files(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        contrast_limits=contrast_limits,
        match_method=args.match_method,
        collapse_delimiter=args.collapse_delimiter,
        no_parallel=args.no_parallel,
        output_extension=args.output_file_name_extension,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()