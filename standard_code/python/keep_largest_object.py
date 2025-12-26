"""
Keep Largest N Objects in Binary Masks (TZYX Connected Components)

Labels connected components across time and Z (TZYX) and keeps only the N largest objects.
Useful for filtering segmentation masks to retain only the primary tracked objects.

Uses 4D connected component labeling (time and Z as spatial dimensions) to track objects
across timepoints and Z-slices, then filters by total volume to keep the top N largest.

Author: BIPHUB, University of Oslo
Written by: Øyvind Ødegård Fougner
License: MIT
"""

from __future__ import annotations

import os
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage
import tifffile

import bioimage_pipeline_utils as rp


# Module-level logger
logger = logging.getLogger(__name__)


def keep_largest_n_objects_tzyx(
    mask: np.ndarray,
    n: int = 1,
    connectivity: int = 1
) -> np.ndarray:
    """
    Label objects in TZYX and keep only the N largest by total volume.
    
    Performs 4D connected component labeling treating T and Z as spatial dimensions,
    effectively tracking objects across time and Z-slices. Keeps only the top N objects
    with the largest total volumes.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask array of shape (T, Z, Y, X) with values 0 or 255.
    n : int
        Number of largest objects to keep. Default is 1.
    connectivity : int
        Connectivity for labeling (1 for face neighbors, 2 for face+edge, 3 for all).
        Default is 1 (8-connectivity in 4D).
    
    Returns
    -------
    filtered_mask : np.ndarray
        Binary mask with only the N largest objects retained (0 or 255).
    
    Examples
    --------
    >>> # Keep largest tracked object across time and Z
    >>> mask_tzyx = load_mask("tracked_cells.tif")  # TZYX
    >>> filtered = keep_largest_n_objects_tzyx(mask_tzyx, n=1)
    
    >>> # Keep top 3 largest objects
    >>> filtered = keep_largest_n_objects_tzyx(mask_tzyx, n=3)
    
    Notes
    -----
    - Treats time and Z as spatial dimensions for connectivity
    - Objects connected across time/Z are considered the same object
    - Useful for multi-object tracking or removing spurious detections
    """
    if mask.ndim != 4:
        raise ValueError(f"Expected 4D TZYX array, got shape {mask.shape}")
    
    # Convert to binary (handle both 0/1 and 0/255)
    binary = mask > 0
    
    # Handle empty mask
    if not binary.any():
        logger.warning("Mask is empty, returning zeros")
        return np.zeros_like(mask, dtype=np.uint8)
    
    # Label connected components in 4D (TZYX)
    logger.debug(f"Labeling components (connectivity={connectivity})")
    labeled, num_features = ndimage.label(binary, structure=ndimage.generate_binary_structure(4, connectivity))
    
    if num_features == 0:
        logger.warning("No objects found after labeling")
        return np.zeros_like(mask, dtype=np.uint8)
    
    logger.info(f"Found {num_features} connected objects in TZYX")
    
    # Count pixels per label (total volume across all T and Z)
    label_volumes = np.bincount(labeled.ravel())
    label_volumes[0] = 0  # Ignore background
    
    # Find top N largest objects
    n_to_keep = min(n, num_features)
    if n_to_keep < n:
        logger.warning(f"Requested to keep {n} objects, but only {num_features} found. Keeping all.")
    
    # Get indices of top N largest (excluding background at index 0)
    top_n_labels = np.argsort(label_volumes)[-n_to_keep:]
    
    logger.info(f"Keeping top {n_to_keep} largest objects:")
    for rank, label_idx in enumerate(reversed(top_n_labels), 1):
        logger.info(f"  Rank {rank}: label={label_idx}, volume={label_volumes[label_idx]} pixels")
    
    # Keep only top N objects
    filtered = np.isin(labeled, top_n_labels).astype(np.uint8) * 255
    
    return filtered


def save_imagej_tif(mask: np.ndarray, output_path: str) -> None:
    """
    Save mask as ImageJ-compatible TIFF (non-OME for thumbnail support).
    
    Args:
        mask: 5D mask (T, C, Z, Y, X) in TCZYX order
        output_path: Output file path
    """
    # Convert to binary 0/255 format
    mask_out = (mask > 0).astype(np.uint8) * 255
    
    # Remove C dimension if single channel and convert to TZYX for ImageJ
    if mask_out.shape[1] == 1:
        mask_out = mask_out[:, 0, :, :, :]
    
    logger.info(f"Saving mask (TZYX): shape={mask_out.shape}, dtype={mask_out.dtype}, max={mask_out.max()}")
    
    # Save as ImageJ-compatible TIFF (ImageJ expects TZYX order for stacks)
    tifffile.imwrite(
        output_path,
        mask_out,
        imagej=True,
        metadata={'axes': 'TZYX'},
        compression='deflate'
    )
    
    logger.info(f"Saved: {output_path}")


def process_single_file(
    input_path: str,
    output_path: str,
    channels: list[int] | None,
    n: int,
    connectivity: int,
    force: bool,
    output_suffix: str = "_largest"
) -> bool:
    """
    Process a single mask file: load, filter top N largest objects per channel, save.
    
    Parameters
    ----------
    input_path : str
        Path to input binary mask file.
    output_path : str
        Path where filtered mask will be saved.
    channels : list[int] | None
        Channel indices to process. If None, processes all channels.
    n : int
        Number of largest objects to keep.
    connectivity : int
        Connectivity for component labeling (1, 2, or 3).
    force : bool
        If True, overwrite existing output files.
    
    Returns
    -------
    success : bool
        True if processing succeeded, False otherwise.
    """
    try:
        logger.info(f"Processing: {os.path.basename(input_path)}")
        
        if os.path.exists(output_path) and not force:
            logger.info(f"Output exists, skipping: {output_path}")
            return True
        
        # Load mask
        img = rp.load_tczyx_image(input_path)
        T, C, Z, Y, X = img.shape
        logger.info(f"Dimensions: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
        
        # Determine which channels to process
        if channels is None:
            channels_to_process = list(range(C))
        else:
            channels_to_process = [c for c in channels if 0 <= c < C]
            if not channels_to_process:
                raise ValueError(
                    f"No valid channels found. Mask has {C} channels, requested {channels}"
                )
        
        logger.info(f"Processing channels: {channels_to_process}")
        logger.info(f"Keeping top {n} largest object(s) per channel")
        
        # Create output array (copy to preserve non-processed channels)
        output_data = img.data.copy()
        
        # Process each channel
        for c in channels_to_process:
            logger.info(f"  Channel {c}:")
            
            # Process entire TZYX volume for this channel
            mask_tzyx = img.data[:, c, :, :, :]  # Shape: (T, Z, Y, X)
            
            # Skip empty channels
            if not mask_tzyx.any():
                logger.info(f"    Empty, skipping")
                continue
            
            filtered = keep_largest_n_objects_tzyx(mask_tzyx, n=n, connectivity=connectivity)
            output_data[:, c, :, :, :] = filtered
        
        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_imagej_tif(output_data, output_path)
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_files(
    input_pattern: str,
    output_folder: str | None,
    channels: list[int] | None,
    n: int,
    connectivity: int,
    no_parallel: bool,
    dry_run: bool,
    force: bool,
    output_suffix: str = "_largest"
) -> None:
    """
    Process multiple mask files with optional parallel execution.
    
    Parameters
    ----------
    input_pattern : str
        Input file search pattern (supports wildcards, '**' for recursive).
    output_folder : str | None
        Output directory. If None, uses input_folder + '_largest'.
    channels : list[int] | None
        Channel indices to process. If None, processes all channels.
    n : int
        Number of largest objects to keep.
    connectivity : int
        Connectivity for component labeling (1, 2, or 3).
    no_parallel : bool
        If True, disable parallel processing.
    dry_run : bool
        If True, only print planned actions without executing.
    force : bool
        If True, overwrite existing output files.
    output_suffix : str
        Suffix to add to output filenames (default: '_largest'). Use empty string '' to overwrite original files.
    """
    search_subfolders = "**" in input_pattern
    input_files = rp.get_files_to_process2(input_pattern, search_subfolders=search_subfolders)
    
    if not input_files:
        logger.error(f"No files matched pattern: {input_pattern}")
        return
    
    logger.info(f"Found {len(input_files)} files")
    
    # Default output folder
    if output_folder is None:
        base_dir = os.path.dirname(input_pattern.replace("**/", "").replace("*", ""))
        output_folder = (base_dir or ".") + "_largest"
    
    logger.info(f"Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    # Build task list
    tasks: list[tuple[str, str]] = []
    for input_path in input_files:
        basename = os.path.splitext(os.path.basename(input_path))[0]
        if output_suffix:
            output_filename = f"{basename}{output_suffix}.tif"
        else:
            # Empty suffix: use original filename to overwrite
            output_filename = f"{basename}.tif"
        output_path = os.path.join(output_folder, output_filename)
        tasks.append((input_path, output_path))
    
    # Dry run
    if dry_run:
        print(f"[DRY RUN] Would process {len(tasks)} files")
        print(f"[DRY RUN] Output folder: {output_folder}")
        print(f"[DRY RUN] Keep top N objects: {n}")
        print(f"[DRY RUN] Connectivity: {connectivity}")
        if channels is None:
            print("[DRY RUN] Channels: all")
        else:
            print(f"[DRY RUN] Channels: {channels}")
        for inp, out in tasks[:5]:
            print(f"[DRY RUN]   {os.path.basename(inp)} -> {os.path.basename(out)}")
        if len(tasks) > 5:
            print(f"[DRY RUN]   ... and {len(tasks) - 5} more files")
        return
    
    # Sequential processing
    if no_parallel or len(tasks) == 1:
        logger.info("Processing files sequentially")
        ok = 0
        for inp, out in tasks:
            if process_single_file(inp, out, channels, n, connectivity, force, output_suffix):
                ok += 1
        logger.info(f"Done: {ok} succeeded, {len(tasks)-ok} failed")
        return
    
    # Parallel processing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    cpu_count = os.cpu_count() or 1
    max_workers = max(cpu_count - 1, 1)
    logger.info(f"Processing files in parallel (workers={max_workers})")
    
    ok = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                process_single_file,
                inp, out, channels, n, connectivity, force, output_suffix
            )
            for inp, out in tasks
        ]
        for f in as_completed(futures):
            try:
                if f.result():
                    ok += 1
            except Exception as e:
                logger.error(f"Task failed: {e}")
    
    logger.info(f"Done: {ok} succeeded, {len(tasks)-ok} failed")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Keep top N largest objects in binary masks using TZYX connected components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Keep largest tracked object
  environment: uv@3.11:mask-filtering
  commands:
  - python
  - '%REPO%/standard_code/python/keep_largest_object.py'
  - --input-search-pattern: '%YAML%/masks/**/*_mask.tif'
  - --output-folder: '%YAML%/masks_largest'

- name: Keep top 3 largest objects
  environment: uv@3.11:mask-filtering
  commands:
  - python
  - '%REPO%/standard_code/python/keep_largest_object.py'
  - --input-search-pattern: '%YAML%/masks/**/*_mask.tif'
  - --output-folder: '%YAML%/masks_top3'
  - --keep-n: 3

- name: Keep largest object in channel 0 only
  environment: uv@3.11:mask-filtering
  commands:
  - python
  - '%REPO%/standard_code/python/keep_largest_object.py'
  - --input-search-pattern: '%YAML%/masks/**/*_mask.tif'
  - --output-folder: '%YAML%/masks_largest_ch0'
  - --channels
  - '0'

- name: Keep top 2 objects with high connectivity (face+edge+corner)
  environment: uv@3.11:mask-filtering
  commands:
  - python
  - '%REPO%/standard_code/python/keep_largest_object.py'
  - --input-search-pattern: '%YAML%/masks/**/*_mask.tif'
  - --output-folder: '%YAML%/masks_top2'
  - --keep-n: 2
  - --connectivity: 3

Description:
  Labels connected components across time and Z (TZYX) and keeps only the top N largest objects.
  Useful for filtering segmentation masks to retain only the primary tracked objects.

Algorithm:
  1. Treats T (time) and Z as spatial dimensions for 4D connected component labeling
  2. Labels all objects in TZYX (objects connected across time/Z = same object)
  3. Counts total volume (pixels) for each object across all timepoints and Z-slices
  4. Keeps only the top N objects with the largest total volumes
  5. Sets all other objects to 0 (background)

Connectivity:
  - 1 (default): Face neighbors only (8-connectivity in 4D)
  - 2: Face + edge neighbors (higher connectivity in 4D)
  - 3: Face + edge + corner neighbors (full 4D connectivity)

Use Cases:
  - Single-cell tracking: Remove debris and keep main cell (--keep-n 1)
  - Multi-object tracking: Keep top N cells (--keep-n N)
  - Spurious detection removal: Clean up over-segmentation
  - Time-series filtering: Ensure temporal and spatial continuity

Notes:
  - Input must be binary masks (0/255 or 0/1)
  - Each channel is processed independently
  - Objects must be connected across time and/or Z to be considered the same object
  - TZYX connectivity means objects can span multiple Z-slices and timepoints
        """
    )
    
    parser.add_argument(
        "--input-search-pattern",
        required=True,
        help="Input file pattern (supports wildcards, use '**' for recursive)"
    )
    
    parser.add_argument(
        "--output-folder",
        help="Output folder (default: input_folder + '_largest')"
    )
    
    parser.add_argument(
        "--channels",
        type=int,
        nargs='+',
        default=None,
        help=(
            "Channel indices to process (0-based). If not specified, processes all channels. "
            "Example: --channels 0 2"
        )
    )
    
    parser.add_argument(
        "--keep-n",
        type=int,
        default=1,
        help="Number of largest objects to keep per channel. Default: 1"
    )
    
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help=(
            "Connectivity for component labeling: 1 (face), 2 (face+edge), 3 (face+edge+corner). "
            "Default: 1"
        )
    )
    
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_largest",
        help="Suffix to add to output filenames (default: '_largest'). Use empty string '' to overwrite original files in output folder."
    )
    
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing"
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
            version = version_file.read_text(encoding="utf-8").strip()
        except Exception:
            version = "unknown"
        print(f"keep_largest_object.py version: {version}")
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
        channels=args.channels,
        n=args.keep_n,
        connectivity=args.connectivity,
        no_parallel=args.no_parallel,
        dry_run=args.dry_run,
        force=args.force,
        output_suffix=args.output_suffix
    )


if __name__ == "__main__":
    main()
