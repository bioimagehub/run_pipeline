"""
Keep N Closest Objects to ROI Points (TZYX Connected Components)

Labels connected components across time and Z (TZYX) and keeps only the N closest objects
to ROI point(s) defined in metadata YAML files. Useful for filtering segmentation masks
based on spatial proximity to points of interest.

Uses 4D connected component labeling (time and Z as spatial dimensions) to track objects,
calculates centroid distances to ROI points, and filters by proximity.

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
import yaml
import tifffile

import bioimage_pipeline_utils as rp


# Module-level logger
logger = logging.getLogger(__name__)


def load_roi_points_from_yaml(yaml_path: str) -> list[tuple[float, float, float]]:
    """
    Load ROI points from metadata YAML file.
    
    Expected YAML structures:
    
    1. Standard pipeline format (Image metadata structure):
        Image metadata:
          ROIs:
            - Roi:
                Positions:
                  x: 319.37
                  y: 102.78
                Size:
                  x: 11.27
                  y: 11.27
                Shape: circle
                Type: stimulation
    
    2. Simple structured format:
        rois:
          - name: "ROI1"
            x: 512.5
            y: 384.2
            z: 10.0
    
    3. Simple list format:
        roi_points:
          - [512.5, 384.2, 10.0]  # [x, y, z]
    
    Parameters
    ----------
    yaml_path : str
        Path to YAML metadata file.
    
    Returns
    -------
    points : list[tuple[float, float, float]]
        List of (x, y, z) coordinates for each ROI point.
        Z defaults to 0.0 if not provided in YAML.
    """
    try:
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        points = []
        
        # Try standard pipeline format: Image metadata -> ROIs -> Roi -> Positions
        if 'Image metadata' in metadata:
            img_meta = metadata['Image metadata']
            if 'ROIs' in img_meta and isinstance(img_meta['ROIs'], list):
                for roi_item in img_meta['ROIs']:
                    if 'Roi' in roi_item:
                        roi = roi_item['Roi']
                        if 'Positions' in roi:
                            pos = roi['Positions']
                            x = float(pos.get('x', 0))
                            y = float(pos.get('y', 0))
                            z = float(pos.get('z', 0))  # Default to 0 if no Z
                            
                            roi_type = roi.get('Type', 'unknown')
                            roi_shape = roi.get('Shape', 'unknown')
                            points.append((x, y, z))
                            logger.debug(f"  Found ROI (type={roi_type}, shape={roi_shape}) at ({x:.2f}, {y:.2f}, {z:.2f})")
        
        # Try simple structured format
        if not points and 'rois' in metadata and isinstance(metadata['rois'], list):
            for roi in metadata['rois']:
                if 'x' in roi and 'y' in roi:
                    x = float(roi['x'])
                    y = float(roi['y'])
                    z = float(roi.get('z', 0))  # Default to 0 if no Z
                    points.append((x, y, z))
                    logger.debug(f"  Found ROI '{roi.get('name', 'unnamed')}' at ({x:.2f}, {y:.2f}, {z:.2f})")
        
        # Try simple list format
        if not points and 'roi_points' in metadata and isinstance(metadata['roi_points'], list):
            for point in metadata['roi_points']:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    x = float(point[0])
                    y = float(point[1])
                    z = float(point[2]) if len(point) >= 3 else 0.0
                    points.append((x, y, z))
                    logger.debug(f"  Found ROI point at ({x:.2f}, {y:.2f}, {z:.2f})")
        
        if not points:
            logger.warning(f"No ROI points found in {yaml_path}")
        else:
            logger.info(f"Loaded {len(points)} ROI point(s) from {yaml_path}")
        
        return points
        
    except Exception as e:
        logger.error(f"Failed to load ROI points from {yaml_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


def calculate_object_centroids_tzyx(
    labeled: np.ndarray,
    num_labels: int
) -> dict[int, tuple[float, float, float]]:
    """
    Calculate centroids for each labeled object in TZYX.
    
    Centroid is averaged across all timepoints, returning (x, y, z) in pixel coordinates.
    
    Parameters
    ----------
    labeled : np.ndarray
        Labeled mask array of shape (T, Z, Y, X).
    num_labels : int
        Number of labeled objects (excluding background).
    
    Returns
    -------
    centroids : dict[int, tuple[float, float, float]]
        Mapping from label ID to (x, y, z) centroid coordinates.
    """
    centroids = {}
    
    for label_id in range(1, num_labels + 1):
        coords = np.argwhere(labeled == label_id)  # Returns (T, Z, Y, X) indices
        
        if len(coords) == 0:
            continue
        
        # Average across all timepoints to get (z, y, x) centroid
        t_mean = coords[:, 0].mean()
        z_mean = coords[:, 1].mean()
        y_mean = coords[:, 2].mean()
        x_mean = coords[:, 3].mean()
        
        # Return as (x, y, z) to match ROI point format
        centroids[label_id] = (x_mean, y_mean, z_mean)
        logger.debug(f"  Label {label_id}: centroid at ({x_mean:.1f}, {y_mean:.1f}, {z_mean:.1f}), T_avg={t_mean:.1f}")
    
    return centroids


def euclidean_distance_3d(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> float:
    """Calculate Euclidean distance between two 3D points (x, y, z)."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


def keep_closest_n_objects_to_rois(
    mask: np.ndarray,
    roi_points: list[tuple[float, float, float]],
    n: int = 1,
    connectivity: int = 1
) -> np.ndarray:
    """
    Label objects in TZYX and keep only the N closest objects to each ROI point.
    
    Performs 4D connected component labeling, calculates centroid for each object,
    and keeps the N closest objects to each ROI point (union across all ROI points).
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask array of shape (T, Z, Y, X) with values 0 or 255.
    roi_points : list[tuple[float, float, float]]
        List of (x, y, z) coordinates for ROI points.
    n : int
        Number of closest objects to keep per ROI point. Default is 1.
    connectivity : int
        Connectivity for labeling (1 for face neighbors, 2 for face+edge, 3 for all).
        Default is 1 (8-connectivity in 4D).
    
    Returns
    -------
    filtered_mask : np.ndarray
        Binary mask with only the N closest objects to ROI points retained (0 or 255).
    
    Examples
    --------
    >>> # Keep closest object to each ROI point
    >>> mask_tzyx = load_mask("tracked_cells.tif")
    >>> roi_points = [(512, 384, 10), (256, 512, 5)]
    >>> filtered = keep_closest_n_objects_to_rois(mask_tzyx, roi_points, n=1)
    
    >>> # Keep 3 closest objects to each ROI
    >>> filtered = keep_closest_n_objects_to_rois(mask_tzyx, roi_points, n=3)
    
    Notes
    -----
    - Treats time and Z as spatial dimensions for connectivity
    - Objects connected across time/Z are considered the same object
    - Centroids are averaged across all timepoints
    - If multiple ROI points exist, keeps union of N closest objects to each point
    """
    if mask.ndim != 4:
        raise ValueError(f"Expected 4D TZYX array, got shape {mask.shape}")
    
    if not roi_points:
        logger.warning("No ROI points provided, returning empty mask")
        return np.zeros_like(mask, dtype=np.uint8)
    
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
    
    # Calculate centroids for all objects
    centroids = calculate_object_centroids_tzyx(labeled, num_features)
    
    # For each ROI point, find N closest objects
    labels_to_keep = set()
    
    for roi_idx, roi_point in enumerate(roi_points, 1):
        logger.info(f"ROI {roi_idx} at ({roi_point[0]:.1f}, {roi_point[1]:.1f}, {roi_point[2]:.1f}):")
        
        # Calculate distances from this ROI to all object centroids
        distances = {}
        for label_id, centroid in centroids.items():
            dist = euclidean_distance_3d(roi_point, centroid)
            distances[label_id] = dist
        
        # Find N closest objects
        n_to_keep = min(n, len(distances))
        closest_labels = sorted(distances.items(), key=lambda x: x[1])[:n_to_keep]
        
        for rank, (label_id, dist) in enumerate(closest_labels, 1):
            logger.info(f"  Rank {rank}: label={label_id}, distance={dist:.2f} pixels, centroid={centroids[label_id]}")
            labels_to_keep.add(label_id)
    
    logger.info(f"Keeping {len(labels_to_keep)} object(s) total (union across all ROI points)")
    
    # Keep only selected objects
    filtered = np.isin(labeled, list(labels_to_keep)).astype(np.uint8) * 255
    
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


def process_single_file_group(
    mask_path: str,
    yaml_path: str,
    output_path: str,
    channels: list[int] | None,
    n: int,
    connectivity: int,
    force: bool,
    output_suffix: str = "_closest"
) -> bool:
    """
    Process a single mask file with its metadata: load mask and ROI points, filter, save.
    
    Parameters
    ----------
    mask_path : str
        Path to input binary mask file.
    yaml_path : str
        Path to metadata YAML file containing ROI points.
    output_path : str
        Path where filtered mask will be saved.
    channels : list[int] | None
        Channel indices to process. If None, processes all channels.
    n : int
        Number of closest objects to keep per ROI point.
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
        logger.info(f"Processing: {os.path.basename(mask_path)}")
        logger.info(f"  Output path: {output_path}")
        
        if os.path.exists(output_path) and not force:
            logger.info(f"Output exists, skipping: {output_path}")
            return True
        
        # Load ROI points from YAML
        roi_points = load_roi_points_from_yaml(yaml_path)
        if not roi_points:
            logger.warning(f"No ROI points found in {yaml_path}, skipping")
            return False
        
        # Load mask
        img = rp.load_tczyx_image(mask_path)
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
        logger.info(f"Keeping top {n} closest object(s) to each of {len(roi_points)} ROI point(s)")
        
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
            
            filtered = keep_closest_n_objects_to_rois(
                mask_tzyx, 
                roi_points=roi_points,
                n=n, 
                connectivity=connectivity
            )
            output_data[:, c, :, :, :] = filtered
        
        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_imagej_tif(output_data, output_path)
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {mask_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_files(
    mask_pattern: str,
    yaml_pattern: str,
    output_folder: str | None,
    channels: list[int] | None,
    n: int,
    connectivity: int,
    no_parallel: bool,
    dry_run: bool,
    force: bool,
    output_suffix: str = "_closest"
) -> None:
    """
    Process multiple mask files with their metadata files using grouped file matching.
    
    Parameters
    ----------
    mask_pattern : str
        Mask file search pattern (supports wildcards, '**' for recursive).
    yaml_pattern : str
        YAML metadata file search pattern.
    output_folder : str | None
        Output directory. If None, uses mask_folder + '_closest'.
    channels : list[int] | None
        Channel indices to process. If None, processes all channels.
    n : int
        Number of closest objects to keep per ROI point.
    connectivity : int
        Connectivity for component labeling (1, 2, or 3).
    no_parallel : bool
        If True, disable parallel processing.
    dry_run : bool
        If True, only print planned actions without executing.
    force : bool
        If True, overwrite existing output files.
    output_suffix : str
        Suffix to add to output filenames (default: '_closest'). Use empty string '' to overwrite original files.
    """
    # Build search patterns dictionary for grouping
    search_patterns = {
        'mask': mask_pattern,
        'yaml': yaml_pattern
    }
    
    search_subfolders = "**" in mask_pattern or "**" in yaml_pattern
    grouped_files = rp.get_grouped_files_to_process(search_patterns, search_subfolders)
    
    if not grouped_files:
        logger.error(f"No matching file groups found")
        return
    
    logger.info(f"Found {len(grouped_files)} file groups")
    logger.info(f"Mask pattern: {mask_pattern}")
    logger.info(f"YAML pattern: {yaml_pattern}")
    
    # Default output folder
    if output_folder is None:
        base_dir = os.path.dirname(mask_pattern.replace("**/", "").replace("*", ""))
        output_folder = (base_dir or ".") + "_closest"
    
    logger.info(f"Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    # Build task list with deduplication by mask file path
    tasks: list[tuple[str, str, str]] = []
    processed_masks = set()  # Track which mask files we've already added
    
    # Debug: Show first few groups
    for i, (basename, files) in enumerate(list(grouped_files.items())[:5]):
        logger.info(f"Group {i}: basename='{basename}', mask={files.get('mask', 'MISSING')}, yaml={files.get('yaml', 'MISSING')}")
    
    for basename, files in grouped_files.items():
        if 'mask' not in files or 'yaml' not in files:
            logger.warning(f"Skipping {basename}: missing mask or yaml file")
            continue
        
        mask_path = files['mask']
        yaml_path = files['yaml']
        
        # Skip if we've already processed this mask file
        if mask_path in processed_masks:
            logger.info(f"Skipping duplicate mask: {os.path.basename(mask_path)}")
            continue
        processed_masks.add(mask_path)
        
        mask_basename = os.path.splitext(os.path.basename(mask_path))[0]
        if output_suffix:
            output_filename = f"{mask_basename}{output_suffix}.tif"
        else:
            # Empty suffix: use original filename to overwrite
            output_filename = f"{mask_basename}.tif"
        output_path = os.path.join(output_folder, output_filename)
        
        tasks.append((mask_path, yaml_path, output_path))
    
    # Debug: Show first few tasks
    logger.info(f"Built {len(tasks)} tasks")
    for i, (mask_p, yaml_p, out_p) in enumerate(tasks[:5]):
        logger.info(f"Task {i}: mask={os.path.basename(mask_p)}, output={os.path.basename(out_p)}")
    
    if not tasks:
        logger.error("No valid file groups to process")
        return
    
    # Dry run
    if dry_run:
        print(f"[DRY RUN] Would process {len(tasks)} file groups")
        print(f"[DRY RUN] Output folder: {output_folder}")
        print(f"[DRY RUN] Keep N closest objects per ROI: {n}")
        print(f"[DRY RUN] Connectivity: {connectivity}")
        if channels is None:
            print("[DRY RUN] Channels: all")
        else:
            print(f"[DRY RUN] Channels: {channels}")
        for mask_p, yaml_p, out_p in tasks[:5]:
            print(f"[DRY RUN]   Mask: {os.path.basename(mask_p)}")
            print(f"[DRY RUN]   YAML: {os.path.basename(yaml_p)}")
            print(f"[DRY RUN]   Out:  {os.path.basename(out_p)}")
        if len(tasks) > 5:
            print(f"[DRY RUN]   ... and {len(tasks) - 5} more file groups")
        return
    
    # Sequential processing
    if no_parallel or len(tasks) == 1:
        logger.info("Processing files sequentially")
        ok = 0
        for mask_p, yaml_p, out_p in tasks:
            if process_single_file_group(mask_p, yaml_p, out_p, channels, n, connectivity, force, output_suffix):
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
                process_single_file_group,
                mask_p, yaml_p, out_p, channels, n, connectivity, force, output_suffix
            )
            for mask_p, yaml_p, out_p in tasks
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
        description="Keep N closest objects to ROI points using TZYX connected components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Keep closest object to ROI points
  environment: uv@3.11:mask-filtering
  commands:
  - python
  - '%REPO%/standard_code/python/keep_closest_object.py'
  - --mask-search-pattern: '%YAML%/masks/**/*_mask.tif'
  - --yaml-search-pattern: '%YAML%/metadata/**/*_metadata.yaml'
  - --output-folder: '%YAML%/masks_closest'

- name: Keep 3 closest objects to each ROI point
  environment: uv@3.11:mask-filtering
  commands:
  - python
  - '%REPO%/standard_code/python/keep_closest_object.py'
  - --mask-search-pattern: '%YAML%/masks/**/*_mask.tif'
  - --yaml-search-pattern: '%YAML%/metadata/**/*_metadata.yaml'
  - --output-folder: '%YAML%/masks_closest3'
  - --keep-n: 3

- name: Keep closest objects in channel 0 only
  environment: uv@3.11:mask-filtering
  commands:
  - python
  - '%REPO%/standard_code/python/keep_closest_object.py'
  - --mask-search-pattern: '%YAML%/masks/**/*_mask.tif'
  - --yaml-search-pattern: '%YAML%/metadata/**/*_metadata.yaml'
  - --output-folder: '%YAML%/masks_closest_ch0'
  - --channels
  - '0'

Description:
  Labels connected components across time and Z (TZYX) and keeps only the N closest objects
  to ROI point(s) defined in metadata YAML files. Useful for filtering segmentation masks
  based on spatial proximity to points of interest.

Algorithm:
  1. Load ROI points from metadata YAML file
  2. Treat T (time) and Z as spatial dimensions for 4D connected component labeling
  3. Calculate centroid (x, y, z) for each object (averaged across all timepoints)
  4. For each ROI point, calculate distances to all object centroids
  5. Keep the N closest objects to each ROI point (union if multiple ROI points)
  6. Set all other objects to 0 (background)

YAML Format:
  Structured format:
    rois:
      - name: "ROI1"
        x: 512.5
        y: 384.2
        z: 10.0
      - name: "ROI2"
        x: 256.0
        y: 512.0
        z: 5.0
  
  Simple list format:
    roi_points:
      - [512.5, 384.2, 10.0]  # [x, y, z]
      - [256.0, 512.0, 5.0]

Connectivity:
  - 1 (default): Face neighbors only (8-connectivity in 4D)
  - 2: Face + edge neighbors (higher connectivity in 4D)
  - 3: Face + edge + corner neighbors (full 4D connectivity)

Use Cases:
  - Region-specific tracking: Keep cells near specific anatomical landmarks
  - Multi-ROI analysis: Filter objects around multiple points of interest
  - Proximity-based filtering: Remove distant objects from analysis
  - Targeted segmentation cleanup: Keep only relevant objects near ROI

Notes:
  - Input must be binary masks (0/255 or 0/1)
  - Each channel is processed independently
  - Centroids are averaged across all timepoints
  - If multiple ROI points exist, returns union of N closest objects to each point
  - Mask and YAML files are matched by common basename
        """
    )
    
    parser.add_argument(
        "--mask-search-pattern",
        required=True,
        help="Mask file pattern (supports wildcards, use '**' for recursive)"
    )
    
    parser.add_argument(
        "--yaml-search-pattern",
        required=True,
        help="YAML metadata file pattern with ROI points (supports wildcards, use '**' for recursive)"
    )
    
    parser.add_argument(
        "--output-folder",
        help="Output folder (default: mask_folder + '_closest')"
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
        help="Number of closest objects to keep per ROI point. Default: 1"
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
        default="_closest",
        help="Suffix to add to output filenames (default: '_closest'). Use empty string '' to overwrite original files in output folder."
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
        print(f"keep_closest_object.py version: {version}")
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Process files
    process_files(
        mask_pattern=args.mask_search_pattern,
        yaml_pattern=args.yaml_search_pattern,
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
