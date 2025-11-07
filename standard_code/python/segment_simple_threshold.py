"""
Simple threshold-based segmentation (simplified to match ImageJ macro approach).
Python implementation of simple_treshold.ijm for reproducible batch processing.

MIT License - BIPHUB, University of Oslo
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy import ndimage
from skimage import filters, measure

import bioimage_pipeline_utils as rp

# Module-level logger
logger = logging.getLogger(__name__)


def median_filter_3d(data: np.ndarray, xy_size: int = 8, z_size: int = 2) -> np.ndarray:
    """
    Apply 3D median filter to image stack.
    
    Args:
        data: Input image array (3D: ZYX)
        xy_size: Median filter size for X and Y dimensions
        z_size: Median filter size for Z dimension
    
    Returns:
        Filtered image array
    """
    from scipy.ndimage import median_filter
    
    # Create footprint: (z, y, x)
    footprint = np.ones((z_size, xy_size, xy_size), dtype=bool)
    return median_filter(data, footprint=footprint)


def segment_single_file(
    input_path: str,
    output_path: str,
    minsize: int = 35000,
    maxsize: int = 120000,
    median_xy: int = 8,
    median_z: int = 2,
    threshold_method: str = 'li',
    max_objects: int = 1,
    save_roi: bool = True,
    use_watershed: bool = False,
    channel: int = 0
) -> bool:
    """
    Segment a single image file using simple threshold (like ImageJ macro).
    
    Args:
        input_path: Path to input image file
        output_path: Path to output mask file
        minsize: Minimum object size in pixels
        maxsize: Maximum object size in pixels
        median_xy: Median filter size for X and Y
        median_z: Median filter size for Z
        threshold_method: Threshold method ('li', 'otsu', 'yen')
        max_objects: Expected number of objects per slice (-1 to skip check)
        save_roi: Whether to save ROI coordinates as sidecar
        use_watershed: Apply watershed if object count doesn't match
        channel: Channel index to process (0-based, -1 for all channels)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing: {os.path.basename(input_path)}")
        
        # Check if output already exists
        if os.path.exists(output_path):
            logger.info(f"Output exists, skipping: {output_path}")
            return True
        
        # Load image
        img = rp.load_tczyx_image(input_path)
        
        # Process each timepoint and channel
        T, C, Z, Y, X = img.shape
        logger.info(f"Dimensions: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
        
        # Determine which channels to process
        if channel == -1:
            # Process all channels
            channels_to_process = list(range(C))
            logger.info(f"Processing all {C} channels")
        else:
            # Process single channel
            if channel >= C:
                logger.error(f"Channel {channel} does not exist (image has {C} channels)")
                return False
            channels_to_process = [channel]
            logger.info(f"Processing channel {channel} only")
        
        # Initialize output mask (single channel output)
        output_mask = np.zeros((T, 1, Z, Y, X), dtype=np.uint8)
        
        for t in range(T):
            for c_idx, c in enumerate(channels_to_process):
                logger.info(f"Processing T={t}, C={c}")
                
                # Get data for this TC
                data = img.data[t, c]  # ZYX
                
                # Convert to float for processing
                if data.max() > 0:
                    data_float = data.astype(np.float32)
                else:
                    logger.warning(f"T={t}, C={c}: No signal detected, skipping")
                    continue
                
                # Step 1: Median filter preprocessing
                preprocessed = median_filter_3d(data_float, median_xy, median_z)
                
                # Step 2: Auto threshold per slice
                intensity_mask = np.zeros_like(preprocessed, dtype=bool)
                for z in range(Z):
                    slice_data = preprocessed[z]
                    if slice_data.max() > 0:
                        if threshold_method == 'li':
                            thresh_val = filters.threshold_li(slice_data)
                        elif threshold_method == 'otsu':
                            thresh_val = filters.threshold_otsu(slice_data)
                        elif threshold_method == 'yen':
                            thresh_val = filters.threshold_yen(slice_data)
                        else:
                            thresh_val = filters.threshold_li(slice_data)
                        
                        intensity_mask[z] = slice_data > thresh_val
                
                # Step 3: Fill holes (3D)
                intensity_mask = ndimage.binary_fill_holes(intensity_mask)
                
                # Step 4: Analyze particles and filter by size
                final_mask = np.zeros_like(intensity_mask, dtype=bool)
                for z in range(Z):
                    labeled = measure.label(intensity_mask[z])
                    props = measure.regionprops(labeled)
                    
                    # Count objects in size range
                    valid_labels = []
                    for prop in props:
                        if minsize <= prop.area <= maxsize:
                            valid_labels.append(prop.label)
                    
                    n_objects = len(valid_labels)
                    
                    # Step 5: If wrong object count and watershed enabled, try watershed
                    if use_watershed and max_objects > 0 and n_objects != max_objects:
                        # Check if we have oversized objects
                        oversized = any(prop.area > maxsize for prop in props)
                        
                        if oversized:
                            logger.info(f"T={t}, C={c}, Z={z}: Applying watershed (found {n_objects}, expected {max_objects})")
                            
                            # Apply watershed on this slice
                            from scipy import ndimage as ndi
                            from skimage.segmentation import watershed
                            from skimage.feature import peak_local_max
                            
                            # Distance transform
                            distance = ndi.distance_transform_edt(intensity_mask[z])
                            
                            # Find peaks
                            coords = peak_local_max(distance, min_distance=20, labels=intensity_mask[z])
                            mask_peaks = np.zeros(distance.shape, dtype=bool)
                            mask_peaks[tuple(coords.T)] = True
                            markers = measure.label(mask_peaks)
                            
                            # Watershed
                            labels_ws = watershed(-distance, markers, mask=intensity_mask[z])
                            
                            # Re-analyze after watershed
                            props = measure.regionprops(labels_ws)
                            valid_labels = []
                            for prop in props:
                                if minsize <= prop.area <= maxsize:
                                    valid_labels.append(prop.label)
                            
                            labeled = labels_ws
                    
                    # Create final mask for this slice
                    slice_mask = np.zeros_like(intensity_mask[z], dtype=bool)
                    for label in valid_labels:
                        slice_mask[labeled == label] = True
                    
                    final_mask[z] = slice_mask
                    
                    # Log object count warnings
                    if max_objects > 0:
                        n_final = len(valid_labels)
                        if n_final != max_objects:
                            logger.warning(
                                f"T={t}, C={c}, Z={z}: Found {n_final} objects, expected {max_objects}"
                            )
                
                # Store mask (convert to uint8: 0 or 255)
                # Note: c_idx is the index in output (always 0 for single channel output)
                output_mask[t, 0] = final_mask.astype(np.uint8) * 255
        
        # Save mask
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rp.save_tczyx_image(output_mask, output_path)
        logger.info(f"Saved mask: {output_path}")
        
        # Save ROI if requested
        if save_roi:
            roi_path = os.path.splitext(output_path)[0] + "_roi.npz"
            try:
                # Extract ROI coordinates for each slice
                rois = []
                for t in range(T):
                    for z in range(Z):
                        labeled = measure.label(output_mask[t, 0, z] > 0)
                        props = measure.regionprops(labeled)
                        
                        for prop in props:
                            roi_info = {
                                't': t,
                                'c': channel if channel >= 0 else 0,
                                'z': z,
                                'label': prop.label,
                                'area': prop.area,
                                'centroid': prop.centroid,
                                'bbox': prop.bbox,
                            }
                            rois.append(roi_info)
                
                # Save as compressed numpy archive
                np.savez_compressed(roi_path, rois=rois)
                logger.info(f"Saved ROI data: {roi_path}")
            except Exception as e:
                logger.warning(f"Failed to save ROI data: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to segment {input_path}: {e}", exc_info=True)
        return False


def process_files(
    input_pattern: str,
    output_folder: Optional[str] = None,
    minsize: int = 35000,
    maxsize: int = 120000,
    median_xy: int = 8,
    median_z: int = 2,
    threshold_method: str = 'li',
    max_objects: int = 1,
    collapse_delimiter: str = "__",
    no_parallel: bool = False,
    save_roi: bool = True,
    output_extension: str = "_mask",
    use_watershed: bool = False,
    channel: int = 0,
    dry_run: bool = False
) -> None:
    """
    Process multiple files matching a pattern.
    
    Args:
        input_pattern: File search pattern (supports ** for recursive)
        output_folder: Output directory (default: input_dir + '_threshold')
        minsize: Minimum object size in pixels
        maxsize: Maximum object size in pixels
        median_xy: Median filter size for X and Y
        median_z: Median filter size for Z
        threshold_method: Threshold method ('li', 'otsu', 'yen')
        max_objects: Expected number of objects per slice (-1 to skip check)
        collapse_delimiter: Delimiter for collapsing subfolder paths
        no_parallel: Disable parallel processing
        save_roi: Whether to save ROI coordinates
        output_extension: Extension to add before .tif
        use_watershed: Apply watershed if object count doesn't match
        channel: Channel index to process (0-based, -1 for all channels)
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
        output_folder = base_folder + "_threshold"
    
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
        print(f"[DRY RUN] Parameters: minsize={minsize}, maxsize={maxsize}, "
              f"median_xy={median_xy}, median_z={median_z}, threshold={threshold_method}, "
              f"channel={channel}")
        for src, dst in file_pairs:
            print(f"[DRY RUN] {src} -> {dst}")
        return
    
    # Process files
    if no_parallel or len(file_pairs) == 1:
        # Sequential processing
        for src, dst in file_pairs:
            segment_single_file(
                src, dst,
                minsize=minsize,
                maxsize=maxsize,
                median_xy=median_xy,
                median_z=median_z,
                threshold_method=threshold_method,
                max_objects=max_objects,
                save_roi=save_roi,
                use_watershed=use_watershed,
                channel=channel
            )
    else:
        # Parallel processing
        max_workers = min(os.cpu_count() or 4, len(file_pairs))
        logger.info(f"Processing with {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    segment_single_file,
                    src, dst,
                    minsize=minsize,
                    maxsize=maxsize,
                    median_xy=median_xy,
                    median_z=median_z,
                    threshold_method=threshold_method,
                    max_objects=max_objects,
                    save_roi=save_roi,
                    use_watershed=use_watershed,
                    channel=channel
                ): (src, dst)
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
        description="Simple threshold-based segmentation (like ImageJ macro).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Segment images (default settings)
  environment: uv@3.11:segment-simple-threshold
  commands:
  - python
  - '%REPO%/standard_code/python/segment_simple_threshold.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_masks'
  
- name: Segment images (with watershed)
  environment: uv@3.11:segment-simple-threshold
  commands:
  - python
  - '%REPO%/standard_code/python/segment_simple_threshold.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_masks'
  - --channel: 1
  - --minsize: 35000
  - --maxsize: 120000
  - --median-xy: 8
  - --median-z: 2
  - --threshold-method: li
  - --max-objects: 1
  - --use-watershed
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
        help="Output folder (default: input_folder + '_threshold')"
    )
    
    parser.add_argument(
        "--minsize",
        type=int,
        default=35000,
        help="Minimum object size in pixels (default: 35000)"
    )
    
    parser.add_argument(
        "--maxsize",
        type=int,
        default=120000,
        help="Maximum object size in pixels (default: 120000)"
    )
    
    parser.add_argument(
        "--median-xy",
        type=int,
        default=8,
        help="Median filter size for X and Y dimensions (default: 8)"
    )
    
    parser.add_argument(
        "--median-z",
        type=int,
        default=2,
        help="Median filter size for Z dimension (default: 2)"
    )
    
    parser.add_argument(
        "--threshold-method",
        type=str,
        default="li",
        choices=["li", "otsu", "yen"],
        help="Auto threshold method (default: li)"
    )
    
    parser.add_argument(
        "--max-objects",
        type=int,
        default=1,
        help="Expected number of objects per slice, -1 to skip check (default: 1)"
    )
    
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index to process (0-based), use -1 to process all channels (default: 0)"
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
        "--no-roi",
        action="store_true",
        help="Skip saving ROI coordinate sidecars"
    )
    
    parser.add_argument(
        "--use-watershed",
        action="store_true",
        help="Apply watershed if object count doesn't match expected"
    )
    
    parser.add_argument(
        "--output-file-name-extension",
        type=str,
        default="_mask",
        help="Additional extension to add before .tif (default: '_mask')"
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
        print(f"segment_simple_threshold.py version: {version}")
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
        minsize=args.minsize,
        maxsize=args.maxsize,
        median_xy=args.median_xy,
        median_z=args.median_z,
        threshold_method=args.threshold_method,
        max_objects=args.max_objects,
        collapse_delimiter=args.collapse_delimiter,
        no_parallel=args.no_parallel,
        save_roi=not args.no_roi,
        output_extension=args.output_file_name_extension,
        use_watershed=args.use_watershed,
        channel=args.channel,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
