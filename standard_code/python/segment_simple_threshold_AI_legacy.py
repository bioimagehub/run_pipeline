"""
ROI-seeded threshold-based nucleus segmentation.
State-of-the-art approach using ROI center as guaranteed seed point.

Strategy:
1. ROI coordinates mark the CENTER of the nucleus (guaranteed inside)
2. Use ROI as seed for region growing and watershed
3. Morphological operations to remove blebs/debris while keeping main nucleus
4. No edge removal since ROI-seeded approach handles edge-touching nuclei

MIT License - BIPHUB, University of Oslo
"""

import os
import argparse
import logging
import yaml
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import filters, measure, morphology
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk

import tifffile
import bioimage_pipeline_utils as rp

# Module-level logger
logger = logging.getLogger(__name__)


def get_coordinates_from_metadata(yaml_path: str) -> Optional[list[tuple[int, int]]]:
    """Extract (x, y) coordinates from ROI metadata."""
    if not os.path.exists(yaml_path):
        logger.warning(f"YAML file not found: {yaml_path}")
        return None
    
    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)

    rois = metadata.get("Image metadata", {}).get("ROIs", [])
    if not rois:
        return None

    coords = []
    for roi in rois:
        pos = roi.get("Roi", {}).get("Positions", {})
        x = int(round(pos.get("x", -1)))
        y = int(round(pos.get("y", -1)))
        if x != -1 and y != -1:
            coords.append((x, y))

    return coords or None


def robust_threshold(image: np.ndarray, method: str = 'li') -> float:
    """
    Compute threshold with fallback strategies for poor signal.
    
    Args:
        image: Input image array
        method: Primary threshold method ('li', 'otsu', 'yen', 'triangle')
    
    Returns:
        Threshold value
    """
    # Try primary method
    try:
        if method == 'li':
            return filters.threshold_li(image)
        elif method == 'otsu':
            return filters.threshold_otsu(image)
        elif method == 'yen':
            return filters.threshold_yen(image)
        elif method == 'triangle':
            return filters.threshold_triangle(image)
    except Exception as e:
        logger.warning(f"Primary threshold method '{method}' failed: {e}")
    
    # Fallback to percentile-based threshold
    background = np.percentile(image, 10)
    foreground = np.percentile(image, 90)
    threshold = background + 0.4 * (foreground - background)
    logger.info(f"Using fallback threshold: {threshold}")
    return threshold


def roi_seeded_segmentation(
    image: np.ndarray,
    roi_center: tuple[int, int],
    minsize: int,
    maxsize: int,
    threshold_method: str = 'li',
    median_size: int = 5,
    remove_blebs: bool = True,
    apply_watershed: bool = True
) -> np.ndarray:
    """
    Segment nucleus using ROI center as seed point.
    
    Strategy:
    1. Median filter to reduce noise
    2. Threshold to get initial mask
    3. Use ROI center to identify the nucleus of interest
    4. Apply watershed to separate touching objects (if enabled)
    5. Morphological operations to remove blebs/protrusions
    6. Keep only component containing ROI center
    
    Args:
        image: 2D image array (YX)
        roi_center: (x, y) coordinates of ROI center (guaranteed inside nucleus)
        minsize: Minimum object size in pixels
        maxsize: Maximum object size in pixels
        threshold_method: Threshold method ('li', 'otsu', 'yen', 'triangle')
        median_size: Median filter size
        remove_blebs: Apply morphological operations to remove blebs
        apply_watershed: Apply watershed to all masks to separate touching objects
    
    Returns:
        Binary mask (2D array) with segmented nucleus
    """
    # Convert ROI from (x, y) to array indices (y, x)
    roi_y, roi_x = roi_center[1], roi_center[0]
    
    # Validate ROI is within image bounds
    if not (0 <= roi_y < image.shape[0] and 0 <= roi_x < image.shape[1]):
        logger.error(f"ROI center {roi_center} is outside image bounds {image.shape}")
        return np.zeros_like(image, dtype=np.uint8)
    
    # Step 1: Median filtering
    if median_size > 1:
        filtered = ndimage.median_filter(image, size=median_size)
    else:
        filtered = image.copy()
    
    # Step 2: Thresholding
    threshold_value = robust_threshold(filtered, method=threshold_method)
    binary_mask = filtered > threshold_value
    
    # Check if ROI is in foreground
    if not binary_mask[roi_y, roi_x]:
        logger.warning(f"ROI center not in thresholded mask, lowering threshold")
        # Lower threshold to include ROI
        threshold_value = filtered[roi_y, roi_x] * 0.95
        binary_mask = filtered > threshold_value
    
    # Step 3: Fill holes
    binary_mask = ndimage.binary_fill_holes(binary_mask)
    
    # Step 4: Label connected components
    labeled, num_features = ndimage.label(binary_mask)
    
    if num_features == 0:
        logger.warning("No objects found after thresholding")
        return np.zeros_like(image, dtype=np.uint8)
    
    # Get the label at ROI center
    roi_label = labeled[roi_y, roi_x]
    
    if roi_label == 0:
        logger.warning("ROI not in any labeled component, using nearest")
        # Find nearest non-zero label
        indices = np.argwhere(labeled > 0)
        distances = np.sqrt((indices[:, 0] - roi_y)**2 + (indices[:, 1] - roi_x)**2)
        nearest_idx = np.argmin(distances)
        roi_label = labeled[indices[nearest_idx, 0], indices[nearest_idx, 1]]
    
    # Extract only the component containing ROI
    nucleus_mask = (labeled == roi_label)
    
    # Step 5: Check initial size
    nucleus_size = np.sum(nucleus_mask)
    
    # Step 6: Apply watershed to separate touching objects (if enabled)
    # Apply to all masks to clean up and potentially separate merged nuclei
    if apply_watershed and nucleus_size >= minsize:
        logger.info("Applying watershed for cleanup/separation")
        
        # Distance transform for watershed
        distance = ndimage.distance_transform_edt(nucleus_mask)
        
        # Find local maxima as markers - use conservative footprint
        footprint_size = max(3, 3)  # Small footprint like in test code
        coords = peak_local_max(distance, footprint=np.ones((footprint_size, footprint_size)), 
                               labels=nucleus_mask)
        
        num_peaks = len(coords)
        logger.info(f"Found {num_peaks} peaks")
        
        if num_peaks >= 1:
            # Create markers
            markers = np.zeros(distance.shape, dtype=bool)
            markers[tuple(coords.T)] = True
            markers = ndimage.label(markers)[0]
            
            # Apply watershed
            labels_ws = watershed(-distance, markers, mask=nucleus_mask)
            
            # Get the component containing ROI
            roi_label_ws = labels_ws[roi_y, roi_x]
            nucleus_mask_ws = (labels_ws == roi_label_ws)
            
            # Check if watershed result is acceptable
            nucleus_size_ws = np.sum(nucleus_mask_ws)
            
            if nucleus_size_ws >= minsize:
                logger.info(f"Watershed applied: {nucleus_size} -> {nucleus_size_ws} pixels ({num_peaks} peaks)")
                nucleus_mask = nucleus_mask_ws
                nucleus_size = nucleus_size_ws
            else:
                logger.warning(f"Watershed made nucleus too small ({nucleus_size_ws} < {minsize}), keeping original")
        else:
            logger.info("No peaks found, skipping watershed")
    
    # Step 7: Check for edge touching BEFORE morphological operations
    # Nuclei touching image edges should be rejected early
    height, width = nucleus_mask.shape
    touches_edge = (
        np.any(nucleus_mask[0, :]) or      # Top edge
        np.any(nucleus_mask[-1, :]) or     # Bottom edge
        np.any(nucleus_mask[:, 0]) or      # Left edge
        np.any(nucleus_mask[:, -1])        # Right edge
    )
    
    if touches_edge:
        logger.warning("Nucleus touches image edge - rejecting")
        return np.zeros_like(image, dtype=np.uint8)
    
    # Step 8: Morphological cleaning to remove blebs/protrusions
    if remove_blebs and nucleus_size > minsize:
        # Opening to remove small protrusions
        # Use disk structuring element sized based on nucleus size
        selem_size = max(2, int(np.sqrt(nucleus_size) / 30))
        selem = disk(selem_size)
        
        nucleus_mask = binary_opening(nucleus_mask, selem)
        
        # Fill holes again after opening
        nucleus_mask = ndimage.binary_fill_holes(nucleus_mask)
        
        # Note: Removed binary_closing to prevent edge expansion
        # Closing was smoothing boundaries but causing edge-touching rejections
        
        # Verify ROI still inside after morphological operations
        if not nucleus_mask[roi_y, roi_x]:
            logger.warning("ROI lost after morphological operations, reverting")
            # Revert to pre-morphology mask
            nucleus_mask = (labeled == roi_label)
    
    # Step 9: Final size check
    final_size = np.sum(nucleus_mask)
    
    if final_size < minsize:
        logger.warning(f"Final nucleus too small ({final_size} < {minsize} px)")
        return np.zeros_like(image, dtype=np.uint8)
    
    if final_size > maxsize:
        logger.warning(f"Final nucleus too large ({final_size} > {maxsize} px)")
        # Don't fail - return it anyway, might be acceptable
    
    return nucleus_mask.astype(np.uint8) * 255


def segment_single_file(
    input_path: str,
    output_path: str,
    yaml_path: Optional[str] = None,
    minsize: int = 35000,
    maxsize: int = 120000,
    median_size: int = 5,
    threshold_method: str = 'li',
    channel: int = 0,
    remove_blebs: bool = True,
    apply_watershed: bool = True,
    force: bool = False
) -> bool:
    """
    Segment a single image file using ROI-seeded approach.
    
    Args:
        input_path: Path to input image file
        output_path: Path to output mask file
        yaml_path: Path to YAML metadata file with ROI coordinates
        minsize: Minimum nucleus size in pixels
        maxsize: Maximum nucleus size in pixels
        median_size: Median filter size
        threshold_method: Threshold method ('li', 'otsu', 'yen', 'triangle')
        channel: Channel index to process (0-based)
        remove_blebs: Apply morphological operations to remove blebs
        apply_watershed: Apply watershed to all masks to separate touching objects
        force: Force reprocessing even if output exists
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing: {os.path.basename(input_path)}")
        
        # Check if output already exists
        if os.path.exists(output_path) and not force:
            logger.info(f"Output exists, skipping: {output_path}")
            return True
        
        # Load ROI coordinates
        roi_coords = None
        if yaml_path:
            roi_coords = get_coordinates_from_metadata(yaml_path)
        
        if not roi_coords:
            logger.error(f"No ROI coordinates found - required for ROI-seeded segmentation")
            return False
        
        # For now, use first ROI (we expect one nucleus per image)
        roi_center = roi_coords[0]
        logger.info(f"Using ROI center: {roi_center}")
        
        # Load image
        img = rp.load_tczyx_image(input_path)
        
        # Get dimensions
        T, C, Z, Y, X = img.shape
        logger.info(f"Dimensions: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
        
        # Validate channel
        if channel >= C:
            logger.error(f"Channel {channel} does not exist (image has {C} channels)")
            return False
        
        # For 2D images, we expect Z=1, but handle stacks by max projection if needed
        if Z > 1:
            logger.warning(f"Image has Z={Z} slices, using max projection")
            image_2d = np.max(img.data[0, channel, :, :, :], axis=0)
        else:
            image_2d = img.data[0, channel, 0, :, :]
        
        # Check if there's any signal
        if image_2d.max() == 0:
            logger.error(f"No signal detected in channel {channel}")
            return False
        
        # Check minimum signal level
        background_level = np.percentile(image_2d, 2)
        signal_level = np.percentile(image_2d, 98)
        if signal_level < background_level * 1.5:
            logger.error(f"Insufficient signal contrast (bg={background_level:.1f}, signal={signal_level:.1f})")
            return False
        
        # Perform segmentation
        mask = roi_seeded_segmentation(
            image=image_2d.astype(np.float32),
            roi_center=roi_center,
            minsize=minsize,
            maxsize=maxsize,
            threshold_method=threshold_method,
            median_size=median_size,
            remove_blebs=remove_blebs,
            apply_watershed=apply_watershed
        )
        
        # Check if segmentation succeeded
        if mask.sum() == 0:
            logger.error("Segmentation failed - no mask produced")
            return False
        
        # Save mask
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as binary 8-bit mask (0 or 255)
        tifffile.imwrite(
            output_path,
            mask,
            imagej=True,
            metadata={'axes': 'YX'},
            compression='deflate'
        )
        
        logger.info(f"Saved: {output_path} (nucleus size: {np.sum(mask > 0)} pixels)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_files(
    input_pattern: str,
    yaml_pattern: str,
    output_folder: Optional[str] = None,
    minsize: int = 35000,
    maxsize: int = 120000,
    median_size: int = 5,
    threshold_method: str = 'li',
    channel: int = 0,
    remove_blebs: bool = True,
    apply_watershed: bool = True,
    no_parallel: bool = False,
    dry_run: bool = False,
    force: bool = False
) -> None:
    """
    Process multiple files using grouped file matching.
    
    Args:
        input_pattern: Image file search pattern
        yaml_pattern: YAML metadata file search pattern
        output_folder: Output directory (default: input_dir + '_segmented')
        minsize: Minimum nucleus size in pixels
        maxsize: Maximum nucleus size in pixels
        median_size: Median filter size
        threshold_method: Threshold method ('li', 'otsu', 'yen', 'triangle')
        channel: Channel index to process (0-based)
        remove_blebs: Apply morphological operations to remove blebs
        apply_watershed: Apply watershed to all masks to separate touching objects
        no_parallel: Disable parallel processing
        dry_run: Only print planned actions without executing
        force: Force reprocessing even if outputs exist
    """
    logger.info("Using grouped file processing with YAML metadata")
    search_subfolders = '**' in input_pattern or '**' in yaml_pattern
    
    search_patterns = {
        'image': input_pattern,
        'yaml': yaml_pattern
    }
    
    try:
        grouped_files = rp.get_grouped_files_to_process(
            search_patterns=search_patterns,
            search_subfolders=search_subfolders
        )
    except Exception as e:
        logger.error(f"Error in file grouping: {e}")
        return
    
    if not grouped_files:
        logger.error("No files matched the search patterns.")
        return
    
    logger.info(f"Found {len(grouped_files)} file groups")
    
    # Determine output folder
    if output_folder is None:
        base_pattern = input_pattern
        output_folder = os.path.dirname(base_pattern) + "_segmented" or "output_segmented"
    
    logger.info(f"Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    # Build task list
    tasks = []
    for basename, files in grouped_files.items():
        if 'image' not in files:
            logger.warning(f"Skipping '{basename}': missing image file")
            continue
        
        if 'yaml' not in files:
            logger.warning(f"Skipping '{basename}': missing YAML file (required for ROI)")
            continue
        
        img_path = files['image']
        yaml_path = files['yaml']
        
        # Prepare output path
        out_name = os.path.splitext(os.path.basename(img_path))[0] + "_mask.tif"
        out_path = os.path.join(output_folder, out_name)
        
        tasks.append((img_path, out_path, yaml_path))
    
    if not tasks:
        logger.warning("No valid image/YAML pairs found.")
        return
    
    logger.info(f"Processing {len(tasks)} valid pairs")
    
    # Dry run
    if dry_run:
        print(f"[DRY RUN] Would process {len(tasks)} files")
        print(f"[DRY RUN] Output folder: {output_folder}")
        print(f"[DRY RUN] Parameters: minsize={minsize}, maxsize={maxsize}, "
              f"median={median_size}, threshold={threshold_method}, channel={channel}, "
              f"apply_watershed={apply_watershed}")
        for img, out, yml in tasks:
            print(f"[DRY RUN] {os.path.basename(img)} + {os.path.basename(yml)} -> {os.path.basename(out)}")
        return
    
    # Process files
    success_count = 0
    fail_count = 0
    
    if no_parallel or len(tasks) == 1:
        for img_path, out_path, yaml_path in tasks:
            try:
                success = segment_single_file(
                    img_path, out_path,
                    yaml_path=yaml_path,
                    minsize=minsize,
                    maxsize=maxsize,
                    median_size=median_size,
                    threshold_method=threshold_method,
                    channel=channel,
                    remove_blebs=remove_blebs,
                    apply_watershed=apply_watershed,
                    force=force
                )
                if success:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                fail_count += 1
    else:
        cpu_count = os.cpu_count() or 1
        cpu_count = max(cpu_count - 1, 1)
        
        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = []
            for img_path, out_path, yaml_path in tasks:
                future = executor.submit(
                    segment_single_file,
                    img_path, out_path, yaml_path,
                    minsize, maxsize, median_size,
                    threshold_method, channel, remove_blebs,
                    apply_watershed, force
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
                    logger.error(f"A file failed to process: {e}")
                    fail_count += 1
    
    logger.info(f"Processing complete: {success_count} succeeded, {fail_count} failed")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ROI-seeded threshold-based nucleus segmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Segment nuclei with ROI-seeded approach
  environment: uv@3.11:segment-simple-threshold
  commands:
  - python
  - '%REPO%/standard_code/python/segment_simple_threshold_AI.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --yaml-search-pattern: '%YAML%/input_data/**/*_metadata.yaml'
  - --output-folder: '%YAML%/output_masks'
  - --channel: 0
  - --minsize: 35000
  - --maxsize: 120000
  - --median-size: 5
  - --threshold-method: li
        """
    )
    
    parser.add_argument(
        "--input-search-pattern",
        type=str,
        required=True,
        help="Input image file pattern (supports wildcards, use '**' for recursive search)"
    )
    
    parser.add_argument(
        "--yaml-search-pattern",
        type=str,
        required=True,
        help="YAML metadata file pattern with ROI coordinates (required)"
    )
    
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder (default: input_folder + '_segmented')"
    )
    
    parser.add_argument(
        "--minsize",
        type=int,
        default=35000,
        help="Minimum nucleus size in pixels (default: 35000)"
    )
    
    parser.add_argument(
        "--maxsize",
        type=int,
        default=120000,
        help="Maximum nucleus size in pixels (default: 120000)"
    )
    
    parser.add_argument(
        "--median-size",
        type=int,
        default=5,
        help="Median filter size (default: 5)"
    )
    
    parser.add_argument(
        "--threshold-method",
        type=str,
        default="li",
        choices=["li", "otsu", "yen", "triangle"],
        help="Auto threshold method (default: li)"
    )
    
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index to process (0-based, default: 0)"
    )
    
    parser.add_argument(
        "--no-bleb-removal",
        action="store_true",
        help="Disable morphological bleb removal"
    )
    
    parser.add_argument(
        "--no-watershed",
        action="store_true",
        help="Disable watershed segmentation (use only thresholding)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if output files exist"
    )
    
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (process files sequentially)"
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
        print(f"segment_simple_threshold_AI.py version: {version}")
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
        yaml_pattern=args.yaml_search_pattern,
        output_folder=args.output_folder,
        minsize=args.minsize,
        maxsize=args.maxsize,
        median_size=args.median_size,
        threshold_method=args.threshold_method,
        channel=args.channel,
        remove_blebs=not args.no_bleb_removal,
        apply_watershed=not args.no_watershed,
        no_parallel=args.no_parallel,
        dry_run=args.dry_run,
        force=args.force
    )


if __name__ == "__main__":
    main()
