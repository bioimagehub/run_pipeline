"""
Fill Segmentation Gaps

Fills missing timepoints in tracked masks by interpolating between neighboring frames.
Handles multiple objects per timepoint and prevents overlap.

Author: BIPHUB - Bioimage Informatics Hub, University of Oslo
License: MIT
"""

import os
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import logging
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
from skimage.morphology import disk

# Local imports
import bioimage_pipeline_utils as rp
from track_indexed_mask import track_labels_with_trackpy


def find_segmentation_gaps(mask_data: np.ndarray, max_distance: float = 50.0) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    Find gaps caused by segmentation failures where tracking broke continuity.
    
    Detects when:
    1. An object disappears (segmentation failed)
    2. Zero or empty frames exist
    3. A new object appears in roughly the same location (tracking gave it a new ID)
    
    Args:
        mask_data: 5D TCZYX mask array
        max_distance: Maximum centroid distance (pixels) to consider objects as the same cell
        
    Returns:
        Dictionary mapping original_object_id to list of (gap_start, gap_end, new_object_id) tuples
        where gap_start/gap_end are the FIRST and LAST EMPTY frames
    """
    T, C, Z, Y, X = mask_data.shape
    
    # Calculate centroids for all objects at all timepoints
    centroids = {}  # {(t, label_id): centroid}
    for t in range(T):
        labels = np.unique(mask_data[t, 0, :, :, :])
        for label_id in labels:
            if label_id == 0:
                continue
            mask_binary = mask_data[t, 0, :, :, :] == label_id
            if np.any(mask_binary):
                coords = np.argwhere(mask_binary)
                centroid = coords.mean(axis=0)
                centroids[(t, label_id)] = centroid
    
    logging.info(f"Found {len(centroids)} object instances across {T} timepoints")
    
    # Find gaps: when an object ends and another begins nearby
    gaps = {}  # {original_id: [(gap_start, gap_end, new_id), ...]}
    
    for t in range(T - 1):
        # Get objects in current frame
        current_labels = set(np.unique(mask_data[t, 0, :, :, :]))
        current_labels.discard(0)
        
        if not current_labels:
            continue
        
        for obj_id in current_labels:
            # Check if this object disappears
            # Look ahead to find when it (or a new object in same location) reappears
            obj_centroid = centroids.get((t, obj_id))
            if obj_centroid is None:
                continue
            
            # Check next frame
            next_labels = set(np.unique(mask_data[t + 1, 0, :, :, :]))
            next_labels.discard(0)
            
            # If object continues with same ID, no gap
            if obj_id in next_labels:
                continue
            
            # Object disappeared! Look for a gap and reappearance
            gap_start = t + 1  # First empty frame
            gap_found = False
            
            # Look ahead up to reasonable distance
            for future_t in range(t + 1, min(t + 10, T)):  # Look max 10 frames ahead
                future_labels = set(np.unique(mask_data[future_t, 0, :, :, :]))
                future_labels.discard(0)
                
                if not future_labels:
                    # Empty frame, continue looking
                    continue
                
                # Check if any object in future frame is near the original location
                for future_id in future_labels:
                    future_centroid = centroids.get((future_t, future_id))
                    if future_centroid is None:
                        continue
                    
                    distance = np.linalg.norm(obj_centroid - future_centroid)
                    
                    if distance < max_distance:
                        # Found a nearby object! This is likely the same cell with a new ID
                        gap_end = future_t - 1  # Last empty frame before reappearance
                        
                        if obj_id not in gaps:
                            gaps[obj_id] = []
                        
                        gaps[obj_id].append((gap_start, gap_end, future_id))
                        logging.info(f"  Found gap: Object {obj_id} last seen at T{t}, empty frames T{gap_start}-{gap_end}, reappears as Object {future_id} at T{future_t} (distance={distance:.1f}px)")
                        gap_found = True
                        break
                
                if gap_found:
                    break
    
    return gaps


def interpolate_mask_morphology(
    mask_before: np.ndarray,
    mask_after: np.ndarray,
    n_steps: int,
    object_id_before: int,
    object_id_after: int = None
) -> List[np.ndarray]:
    """
    Interpolate mask shape between two timepoints using simple blending.
    
    Args:
        mask_before: Labeled mask from frame before gap (2D: YX or 3D: ZYX)
        mask_after: Labeled mask from frame after gap (2D: YX or 3D: ZYX)
        n_steps: Number of intermediate frames to create
        object_id_before: The object ID in the before frame
        object_id_after: The object ID in the after frame (defaults to object_id_before)
        
    Returns:
        List of interpolated masks (one per intermediate frame), all labeled with object_id_before
    """
    if object_id_after is None:
        object_id_after = object_id_before
        
    # Convert to binary masks for the TWO DIFFERENT object IDs
    mask_before_binary = (mask_before == object_id_before).astype(bool)
    mask_after_binary = (mask_after == object_id_after).astype(bool)
    
    # If object doesn't exist in both frames, return empty
    if not np.any(mask_before_binary) or not np.any(mask_after_binary):
        return [np.zeros_like(mask_before, dtype=mask_before.dtype) for _ in range(n_steps)]
    
    interpolated = []
    for step in range(n_steps):
        alpha = (step + 1) / (n_steps + 1)  # 0 < alpha < 1, smoothly transitions from 0 to 1
        
        # Simple approach: blend the two binary masks and threshold
        # This creates a smooth transition between the two shapes
        blended = mask_before_binary.astype(float) * (1 - alpha) + mask_after_binary.astype(float) * alpha
        
        # Threshold at 0.3 to be more inclusive (creates union-like behavior)
        interpolated_mask = (blended > 0.3).astype(bool)
        
        # Convert back to labeled mask WITH THE BEFORE ID (tracking will reconnect later)
        result = np.zeros_like(mask_before, dtype=mask_before.dtype)
        result[interpolated_mask] = object_id_before
        interpolated.append(result)
    
    return interpolated


def fill_gaps_in_mask_data(
    mask_data: np.ndarray,
    gaps: Dict[int, List[Tuple[int, int, int]]],
    max_gap_size: int = 2,
    z_slice: int = None
) -> Tuple[np.ndarray, int, int]:
    """
    Fill detected gaps in mask data by interpolating between frames.
    
    Args:
        mask_data: 5D TCZYX mask array (will be modified in-place)
        gaps: Dictionary from find_segmentation_gaps()
        max_gap_size: Maximum gap size to fill (in frames)
        z_slice: If specified, only process this Z slice
        
    Returns:
        Tuple of (modified_mask_data, filled_count, skipped_count)
    """
    T, C, Z, Y, X = mask_data.shape
    filled_count = 0
    skipped_count = 0
    
    for object_id, gap_list in gaps.items():
        for gap_start, gap_end, new_object_id in gap_list:
            gap_size = gap_end - gap_start + 1
            
            # Validate gap
            if gap_size <= 0:
                logging.warning(f"  Object {object_id}: Invalid gap T={gap_start}-{gap_end} (negative size), skipping")
                skipped_count += 1
                continue
            
            if gap_size > max_gap_size:
                logging.warning(f"  Object {object_id}: Gap at T={gap_start}-{gap_end} (size {gap_size}) exceeds max ({max_gap_size}), skipping")
                skipped_count += 1
                continue
            
            logging.info(f"  Filling object {object_id}: Gap at T={gap_start}-{gap_end} (size {gap_size} empty frames)")
            
            # Handle edge cases (gaps at start or end)
            if gap_start == 0:
                # Gap at start - copy from first valid frame
                first_valid = gap_end + 1
                if first_valid < T:
                    logging.info(f"    Gap at start, copying from T={first_valid}")
                    for t in range(gap_start, gap_end + 1):
                        mask_data[t, 0, :, :, :][mask_data[first_valid, 0, :, :, :] == object_id] = object_id
                filled_count += 1
                continue
            
            if gap_end == T - 1:
                # Gap at end - copy from last valid frame
                last_valid = gap_start - 1
                logging.info(f"    Gap at end, copying from T={last_valid}")
                for t in range(gap_start, gap_end + 1):
                    mask_data[t, 0, :, :, :][mask_data[last_valid, 0, :, :, :] == object_id] = object_id
                filled_count += 1
                continue
            
            # Normal gap - interpolate
            before_t = gap_start - 1
            after_t = gap_end + 1
            
            logging.info(f"    Interpolating between T={before_t} (ID {object_id}) and T={after_t} (ID {new_object_id})")
            
            # Process each Z slice
            for z in range(Z):
                if z_slice is not None and z != z_slice:
                    continue
                
                mask_before = mask_data[before_t, 0, z, :, :]
                mask_after = mask_data[after_t, 0, z, :, :]
                
                # Check if objects exist in before/after frames with correct IDs
                has_before = np.any(mask_before == object_id)
                has_after = np.any(mask_after == new_object_id)
                
                if not has_before or not has_after:
                    logging.warning(f"    Z={z}: Missing reference frames (before ID {object_id}={has_before}, after ID {new_object_id}={has_after})")
                    continue
                
                # Interpolate using BOTH IDs
                interpolated = interpolate_mask_morphology(
                    mask_before, mask_after, gap_size, object_id, new_object_id
                )
                
                # Insert interpolated frames
                for i, interp_mask in enumerate(interpolated):
                    t = gap_start + i
                    # Count pixels to fill
                    pixels_to_fill = np.sum((mask_data[t, 0, z, :, :] == 0) & (interp_mask == object_id))
                    if pixels_to_fill > 0:
                        mask_data[t, 0, z, :, :][mask_data[t, 0, z, :, :] == 0] = interp_mask[mask_data[t, 0, z, :, :] == 0]
                        logging.info(f"    Filled {pixels_to_fill} pixels at T={t}, Z={z}")
            
            filled_count += 1
    
    return mask_data, filled_count, skipped_count


def reconnect_objects_with_tracking(mask_data: np.ndarray) -> np.ndarray:
    """
    Run trackpy tracking to reconnect objects after gap filling.
    
    Args:
        mask_data: 5D TCZYX mask array
        
    Returns:
        Tracked mask data with consistent object IDs
    """
    logging.info("Running tracking to reconnect objects...")
    
    # Debug: print max values before tracking
    max_values = np.max(mask_data, axis=(1, 2, 3, 4))
    logging.info(f"Max object IDs per timepoint BEFORE tracking: {max_values}")
    
    try:
        _, mask_data = track_labels_with_trackpy(mask_data)
        logging.info("✓ Tracking completed successfully")
    except Exception as e:
        logging.warning(f"✗ Tracking failed: {e}")
        import traceback
        logging.warning(traceback.format_exc())
    
    # Debug: print max values after tracking
    max_values = np.max(mask_data, axis=(1, 2, 3, 4))
    logging.info(f"Max object IDs per timepoint AFTER tracking: {max_values}")
    
    return mask_data


def generate_tracking_error_report(
    mask_data: np.ndarray,
    output_path: str,
    mask_path: str
) -> None:
    """
    Check for remaining gaps and generate error report if needed.
    
    Args:
        mask_data: 5D TCZYX mask array after filling and tracking
        output_path: Path where the mask will be saved
        mask_path: Original mask path (for report filename)
    """
    logging.info("Checking for remaining gaps after tracking...")
    remaining_gaps = find_segmentation_gaps(mask_data, max_distance=50.0)
    
    if not remaining_gaps:
        logging.info("✓ No remaining gaps! Tracking is complete and consistent.")
        return
    
    # Generate error report
    error_report_path = output_path.replace('.tif', '_tracking_errors.txt').replace('.tiff', '_tracking_errors.txt')
    total_remaining_gaps = sum(len(gap_list) for gap_list in remaining_gaps.values())
    
    with open(error_report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRACKING ERROR REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"File: {Path(mask_path).name}\n")
        f.write(f"Total remaining gaps: {total_remaining_gaps}\n")
        f.write(f"Objects with gaps: {len(remaining_gaps)}\n")
        f.write("\n")
        f.write("DETAILS:\n")
        f.write("-" * 80 + "\n")
        
        for obj_id, gap_list in remaining_gaps.items():
            f.write(f"\nObject {obj_id}:\n")
            for gap_start, gap_end, new_id in gap_list:
                gap_size = gap_end - gap_start + 1
                f.write(f"  - Gap at T{gap_start}-T{gap_end} (size: {gap_size} frames)\n")
                f.write(f"    Object reappears as ID {new_id} at T{gap_end + 1}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("- Review these timepoints in the original images\n")
        f.write("- Check segmentation quality for these specific frames\n")
        f.write("- Consider increasing --max-gap-size if gaps are due to temporary segmentation failures\n")
        f.write("- Consider re-running segmentation with adjusted parameters\n")
    
    logging.warning(f"✗ Tracking errors detected! Report saved to: {error_report_path}")
    logging.warning(f"  {total_remaining_gaps} gaps remain across {len(remaining_gaps)} objects")


def process_file(
    mask_path: str,
    output_path: str,
    max_gap_size: int = 2,
    z_slice: int = None
) -> None:
    """
    Process a single mask file: detect gaps, fill them, run tracking, and validate.
    
    This is the main pipeline function that orchestrates:
    1. Loading the mask
    2. Finding gaps
    3. Filling gaps
    4. Running tracking to reconnect objects
    5. Validating results and generating reports
    
    Args:
        mask_path: Path to input mask file
        output_path: Path to save filled mask
        max_gap_size: Maximum gap size to fill (in frames)
        z_slice: If specified, only process this Z slice. Otherwise processes all.
    """
    # Step 1: Load mask
    logging.info(f"Loading mask: {mask_path}")
    mask_img = rp.load_tczyx_image(mask_path)
    mask_data = mask_img.data.copy()  # Make a copy to modify
    
    T, C, Z, Y, X = mask_data.shape
    logging.info(f"Mask shape: {mask_data.shape}")
    
    # Step 2: Find all gaps
    logging.info("Step 1: Detecting gaps...")
    gaps = find_segmentation_gaps(mask_data)
    
    if not gaps:
        logging.info("No gaps found! Mask is complete.")
        rp.save_tczyx_image(mask_data, output_path)
        return
    
    total_gaps = sum(len(gap_list) for gap_list in gaps.values())
    logging.info(f"Found {total_gaps} gaps across {len(gaps)} objects")
    
    # Step 3: Fill gaps
    logging.info("Step 2: Filling gaps...")
    mask_data, filled_count, skipped_count = fill_gaps_in_mask_data(
        mask_data, gaps, max_gap_size, z_slice
    )
    logging.info(f"Filled {filled_count} gaps, skipped {skipped_count} (too large or invalid)")
    
    # Step 4: Run tracking to reconnect objects
    logging.info("Step 3: Reconnecting objects with tracking...")
    mask_data = reconnect_objects_with_tracking(mask_data)
    
    # Step 5: Validate and generate report
    logging.info("Step 4: Validating results...")
    generate_tracking_error_report(mask_data, output_path, mask_path)
    
    # Step 6: Save result
    logging.info(f"Saving filled mask to: {output_path}")
    rp.save_tczyx_image(mask_data, output_path)
    logging.info("Done!")


def fill_segmentation_gaps(
    mask_path: str,
    output_path: str,
    max_gap_size: int = 2,
    z_slice: int = None
) -> None:
    """
    Fill gaps in tracked masks by interpolating missing timepoints.
    
    DEPRECATED: Use process_file() instead. This function is kept for backwards compatibility.
    
    Args:
        mask_path: Path to input mask file
        output_path: Path to save filled mask
        max_gap_size: Maximum gap size to fill (in frames)
        z_slice: If specified, only process this Z slice. Otherwise processes all.
    """
    logging.warning("fill_segmentation_gaps() is deprecated. Use process_file() instead.")
    process_file(mask_path, output_path, max_gap_size, z_slice)


def main():
    parser = argparse.ArgumentParser(
        description='Fill gaps in tracked masks by interpolating missing timepoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fill small gaps (max 2 frames)
  python fill_mask_gaps.py --input-search-pattern "./masks/*_segmentation.tif" --max-gap-size 2
  
  # Fill larger gaps
  python fill_mask_gaps.py --input-search-pattern "./masks/*_segmentation.tif" --max-gap-size 5
  
  # Specify output folder
  python fill_mask_gaps.py --input-search-pattern "./masks/*_segmentation.tif" --output-folder ./masks_filled
        """
    )
    
    parser.add_argument('--input-search-pattern', type=str, required=True,
                       help='Glob pattern for input mask images')
    parser.add_argument('--output-folder', type=str, default=None,
                       help='Output folder for filled masks. If not specified, adds "_filled" suffix to input folder.')
    parser.add_argument('--max-gap-size', type=int, default=2,
                       help='Maximum gap size (in frames) to fill (default: 2)')
    parser.add_argument('--search-subfolders', action='store_true',
                       help='Enable recursive search for files')
    parser.add_argument('--suffix', type=str, default='_filled',
                       help='Suffix to add to output filenames (default: "_filled")')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Find input files
    mask_files = rp.get_files_to_process2(args.input_search_pattern, args.search_subfolders)
    
    if not mask_files:
        raise FileNotFoundError(f"No files found matching pattern: {args.input_search_pattern}")
    
    logging.info(f"Found {len(mask_files)} mask files")
    
    # Determine output folder
    if args.output_folder is None:
        # Use input folder with suffix
        first_file_dir = os.path.dirname(mask_files[0])
        output_folder = first_file_dir + args.suffix
    else:
        output_folder = args.output_folder
    
    os.makedirs(output_folder, exist_ok=True)
    logging.info(f"Output folder: {output_folder}")
    
    # Process each file
    for mask_path in mask_files:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing: {Path(mask_path).name}")
        
        output_name = Path(mask_path).stem + args.suffix + Path(mask_path).suffix
        output_path = os.path.join(output_folder, output_name)
        
        try:
            process_file(
                mask_path,
                output_path,
                max_gap_size=args.max_gap_size
            )
        except Exception as e:
            logging.error(f"Error processing {Path(mask_path).name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logging.info(f"\n{'='*60}")
    logging.info("Processing complete!")


if __name__ == "__main__":
    main()
