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
# from scipy.ndimage import median_filter, 

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import filters, measure

import tifffile
import yaml
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
    
    # Create footprint: (z, y, x)
    footprint = np.ones((z_size, xy_size, xy_size), dtype=bool)
    return ndimage.median_filter(data, footprint=footprint)


def apply_watershed_segmentation(
    binary_mask: np.ndarray,
    minsize: int,
    maxsize: int,
    min_distance: int = 1,
    tolerance: float = 0.5
) -> tuple[np.ndarray, list[int]]:
    """
    Apply watershed segmentation to separate touching objects in a binary mask.
    Uses distance transform and local maxima detection for marker-based watershed.
    
    Args:
        binary_mask: 2D binary mask (YX) with objects to segment
        minsize: Minimum object size in pixels
        maxsize: Maximum object size in pixels
        min_distance: Minimum distance between watershed peaks in pixels (default: 1).
                     Increase to reduce over-segmentation (e.g., 5-10 for touching cells).
        tolerance: Not used in this implementation (kept for API compatibility)
    
    Returns:
        Tuple of:
        - labeled_mask: Labeled watershed result (2D array)
        - valid_labels: List of label IDs that meet size criteria
    """
    if binary_mask.sum() == 0:
        # Empty mask - return zeros and empty list
        return np.zeros_like(binary_mask, dtype=np.int32), []
    
    # Compute distance transform
    distance = ndimage.distance_transform_edt(binary_mask)
    
    # Find local maxima (markers for watershed)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary_mask)
    
    if len(coords) == 0:
        # No peaks found - return original labeled mask without watershed
        labeled_no_ws = measure.label(binary_mask)
        props = measure.regionprops(labeled_no_ws)
        valid_labels = [p.label for p in props if minsize <= p.area <= maxsize]
        return labeled_no_ws, valid_labels
    
    # Create markers from peaks
    markers = np.zeros(distance.shape, dtype=bool)
    markers[tuple(coords.T)] = True
    markers = ndimage.label(markers)[0]
    
    # Apply watershed
    labeled = watershed(-distance, markers, mask=binary_mask)
    
    # Filter by size criteria
    props = measure.regionprops(labeled)
    valid_labels = []
    for prop in props:
        if minsize <= prop.area <= maxsize:
            valid_labels.append(prop.label)
    
    return labeled, valid_labels


def select_primary_object(
    labeled_mask: np.ndarray,
    valid_labels: list[int],
    selection_method: str = "largest"
) -> int | None:
    """
    Select the primary object from multiple candidates.
    
    Useful when you expect 1 main object but may have touching contaminants
    after watershed splitting.
    
    Args:
        labeled_mask: Labeled mask with multiple objects
        valid_labels: List of valid object label IDs
        selection_method: How to choose the primary object:
                         - "largest": Pick the largest object (default)
                         - "central": Pick the most centrally located object
                         - "brightest": Pick based on highest mean intensity (requires intensity image)
    
    Returns:
        Label ID of the selected primary object, or None if no valid objects
    """
    if not valid_labels:
        return None
    
    if len(valid_labels) == 1:
        return valid_labels[0]
    
    props = measure.regionprops(labeled_mask)
    valid_props = [p for p in props if p.label in valid_labels]
    
    if selection_method == "largest":
        # Select largest object by area
        largest = max(valid_props, key=lambda p: p.area)
        return largest.label
    
    elif selection_method == "central":
        # Select object closest to image center
        img_center = np.array(labeled_mask.shape) / 2
        closest = min(valid_props, key=lambda p: np.linalg.norm(np.array(p.centroid) - img_center))
        return closest.label
    
    else:
        # Default to largest
        largest = max(valid_props, key=lambda p: p.area)
        return largest.label


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
    channel: int = 0,
    roi_coords: Optional[list[tuple[int, int]]] = None
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
        roi_coords: Optional list of (x, y) ROI coordinates from metadata
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing: {os.path.basename(input_path)}")
        
        # Log ROI coordinates if provided
        if roi_coords:
            logger.info(f"Using {len(roi_coords)} ROI coordinate(s): {roi_coords}")
        
        # Check if output already exists
        if os.path.exists(output_path):
            logger.info(f"Output exists, skipping: {output_path}")
            return True
        
        # Load image
        img = rp.load_tczyx_image(input_path)
        # rp.show_image(image=img, title="Input Image")
        
        # Get dimensions
        T, C, Z, Y, X = img.shape
        logger.info(f"Dimensions: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
        
        # Validate channel
        if channel >= C:
            logger.error(f"Channel {channel} does not exist (image has {C} channels)")
            return False
        
        logger.info(f"Processing channel {channel} across all {T} timepoints")
        
        # Get all timepoints for this channel at once: (T, C=1, Z, Y, X)
        data_all_t = img.data[:, channel:channel+1, :, :, :]
        
        # Check if there's any signal
        if data_all_t.max() == 0:
            logger.error(f"Channel {channel}: No signal detected in any timepoint")
            return False
        
        # Convert to float for processing - vectorized across all timepoints
        data_float = data_all_t.astype(np.float32)


        ##### STEP 0: #######
        # Check that there are at least minsize pixels above 1.5x background level
        background_level = np.percentile(data_float, 2)
        signal_mask = data_float > (1.5 * background_level)
        if np.count_nonzero(signal_mask) < minsize:
            # rp.show_image(image=data_float, mask=signal_mask, title="ERROR: Not enough signal pixels above 1.5x background")
            logger.error(f"Step 0: Not enough signal pixels above 1.5x background level")
            return False

        ##### STEP 1: #######
        # Median filter preprocessing - apply to all timepoints at once

        logger.info(f"Step 1: Median filtering all {T} timepoints...")
        # Use 5D median filter on (T, C=1, Z, Y, X) data
        footprint_5d = np.ones((1, 1, median_z, median_xy, median_xy), dtype=bool)  # (T, C, Z, Y, X)
        preprocessed = ndimage.median_filter(data_float, footprint=footprint_5d)

        # Show preprocessed image
        # rp.show_image(image=preprocessed, title="Preprocessed Image")

        ##### STEP 2: #######
        # Auto threshold per slice - vectorized where possible

        logger.info(f"Step 2: Thresholding all {T} timepoints...")
        intensity_mask = np.zeros_like(preprocessed, dtype=bool)  # (T, C=1, Z, Y, X)
        
        # get threshold for entire TCZ stack at once
        if threshold_method == 'li':
            threshold_value = filters.threshold_li(preprocessed)
        if threshold_method == 'otsu':
            threshold_value = filters.threshold_otsu(preprocessed)
        if threshold_method == 'yen':
            threshold_value = filters.threshold_yen(preprocessed)
        else: # Default to 'li'
            threshold_value = filters.threshold_li(preprocessed)

        logger.info(f"Global threshold value: {threshold_value} ")

        # Apply threshold to create mask
        intensity_mask = preprocessed > threshold_value

        # Catch errors if intensity_mask is empty in one or more frames
        intensity_sum = [np.count_nonzero(intensity_mask[t]) for t in range(T)]
        
        # If all of the timepoints are empty
        if not any(intensity_sum):
            # rp.show_image(image=preprocessed, mask=intensity_mask, title="ERROR STEP 2: All timepoints empty intensity mask")
            logger.error(f"Step 2: All timepoints have empty intensity mask after thresholding")
            return False

        # if just some are empty we may be able to resolve it...
        if not all(intensity_sum):
            # rp.show_image(image=preprocessed, mask=intensity_mask, title="ERROR STEP 2: Empty intensity mask in one or more timepoints")
            logger.warning(f"Step 2: Empty intensity mask in one or more timepoints")


        
        ##### STEP 3: #######
        # Try Connected components in TZYX
        # If this fails we may have to redo per timepoint or something else

        logger.info(f"Step 3: Finding connected components for all {T} timepoints...")

        labeled_mask, num_features = ndimage.label(intensity_mask[:, 0, :, :, :]) # pyright: ignore[reportGeneralTypeIssues]
        labeled_mask = labeled_mask[:, np.newaxis, :, :, :]
        # rp.show_image(image=preprocessed, mask=labeled_mask, title="Intensity Mask after Thresholding")


        ##### STEP 4: #######
        # Fill holes per TCZ slice
        logger.info(f"Step 4: Filling holes for all {T} timepoints... per TCZ slice")
        for t in range(T):
            for z in range(Z):
                for label in np.unique(labeled_mask[t, 0, z]):
                    if label == 0:
                        continue  # Skip background
                    single_object_mask = (labeled_mask[t, 0, z] == label)
                    filled_mask = ndimage.binary_fill_holes(single_object_mask)
                    labeled_mask[t, 0, z][filled_mask] = label

        ##### STEP 5: #######
        # Filter components by size, apply watershed on-the-fly if oversized components found
        tmp_labeled_mask = labeled_mask.copy()  
        logger.info(f"Step 5: Filtering components by size for all {T} timepoints... per TCZ slice")
        
        watershed_count = 0  # Track how many slices needed watershed
        
        for t in range(T):
            for z in range(Z):
                component_sizes = np.bincount(tmp_labeled_mask[t, 0, z].ravel())
                
                # Check if any components exceed maxsize
                oversized_components = np.where(component_sizes > maxsize)[0]
                
                if len(oversized_components) > 0 and use_watershed:
                    # Apply watershed immediately to this slice
                    logger.info(f"T={t}, Z={z}: Found {len(oversized_components)} oversized component(s), applying watershed")
                    watershed_count += 1
                    
                    labels_ws, valid_labels = apply_watershed_segmentation(
                        binary_mask=intensity_mask[t, 0, z],
                        minsize=minsize,
                        maxsize=maxsize,
                        min_distance=1,  # ImageJ default: all local maxima above tolerance
                        tolerance=0.5     # ImageJ MaximumFinder default tolerance
                    )
                    
                    # Filter watershed result to keep only valid components
                    mask_filtered = np.isin(labels_ws, valid_labels)
                    tmp_labeled_mask[t, 0, z] = labels_ws * mask_filtered
                    logger.info(f"  T={t}, Z={z}: Watershed produced {len(valid_labels)} valid components")
                else:
                    # No watershed needed - just filter by size as normal
                    valid_components = np.where((component_sizes >= minsize) & (component_sizes <= maxsize))[0]
                    mask_filtered = np.isin(tmp_labeled_mask[t, 0, z], valid_components)
                    tmp_labeled_mask[t, 0, z] = tmp_labeled_mask[t, 0, z] * mask_filtered
        
        if watershed_count > 0:
            logger.info(f"Step 5: Applied watershed to {watershed_count} slices with oversized components")
            rp.show_image(image=preprocessed, mask=tmp_labeled_mask, title=f"After Watershed")
        
        # rp.show_image(image=preprocessed, mask=tmp_labeled_mask, title=f"No Watershed")


        
        # Update labeled_mask with the filtered/watershed results
        labeled_mask = tmp_labeled_mask
        
        # Check if empty in one or more frames after watershed
        empty_frames = [t for t in range(T) if np.count_nonzero(labeled_mask[t, 0]) == 0]
        if len(empty_frames) > 0:
            logger.warning(f"Empty frames after watershed: {empty_frames}")
        
        

        ##### STEP 6: #######
        # Remove objects on XY edges per TCZ slice
        # Strategy: Try edge removal first. If it eliminates all signal, apply watershed then remove edges.
        
        logger.info(f"Step 6: Removing edge-touching objects for all {T} timepoints... per TCZ slice")
        
        tmp_labeled_mask = labeled_mask.copy()
        edge_removal_failed = False  # Track if edge removal killed everything
        
        for t in range(T):
            for z in range(Z):
                edge_labels = set()
                # Check edges
                edges = [
                    labeled_mask[t, 0, z, 0, :],    # Top edge
                    labeled_mask[t, 0, z, -1, :],   # Bottom edge
                    labeled_mask[t, 0, z, :, 0],    # Left edge
                    labeled_mask[t, 0, z, :, -1]    # Right edge
                ]
                for edge in edges:
                    edge_labels.update(np.unique(edge))
                
                edge_labels.discard(0)  # Remove background
                
                # Remove edge-touching objects
                for label in edge_labels:
                    tmp_labeled_mask[t, 0, z][tmp_labeled_mask[t, 0, z] == label] = 0
        
        # Check if edge removal killed everything
        if tmp_labeled_mask.sum() == 0:
            logger.warning("Step 6: Edge removal eliminated all objects - applying watershed fallback")
            edge_removal_failed = True
            
            # Fallback: Apply watershed to original mask, THEN remove edges
            logger.info("Step 6 Fallback: Applying watershed to recover signal...")
            watershed_fallback_count = 0
            
            for t in range(T):
                for z in range(Z):
                    if labeled_mask[t, 0, z].sum() > 0:
                        # Apply watershed to this slice
                        labels_ws, valid_labels = apply_watershed_segmentation(
                            binary_mask=intensity_mask[t, 0, z],
                            minsize=minsize,
                            maxsize=maxsize,
                            min_distance=1,   # ImageJ default: all local maxima above tolerance
                            tolerance=0.5      # ImageJ MaximumFinder default tolerance
                        )
                        
                        # Filter to keep only valid components
                        mask_filtered = np.isin(labels_ws, valid_labels)
                        tmp_labeled_mask[t, 0, z] = labels_ws * mask_filtered
                        
                        # NOW remove edges from watershed result
                        edge_labels = set()
                        edges = [
                            tmp_labeled_mask[t, 0, z, 0, :],
                            tmp_labeled_mask[t, 0, z, -1, :],
                            tmp_labeled_mask[t, 0, z, :, 0],
                            tmp_labeled_mask[t, 0, z, :, -1]
                        ]
                        for edge in edges:
                            edge_labels.update(np.unique(edge))
                        edge_labels.discard(0)
                        
                        for label in edge_labels:
                            tmp_labeled_mask[t, 0, z][tmp_labeled_mask[t, 0, z] == label] = 0
                        
                        if len(valid_labels) > 0:
                            watershed_fallback_count += 1
            
            logger.info(f"Step 6 Fallback: Watershed applied to {watershed_fallback_count} slices")
            
            # Check if we still have signal after fallback
            if tmp_labeled_mask.sum() == 0:
                logger.error("Step 6 Fallback: Still no signal after watershed - segmentation failed")
                rp.show_image(image=preprocessed, mask=tmp_labeled_mask, 
                            title="ERROR STEP 6: No signal after edge removal + watershed fallback")

                
                
                labeled_mask_squeezed = labeled_mask[:, 0, :, :, :]
                
                # Convert to binary (0 or 255) 8-bit mask
                labeled_mask_out = (labeled_mask_squeezed > 0).astype(np.uint8) * 255
                
                logger.info(f"Mask shape before save: {labeled_mask_out.shape}, dtype: {labeled_mask_out.dtype}, max_value: {labeled_mask_out.max()}")

                tifffile.imwrite(
                    os.path.splitext(output_path)[0] + "_before_failed_edge_removal.tif",
                    labeled_mask_out,
                    imagej=True,  # ImageJ-compatible format
                    metadata={'axes': 'TZYX'},  # Specify axis order (4D: T, Z, Y, X)
                    compression='deflate'  # Lossless compression
                )
                rp.show_image(image=preprocessed, mask=labeled_mask,
                              title="ERROR STEP 6: It was like this before")
                return False
        
        # Update labeled_mask with edge-cleaned result
        labeled_mask = tmp_labeled_mask
        
        if edge_removal_failed:
            logger.info("Step 6: Fallback successful - continuing with watershed-recovered mask")
        else:
            logger.info(f"Step 6: Edge removal successful - {np.count_nonzero(labeled_mask)} pixels remaining")



        ##### STEP 7: #######
        # Are all the same objects kept in all frames?
        logger.info(f"Step 7: Checking component ID-consistency across {T} timepoints...")
        components_in_frame_1 = set(np.unique(labeled_mask[0, 0, :, :, :])) - {0}
        for t in range(1, T):
            components_in_frame_t = set(np.unique(labeled_mask[t, 0, :, :, :])) - {0}
            if components_in_frame_t != components_in_frame_1:
                logger.warning(f"Components differ in frame {t}: {components_in_frame_t}")
                rp.show_image(image=preprocessed, mask=labeled_mask, title="ERROR: Components ID differ between timepoints")        
        
        ##### STEP 8: #######
        # Does the size of each component size change less than a threshold over time?
        logger.info(f"Step 8: Checking component size consistency across {T} timepoints...")
        size_change_threshold = 0.1  # Example threshold
        for component_id in components_in_frame_1:
            sizes_over_time = [np.sum(labeled_mask[t, 0] == component_id) for t in range(T)]
            if max(sizes_over_time) / min(sizes_over_time) < (1 + size_change_threshold):
                logger.info(f"Component {component_id} size change is within threshold over time.")
            else:
                logger.warning(f"Component {component_id} size change is outside threshold over time.")
                rp.show_image(image=preprocessed, mask=labeled_mask, title="ERROR: Large size change over time")

        ##### STEP 9: #######
        # particle analysis per TCZ slice
        logger.info(f"Step 9: Performing particle analysis for all {T} timepoints... per TCZ slice")
        results = []
        for t in range(T):
            for z in range(Z):
                props = measure.regionprops(labeled_mask[t, 0, z])
                for prop in props:
                    results.append({
                        'timepoint': t,
                        'z_slice': z,
                        'label': prop.label,
                        'area': prop.area,
                        'centroid': prop.centroid
                    })  

        
        # Save mask
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        #rp.save_tczyx_image(output_mask, output_path)
        # Lets make an exeption for saving these masks as normal tif instead of ome.tif
        # since normal tifs gets a preview in Windows explorer
        
        # Save as binary 8-bit mask (0 or 255) for ImageJ compatibility
        # Squeeze out the C dimension: (T, C=1, Z, Y, X) -> (T, Z, Y, X)
        labeled_mask_squeezed = labeled_mask[:, 0, :, :, :]
        #rp.show_image(image=data_float, mask=labeled_mask_squeezed, title="Final Mask to be saved")

        
        # Convert to binary (0 or 255) 8-bit mask
        labeled_mask_out = (labeled_mask_squeezed > 0).astype(np.uint8) * 255
        
        logger.info(f"Mask shape before save: {labeled_mask_out.shape}, dtype: {labeled_mask_out.dtype}, max_value: {labeled_mask_out.max()}")

        tifffile.imwrite(
            output_path,
            labeled_mask_out,
            imagej=True,  # ImageJ-compatible format
            metadata={'axes': 'TZYX'},  # Specify axis order (4D: T, Z, Y, X)
            compression='deflate'  # Lossless compression
        )
        
        logger.info(f"Saved mask: {output_path}")

        # Save max projections for quick visualization
        
        # Save imageJ ROIs if requested
        if save_roi:
            roi_folder = os.path.splitext(output_path)[0] + "_rois"
            try:
                # Use helper function to save all ROIs from the mask
                roi_count = rp.save_imagej_rois_from_mask(
                    mask=labeled_mask,
                    output_path=roi_folder,
                    name_pattern=f"T{{t}}_C{channel if channel >= 0 else 0}_Z{{z}}_obj{{label}}.roi"
                )
                
                logger.info(f"Saved {roi_count} ImageJ ROI files to: {roi_folder}")
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
    dry_run: bool = False,
    yaml_pattern: Optional[str] = None
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
        yaml_pattern: Optional YAML search pattern for ROI metadata
    """
    # If YAML pattern provided, use grouped file processing
    if yaml_pattern:
        logger.info("Using grouped file processing with YAML metadata")
        search_subfolders = '**' in input_pattern or '**' in yaml_pattern
        
        search_patterns = {
            'image': input_pattern,
            'yaml': yaml_pattern
        }
        
        grouped_files = rp.get_grouped_files_to_process(
            search_patterns=search_patterns,
            search_subfolders=search_subfolders
        )
        
        if not grouped_files:
            logger.error("No files matched the search patterns.")
            return
        
        logger.info(f"Found {len(grouped_files)} file groups to process")
        
        # Determine output folder
        if output_folder is None:
            base_pattern = input_pattern or yaml_pattern
            output_folder = os.path.dirname(base_pattern) + "_threshold" or "output_threshold"
        
        logger.info(f"Output folder: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)
        
        # Build task list from groups
        tasks = []
        for basename, files in grouped_files.items():
            if 'image' in files:
                img_path = files['image']
                yaml_path = files.get('yaml', None)
                
                # Get ROI coordinates if YAML exists
                roi_coords = None
                if yaml_path:
                    roi_coords = get_coordinates_from_metadata(yaml_path)
                    if roi_coords:
                        logger.debug(f"Found {len(roi_coords)} ROI(s) in {basename}")
                
                # Prepare output path
                out_name = os.path.splitext(os.path.basename(img_path))[0] + output_extension + ".tif"
                out_path = os.path.join(output_folder, out_name)
                
                tasks.append((img_path, out_path, roi_coords))
            else:
                logger.warning(f"Skipping '{basename}': missing image file")
        
        if not tasks:
            logger.warning("No valid image files were found after grouping.")
            return
        
        logger.info(f"Processing {len(tasks)} valid files")
        
        # Dry run
        if dry_run:
            print(f"[DRY RUN] Would process {len(tasks)} files")
            print(f"[DRY RUN] Output folder: {output_folder}")
            print(f"[DRY RUN] Parameters: minsize={minsize}, maxsize={maxsize}, "
                  f"median_xy={median_xy}, median_z={median_z}, threshold={threshold_method}, "
                  f"channel={channel}")
            for img, out, coords in tasks:
                coord_info = f" (with {len(coords)} ROI coordinates)" if coords else " (no ROI)"
                print(f"[DRY RUN] {img} -> {out}{coord_info}")
            return
        
        # Process files
        if no_parallel or len(tasks) == 1:
            for img_path, out_path, roi_coords in tasks:
                try:
                    segment_single_file(
                        img_path, out_path,
                        minsize=minsize,
                        maxsize=maxsize,
                        median_xy=median_xy,
                        median_z=median_z,
                        threshold_method=threshold_method,
                        max_objects=max_objects,
                        save_roi=save_roi,
                        use_watershed=use_watershed,
                        channel=channel,
                        roi_coords=roi_coords
                    )
                except Exception as e:
                    logger.error(f"Failed to process {img_path}: {e}")
        else:
            cpu_count = os.cpu_count() or 1
            cpu_count = max(cpu_count - 1, 1)
            
            with ProcessPoolExecutor(max_workers=cpu_count) as executor:
                futures = []
                for img_path, out_path, roi_coords in tasks:
                    future = executor.submit(
                        segment_single_file,
                        img_path, out_path,
                        minsize, maxsize, median_xy, median_z,
                        threshold_method, max_objects, save_roi,
                        use_watershed, channel, roi_coords
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"A file failed to process: {e}")
        return
    
    # Original single-pattern processing (no YAML)
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
        "--yaml-search-pattern",
        type=str,
        required=False,
        help="Glob for YAML ROI metadata files, e.g. './input_data/**/*_metadata.yaml'"
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
        help="Channel index to process (0-based, default: 0)"
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
        dry_run=args.dry_run,
        yaml_pattern=args.yaml_search_pattern
    )


if __name__ == "__main__":
    main()
