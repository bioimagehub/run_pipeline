import subprocess
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import numpy as np
from bioio.writers import OmeTiffWriter
from skimage.measure import label as skimage_label
from scipy.ndimage import binary_fill_holes, label as ndimage_label
import logging
from skimage.transform import resize
import tempfile
import time
import uuid

from segment_threshold import LabelInfo, remove_small_or_large_labels, remove_on_edges
from track_indexed_mask import track_labels_with_trackpy
import bioimage_pipeline_utils as rp
from skimage.filters import median
from skimage.morphology import disk

import yaml
# Add tqdm for progress bars (fallback to no-op if not installed)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

# Configure logging

logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')


# Holes defined as pixels around which are all non-zero, but the pixel itself is zero
def fill_xy_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes within each labeled region independently.
    
    Works on both binary (0/255) and indexed (0, 1, 2, 3, ...) masks.
    For binary masks, fills all holes. For indexed, fills holes per label.
    """
    # Start with a copy of the original to preserve existing labels
    filled = np.copy(mask)
    
    for t_idx in range(mask.shape[0]):
        for c_idx in range(mask.shape[1]):
            for z_idx in range(mask.shape[2]):
                slice_ = mask[t_idx, c_idx, z_idx]
                unique_labels = np.unique(slice_)
                
                # For binary masks (only 0 and one other value), just fill once
                if len(unique_labels) <= 2:
                    binary_mask = slice_ > 0
                    filled_region = binary_fill_holes(binary_mask)
                    # Fill holes with the non-zero value
                    nonzero_val = unique_labels[unique_labels > 0][0] if len(unique_labels) > 1 else 0
                    holes = filled_region & ~binary_mask
                    filled[t_idx, c_idx, z_idx][holes] = nonzero_val
                else:
                    # For indexed masks, fill holes per label
                    for label_id in unique_labels:
                        if label_id == 0:
                            continue  # Skip background
                        region_mask = slice_ == label_id
                        filled_region = binary_fill_holes(region_mask)
                        # Only fill where holes exist (filled but not in original)
                        holes = filled_region & ~region_mask
                        filled[t_idx, c_idx, z_idx][holes] = label_id
                    
    return filled

# gaps are defined as pixels that are zero but have non-zero labels at both t-gap and t+gap
def fill_temporal_gaps(output_data: np.ndarray, max_time_gap: int = 1, require_matching_labels: bool = True) -> tuple[np.ndarray, int]:
    """
    Fill temporal gaps in mask data based on spatial overlap.
    
    If a pixel is background at time t, but has non-zero values at both t-gap and t+gap,
    fill it with the label from t-gap. Supports multi-gap filling.
    
    Args:
        output_data: 5D array in TCZYX order
        max_time_gap: Maximum temporal gap to fill (e.g., 2 means check up to t±2)
        require_matching_labels: If True, only fill if labels match at t-gap and t+gap.
                                If False, fill based on spatial overlap (any labels present).
                                Use True for indexed masks, False for binary masks.
    
    Returns:
        tuple: (filled_data, total_pixels_filled)
    """
    fill_count = 0
    
    for c in range(output_data.shape[1]):
        for z in range(output_data.shape[2]):
            if output_data.shape[0] < (2 * max_time_gap + 1):  # Need enough time points
                continue
            
            # Iterate through each time point
            for t in range(output_data.shape[0]):
                t_current = output_data[t, c, z, :, :]
                
                # Only process background pixels
                is_background = (t_current == 0)
                if not np.any(is_background):
                    continue
                
                # Check for labels within max_time_gap frames before and after
                for gap in range(1, max_time_gap + 1):
                    t_before_idx = t - gap
                    t_after_idx = t + gap
                    
                    # Check if indices are valid
                    if t_before_idx < 0 or t_after_idx >= output_data.shape[0]:
                        continue
                    
                    t_before = output_data[t_before_idx, c, z, :, :]
                    t_after = output_data[t_after_idx, c, z, :, :]
                    
                    # Find pixels that:
                    # 1. Are background at current time
                    # 2. Have non-zero values at both t-gap and t+gap
                    # 3. (Optional) Labels match at t-gap and t+gap
                    has_before = (t_before > 0)
                    has_after = (t_after > 0)
                    
                    if require_matching_labels:
                        # For indexed masks: require matching labels to prevent merging different objects
                        match_mask = (t_before == t_after) & (t_before > 0)
                        fill_mask = is_background & match_mask
                    else:
                        # For binary masks: fill based on spatial overlap
                        fill_mask = is_background & has_before & has_after
                    
                    # Count and fill
                    n_filled = np.sum(fill_mask)
                    if n_filled > 0:
                        # Fill with the label from t-gap
                        t_current[fill_mask] = t_before[fill_mask]
                        output_data[t, c, z, :, :] = t_current
                        fill_count += n_filled
                        
                        # Update is_background for next gap iteration
                        is_background = (t_current == 0)
                        if not np.any(is_background):
                            break  # No more background pixels to fill
    
    return output_data, fill_count

# gaps are defined as pixels that are zero but have the same non-zero label directly above and below in Z
def fill_z_gaps(mask: np.ndarray, max_z_gap: int = 1) -> tuple[np.ndarray, int]:
    """Fill holes in the Z direction based on matching labels above and below.
    
    If a pixel at position (t, c, z, y, x) is background (0), but the pixels at 
    z-gap and z+gap have the same non-zero label, fill the pixel with that label.
    Supports multi-gap filling.
    
    Args:
        mask: 5D array in TCZYX order
        max_z_gap: Maximum Z gap to fill (e.g., 2 means check up to z±2)
        
    Returns:
        tuple: (filled_mask, total_pixels_filled)
    """
    fill_count = 0
    filled = np.copy(mask)
    
    for t in range(filled.shape[0]):
        for c in range(filled.shape[1]):
            if filled.shape[2] < (2 * max_z_gap + 1):  # Need enough z-slices
                continue
            
            # Iterate through each z-slice
            for z in range(filled.shape[2]):
                z_current = filled[t, c, z, :, :]
                
                # Only process background pixels
                is_background = (z_current == 0)
                if not np.any(is_background):
                    continue
                
                # Check for labels within max_z_gap slices before and after
                for gap in range(1, max_z_gap + 1):
                    z_below_idx = z - gap
                    z_above_idx = z + gap
                    
                    # Check if indices are valid
                    if z_below_idx < 0 or z_above_idx >= filled.shape[2]:
                        continue
                    
                    z_below = filled[t, c, z_below_idx, :, :]
                    z_above = filled[t, c, z_above_idx, :, :]
                    
                    # Find pixels that:
                    # 1. Are background at current z
                    # 2. Have matching non-zero labels at both z-gap and z+gap
                    match_mask = (z_below == z_above) & (z_below > 0)
                    
                    # Fill current with the matching label where current is background (0)
                    fill_mask = is_background & match_mask
                    
                    # Count and fill
                    n_filled = np.sum(fill_mask)
                    if n_filled > 0:
                        # Fill with the label from z-gap
                        z_current[fill_mask] = z_below[fill_mask]
                        filled[t, c, z, :, :] = z_current
                        fill_count += n_filled
                        
                        # Update is_background for next gap iteration
                        is_background = (z_current == 0)
                        if not np.any(is_background):
                            break  # No more background pixels to fill
    
    return filled, fill_count


def _save_empty_output(args, input_file: str, output_tif_file_path: str) -> None:
    """Write an empty OME-TIFF (all zeros) with TCZYX order so every input yields an output.
    Tries to infer T, Z, Y, X from the input npy (expected TZYXC) and uses C=1.
    Also writes a metadata YAML mirroring the regular path.
    """
    # Infer shape from input where possible
    try:
        empty = None
        if os.path.splitext(input_file)[1].lower() == ".npy":
            arr = np.load(input_file, allow_pickle=False)
            if arr.ndim == 5:
                t, z, y, x, _ = arr.shape
                empty = np.zeros((t, 1, z, y, x), dtype=np.uint8)
            elif arr.ndim == 4:
                t, z, y, x = arr.shape
                empty = np.zeros((t, 1, z, y, x), dtype=np.uint8)
        if empty is None:
            empty = np.zeros((1, 1, 1, 1, 1), dtype=np.uint8)
    except Exception:
        empty = np.zeros((1, 1, 1, 1, 1), dtype=np.uint8)

    # Save empty image
    try:
        rp.save_tczyx_image(empty, output_tif_file_path, dim_order="TCZYX")
    except Exception as e:
        logging.error(f"Failed to save empty output for {input_file}: {e}")
        return

    # Write metadata YAML (best-effort)
    try:
        input_metadata_file_path = os.path.splitext(input_file)[0] + "_metadata.yaml"
        output_metadata_file_path = os.path.splitext(output_tif_file_path)[0] + "_metadata.yaml"
        metadata = {}
        if os.path.exists(input_metadata_file_path):
            with open(input_metadata_file_path, 'r') as f:
                try:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        metadata = loaded
                except Exception:
                    metadata = {}
        metadata["Segmentation Ilastik"] = {
            "Method": "Ilastik headless segmentation",
            "Project_path": getattr(args, 'project_path', None),
            "Min_sizes": getattr(args, 'min_sizes', None),
            "Max_sizes": getattr(args, 'max_sizes', None),
            "Remove_xy_edges": getattr(args, 'remove_xy_edges', False),
            "Remove_z_edges": getattr(args, 'remove_z_edges', False),
            "Max_z_gap": getattr(args, 'max_z_gap', -1),
            "Max_time_gap": getattr(args, 'max_time_gap', -1),
            "Output_format": "OME-TIFF",
            "Output_dim_order": "TCZYX",
            "Tracking": "4D connected component labeling (automatic)",
            "Hole_filling": "scipy.ndimage.binary_fill_holes",
            "Edge_smoothing": "median filter (disk(3))",
            "Note": "Empty placeholder written because segmentation failed or produced no output."
        }
        with open(output_metadata_file_path, 'w') as f:
            yaml.dump(metadata, f)
    except Exception:
        # Non-fatal if metadata can't be written
        pass


def process_file(args, input_file):
    # Removed per-file print; progress is handled by tqdm in process_folder
    # Define output file paths
    
    in_dir = os.path.dirname(input_file)
    out_np_file = os.path.join(in_dir, os.path.splitext(os.path.basename(input_file))[0] + "_segmentation.np")
    output_tif_file_path = os.path.join(args.output_folder, os.path.basename(os.path.splitext(out_np_file)[0]) + ".tif")
    h5_path = os.path.splitext(out_np_file)[0] + ".h5"

    # Prepare a unique Ilastik log directory per process to avoid Windows file-lock races
    unique_log_dir = os.path.join(tempfile.gettempdir(), f"ilastik_logs_{uuid.uuid4().hex}")
    os.makedirs(unique_log_dir, exist_ok=True)
    env = os.environ.copy()
    env["ILASTIK_LOG_DIR"] = unique_log_dir

    # Run Ilastik headless, suppress warnings by filtering stderr (keep real errors)
    proc = subprocess.run([
        args.ilastik_path,
        '--headless',
        f'--project={args.project_path}',
        '--export_source=simple segmentation',
        f'--raw_data={input_file}',
        f'--output_filename_format={out_np_file}'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env)

    # Filter and print only non-warning stderr lines
    if proc.stderr:
        non_warning_lines = [
            line for line in proc.stderr.splitlines()
            if "WARNING" not in line and "UserWarning" not in line
        ]
        non_warning_output = "\n".join(non_warning_lines).strip()
        if non_warning_output:
            print(non_warning_output)

    # Give the filesystem a brief moment for the output to appear
    if not os.path.exists(h5_path):
        for _ in range(10):  # wait up to ~10s
            time.sleep(1)
            if os.path.exists(h5_path):
                break

    # If Ilastik failed and no output exists, write empty output for this file
    if not os.path.exists(h5_path):
        if proc.returncode != 0:
            print(f"Ilastik failed (code {proc.returncode}) and produced no output for: {input_file}")
        else:
            print(f"Ilastik returned success but output is missing for: {input_file}. Expected: {h5_path}")
        _save_empty_output(args, input_file, output_tif_file_path)
        return

    # Read Ilastik output
    with h5py.File(h5_path, 'r') as f:
        group_key = list(f.keys())[0]
        data = np.array(f[group_key])  # Expected: TZYXC

    # Rearrange to TCZYX (be tolerant to 4D TZYX)
    if data.ndim == 5:
        data_tczyx = np.transpose(data, (0, 4, 1, 2, 3))
    elif data.ndim == 4:
        # Assume T, Z, Y, X -> add a singleton channel
        data_tczyx = data[:, np.newaxis, :, :, :]
    else:
        logging.warning(f"Unexpected data shape: {data.shape}, expected 5D (T, Z, Y, X, C) or 4D (T, Z, Y, X)")
        _save_empty_output(args, input_file, output_tif_file_path)
        return

    # C should always be 1 for segmentation; if not, keep the first channel and continue
    if data_tczyx.shape[1] != 1:
        logging.warning(f"Unexpected number of channels: {data_tczyx.shape[1]}, expected 1 for segmentation; using first channel.")
        data_tczyx = data_tczyx[:, :1, ...]

    # Determine which ilastik labels to keep
    all_values = np.unique(data_tczyx)
    nonzero_values = all_values[all_values > 0]  # Exclude background (0)
    keep = getattr(args, 'keep_labels', None)
    if keep:
        keep_set = set(keep)
        unique_values = np.array([v for v in nonzero_values if v in keep_set], dtype=nonzero_values.dtype)
    else:
        unique_values = nonzero_values

    
    # return if no unique values found
    if len(unique_values) == 0:
        logging.warning(f"No unique values found in {input_file}.")
        #rp.save_tczyx_image(np.zeros_like(data_tczyx), output_tif_file_path, dim_order="TCZYX")
        _save_empty_output(args, input_file, output_tif_file_path)  # Save metadata for empty output
        return

    # make a dataset like data_tczyx but with the len(unique_values) channels
    # Use uint16 to support more than 255 unique objects
    output_data = np.zeros((data_tczyx.shape[0], len(unique_values), data_tczyx.shape[2], data_tczyx.shape[3], data_tczyx.shape[4]), dtype=np.uint16)

    # loop over unique values and create BINARY masks (0 or 255) for each ilastik class
    for i, value in enumerate(unique_values):
        output_data[:, i, :, :, :] = (data_tczyx[:,0,:,:,:] == value).astype(np.uint16) * 255  # Convert boolean to uint16

    # --- Fill holes in BINARY data BEFORE labeling ---
    # This creates more complete objects before we assign unique IDs
    
    # Fill holes in yx (per-frame, per-channel)
    logging.info("Filling XY holes in binary masks...")
    output_data = fill_xy_holes(output_data)
    
    # Fill holes in z direction
    max_z_gap = getattr(args, 'max_z_gap', -1)
    if max_z_gap > 0:
        logging.info(f"Filling Z gaps in binary masks (max_gap={max_z_gap})...")
        output_data, z_fill_count = fill_z_gaps(output_data, max_z_gap)
        logging.info(f"Filled {z_fill_count} pixels in Z-direction gaps")
    
    # Fill holes in time direction
    max_time_gap = getattr(args, 'max_time_gap', -1)
    if max_time_gap > 0:
        logging.info(f"Filling temporal gaps in binary masks (max_gap={max_time_gap})...")
        # For binary masks, we don't require matching labels (since they're all 255)
        output_data, temporal_fill_count = fill_temporal_gaps(output_data, max_time_gap, require_matching_labels=False)
        logging.info(f"Filled {temporal_fill_count} pixels in temporal gaps")
    
    # NOW label connected components on the FILLED binary masks
    # For simple tracking (no divisions/merging), label across TZYX as one 4D volume
    logging.info("Labeling connected components in 4D (TZYX)...")
    for i, value in enumerate(unique_values):
        # Convert to binary (0 or 1) for labeling
        binary_mask = (output_data[:, i, :, :, :] > 0).astype(np.uint8)
        
        # Label in 4D - this gives automatic temporal tracking!
        # Each connected component across time gets a unique ID
        # Use scipy.ndimage.label which supports N-dimensional arrays
        # Structure defines connectivity: 1 for face connectivity (6-connectivity in 3D, 8 in 4D)
        labeled_4d, n_objects = ndimage_label(binary_mask)
        
        logging.info(f"  Channel {i} (ilastik label {value}): {n_objects} unique objects found")
        
        # Check if we need uint16 or can use uint8
        if n_objects > 255:
            logging.warning(f"    More than 255 objects ({n_objects}), using uint16")
            output_data[:, i, :, :, :] = labeled_4d.astype(np.uint16)
        else:
            output_data[:, i, :, :, :] = labeled_4d.astype(np.uint16)
    
    
    if data_tczyx.shape[1] != 1:
        logging.warning(f"Unexpected number of channels: {data_tczyx.shape[1]}, expected 1 for segmentation")
        # Do not return; continue with computed output_data
    
    min_sizes = args.min_sizes if hasattr(args, 'min_sizes') else [0]
    max_sizes = args.max_sizes if hasattr(args, 'max_sizes') else [float('inf')]
    if isinstance(min_sizes, int):
        min_sizes = [min_sizes]
    if isinstance(max_sizes, int):
        max_sizes = [max_sizes]
    channels = list(range(output_data.shape[1]))
    label_info_list =LabelInfo.from_mask(output_data)
    output_data, label_info_list_new = remove_small_or_large_labels(output_data, label_info_list, channels=channels, min_sizes=min_sizes, max_sizes=max_sizes)

    
    # return if no labels left after filtering min max sizes
    if not label_info_list_new:

        print("Unique npixels before filtering:")
        npixels_list = [li.npixels for li in label_info_list]
        unique_npixels = sorted(set(npixels_list))
        print(unique_npixels)

        print(f"No labels left after min-max filtering in {input_file}.")
        # rp.save_tczyx_image(np.zeros_like(data_tczyx), output_tif_file_path, dim_order="TCZYX")
        _save_empty_output(args, input_file, output_tif_file_path)  # Save metadata for empty output

        return
    
    label_info_list = label_info_list_new


    # --- Edge removal ---
    if getattr(args, 'remove_xy_edges', False) or getattr(args, 'remove_z_edges', False):
        output_data, label_info_list = remove_on_edges(output_data, label_info_list, remove_xy_edges=getattr(args, 'remove_xy_edges', False), remove_z_edges=getattr(args, 'remove_z_edges', False))


    # --- Smoothing out edges ---
    # This is done by median filtering the edges
    # Apply median filter to all YX planes for each t, c, z
    for t in range(output_data.shape[0]):
        for c in range(output_data.shape[1]):
            for z in range(output_data.shape[2]):
                # Only smooth non-background pixels
                mask = output_data[t, c, z] > 0
                smoothed = median(mask.astype(np.uint8), disk(3))
                # Keep original label values, but set background to 0 after smoothing
                output_data[t, c, z] *= smoothed
    
    # --- Ensure final labels are contiguous starting from 1 ---
    for c in range(output_data.shape[1]):  
        unique_labels = np.unique(output_data[:, c, :, :, :])
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background
        new_label = 1
        label_mapping = {}
        for ul in unique_labels:
            label_mapping[ul] = new_label
            new_label += 1
        # Apply mapping
        for old_label, new_label in label_mapping.items():
            output_data[:, c, :, :, :][output_data[:, c, :, :, :] == old_label] = new_label 
    
    # --- Apply output scaling if needed ---
    scale = getattr(args, 'output_scale', 1)
    if scale != 1:
        # output_data shape: (T, C, Z, Y, X)
        t, c, z, y, x = output_data.shape
        new_y = int(y * scale)
        new_x = int(x * scale)
        scaled = np.zeros((t, c, z, new_y, new_x), dtype=output_data.dtype)
        for ti in range(t):
            for ci in range(c):
                for zi in range(z):
                    # Use order=0 for nearest-neighbor (preserve labels)
                    scaled[ti, ci, zi] = resize(
                        output_data[ti, ci, zi],
                        (new_y, new_x),
                        order=0,
                        preserve_range=True,
                        anti_aliasing=False
                    ).astype(output_data.dtype)
        output_data = scaled

    # Save as OME-TIFF
    # This is now done per-class outputs
    # rp.save_tczyx_image(output_data, output_tif_file_path, dim_order="TCZYX")


    # --- Save metadata YAML ---
    input_metadata_file_path = os.path.splitext(input_file)[0] + "_metadata.yaml"
    output_metadata_file_path = os.path.splitext(output_tif_file_path)[0] + "_metadata.yaml"
    
    metadata = {}
    if os.path.exists(input_metadata_file_path):
        with open(input_metadata_file_path, 'r') as f:
            metadata = yaml.safe_load(f)

    # Add segmentation info
    metadata["Segmentation Ilastik"] = {
        "Method": "Ilastik headless segmentation",
        "Project_path": args.project_path,
        "Min_sizes": args.min_sizes,
        "Max_sizes": args.max_sizes,
        "Remove_xy_edges": getattr(args, 'remove_xy_edges', False),
        "Remove_z_edges": getattr(args, 'remove_z_edges', False),
        "Max_z_gap": getattr(args, 'max_z_gap', -1),
        "Max_time_gap": getattr(args, 'max_time_gap', -1),
        "Output_format": "OME-TIFF",
        "Output_dim_order": "TCZYX",
        "Tracking": "4D connected component labeling (automatic)",
        "Hole_filling": "scipy.ndimage.binary_fill_holes",
        "Edge_smoothing": "median filter (disk(3))",
        "Kept_labels": [int(v) for v in (unique_values.tolist() if hasattr(unique_values, 'tolist') else list(unique_values))]
    }
    with open(output_metadata_file_path, 'w') as f:
        yaml.dump(metadata, f)

    # Additionally, save per-class outputs so each kept label gets its own file (C=1)
    # If only one label is kept, don't add the _class{value} suffix
    try:
        for i, value in enumerate(unique_values):
            per_class = output_data[:, i:i+1, ...]  # keep channel dimension as 1
            
            # Determine filename based on number of unique values
            if len(unique_values) == 1:
                # Single label: use base filename without class suffix
                per_class_path = output_tif_file_path
            else:
                # Multiple labels: add class suffix
                per_class_path = os.path.splitext(output_tif_file_path)[0] + f"_class{int(value)}.tif"
            
            rp.save_tczyx_image(per_class, per_class_path, dim_order="TCZYX")
            
            # # Write per-class metadata
            # per_class_meta_path = os.path.splitext(per_class_path)[0] + "_metadata.yaml"
            # per_meta = dict(metadata) if isinstance(metadata, dict) else {}
            # per_meta.setdefault("Segmentation Ilastik", {})
            # per_meta["Segmentation Ilastik"]["Kept_label"] = int(value)
            # with open(per_class_meta_path, 'w') as f:
            #     yaml.dump(per_meta, f)
    except Exception as e:
        logging.warning(f"Per-class saving failed for {input_file}: {e}")

def process_folder(args: argparse.Namespace) -> None:
    # Find files to process using glob pattern
    pattern = args.input_search_pattern
    files_to_process = rp.get_files_to_process2(pattern, False)

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    if args.no_parallel:
        for input_file in tqdm(files_to_process, desc="Processing files", unit="file"):
            process_file(args, input_file)
    else:
        with ProcessPoolExecutor() as executor:
            future_to_file = {executor.submit(process_file, args, input_file): input_file for input_file in files_to_process}
            for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing files", unit="file"):
                input_file = future_to_file[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f'File {input_file} generated an exception: {exc}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BioImage files.")
    parser.add_argument("--ilastik-path", type=str, required=True, help="Path to the ilastik executable")
    parser.add_argument("--input-search-pattern", type=str, required=True, help="Glob pattern for input images, e.g. './input_Ilastik/*.npy'")
    parser.add_argument("--project-path", type=str, required=True, help="Path to already trained ilp project")
    parser.add_argument("--min-sizes", type=rp.split_comma_separated_intstring, default=[0], help="Minimum size for processing. Comma-separated list, one per channel, or single value for all.")
    parser.add_argument("--max-sizes", type=rp.split_comma_separated_intstring, default=[99999999], help="Maximum size for processing. Comma-separated list, one per channel, or single value for all.")
    # Select which ilastik label values to keep (default: all non-zero). Will also write per-class files with suffix _class{v}.tif
    parser.add_argument("--keep-labels", type=rp.split_comma_separated_intstring, help="Comma-separated ilastik label values to keep (e.g. 2,3). Default: keep all non-zero; note 1 is often background.")
    parser.add_argument("--remove-xy-edges", action="store_true", help="Remove edges in XY")
    parser.add_argument("--remove-z-edges", action="store_true", help="Remove edges in Z")
    parser.add_argument("--max-z-gap", type=int, default=-1, help="Fill Z gaps up to this many slices (e.g., 1 or 2). Default -1 (disabled). Fills gaps where pixels have matching labels at z-gap and z+gap.")
    parser.add_argument("--max-time-gap", type=int, default=-1, help="Fill temporal gaps up to this many frames (e.g., 1 or 2). Default -1 (disabled). Fills gaps where pixels have labels at t-gap and t+gap.")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--output-folder", type=str, help="Output folder for processed files")
    parser.add_argument("--output-scale", type=float, default=1, help="Over or under-sampling factor for the output image. Default is 1 (no scaling).")


    args = parser.parse_args()

    # Process the folder
    process_folder(args)
