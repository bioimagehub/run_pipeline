import subprocess
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import numpy as np
from bioio.writers import OmeTiffWriter
from skimage.measure import label
from scipy.ndimage import binary_fill_holes
import logging
from skimage.transform import resize

from segment_threshold import LabelInfo, remove_small_or_large_labels, remove_on_edges, fill_holes_indexed
from track_indexed_mask import track_labels_with_trackpy
import run_pipeline_helper_functions as rp
from skimage.filters import median
from skimage.morphology import disk

import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def process_file(args, input_file):
    print(f"Processing file: {input_file}")
    # Define output file paths
    out_np_file = os.path.join(args.input_folder, os.path.splitext(os.path.basename(input_file))[0] + "_segmentation.np")
    output_tif_file_path = os.path.join(args.output_folder, os.path.basename(os.path.splitext(out_np_file)[0]) + ".tif")

    subprocess.run([
        args.ilastik_path,
        '--headless',
        f'--project="{args.project_path}"',
        '--export_source=simple segmentation',
        f'--raw_data="{input_file}"',
        f'--output_filename_format="{out_np_file}"'
    ], stdout=subprocess.DEVNULL)


    f = h5py.File(os.path.splitext(out_np_file)[0] + ".h5", 'r')
    group_key = list(f.keys())[0]
    data = np.array(f[group_key]) # TZYXC

    # Rearrange to TCZYX
    if data.ndim == 5:
        data_tczyx = np.transpose(data, (0, 4, 1, 2, 3))
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}, expected 5D (T, Z, Y, X, C)")

    # C should always be 1 for segmentation if not contunue to next step
    if data_tczyx.shape[1] != 1:
        print(f"Unexpected number of channels: {data_tczyx.shape[1]}, expected 1 for segmentation")
        return
    

    unique_values = np.unique(data_tczyx)
    unique_values = unique_values[unique_values > 1]  # Exclude background (0)

    # return if no unique values found
    if len(unique_values) == 0:
        print(f"No unique values found in {input_file}.")
        OmeTiffWriter.save(np.zeros_like(data_tczyx), output_tif_file_path, dim_order="TCZYX")
        return

    # make a dataset like data_tczyx byt with the len(unique_values) channels
    output_data = np.zeros((data_tczyx.shape[0], len(unique_values), data_tczyx.shape[2], data_tczyx.shape[3], data_tczyx.shape[4]), dtype=np.uint8)

    # loop over unique values and set the corresponding channel to 255 where the value is value
    for i, value in enumerate(unique_values):
        output_data[:, i, :, :, :] = (data_tczyx[:,0,:,:,:] == value).astype(np.uint8) * 255  # Convert boolean to uint8
        
        # ADD a unique index to each connected component
        for t in range(output_data.shape[0]):
            output_data[t, i, :, :, :] = label(output_data[t, i, :, :, :], connectivity=1)  # Label connected components

    output_data = fill_holes_indexed(output_data)
    if data_tczyx.shape[1] != 1:
        logging.warning(f"Unexpected number of channels: {data_tczyx.shape[1]}, expected 1 for segmentation")
        return
    
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
        OmeTiffWriter.save(np.zeros_like(data_tczyx), output_tif_file_path, dim_order="TCZYX")
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


    # --- Fill holes using scipy.ndimage.binary_fill_holes ---
    for t in range(output_data.shape[0]):
        for c in range(output_data.shape[1]):
            for z in range(output_data.shape[2]):
                # Fill holes for each label separately
                labels = np.unique(output_data[t, c, z])
                for lbl in labels:
                    if lbl == 0:
                        continue
                    mask = output_data[t, c, z] == lbl
                    filled = binary_fill_holes(mask)
                    output_data[t, c, z][mask] = 0  # Clear old label
                    output_data[t, c, z][filled] = lbl  # Set filled label

    # return if no labels left after edge removal
    if not label_info_list:
        print(f"No labels left after edge removal in {input_file}. Skipping tracking.")
    else:
        try:
            _, output_data = track_labels_with_trackpy(output_data)
        except Exception as e:
            print(f"Tracking skipped for file {input_file}: {e}")



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
    OmeTiffWriter.save(output_data, output_tif_file_path, dim_order="TCZYX")

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
        "Output_format": "OME-TIFF",
        "Output_dim_order": "TCZYX",
        "Tracking": not (not label_info_list),
        "Hole_filling": "scipy.ndimage.binary_fill_holes",
        "Edge_smoothing": "median filter (disk(3))"
    }
    with open(output_metadata_file_path, 'w') as f:
        yaml.dump(metadata, f)

def process_folder(args: argparse.Namespace) -> None:
    # Find files to process using glob pattern
    if hasattr(args, 'input_search_pattern') and args.input_search_pattern:
        pattern = args.input_search_pattern
    else:
        # Backward compatibility: construct pattern from folder + suffix
        pattern = os.path.join(args.input_folder, f"*{args.input_suffix}")
    files_to_process = rp.get_files_to_process2(pattern, False)

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)  
    
    if args.no_parallel:
        for input_file in files_to_process:
            process_file(args, input_file)
    else:
        with ProcessPoolExecutor() as executor:
            future_to_file = {executor.submit(process_file, args, input_file): input_file for input_file in files_to_process}
            for future in as_completed(future_to_file):
                input_file = future_to_file[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f'File {input_file} generated an exception: {exc}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BioImage files.")
    parser.add_argument("--ilastik-path", type=str, required=True, help="Path to the ilastik executable")
    parser.add_argument("--input-search-pattern", type=str, required=False, help="Glob pattern for input images, e.g. 'folder/*.tif'")
    parser.add_argument("--input-folder", type=str, required=False, help="Deprecated: Path to input folder (use --input-search-pattern)")
    parser.add_argument("--input-suffix", type=str, required=False, help="Deprecated: File ending e.g. .tif (use --input-search-pattern)")
    parser.add_argument("--project-path", type=str, required=True, help="Path to already trained ilp project")
    parser.add_argument("--min-sizes", type=rp.split_comma_separated_intstring, default=[0], help="Minimum size for processing. Comma-separated list, one per channel, or single value for all.")
    parser.add_argument("--max-sizes", type=rp.split_comma_separated_intstring, default=[99999999], help="Maximum size for processing. Comma-separated list, one per channel, or single value for all.")
    parser.add_argument("--remove-xy-edges", action="store_true", help="Remove edges in XY")
    parser.add_argument("--remove-z-edges", action="store_true", help="Remove edges in Z")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--output-folder", type=str, help="Output folder for processed files")
    parser.add_argument("--output-scale", type=float, default=1, help="Over or under-sampling factor for the output image. Default is 1 (no scaling).")


    args = parser.parse_args()

    # Process the folder
    process_folder(args)
