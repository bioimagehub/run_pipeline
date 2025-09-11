import os
import argparse
from tqdm import tqdm
import numpy as np
from scipy.ndimage import distance_transform_edt
from bioio.writers import OmeTiffWriter


import run_pipeline_helper_functions as rp

def process_file(mask_path: str, output_folder_path: str) -> None:
    output_file_basename = os.path.join(output_folder_path, os.path.splitext(os.path.basename(mask_path))[0])
    
    mask = rp.load_bioio(mask_path)  # TCZYX
    # Check if mask is correctly loaded and has valid data
    if mask is None or mask.data is None:
        print(f"Error: Mask data is None for file {mask_path}. Skipping this file.")
        return

    # Image dimensions
    try:
        t, c, z = mask.dims.T, mask.dims.C, mask.dims.Z
    except AttributeError:
        print(f"Error: Invalid dimensions in mask for file {mask_path}. Skipping this file.")
        return

    # Get metadata
    try:
        physical_pixel_sizes = mask.physical_pixel_sizes if mask.physical_pixel_sizes is not None else (None, None, None)
    except Exception as e:
        print(f"Error retrieving physical pixel sizes: {e} for file {mask_path}. Using None.")
        physical_pixel_sizes = (None, None, None)

    # Prepare to store overall distance matrix
    overall_distance_matrix = np.zeros_like(mask.data)

    # Compute distances
    for mask_frame in range(t):
        for mask_channel in range(c):
            for mask_zslice in range(z):
                # Extract the 2D mask for the current frame, channel, and z-slice
                mask_2d = mask.data[mask_frame, mask_channel, mask_zslice, :, :]

                objects = np.unique(mask_2d)
                objects = objects[objects > 0]  # Exclude background (0)

                # Initialize a 2D distance matrix for the current position
                distance_matrix_for_frame = np.zeros_like(mask_2d, dtype=np.float32)

                # Loop through all objects in the indexed mask
                for object_id in objects:
                    # Create a mask for the current object
                    object_mask = (mask_2d == object_id).astype(np.uint8)

                    # Calculate the distance to the edge of the object
                    distance_to_edge_mask = distance_transform_edt(object_mask == object_id)  # Distance to background (0)

                    # Assign the distance to the corresponding pixels in the overall distance matrix
                    if distance_to_edge_mask is not None:
                        distance_matrix_for_frame[mask_2d == object_id] = distance_to_edge_mask[mask_2d == object_id]
                    else:
                        print(f"Warning: Distance transform returned None for object {object_id} in file {mask_path}.")

                # Store the distance matrix for the current frame, channel, and z-slice
                overall_distance_matrix[mask_frame, mask_channel, mask_zslice, :, :] = distance_matrix_for_frame

    # Save the overall distance matrix
    output_file_path = f"{output_file_basename}_distance_matrix.tif"
    OmeTiffWriter.save(overall_distance_matrix, output_file_path, dim_order="TCZYX", physical_pixel_sizes=physical_pixel_sizes)



  
def process_folder(args: argparse.Namespace):
    files_to_process = rp.get_files_to_process2(args.input_search_pattern, args.search_subfolders)
    os.makedirs(args.output_folder, exist_ok=True)

    for input_file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
        process_file(mask_path=input_file_path, output_folder_path=args.output_folder)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process masks and convert indexed masks to distance matrix")
    parser.add_argument("--input-search-pattern", type=str, required=True, help="Glob pattern for input masks, e.g. './output_masks/*_segmentation.tif'")
    parser.add_argument("--search-subfolders", action="store_true", help="Enable recursive search (only if pattern doesn't already include '**')")
    parser.add_argument("--output-folder", type=str, help="Path to the output folder.")
    parsed_args = parser.parse_args()

    if not parsed_args.output_folder:
        parsed_args.output_folder = os.path.dirname(parsed_args.input_search_pattern) or "."

    process_folder(parsed_args)





