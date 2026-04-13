import os
import argparse
import logging
from tqdm import tqdm
import numpy as np
from scipy.ndimage import distance_transform_edt, label
from bioio.writers import OmeTiffWriter
from concurrent.futures import ProcessPoolExecutor, as_completed
import tifffile


import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)



def process_file(mask_path: str, output_folder_path: str, output_suffix: str) -> None:
    output_filename = os.path.basename(
        rp.resolve_output_path(mask_path, extension='.tif', suffix=output_suffix)
    )
    
    mask = rp.load_tczyx_image(mask_path)  # TCZYX
    # Check if mask is correctly loaded and has valid data
    if mask is None or mask.data is None:
        logger.error(f"Mask data is None for file {mask_path}. Skipping this file.")
        return

    # Image dimensions
    try:
        t, c, z = mask.dims.T, mask.dims.C, mask.dims.Z
    except AttributeError:
        logger.error(f"Invalid dimensions in mask for file {mask_path}. Skipping this file.")
        return

    # Get metadata
    try:
        physical_pixel_sizes = mask.physical_pixel_sizes if mask.physical_pixel_sizes is not None else (None, None, None)
    except Exception as e:
        logger.error(f"Error retrieving physical pixel sizes: {e} for file {mask_path}. Using None.")
        physical_pixel_sizes = (None, None, None)

    # Prepare to store overall distance matrix
    overall_distance_matrix = np.zeros_like(mask.data)

    # Compute distances
    for mask_frame in range(t):
        for mask_channel in range(c):
            for mask_zslice in range(z):
                # Extract the 2D mask for the current frame, channel, and z-slice
                mask_2d = mask.data[mask_frame, mask_channel, mask_zslice, :, :]

                # Create binary mask (any non-zero value is foreground)
                binary_mask = (mask_2d > 0).astype(np.uint8)
                
                # Label connected components to separate distinct objects
                labeled_mask, num_objects = label(binary_mask)

                # Initialize a 2D distance matrix for the current position
                distance_matrix_for_frame = np.zeros_like(mask_2d, dtype=np.float32)

                # Loop through each separate connected component
                for object_id in range(1, num_objects + 1):
                    # Create binary mask for this specific connected component
                    object_mask = (labeled_mask == object_id).astype(np.uint8)

                    # Calculate the distance to the edge of the object
                    distance_to_edge_mask = distance_transform_edt(object_mask)

                    # Assign the distance to the corresponding pixels in the overall distance matrix
                    if distance_to_edge_mask is not None:
                        distance_matrix_for_frame[object_mask == 1] = distance_to_edge_mask[object_mask == 1]
                    else:
                        logger.warning(f"Distance transform returned None for object {object_id} in file {mask_path}.")

                # Store the distance matrix for the current frame, channel, and z-slice
                overall_distance_matrix[mask_frame, mask_channel, mask_zslice, :, :] = distance_matrix_for_frame

    # Save the overall distance matrix
    output_file_path = os.path.join(output_folder_path, output_filename)
    # rp.save_mask(overall_distance_matrix, output_file_path, as_binary=False)
    rp.save_tczyx_image(
        overall_distance_matrix,
        output_file_path,
        physical_pixel_sizes=physical_pixel_sizes,
        ome_xml=None
    )

  
def process_folder(args: argparse.Namespace):
    files_to_process = rp.get_files_to_process2(args.input_search_pattern, False)
    os.makedirs(args.output_folder, exist_ok=True)

    if args.no_parallel:
        # Sequential processing
        for input_file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
            process_file(mask_path=input_file_path, output_folder_path=args.output_folder, output_suffix=args.output_suffix)
    else:
        # Parallel processing
        max_workers = rp.resolve_maxcores(args.maxcores, len(files_to_process))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for input_file_path in files_to_process:
                future = executor.submit(
                    process_file,
                    mask_path=input_file_path,
                    output_folder_path=args.output_folder,
                    output_suffix=args.output_suffix
                )
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files", unit="file"):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing file: {e}")



if __name__ == "__main__":
        parser = argparse.ArgumentParser(
                description="Process masks and convert indexed masks to distance matrix",
                formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Compute per-object distance maps
    environment: uv@3.11:default
    commands:
    - python
    - '%REPO%/standard_code/python/mask_get_distance_matrix.py'
    - --input-search-pattern: '%YAML%/masks/**/*_mask.tif'
    - --output-folder: '%YAML%/distance_maps'

- name: Compute distance maps sequentially
    environment: uv@3.11:default
    commands:
    - python
    - '%REPO%/standard_code/python/mask_get_distance_matrix.py'
    - --input-search-pattern: '%YAML%/masks/**/*.tif'
    - --output-folder: '%YAML%/distance_maps'
    - --no-parallel
                """
        )
    parser.add_argument("--input-search-pattern", type=str, required=True, help="Glob pattern for input masks, e.g. './output_masks/*_segmentation.tif'")
    parser.add_argument("--output-folder", type=str, help="Path to the output folder.")
    parser.add_argument("--output-suffix", type=str, default="_distance_matrix", help="Suffix appended to output filenames before the extension")
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing (default: parallel enabled)')
    parser.add_argument('--maxcores', type=int, default=None, help='Maximum CPU cores to use for parallel processing (default: all available CPU cores minus 1). Ignored if --no-parallel is set.')
    parser.add_argument('--log-level', type=str, default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level (default: WARNING)')

    parsed_args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, parsed_args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if not parsed_args.output_folder:
        parsed_args.output_folder = os.path.dirname(parsed_args.input_search_pattern) or "."

    process_folder(parsed_args)





