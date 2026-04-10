import os
import argparse
import logging
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from scipy.ndimage import distance_transform_edt
from bioio.writers import OmeTiffWriter

import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)

def process_file(mask_path: str, output_folder_path: str, distance_inside: int, distance_outside: int, output_suffix: str) -> None:
    output_file_basename = os.path.join(output_folder_path, os.path.splitext(os.path.basename(mask_path))[0])
    
    mask = rp.load_tczyx_image(mask_path)  # Load the mask
    
    if mask is None or mask.data is None:
        logger.error(f"Mask data is None for file {mask_path}. Skipping this file.")
        return

    # Image dimensions
    try:
        t, c, z = mask.dims.T, mask.dims.C, mask.dims.Z
    except AttributeError:
        logger.error(f"Invalid dimensions in mask for file {mask_path}. Skipping this file.")
        return

    # Metadata
    try:
        physical_pixel_sizes = mask.physical_pixel_sizes if mask.physical_pixel_sizes is not None else (None, None, None)
    except Exception as e:
        logger.error(f"Error retrieving physical pixel sizes: {e} for file {mask_path}. Using None.")
        physical_pixel_sizes = (None, None, None)

    # Prepare to store the indexed mask with edges defined
    indexed_mask = np.zeros_like(mask.data, dtype=np.int8)

    # Compute edges and defined distances
    for mask_frame in range(t):
        for mask_channel in range(c):
            for mask_zslice in range(z):
                # Extract the 2D mask for the current frame, channel, and z-slice
                mask_2d = mask.data[mask_frame, mask_channel, mask_zslice, :, :]

                objects = np.unique(mask_2d)
                objects = objects[objects > 0]  # Exclude background (0)

                for object_id in objects:
                    # Create a mask for the current object
                    object_mask = (mask_2d == object_id).astype(np.uint8)

                    # Calculate the distance to the edge of the object
                    result = distance_transform_edt(object_mask == object_id)

                    if isinstance(result, np.ndarray):
                        distance_to_outside = result
                    else:
                        logger.warning("distance_transform_edt returned unexpected type. Skipping this object.")
                        continue
                        
                    result = distance_transform_edt(object_mask == 0)

                    if isinstance(result, np.ndarray):
                        distance_to_inside = result
                    else:
                        logger.warning("distance_transform_edt returned unexpected type. Skipping this object.")
                        continue

                    global_distance_mask = distance_to_outside - distance_to_inside
                    
                    # Set the min and max values for the distance mas
                    min_val, max_val = 0, 0
                    if (distance_inside > 0):
                        max_val = distance_inside
                    if (distance_outside > 0):
                        min_val = 0 - distance_outside
                    
                    # set values outside the range to 0 and inside the range to the object id
                    
                    global_distance_mask[(global_distance_mask < min_val) | (global_distance_mask > max_val)] = 0
                    global_distance_mask[global_distance_mask !=0 ] = object_id # All neg and pos values are set to the object id
                                        
                    # Set the index for the defined edges
                    indexed_mask[mask_frame, mask_channel, mask_zslice, :, :] = global_distance_mask

    # Save the indexed mask
    output_file_path = f"{output_file_basename}{output_suffix}.tif"
    rp.save_tczyx_image(indexed_mask, output_file_path, dim_order="TCZYX", physical_pixel_sizes=physical_pixel_sizes)
  
def process_folder(args: argparse.Namespace):
    files_to_process = rp.get_files_to_process2(args.input_search_pattern, args.search_subfolders)
    os.makedirs(args.output_folder, exist_ok=True)

    for input_file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
        process_file(mask_path = input_file_path, output_folder_path = args.output_folder, distance_inside=args.distance_inside, distance_outside=args.distance_outside, output_suffix=args.output_suffix)

if __name__ == "__main__":
        parser = argparse.ArgumentParser(
                description="Process masks and convert indexed masks to defined edge regions",
                formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Extract inner and outer mask edge bands
    environment: uv@3.11:default
    commands:
    - python
    - '%REPO%/standard_code/python/mask_get_edges.py'
    - --input-search-pattern: '%YAML%/masks/**/*_mask.tif'
    - --output-folder: '%YAML%/edge_masks'

- name: Extract thin outer edge only
    environment: uv@3.11:default
    commands:
    - python
    - '%REPO%/standard_code/python/mask_get_edges.py'
    - --input-search-pattern: '%YAML%/masks/**/*_mask.tif'
    - --output-folder: '%YAML%/edge_masks'
    - --distance-inside: 0
    - --distance-outside: 2
                """
        )
    parser.add_argument("--input-search-pattern", type=str, required=True, help="Glob pattern for input masks, e.g. './output_masks/*_segmentation.tif'")
    parser.add_argument("--search-subfolders", action="store_true", help="Enable recursive search (only if pattern doesn't already include '**')")
    parser.add_argument("--distance-inside", type=int, default= 3, help="How far inside the object should the mask be extended? (default: 3 pixels)")
    parser.add_argument("--distance-outside", type=int, default= 3, help="How far outside the object should the mask be extended? (default: 3 pixels)")
    parser.add_argument("--output-folder", type=str, help="Path to the output folder.")
    parser.add_argument("--output-suffix", type=str, default="_edge", help="Suffix appended to output filenames before the extension")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing (currently unused; processing is sequential).")
    parser.add_argument("--maxcores", type=int, default=None, help="Maximum CPU cores to use for parallel processing (currently unused; processing is sequential). Ignored if --no-parallel is set.")
    parser.add_argument("--log-level", type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level (default: WARNING)")
    parsed_args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, parsed_args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if not parsed_args.output_folder:
        parsed_args.output_folder = os.path.dirname(parsed_args.input_search_pattern) or "."
        
    process_folder(parsed_args)
