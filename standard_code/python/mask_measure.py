import argparse
import os
import numpy as np
import pandas as pd
import shutil
import re
from tqdm import tqdm
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict
from skimage.measure import regionprops
from multiprocessing import Pool


# local imports
import run_pipeline_helper_functions as rp


# def process_folder( image_folder:str,
#                     image_suffix:str,
#                     mask_folder:str,
#                     mask_suffix:str,
#                     mask_names_list:List[str],
#                     output_path:str) -> None:
        
#     image_path_to_process = rp.get_files_to_process(image_folder, image_suffix, search_subfolders=False)


#     mask_paths_to_process: List[List[str]] = []
#     for image_path in image_path_to_process:
#         for ma

from skimage.measure import regionprops

def measure(basename:str, img_data: np.ndarray, mask_data: np.ndarray, mask_name:str, img_dim:List[int], mask_dim:List[int]) -> pd.DataFrame:
    # Measure indexed mask
    object_ids = np.unique(mask_data)
    object_ids = object_ids[object_ids > 0]

    # Initialize an empty list to hold the measurements
    measurements = []

    # For each id in object mask
    for obj_id in object_ids:
        # Create a mask for the current object
        object_mask = (mask_data == obj_id).astype(np.uint8)

        # Calculate basic properties using skimage's regionprops
        properties = regionprops(object_mask, intensity_image=img_data)

        # Assuming there will be only one region per object_id
        if properties:
            prop = properties[0]
            # Binary mask metrics
            area_pixels = prop.area
            perimeter = prop.perimeter
            convex_hull_area = prop.convex_area
            centroid = prop.centroid
            orientation = prop.orientation
            extent = prop.extent
            solidity = area_pixels / convex_hull_area if convex_hull_area > 0 else 0
            
            # Intensity-based metrics
            mean_intensity = prop.mean_intensity
            min_intensity = prop.min_intensity
            max_intensity = prop.max_intensity
            sd_intensity = prop.std_intensity
            #median_intensity = prop.median_intensity
            
            # Roundness and Circularity calculations
            roundness = (4 * np.pi * area_pixels) / (perimeter ** 2) if perimeter != 0 else 0
            circularity = (4 * np.pi * area_pixels) / (prop.equivalent_diameter ** 2) if prop.equivalent_diameter != 0 else 0
            
            # Get eigenvalues/eigenvectors using coordinates
            coords = prop.coords
            if len(coords) > 0:
                cov_matrix = np.cov(coords, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            else:
                eigenvalues, eigenvectors = np.array([]), np.array([])

            # Append the measurements for this object to the list
            measurements.append({
                'File name': basename,
                'Mask suffix': mask_name,
                'Image dim T': int(img_dim[0]),
                'Image dim C': int(img_dim[1]),
                'Image dim Z': int(img_dim[2]),
                'Mask dim T': int(mask_dim[0]),
                'Mask dim C': int(mask_dim[1]),
                'Mask dim Z':int(mask_dim[2]),
                'Object ID': int(obj_id),
                'Area (Pixels)': int(area_pixels),
                'Perimeter': perimeter,
                'Centroid X' : centroid[1],
                'Centroid Y' : centroid[0],
                'Roundness': roundness,
                'Circularity': circularity,
                'Slodity': solidity,
                'Extent': extent,
                'Mean Intensity': mean_intensity,
                'Min Intensity': min_intensity,
                'Max Intensity': max_intensity,
                'Standard Deviation Intensity': sd_intensity,
                #'Median Intensity': median_intensity,
                'Orientation': orientation,
                # Eigenvalues
                'Eigenvalue 1': eigenvalues[0] if len(eigenvalues) > 0 else None,
                'Eigenvalue 2': eigenvalues[1] if len(eigenvalues) > 1 else None,        # Eigenvectors
                'Eigenvector 1 X': eigenvectors[0, 0] if eigenvectors.shape[0] > 0 else None,
                'Eigenvector 1 Y': eigenvectors[0, 1] if eigenvectors.shape[0] > 0 else None,
                'Eigenvector 2 X': eigenvectors[1, 0] if eigenvectors.shape[0] > 1 else None,
                'Eigenvector 2 Y': eigenvectors[1, 1] if eigenvectors.shape[0] > 1 else None,


            })

    # Convert list of measurements to a pandas DataFrame
    measurements_df = pd.DataFrame(measurements)

    return measurements_df

def process_file(arguments: Tuple[str, str, List[str]]):
    image_name, base_name, masks = arguments
    print(f"Processing {image_name}")
    img = rp.load_bioio(image_name)
    
    results = []

    for t in range(img.dims.T):
        for c in range(img.dims.C):
            for z in range(img.dims.Z):
                

                for mask_path in masks:
                    mask = rp.load_bioio(mask_path)
                    mask_suffix = os.path.splitext(os.path.basename(mask_path.replace(base_name, "")))[0]
                    mask_suffix = mask_suffix.replace("_", "")

                    # Determine dimensions for masks
                    mask_T = [0] * mask.dims.T if mask.dims.T == 1 else range(mask.dims.T)
                    mask_C = [0] * mask.dims.C if mask.dims.C == 1 else range(mask.dims.C)
                    mask_Z = [0] * mask.dims.Z if mask.dims.Z == 1 else range(mask.dims.Z)
                    
                    for mt in mask_T:
                        for mc in mask_C:
                            for mz in mask_Z:
                                img_data = img.data[t, c, z]
                                mask_data = mask.data[mt, mc, mz]
                                output:pd.DataFrame = measure(
                                    basename=base_name,
                                    img_data=img_data,
                                    mask_data=mask_data,
                                    mask_name=mask_suffix,
                                    img_dim=[t, c, z],
                                    mask_dim=[mt, mc, mz]
                                )
                                results.append(output)

    # Concatenate all DataFrames in the results list into a single DataFrame
    if results:
        final_results_df = pd.concat(results, ignore_index=True)
        return base_name, final_results_df  # return the base_name and results DataFrame
    else:
        return base_name, pd.DataFrame()  # return empty DataFrame if no results

def process_folder(image_folder: str,
                   image_suffix: str,
                   mask_folders: List[str],
                   mask_suffixes: List[str],
                   output_path: str) -> None:
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Get the files to process in the image folder
    files_to_process = rp.get_files_to_process(image_folder, image_suffix, False)
    
    # Retrieve masks to process for each mask folder and suffix
    masks_to_processes: List[List[str]] = [rp.get_files_to_process(f, s, False) for f, s in zip(mask_folders, mask_suffixes)]
    
    matched_files: Dict[str, List[str]] = {}

    # Populate matched_files with base names from the input images
    for input_file_path in files_to_process:
        base_name = re.sub(re.escape(image_suffix) + r"$", "", os.path.basename(input_file_path))
        if base_name not in matched_files:
            matched_files[base_name] = []

    # Match mask files with corresponding base names
    for i, mask_list in enumerate(masks_to_processes):
        for mask_path in mask_list:
            mask_base_name = re.sub(re.escape(mask_suffixes[i]) + r"$", "", os.path.basename(mask_path))
            if mask_base_name in matched_files:
                matched_files[mask_base_name].append(mask_path)
            else:
                print(f"Warning: Could not find base name '{mask_base_name}' in input folder for mask '{mask_path}'")

    # Prepare arguments for parallel processing
    process_args = [
        (os.path.join(image_folder, f"{base_name}{image_suffix}"), base_name, masks)
        for base_name, masks in matched_files.items()
    ]

    # Use multiprocessing to process files concurrently
    with Pool() as pool:
        results = pool.map(process_file, process_args)

    # Save results to Excel files
    for base_name, df in results:
        if not df.empty:
            output_file_path = os.path.join(output_path, f"{base_name}_results.xlsx")
            # Ensure the parent directory for the output file exists
            output_dir = os.path.dirname(output_file_path)
            os.makedirs(output_dir, exist_ok=True)
            
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
            
            df.to_excel(output_file_path, index=False)
            print(f"Saved: {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, help="Path to the input image folder (E.g. ./input_tif)")
    parser.add_argument("--image-suffix", type=str, help="The extension of the input images, e.g. --image-suffix .tif")
    parser.add_argument("--mask-folders", nargs='+', type=str, help="List of input mask folder paths, e.g. --input-masks /path/to/mask1 /path/to/mask2")
    parser.add_argument("--mask-suffixes", type=rp.split_comma_separated_strstring, help="Suffix to copy from input masks, must be provided for each input mask, e.g. --mask-suffixes _mask.tif, or if several mask folders --mask-suffixes _mask.tif _cp_mask.tif")
    
    #parser.add_argument("--mask-names", type=rp.split_comma_separated_strstring, nargs='+', help="Names of the masks to be processed, e.g. --mask-names mask1 mask2, or of several ch in tif --mask-names protein1,protein2 cytoplasm,nucleus")
    parser.add_argument("--output-folder", type=str, help="Path to the output folder where processed images will be saved")

    parsed_args: argparse.Namespace = parser.parse_args()


    # Validate and process the input arguments

    # Image folder and suffix
    if not parsed_args.image_folder:
        raise ValueError("Input image folder are required. Please provide one path to images.")
    
    
    if not parsed_args.image_suffix:
        raise ValueError("Input image folder are required. Please provide one path to images.")

    # Mask folders, suffixes and names
    if not parsed_args.mask_folders:
        raise ValueError("Input masks are required. Please provide at least one mask path.")
    else:
        mask_folders: List[str] = parsed_args.mask_folders
    
    if not parsed_args.mask_suffixes:
        raise ValueError("Input masks suffixes are required. Please provide at least one suffix for each input mask.")
    elif len(parsed_args.mask_suffixes) != len(parsed_args.mask_folders):
        raise ValueError("Input masks suffixes must be provided for each input mask. Please provide valid suffixes.")
    else:
        mask_suffixes: List[str] = parsed_args.mask_suffixes   
    
    # if not parsed_args.mask_names:
    #     raise ValueError("Input masks names are required. Please provide at least one name for each input mask.")
    # elif len(parsed_args.mask_names) != len(parsed_args.mask_folders):
    #     raise ValueError("Input masks names must be provided for each input mask. Please provide valid names.")
    # else:
    #     mask_names: List[List[str]] = parsed_args.mask_names # List of lists, First level is per mask folder second level is per channel in the image in the mask folder
    
    # Check if the output folder is provided
    if not parsed_args.output_folder:
        raise ValueError("Output folder is required. Please provide a valid output folder path.")

    # Check if the input masks suffixes are provided and are of the same length as input masks
    
    
    process_folder(
                    image_folder=parsed_args.image_folder,
                    image_suffix=parsed_args.image_suffix,
                    mask_folders=mask_folders,
                    mask_suffixes=mask_suffixes,  
                    #mask_names=mask_names,
                    output_path=parsed_args.output_folder
                    )
    

