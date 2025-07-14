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

def measure(
    basename: str,
    img_data: np.ndarray,
    mask_data: np.ndarray,
    mask_name: str,
    img_dim: List[int],
    mask_dim: List[int],
) -> pd.DataFrame:
    measurements = []
    z_planes = range(mask_data.shape[0]) if mask_data.ndim == 3 else [0]

    for z in z_planes:
        img_slice = img_data[z] if img_data.ndim == 3 else img_data
        mask_slice = mask_data[z] if mask_data.ndim == 3 else mask_data

        props = regionprops(mask_slice, intensity_image=img_slice)

        for prop in props:
            area_pixels = prop.area
            perimeter = prop.perimeter
            convex_hull_area = prop.convex_area
            centroid = prop.centroid
            orientation = prop.orientation
            extent = prop.extent
            solidity = area_pixels / convex_hull_area if convex_hull_area > 0 else 0
            mean_intensity = prop.mean_intensity
            min_intensity = prop.min_intensity
            max_intensity = prop.max_intensity
            sd_intensity = prop.std_intensity
            roundness = (4 * np.pi * area_pixels) / (perimeter ** 2) if perimeter != 0 else 0
            circularity = (
                (4 * np.pi * area_pixels) / (prop.equivalent_diameter ** 2)
                if prop.equivalent_diameter != 0 else 0
            )

            coords = prop.coords
            if len(coords) > 0:
                cov_matrix = np.cov(coords, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            else:
                eigenvalues, eigenvectors = np.array([]), np.array([])

            measurements.append({
                'File name': basename,
                'Mask suffix': mask_name,
                'Image dim T': int(img_dim[0]),
                'Image dim C': int(img_dim[1]),
                'Image dim Z': int(z),
                'Mask dim T': int(mask_dim[0]),
                'Mask dim C': int(mask_dim[1]),
                'Mask dim Z': int(z),
                'Object ID': int(prop.label),
                'Area (Pixels)': int(area_pixels),
                'Perimeter': perimeter,
                'Centroid X': centroid[1],
                'Centroid Y': centroid[0],
                'Roundness': roundness,
                'Circularity': circularity,
                'Solidity': solidity,
                'Extent': extent,
                'Mean Intensity': mean_intensity,
                'Min Intensity': min_intensity,
                'Max Intensity': max_intensity,
                'Standard Deviation Intensity': sd_intensity,
                'Orientation': orientation,
                'Eigenvalue 1': eigenvalues[0] if len(eigenvalues) > 0 else None,
                'Eigenvalue 2': eigenvalues[1] if len(eigenvalues) > 1 else None,
                'Eigenvector 1 X': eigenvectors[0, 0] if eigenvectors.shape[0] > 0 else None,
                'Eigenvector 1 Y': eigenvectors[0, 1] if eigenvectors.shape[0] > 0 else None,
                'Eigenvector 2 X': eigenvectors[1, 0] if eigenvectors.shape[0] > 1 else None,
                'Eigenvector 2 Y': eigenvectors[1, 1] if eigenvectors.shape[0] > 1 else None,
            })
        #print(measurements)

    return pd.DataFrame(measurements)


def process_file(arguments: Tuple[str, str, List[str]]) -> Tuple[str, pd.DataFrame]:
    image_name, base_name, masks = arguments
    img = rp.load_bioio(image_name)
    if img is None:
        print(f"Warning: Could not load image {image_name}. Skipping.")
        return base_name, pd.DataFrame()

    results = []

    for t in range(img.dims.T):
        for c in range(img.dims.C):
            for z in range(img.dims.Z):
                img_data = img.data[t, c, z]

                for mask_path in masks:
                    mask = rp.load_bioio(mask_path)
                    if mask is None:
                        print(f"Warning: Could not load mask {mask_path}. Skipping.")
                        continue

                    mask_suffix = os.path.splitext(os.path.basename(mask_path.replace(base_name, "")))[0]
                    mask_suffix = mask_suffix.replace("_", "")

                    try:
                        mask_t = t if mask.dims.T == img.dims.T else 0
                        mask_c = c if mask.dims.C == img.dims.C else 0
                        mask_z = z if mask.dims.Z == img.dims.Z else 0
                    except IndexError:
                        print(f"IndexError: Could not access mask plane for {mask_path} at T={t}, C={c}, Z={z}.")
                        continue

                    if any([
                        mask.dims.T not in (1, img.dims.T),
                        mask.dims.C not in (1, img.dims.C),
                        mask.dims.Z not in (1, img.dims.Z),
                    ]):
                        print(f"Warning: Mismatched dimensions between mask and image. Skipping.")
                        continue

                    mask_data = mask.data[mask_t, mask_c, mask_z]

                    output = measure(
                        basename=base_name,
                        img_data=img_data,
                        mask_data=mask_data,
                        mask_name=mask_suffix,
                        img_dim=[t, c, z],
                        mask_dim=[mask_t, mask_c, mask_z],
                    )
                    results.append(output)

    if results:
        final_df = pd.concat(results, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    return base_name, final_df


def process_folder(
    image_folder: str,
    image_suffix: str,
    mask_folders: List[str],
    mask_suffixes: List[str],
    output_path: str,
    output_suffix: str,
    parallel: bool
) -> None:
    """
    Process a folder of images and matching masks, saving results per file.

    Parameters:
    - image_folder: folder with input images
    - image_suffix: suffix to identify images (e.g., '_img.ome.tif')
    - mask_folders: list of folders with masks
    - mask_suffixes: suffixes to identify masks (e.g., ['_seg1.tif', '_seg2.tif'])
    - output_path: folder where results are saved
    - output_suffix: suffix to append to output filenames (before '.xlsx')
    - parallel: whether to use multiprocessing
    """

    os.makedirs(output_path, exist_ok=True)

    # Get input image files
    files_to_process = rp.get_files_to_process(image_folder, image_suffix, search_subfolders=False)

    # Get mask files for each folder/suffix combo
    masks_to_processes: List[List[str]] = [
        rp.get_files_to_process(f, s, search_subfolders=False) for f, s in zip(mask_folders, mask_suffixes)
    ]

    # Initialize match dictionary by base name
    matched_files: Dict[str, List[str]] = {
        re.sub(re.escape(image_suffix) + r"$", "", os.path.basename(img_path)): []
        for img_path in files_to_process
    }

    # Match masks to base names
    for i, mask_list in enumerate(masks_to_processes):
        for mask_path in mask_list:
            mask_base_name = re.sub(re.escape(mask_suffixes[i]) + r"$", "", os.path.basename(mask_path))
            if mask_base_name in matched_files:
                matched_files[mask_base_name].append(mask_path)
            else:
                print(f"Warning: No matching image for mask '{mask_path}' (base: '{mask_base_name}')")

    # Prepare arguments for processing
    process_args = [
        (os.path.join(image_folder, f"{base_name}{image_suffix}"), base_name, masks)
        for base_name, masks in matched_files.items()
    ]

    # Process files
    if not parallel:
        print("Running in sequential mode...")
        results = [process_file(args) for args in tqdm(process_args, desc="Processing")]
    else:
        print("Running in parallel mode...")
        with Pool() as pool:
            results = list(tqdm(pool.imap(process_file, process_args), total=len(process_args), desc="Processing"))

    # Save results to Excel
    for base_name, df in results:
        if not df.empty:
            output_file_path = os.path.join(output_path, f"{base_name}{output_suffix}.xlsx")
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            if os.path.exists(output_file_path):
                os.remove(output_file_path)

            df.to_excel(output_file_path, index=False)
            print(f"Saved: {output_file_path}")
        else:
            print(f"Skipped: {base_name} (empty result)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, help="Path to the input image folder (E.g. ./input_tif)")
    parser.add_argument("--image-suffix", type=str, help="The extension of the input images, e.g. --image-suffix .tif")
    parser.add_argument("--mask-folders", type=rp.split_comma_separated_strstring, help="List of input mask folder paths, e.g. --input-masks /path/to/mask1")
    parser.add_argument("--mask-suffix", type=rp.split_comma_separated_strstring, help="Suffix to copy from input masks, e.g. --mask-suffixes .tif")
    
    #parser.add_argument("--mask-names", type=rp.split_comma_separated_strstring, nargs='+', help="Names of the masks to be processed, e.g. --mask-names mask1 mask2, or of several ch in tif --mask-names protein1,protein2 cytoplasm,nucleus")
    parser.add_argument("--output-folder", type=str, help="Path to the output folder where processed images will be saved")
    parser.add_argument("--output-suffix", type=str, help="suffix to add to the output .xlsx files, e.g. --output-suffix _results.xlsx", default="_results.xlsx")

    parser.add_argument("--no-parallel", action="store_true", help="Do not use parallel processing")

    
    parsed_args: argparse.Namespace = parser.parse_args()
    parallel = parsed_args.no_parallel == False # inverse


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
        mask_folder: List[str] = parsed_args.mask_folders
    
    if not parsed_args.mask_suffix:
        raise ValueError("Input masks suffixes are required. Please provide at least one suffix for each input mask.")
    
    elif len(parsed_args.mask_suffix) != len(parsed_args.mask_folders):
        raise ValueError(f"Input masks suffixes must be provided for each input mask. Please provide valid suffixes.{len(parsed_args.mask_suffix)=}{len(parsed_args.mask_folders)}")
    else:
        mask_suffix: List[str] = parsed_args.mask_suffix   
    
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
                    mask_folders=mask_folder,
                    mask_suffixes=mask_suffix,  
                    #mask_names=mask_names,
                    output_path=parsed_args.output_folder,
                    output_suffix=parsed_args.output_suffix,
                    parallel = parallel
                    )
    

