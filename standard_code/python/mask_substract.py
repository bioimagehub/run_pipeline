import argparse
import numpy as np
import multiprocessing
from typing import List, Dict, Tuple
import run_pipeline_helper_functions as rp
from bioio.writers import OmeTiffWriter
import re
import os


def process_file(file_paths: List[str], channels: List[int], output_path: str, output_suffix: str = "_subtracted.tif") -> None:
    """
    Subtracts two masks and saves the result to the output path.
    
    Args:
        file_paths (List[str]): List of paths to the masks.
        output_path (str): Path where the resulting mask will be saved.
        output_suffix (str): Suffix for the output file name.
    """
    # print(file_paths)
    
    # Load data
    mask1_img = rp.load_bioio(file_paths[0])
    physical_pixel_sizes = mask1_img.physical_pixel_sizes if mask1_img.physical_pixel_sizes is not None else (None, None, None)
    mask1: np.ndarray = mask1_img.data[:, channels[0], :, :, :]  # TCZYX
    mask2: np.ndarray = rp.load_bioio(file_paths[1]).data[:, channels[1], :, :, :]

    # Ensure both masks have the same shape
    if mask1.shape != mask2.shape:
        raise ValueError("Masks must have the same shape for subtraction.")
    

    # Create the resulting mask: set mask1 pixels to 0 where mask2 is non-zero
    result_mask = np.where(mask2 > 0, 0, mask1)  # Replace mask1 pixels with 0 where mask2 > 0

    # Generate the output file path with the base name and output_suffix
    base_name = re.sub(r'(_cp_masks|_mask)\.tif$', '', os.path.basename(file_paths[0]))
    output_tif_file_path = os.path.join(output_path, f"{base_name}{output_suffix}")
    
    # Save the resulting mask
    OmeTiffWriter.save(result_mask, output_tif_file_path, dim_order="TZYX", physical_pixel_sizes=physical_pixel_sizes)

def process_base_name(args):
    """Helper function to process individual base names for parallel processing."""
    base_name, items, output_path, output_suffix = args
    file_paths = []
    channels = []
    
    # Loop through each tuple in items to unpack the file paths and channels
    for file_path, channel in items:
        file_paths.append(file_path)
        channels.append(channel)

    # print(f"Processing Base name: '{base_name}': file paths: {file_paths}, channels: {channels}")
    # Process the files
    process_file(file_paths, channels, output_path=output_path, output_suffix=output_suffix)


def process_folder(mask_paths: List[str], mask_suffixes: List[str], mask_channels: List[int], output_path: str, output_suffix: str = "_subtracted.tif", parallel: bool = True):
    """
    mask_paths: List of two paths to binary or indexed masks
    mask_suffixes: list of two mask path suffixes (matches mask_paths) files should have a common basename but may vary on the suffix. E.G.  ["_cp_masks.tif", "_mask.tif"]
    mask_channels: The channel in the mask where the subtraction should happen
    output_path: Output path for processed files
    output_suffix: Suffix for output files
    """
    
    mask1_files: List[str] = rp.get_files_to_process(mask_paths[0], mask_suffixes[0], search_subfolders=False)
    mask2_files: List[str] = rp.get_files_to_process(mask_paths[1], mask_suffixes[1], search_subfolders=False)

    # Create a dictionary to store matches
    matched_files: Dict[str, List[Tuple[str, int]]] = {}

    # Process mask1_files and make entries in the dictionary
    for mpf in mask1_files:
        base_name = re.sub(re.escape(mask_suffixes[0]) + r"$", "", os.path.basename(mpf))
        if base_name not in matched_files:
            matched_files[base_name] = []  # Initialize with an empty list
        matched_files[base_name].append((mpf, mask_channels[0]))  # Append tuple (file path, mask channel)
    
    # Process mask2_files and make entries in the dictionary
    for mpf in mask2_files:
        base_name = re.sub(re.escape(mask_suffixes[1]) + r"$", "", os.path.basename(mpf))
        if base_name not in matched_files:
            matched_files[base_name] = []  # Initialize with an empty list
        matched_files[base_name].append((mpf, mask_channels[1]))  # Append tuple (file path, mask channel)
    
    # Now matched_files contains all the files grouped by their base names
    # print("Matched files:", matched_files)

    # Prepare arguments for parallel processing
    jobs = [(base_name, items, output_path, output_suffix) for base_name, items in matched_files.items()]

    if parallel:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.map(process_base_name, jobs)
    else:
        for job in jobs:
            process_base_name(job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-folders", nargs=2, type=str, help="list of paths to mask1 and mask2. E.g. --mask-folders /path/to/mask1 /path/to/mask2")
    parser.add_argument("--mask-suffixes",type=rp.split_comma_separated_strstring, help="comma separated list of suffixes for the masks. E.g. --mask-suffixes,_cp_mask.tif _mask.tif")
    parser.add_argument("--mask-channels", type=rp.split_comma_separated_intstring, help="comma separated list of suffixes for the masks. E.g. --mask-suffixes,_cp_mask.tif _mask.tif")

    parser.add_argument("--output-folder", type=str, help="Path to the output folder where processed images will be saved")
    parser.add_argument("--output-suffix", type=str, help="Suffix for the output file E.g. --output-suffix _cyt.tif")
    parser.add_argument("--no-parallel", action="store_true", help="Do not use parallel processing")
    
    parsed_args: argparse.Namespace = parser.parse_args()
       
    
    #mask_paths = [r"Z:\Schink\Oyvind\colaboration_user_data\20250124_Viola\output_summary", r"Z:\Schink\Oyvind\colaboration_user_data\20250124_Viola\output_summary"]
    # mask_suffixes: List[str] = ["_cp_masks.tif", "_mask.tif"]
    # mask_channels: List[int] = [0, 3]
    # output_folder = r"Z:\Schink\Oyvind\colaboration_user_data\20250124_Viola\output_summary"
    parallel = parsed_args.no_parallel == False # inverse

    # print(parsed_args)

    process_folder(mask_paths=parsed_args.mask_folders, mask_suffixes=parsed_args.mask_suffixes, output_path=parsed_args.output_folder, output_suffix=parsed_args.output_suffix, mask_channels=parsed_args.mask_channels, parallel=parallel)
