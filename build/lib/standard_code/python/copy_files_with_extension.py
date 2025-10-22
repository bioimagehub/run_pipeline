import argparse
import os
import numpy as np
import shutil

from tqdm import tqdm
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict



# local imports
import bioimage_pipeline_utils as rp




def process_folders(input_folders: List[str], suffixes: List[List[str]], output_path: str) -> None:
    # Ensure the output folder exists
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize a list to hold mask paths for later use
    mask_paths: List[List[str]] = []

    # Iterate over each mask folder and its associated suffixes
    for mask_folder, mask_suffixes in zip(input_folders, suffixes):
        current_mask_files: List[str] = []

        # For each suffix, find matching mask files
        for suffix in mask_suffixes:
            pattern = os.path.join(mask_folder, f"*{suffix}")
            mask_files: List[str] = rp.get_files_to_process2(pattern, search_subfolders=False)
            current_mask_files.extend(mask_files)

            # Copy each found mask file to the output folder
            for mask_file in mask_files:
                mask_file_name: str = os.path.basename(mask_file)
                output_mask_path: str = os.path.join(output_path, mask_file_name)

                # Ensure that the output file does not already exist
                if os.path.exists(output_mask_path):
                    # Give the file a new name by appending a number to the end of the file name
                    base_name, ext = os.path.splitext(mask_file_name)
                    counter = 1
                    while os.path.exists(output_mask_path):
                        output_mask_path = os.path.join(output_path, f"{base_name}_{counter}{ext}")
                        counter += 1

                # Copy the mask file to the output folder
                shutil.copyfile(mask_file, output_mask_path) # Assuming you have a function to copy files

        # Append the current list of mask files to the overall list
        mask_paths.append(current_mask_files)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folders", nargs='+', type=str, help="")
    parser.add_argument("--suffixes", type=rp.split_comma_separated_strstring, nargs='+', help="Suffixes to copy from input folders, e.g. --suffixes _mask1 _mask2, or of several from the same folder --suffixes _mask1,_mask2 _mask3,mask4")
    parser.add_argument("--output-folder", type=str, help="Path to the output folder where processed images will be saved")

    parsed_args: argparse.Namespace = parser.parse_args()

    print(parsed_args.suffixes)

    # Check if the input masks are provided
    if not parsed_args.input_folders:
        raise ValueError("Input masks are required. Please provide at least one mask path.")
    else:
        input_folders: List[str] = parsed_args.input_folders
    
    # Check if the output folder is provided
    if not parsed_args.output_folder:
        raise ValueError("Output folder is required. Please provide a valid output folder path.")
    else:
        output_folder: str = parsed_args.output_folder
    # Check if the input masks suffixes are provided and are of the same length as input masks
    
    
    if not parsed_args.suffixes:
        raise ValueError("Input masks suffixes are required. Please provide at least one suffix for each input mask.")
    elif len(parsed_args.suffixes) != len(parsed_args.input_folders):
        raise ValueError("Input masks suffixes must be provided for each input mask. Please provide valid suffixes.")
    else: 
        suffixes: List[List[str]] = parsed_args.suffixes       


    print(f"Input folder: {input_folders}")
    print(f"Input masks suffixes: {suffixes}")
    print(f"Output folder: {output_folder}")

    process_folders(input_folders=input_folders,
                    suffixes=suffixes,
                    output_path=output_folder)

