import argparse
import os
import numpy as np
import shutil

from tqdm import tqdm
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict



# local imports
import run_pipeline_helper_functions as rp


def process_folder( image_folder:str,
                    image_suffix:str,
                    mask_folder:str,
                    mask_suffix:str,
                    mask_names_list:List[str],
                    output_path:str) -> None:
        
    image_path_to_process = rp.get_files_to_process(image_folder, image_suffix, search_subfolders=False)


    mask_paths_to_process: List[List[str]] = []
    for image_path in image_path_to_process:
        for ma







def process_folders(image_folder: str ,
                    image_suffix: str,
                    mask_folders: List[str],
                    mask_suffixes: List[str]   ,  
                    mask_names: List[List[str]],
                    output_path:str) -> None:
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_path, exist_ok=True)  
    
    # Iterate over each mask folder and its associated suffixes
    for mask_folder, mask_suffix, mask_name_list in zip(mask_folders, mask_suffixes, mask_names):
        process_folder(image_folder=image_folder,
                image_suffix=image_suffix,
                mask_folder=mask_folder,
                mask_suffix=mask_suffix,
                mask_names_list=mask_name_list,
                output_path=output_path)
        
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, help="Path to the input image folder (E.g. ./input_tif)")
    parser.add_argument("--image-suffix", type=str, help="The extension of the input images, e.g. --image-suffix .tif")
    parser.add_argument("--mask-folders", nargs='+', type=str, help="List of input mask folder paths, e.g. --input-masks /path/to/mask1 /path/to/mask2")
    parser.add_argument("--mask-suffixes", type=str, nargs='+', help="Suffix to copy from input masks, must be provided for each input mask, e.g. --mask-suffixes _mask.tif, or if several mask folders --mask-suffixes _mask.tif _cp_mask.tif")
    parser.add_argument("--mask-names", type=rp.split_comma_separated_strstring, nargs='+', help="Names of the masks to be processed, e.g. --mask-names mask1 mask2, or of several ch in tif --mask-names protein1,protein2 cytoplasm,nucleus")
    
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
    
    if not parsed_args.mask_names:
        raise ValueError("Input masks names are required. Please provide at least one name for each input mask.")
    elif len(parsed_args.mask_names) != len(parsed_args.mask_folders):
        raise ValueError("Input masks names must be provided for each input mask. Please provide valid names.")
    else:
        mask_names: List[List[str]] = parsed_args.mask_names # List of lists, First level is per mask folder second level is per channel in the image in the mask folder
    
    # Check if the output folder is provided
    if not parsed_args.output_folder:
        raise ValueError("Output folder is required. Please provide a valid output folder path.")

    # Check if the input masks suffixes are provided and are of the same length as input masks
    
    
    process_folders(image_folder = parsed_args.image_folder,
                    image_suffix = parsed_args.image_suffix,
                    mask_folders = mask_folders
                    mask_suffixes = mask_suffixes,
                    mask_names = mask_names,
                    output_path = parsed_args.output_folder)

