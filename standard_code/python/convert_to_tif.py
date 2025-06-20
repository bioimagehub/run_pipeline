import numpy as np
from pystackreg import StackReg
from tqdm import tqdm
import argparse
from joblib import Parallel, delayed
import os
import yaml
from bioio.writers import OmeTiffWriter

# Local imports
import run_pipeline_helper_functions as rp
from extract_metadata import get_all_metadata
from bioio import BioImage

def drift_correct_xy_parallel(video: np.ndarray, drift_correct_channel: int = 0) -> tuple[np.ndarray, np.ndarray]:
    T, C, Z, _, _ = video.shape    
    corrected_video = np.zeros_like(video)
    
    sr = StackReg(StackReg.TRANSLATION)

    # Max-projection along Z for drift correction
    ref_stack = np.max(video[:, drift_correct_channel, :, :, :], axis=1)
    
    tmats = sr.register_stack(ref_stack, reference='mean', verbose=False, axis=0) 
    for t in range(T):
        for c in range(C):
            for z in range(Z):
                corrected_video[t, c, z, :, :] = sr.transform(video[t, c, z, :, :], tmats[t])
                    
    return corrected_video, tmats

def process_multipoint_file(input_file_path: str, output_tif_file_path: str, drift_correct_channel: int = -1, use_parallel: bool = True, projection_method: str = None) -> None:
    """
    Process a multipoint file and convert it to TIF format with optional drift correction.
    """
    img = rp.load_bioio(input_file_path)

    for i, scene in enumerate(img.scenes):
        img.set_scene(scene)
        filename_base = os.path.splitext(os.path.basename(input_file_path))[0]
        output_tif_file_path = os.path.join(os.path.dirname(output_tif_file_path), f"{filename_base}_S{i:02}.tif")
        
        # Process the scene
        process_file(img = img,
                input_file_path = input_file_path, 
                output_tif_file_path = output_tif_file_path, 
                drift_correct_channel = parsed_args.drift_correct_channel, 
                projection_method = parsed_args.projection_method, 
                multipoint_files = parsed_args.multipoint_files)
    

def process_file(img:BioImage, input_file_path: str, output_tif_file_path: str, drift_correct_channel: int = -1, projection_method: str = None, multipoint_files:bool= False) -> None:
    input_metadata_file_path: str = os.path.splitext(input_file_path)[0] + "_metadata.yaml"
    output_metadata_file_path: str = os.path.splitext(output_tif_file_path)[0] + "_metadata.yaml"
    output_shifts_file_path: str = os.path.splitext(output_tif_file_path)[0] + "_shifts.npy"

    if os.path.exists(output_metadata_file_path):
        print(f"Metadata file already exists: {output_metadata_file_path}. Skipping metadata extraction.")
        return


    # img = rp.load_bioio(input_file_path) Accepted as argumet to accomodate multipoint files

    
    # img.physical_pixel_sizes can crash even with else statement, so we use a try-except block
    try:
        physical_pixel_sizes = img.physical_pixel_sizes if img.physical_pixel_sizes is not None else (None, None, None)
    except Exception as e:
        print(f"Error retrieving physical pixel sizes: {e} for file {input_file_path}. Using None.")
        
        # save a txt file with the error
        with open(os.path.splitext(output_tif_file_path)[0] + "_error.txt", 'w') as f:
            f.write(f"Error retrieving physical pixel sizes: {e} for file {input_file_path}. Using None.\n")    

        physical_pixel_sizes = (None, None, None)

    if os.path.exists(input_metadata_file_path):
        with open(input_metadata_file_path, 'r') as f:
            metadata = yaml.safe_load(f)
    else:
        try:        
            metadata = get_all_metadata(input_file_path)
        except Exception as e:
            print(f"Error retrieving metadata: {e} for file {input_file_path}. Using None.")
            metadata = {"Error": f"Error retrieving metadata: {e} for file {input_file_path}. Using None."}
            with open(os.path.splitext(output_tif_file_path)[0] + "_metadata_error.txt", 'w') as f:
                f.write(f"Error retrieving physical pixel sizes: {e} for file {input_file_path}. Using None.\n")    
   
        with open(input_metadata_file_path, 'w') as f:
            yaml.dump(metadata, f)
                
    # Perform projection if requested
    try:
        initial_dtype = img.data.dtype
    except Exception as e:
        print(f"Error: The image data does not have a 'dtype' attribute {e}.\n Skipping this file: {input_file_path}")
        return   
    
    if projection_method == "max":
        img_data = np.max(img.data, axis=2, keepdims=True)
    elif projection_method == "sum":
        print("Warning: Using sum projection and with same dtype may cause pixels to saturate.")
        img_data = np.sum(img.data, axis=2, keepdims=True)
        
        # Change to a dtype of max uint16
        if initial_dtype == np.uint8:
            print(f"Number of saturated pixels: {np.sum(img_data > 255)}")
            img_data = img_data.astype(np.uint8)
            
        elif initial_dtype == np.uint16:
            print(f"Number of saturated pixels: {np.sum(img_data > 65535)}")
            img_data = img_data.astype(np.uint16)
        # elif initial_dtype == np.uint32:
        #     print(f"Number of saturated pixels: {np.sum(img_data > 4294967295)}")
        #     img_data = img_data.astype(np.uint32)
        # elif initial_dtype == np.float32:
        #     img_data = img_data.astype(np.float32)
        else:
            raise ValueError(f"Unsupported dtype: {initial_dtype}")

    elif projection_method == "mean":
        img_data = np.mean(img.data, axis=2, keepdims=True)
    elif projection_method == "median":
        img_data = np.median(img.data, axis=2, keepdims=True)
    elif projection_method == "min":
        img_data = np.min(img.data, axis=2, keepdims=True)
    elif projection_method == "std":
        img_data = np.std(img.data, axis=2, keepdims=True)
    else:
        img_data = img.data
        
    if drift_correct_channel > -1:
        output_img, shifts = drift_correct_xy_parallel(img_data, drift_correct_channel)
        OmeTiffWriter.save(output_img, output_tif_file_path, dim_order="TCZYX", physical_pixel_sizes=physical_pixel_sizes)
        np.save(output_shifts_file_path, shifts)

        metadata["Convert to tif"] = {
            "Drift correction": {
                "Method": "StackReg",
                "Drift_correct_channel": drift_correct_channel,
                "Shifts": os.path.basename(output_shifts_file_path),
            },
            "Projection": {
                "Method": projection_method,
            }
        }   
        with open(output_metadata_file_path, 'w') as f:
            yaml.dump(metadata, f)
    else:
        OmeTiffWriter.save(img_data, output_tif_file_path, dim_order="TCZYX", physical_pixel_sizes=physical_pixel_sizes)

        metadata["Convert to tif"] = {
            "Projection": {
                "Method": projection_method,
            }
        }   
        with open(output_metadata_file_path, 'w') as f:
            yaml.dump(metadata, f)

def process_folder(args: argparse.Namespace, parallel: bool = True) -> None:
    files_to_process = rp.get_files_to_process(args.input_file_or_folder, args.extension, args.search_subfolders)

    destination_folder = args.input_file_or_folder + "_tif"
    os.makedirs(destination_folder, exist_ok=True)

    def process_single_file(input_file_path):
        output_tif_file_path: str = rp.collapse_filename(input_file_path, args.input_file_or_folder, args.collapse_delimiter)
        output_tif_file_path: str = os.path.splitext(output_tif_file_path)[0] + args.output_file_name_extension + ".tif"
        output_tif_file_path: str = os.path.join(destination_folder, output_tif_file_path)

        if(parsed_args.multipoint_files):
            print(f"Processing multipoint file: {input_file_path}.")
            process_multipoint_file(input_file_path, output_tif_file_path, parsed_args.drift_correct_channel, use_parallel=True, projection_method=parsed_args.projection_method)
        else:
            img = rp.load_bioio(input_file_path)
            process_file(img=img,
                         input_file_path=input_file_path, 
                         output_tif_file_path=output_tif_file_path, 
                         drift_correct_channel=parsed_args.drift_correct_channel, 
                         projection_method=parsed_args.projection_method, 
                         multipoint_files=parsed_args.multipoint_files)

    if parallel:
        # Parallel processing for each file
        Parallel(n_jobs=-1)(delayed(process_single_file)(file) for file in tqdm(files_to_process, desc="Processing files", unit="file"))
    else:
        # Sequential processing for each file
        for input_file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
            process_single_file(input_file_path)


def main(parsed_args: argparse.Namespace, parallel: bool = True) -> None:
    
    if os.path.isfile(parsed_args.input_file_or_folder):
        print(f"Processing single file: {parsed_args.input_file_or_folder}")
        output_tif_file_path = os.path.splitext(parsed_args.input_file_or_folder)[0] + parsed_args.output_file_name_extension + ".tif"
        
        if(parsed_args.multipoint_files):
            print("Processing multipoint file.")
            process_multipoint_file(parsed_args.input_file_or_folder, output_tif_file_path, parsed_args.drift_correct_channel, use_parallel=parallel, projection_method=parsed_args.projection_method)
        else:        
            img = rp.load_bioio(parsed_args.input_file_or_folder)
            process_file(img = img,
                         input_file_path = parsed_args.input_file_or_folder, 
                         output_tif_file_path = output_tif_file_path, 
                         drift_correct_channel = parsed_args.drift_correct_channel, 
                         projection_method = parsed_args.projection_method, 
                         multipoint_files = parsed_args.multipoint_files)
        
    elif os.path.isdir(parsed_args.input_file_or_folder):
        print(f"Processing folder: {parsed_args.input_file_or_folder}")
        process_folder(parsed_args, parallel=parallel)
        
    else:
        print("Error: The specified path is neither a file nor a folder.")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a folder (or file) of images and convert them to TIF format with optional drift correction.")
    parser.add_argument("--input-file-or-folder", type=str, help="Path to the file or folder to be processed.")
    parser.add_argument("--extension", type=str, help="File extension to filter files in folder (Only applicable for directory input)")
    parser.add_argument("--search-subfolders", action="store_true", help="Include subfolders in the search for files (Only applicable for directory input)")
    parser.add_argument("--collapse-delimiter", type=str, default="__", help="Delimiter to collapse file paths (Only applicable for directory input)")
    parser.add_argument("--drift-correct-channel", type=int, default=-1, help="Channel for drift correction (default: -1, no correction)")
    parser.add_argument("--projection-method", type=str, default=None, help="Method for projection (options: max, sum, mean, median, min, std)")
    parser.add_argument("--output-file-name-extension", type=str, default="", help="Output file name extension (e.g., '_max', do not include '.tif')")
    parser.add_argument("--multipoint-files", action="store_true", help="Store the output in a multipoint file /series as separate_files (default: False)")
    parser.add_argument("--no-parallel", action="store_true", help="Do not use parallel processing")

    parsed_args = parser.parse_args()

    parallel = parsed_args.no_parallel == False # inverse


    parsed_args.projection_method
    main(parsed_args, parallel=parallel)
