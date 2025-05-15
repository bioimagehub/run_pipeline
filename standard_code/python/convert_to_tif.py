import numpy as np
from pystackreg import StackReg
from tqdm import tqdm
import argparse
from joblib import Parallel, delayed
import os
import yaml
from bioio.writers import OmeTiffWriter

# Local impots
import run_pipeline_helper_functions  as rp 
from extract_metadata import get_all_metadata

def drift_correct_xy_parallel(video: np.ndarray, drift_correct_channel: int = 0, use_parallel: bool = True) -> tuple[np.ndarray, np.ndarray]:
    T, C, Z, Y, X = video.shape    
    corrected_video = np.zeros_like(video)
    
    sr = StackReg(StackReg.RIGID_BODY)

    # Max-projection along Z for drift correction
    ref_stack = np.max(video[:, drift_correct_channel, :, :, :], axis=1)  # shape: (T, Y, X) axis=1 is now Z (after subset C)  -> max projection in z
    # print(ref_stack.shape)
    # Register max projections over time
    print("\nFinding shifts:")
    tmats = sr.register_stack(ref_stack, reference='previous', verbose=False, axis=0) # set to axis=0 -> Time

    # Parallel transform function
    if use_parallel:
        def apply_shift(t):
            local_sr = StackReg(StackReg.RIGID_BODY)  # Create separate instance to avoid shared state
            out = np.empty_like(video[t])
            for c in range(C):
                for z in range(Z):
                    out[c, z] = local_sr.transform(video[t, c, z], tmats[t])
            return out

        # Parallel processing across frames
        corrected_list = Parallel(n_jobs=-1)(
            delayed(apply_shift)(t) for t in tqdm(range(T), desc="Applying shifts")
        )
        corrected_video = np.stack(corrected_list)
    else:
            # Apply transformation to all channels and all Z-slices
        for t in tqdm(range(T), desc="Applying shifts", unit="frame"):
            for c in range(C):
                for z in range(Z):
                    corrected_video[t, c, z, :, :] = sr.transform(video[t, c, z, :, :], tmats[t])
    return corrected_video, tmats

def process_file(input_file_path:str, output_tif_file_path:str, drift_correct_channel:int = -1, use_parallel:bool = True, projection_method:str = None) -> None:
        # Define output file names
        input_metadata_file_path:str = os.path.splitext(input_file_path)[0] + "_metadata.yaml"
        
        output_metadata_file_path:str = os.path.splitext(output_tif_file_path)[0] + "_metadata.yaml"
        output_shifts_file_path:str = os.path.splitext(output_tif_file_path)[0] + "_shifts.npy"

        #  Load image and metadata
        img = rp.load_bioio(input_file_path)
        physical_pixel_sizes = img.physical_pixel_sizes if img.physical_pixel_sizes is not None else (None, None, None)

        # Check if metadata file exists and open/make
        if os.path.exists(input_metadata_file_path):
            with open(input_metadata_file_path, 'r') as f:
                metadata = yaml.safe_load(f)
        else:
            metadata = get_all_metadata(input_file_path)
            #TODO Prompt the user to fill in the channel names etc
            # This is the core metadata without drift correction info
            with open(input_metadata_file_path, 'w') as f:
                yaml.dump(metadata, f)
                
        # Perform projection if requested
        if projection_method == "max":
            img_data = np.max(img.data, axis=2, keepdims=True, dtype=img.data.dtype)
        elif projection_method == "sum":
            print("Warning: Using sum projection and with same dtype may cause pixels to saturate.")
            img_data = np.sum(img.data, axis=2, keepdims=True, dtype=img.data.dtype)
        elif projection_method == "mean":   
            img_data = np.mean(img.data, axis=2, keepdims=True, dtype=img.data.dtype)
        elif projection_method == "median":
            img_data = np.median(img.data, axis=2, keepdims=True, dtype=img.data.dtype)
        elif projection_method == "min":
            img_data = np.min(img.data, axis=2, keepdims=True, dtype=img.data.dtype)
        elif projection_method == "std":
            img_data = np.std(img.data, axis=2, keepdims=True, dtype=img.data.dtype)
        else:
            img_data = img.data
        
        # Apply drift correction if requested or just save the image as tif
        if drift_correct_channel > -1:

            output_img, shifts = drift_correct_xy_parallel(img_data, drift_correct_channel, use_parallel=use_parallel)  # Apply drift correction

            # Save the registered image
            OmeTiffWriter.save(output_img, output_tif_file_path, dim_order="TCZYX",physical_pixel_sizes=physical_pixel_sizes)
            np.save(output_shifts_file_path, shifts)

            # Save drift info to metadata
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
            # Save metadata to YAML file
            with open(output_metadata_file_path, 'w') as f:
                yaml.dump(metadata, f)

        else:
            # Save the image to the specified output file path
            OmeTiffWriter.save(img_data, output_tif_file_path, dim_order="TCZYX",physical_pixel_sizes=physical_pixel_sizes)

            # Save metadata to YAML file
                        # Save drift info to metadata
            metadata["Convert to tif"] = {
                "Projection": {
                    "Method": projection_method,
                }
            }   
            with open(output_metadata_file_path, 'w') as f:
                yaml.dump(metadata, f)

def process_folder(args: argparse.Namespace):
    
    # Find files to process
    files_to_process = rp.get_files_to_process(args.input_folder, args.extension, args.search_subfolders)

    # Make output folder
    destination_folder = args.input_folder + "_tif"
    os.makedirs(destination_folder, exist_ok=True)  # Create the output folder if it doesn't exist

    # Process each file
    for input_file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
        
        # Define output file name
        output_tif_file_path:str = rp.collapse_filename(input_file_path, args.input_folder, args.collapse_delimiter)
        output_tif_file_path:str = os.path.splitext(output_tif_file_path)[0] + args.output_file_name_extension + ".tif"
        output_tif_file_path:str = os.path.join(destination_folder, output_tif_file_path)
        
        # process file
        process_file(input_file_path, output_tif_file_path, args.drift_correct_channel, use_parallel = True, projection_method=args.projection_method)  # Process each file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a folder of images and convert them to TIF format with optional drift correction.")
    parser.add_argument("-p", "--input_folder", type=str, help="Path to the folder to be processed")
    parser.add_argument("-e", "--extension", type=str, help="File extension to filter files in folder")
    parser.add_argument("-R", "--search_subfolders", action="store_true", help="Search for files in subfolders")
    parser.add_argument("--collapse_delimiter", type=str, default="__", help="Delimiter used to collapse file paths")
    parser.add_argument("-drift_ch", "--drift_correct_channel", type=int, default=-1, help="Channel to use for drift correction (default: -1, no correction)")
    parser.add_argument("-pmt", "--projection_method" , type=str, default=None, help="Projection method to use (max, sum, mean, median, min, std)")
    parser.add_argument("-o", "--output_file_name_extension", type=str, default="", help="Output file name extension (default: None, E.g _max [do not include .tif])")

    args = parser.parse_args() 
    process_folder(args)

    


