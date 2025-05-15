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


def drift_correct_xy_parallel(video: np.ndarray, drift_correct_channel: int = 0, use_parallel: bool = True) -> tuple[np.ndarray, np.ndarray]:
    T, C, Z, Y, X = video.shape    
    corrected_video = np.zeros_like(video)
    
    sr = StackReg(StackReg.RIGID_BODY)

    # Max-projection along Z for drift correction
    ref_stack = np.max(video[:, drift_correct_channel, :, :, :], axis=1)
    
    print("\nFinding shifts:")
    tmats = sr.register_stack(ref_stack, reference='previous', verbose=False, axis=0) 

    # Parallel transform function
    if use_parallel:
        def apply_shift(t):
            local_sr = StackReg(StackReg.RIGID_BODY)  
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
        for t in tqdm(range(T), desc="Applying shifts", unit="frame"):
            for c in range(C):
                for z in range(Z):
                    corrected_video[t, c, z, :, :] = sr.transform(video[t, c, z, :, :], tmats[t])
                    
    return corrected_video, tmats

def process_file(input_file_path: str, output_tif_file_path: str, drift_correct_channel: int = -1, use_parallel: bool = True, projection_method: str = None) -> None:
    input_metadata_file_path: str = os.path.splitext(input_file_path)[0] + "_metadata.yaml"
    output_metadata_file_path: str = os.path.splitext(output_tif_file_path)[0] + "_metadata.yaml"
    output_shifts_file_path: str = os.path.splitext(output_tif_file_path)[0] + "_shifts.npy"

    img = rp.load_bioio(input_file_path)
    physical_pixel_sizes = img.physical_pixel_sizes if img.physical_pixel_sizes is not None else (None, None, None)

    if os.path.exists(input_metadata_file_path):
        with open(input_metadata_file_path, 'r') as f:
            metadata = yaml.safe_load(f)
    else:
        metadata = get_all_metadata(input_file_path)
        with open(input_metadata_file_path, 'w') as f:
            yaml.dump(metadata, f)
                
    # Perform projection if requested
    if projection_method == "max":
        img_data = np.max(img.data, axis=2, keepdims=True)
    elif projection_method == "sum":
        print("Warning: Using sum projection and with same dtype may cause pixels to saturate.")
        img_data = np.sum(img.data, axis=2, keepdims=True)
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
        output_img, shifts = drift_correct_xy_parallel(img_data, drift_correct_channel, use_parallel=use_parallel)
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

def process_folder(args: argparse.Namespace):
    files_to_process = rp.get_files_to_process(args.input_file_or_folder, args.extension, args.search_subfolders)

    destination_folder = args.input_file_or_folder + "_tif"
    os.makedirs(destination_folder, exist_ok=True)

    for input_file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
        output_tif_file_path: str = rp.collapse_filename(input_file_path, args.input_file_or_folder, args.collapse_delimiter)
        output_tif_file_path: str = os.path.splitext(output_tif_file_path)[0] + args.output_file_name_extension + ".tif"
        output_tif_file_path: str = os.path.join(destination_folder, output_tif_file_path)
        
        process_file(input_file_path, output_tif_file_path, args.drift_correct_channel, use_parallel=True, projection_method=args.projection_method)

def main(parsed_args: argparse.Namespace):
    print("main")
    if os.path.isfile(parsed_args.input_file_or_folder):
        print(f"Processing single file: {parsed_args.input_file_or_folder}")
        output_tif_file_path = os.path.splitext(parsed_args.input_file_or_folder)[0] + parsed_args.output_file_name_extension + ".tif"
        process_file(parsed_args.input_file_or_folder, output_tif_file_path, parsed_args.drift_correct_channel, use_parallel=True, projection_method=parsed_args.projection_method)
        
    elif os.path.isdir(parsed_args.input_file_or_folder):
        print(f"Processing folder: {parsed_args.input_file_or_folder}")
        process_folder(parsed_args)
        
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

    parsed_args = parser.parse_args()



    parsed_args.projection_method
    main(parsed_args)
