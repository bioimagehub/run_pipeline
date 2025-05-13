import argparse
from bioio import BioImage
from tqdm import tqdm
import json
import numpy as np
from bioio.writers import OmeTiffWriter
import os
from joblib import Parallel, delayed
import yaml

# local imports
import run_pipeline_helper_functions  as rp  
from extract_metadata import get_metadata




def process_file(input_file_path: str, output_tif_file_path: str, merge_channels: str, paralell: bool = True) -> None:
    try:
        # Define output file names
        input_metadata_file_path:str = os.path.splitext(input_file_path)[0] + "_metadata.yaml"
        output_metadata_file_path:str = os.path.splitext(output_tif_file_path)[0] + "_metadata.yaml"  
        
        # Check if metadata file exists and open/make
        if os.path.exists(input_metadata_file_path):
            with open(input_metadata_file_path, 'r') as f:
                metadata = yaml.safe_load(f)
        else:
            metadata = get_metadata(img)
            #TODO Prompt the user to fill in the channel names etc
            # This is the core metadata without drift correction info
            with open(input_metadata_file_path, 'w') as f:
                yaml.dump(metadata, f)
        
        # Load image and metadata
        img = rp.load_bioio(input_file_path)
        physical_pixel_sizes = img.physical_pixel_sizes if img.physical_pixel_sizes is not None else (None, None, None)






        img_np = img.data  # TCZYX
        
        # Parse merge_channels string to list of channel groups
        channel_groups = json.loads(merge_channels)  # e.g. [[0,1], [2], [3,4]]
        channel_groups = [g if isinstance(g, list) else [g] for g in channel_groups] # Ensure all groups are lists


        output_channels = []

        for group in channel_groups:
            group = list(group)  # ensure it's a list in case it's a tuple
            if len(group) == 1:
                # Keep as-is
                output = img_np[:, group[0]:group[0]+1, :, :, :]
            else:
                # Sum-project along channel axis
                output = img_np[:, group, :, :, :].sum(axis=1, keepdims=True, dtype=img_np.dtype)
            output_channels.append(output)

        if os.path.exists(output_tif_file_path):
            os.remove(output_tif_file_path)

        # Concatenate all processed channel groups
        out_img = np.concatenate(output_channels, axis=1)  # TCZYX
        
        # Save output
        OmeTiffWriter.save(out_img, output_tif_file_path, dim_order="TCZYX",physical_pixel_sizes=physical_pixel_sizes)

        # Save metadata to YAML file
        metadata["Merge channels"] = {
            "Method": "Sum projection",
            "Merge_channels": merge_channels,
        }
        # Save metadata to YAML file
        with open(output_metadata_file_path, 'w') as f:
            yaml.dump(metadata, f)

    except Exception as e:
        print(f"Error processing {input_file_path}: {e}")
        return

def process_folder(args: argparse.Namespace, use_parallel = True) -> None:
        
    # Find files to process
    files_to_process = rp.get_files_to_process(args.input_folder, ".tif", search_subfolders=False)
    # files_to_process = [files_to_process[0]] # For debugging
    
    # Make output folder
    os.makedirs(args.output_folder, exist_ok=True)  # Create the output folder if it doesn't exist


    if use_parallel: # Process each file in parallel
        Parallel(n_jobs=-1)(
            delayed(process_file)(input_file_path,
                                  os.path.join(args.output_folder, os.path.basename(input_file_path)),
                                  args.merge_channels) 
            for input_file_path in tqdm(files_to_process, desc="Processing files", unit="file")
        )
    else: # Process each file sequentially        
        for input_file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
            # Define output file name
            output_tif_file_path:str = os.path.join(args.output_folder, os.path.basename(input_file_path))
            # process file
            try:
                process_file(input_file_path, output_tif_file_path, args.merge_channels)  # Process each file
            except Exception as e:
                print(f"Error processing {input_file_path}: {e}")
                continue



if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Process BioImage files.")
    parser.add_argument("-i", "--input_folder", type=str, required=True, help="Path to the input folder containing BioImage files")
    parser.add_argument("-o", "--output_folder", type=str, required=False, help="Path to save the processed files")
    #TODO add merge channels argument
    parser.add_argument("-m", "--merge_channels", type=str, required=True, help="E.g. '[[0,1,2,3], 4] to merge channels 0,1,2,3 and keep channel 4 and remove  >4")

    args = parser.parse_args()

    # Check if output folder path is provided, if not set default
    if args.output_folder is None:
        args.output_folder = os.path.join(args.input_folder, "_merged")

    # Process the folder
    process_folder(args, use_parallel = True)

